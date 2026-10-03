"""
Phase 2D.3 Experiment 1B — Back-to-Back (is_b2b) incremental feature tests.
Tests the correctness, leakage safety, contamination safety, short-history handling,
and train/inference semantic consistency of the is_b2b feature.
"""
import os
import sys
import pytest
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.predict_ratings import construct_prediction_features


def _make_player_df(person_id, ratings, dates):
    """Helper: build a minimal player-game DataFrame with explicit dates."""
    n = len(ratings)
    assert len(dates) == n
    return pd.DataFrame({
        "personId": [person_id] * n,
        "playerName": [f"Player {person_id}"] * n,
        "teamId": [1610612737] * n,
        "gameId": [f"00225{person_id:04d}{i:02d}" for i in range(n)],
        "rating": ratings,
        "game_date": pd.to_datetime(dates),
    })


# ── Test 1: days_since_prev_game == 1 -> is_b2b == 1 ─────────────────────────

def test_is_b2b_positive_case():
    """
    If a player played yesterday (days_since_prev_game == 1),
    is_b2b must equal 1.
    """
    df = _make_player_df(8001, [5.0], ["2026-03-05"])
    feat = construct_prediction_features(df, target_date="2026-03-06")
    assert len(feat) == 1
    row = feat.iloc[0]
    assert row["days_since_prev_game"] == 1
    assert row["is_b2b"] == 1


# ── Test 2: days_since_prev_game != 1 -> is_b2b == 0 ─────────────────────────

def test_is_b2b_negative_case():
    """
    If a player has 2, 3, or more days of rest (days_since_prev_game != 1),
    is_b2b must equal 0.
    """
    # 2 days rest
    df2 = _make_player_df(8002, [4.0], ["2026-03-04"])
    feat2 = construct_prediction_features(df2, target_date="2026-03-06")
    assert feat2.iloc[0]["days_since_prev_game"] == 2
    assert feat2.iloc[0]["is_b2b"] == 0

    # 5 days rest
    df5 = _make_player_df(8003, [6.0], ["2026-03-01"])
    feat5 = construct_prediction_features(df5, target_date="2026-03-06")
    assert feat5.iloc[0]["days_since_prev_game"] == 5
    assert feat5.iloc[0]["is_b2b"] == 0


# ── Test 3: Short history / single prior game ─────────────────────────────────

def test_is_b2b_single_prior_game():
    """
    For a player with only 1 prior completed game:
    is_b2b is well-defined and accurately reflects whether target_date is 1 day later.
    """
    df_b2b = _make_player_df(8004, [3.0], ["2026-03-10"])
    feat_b2b = construct_prediction_features(df_b2b, target_date="2026-03-11")
    assert feat_b2b.iloc[0]["days_since_prev_game"] == 1
    assert feat_b2b.iloc[0]["is_b2b"] == 1

    df_rest = _make_player_df(8005, [3.0], ["2026-03-10"])
    feat_rest = construct_prediction_features(df_rest, target_date="2026-03-13")
    assert feat_rest.iloc[0]["days_since_prev_game"] == 3
    assert feat_rest.iloc[0]["is_b2b"] == 0


# ── Test 4: Cross-player independence ─────────────────────────────────────────

def test_is_b2b_cross_player_independence():
    """
    Player A on back-to-back must receive is_b2b = 1.
    Player B on 3 days rest must receive is_b2b = 0.
    Neither calculation affects the other.
    """
    df_p1 = _make_player_df(8006, [5.0], ["2026-03-05"])  # target 2026-03-06 -> B2B
    df_p2 = _make_player_df(8007, [7.0], ["2026-03-03"])  # target 2026-03-06 -> 3 days rest

    combined = pd.concat([df_p1, df_p2], ignore_index=True)
    feat = construct_prediction_features(combined, target_date="2026-03-06")
    assert len(feat) == 2

    p1_row = feat[feat["personId"] == 8006].iloc[0]
    p2_row = feat[feat["personId"] == 8007].iloc[0]

    assert p1_row["is_b2b"] == 1
    assert p2_row["is_b2b"] == 0


# ── Test 5: Historical training / inference semantic consistency ──────────────

def test_is_b2b_train_inference_consistency():
    """
    Training calculation ((days_since_prev_game == 1).astype(int)) and
    inference calculation must match identically for any game history.
    """
    dates = ["2026-03-01", "2026-03-02", "2026-03-05"]
    ratings = [2.0, 4.0, 6.0]

    df_raw = _make_player_df(8008, ratings, dates)
    df_raw["days_since_prev_game"] = (
        df_raw["game_date"] - df_raw.groupby("personId")["game_date"].shift(1)
    ).dt.days
    df_raw["is_b2b"] = (df_raw["days_since_prev_game"] == 1).astype(int)

    # Game 2 (2026-03-02) was played 1 day after Game 1 (2026-03-01) -> is_b2b = 1
    train_b2b_g2 = df_raw.iloc[1]["is_b2b"]
    assert train_b2b_g2 == 1

    # Game 3 (2026-03-05) was played 3 days after Game 2 (2026-03-02) -> is_b2b = 0
    train_b2b_g3 = df_raw.iloc[2]["is_b2b"]
    assert train_b2b_g3 == 0

    # Inference for game 2 predicting from game 1
    inf_g2 = construct_prediction_features(df_raw.iloc[:1], target_date="2026-03-02")
    assert inf_g2.iloc[0]["is_b2b"] == 1

    # Inference for game 3 predicting from games 1 & 2
    inf_g3 = construct_prediction_features(df_raw.iloc[:2], target_date="2026-03-05")
    assert inf_g3.iloc[0]["is_b2b"] == 0


# ── Test 6: No target-game leakage ───────────────────────────────────────────

def test_is_b2b_no_target_game_leakage():
    """
    Future games and target game ratings/stats must never enter is_b2b.
    """
    df = _make_player_df(8009, [1.0, 2.0], ["2026-03-01", "2026-03-04"])
    feat = construct_prediction_features(df, target_date="2026-03-05")
    row = feat.iloc[0]
    assert row["days_since_prev_game"] == 1
    assert row["is_b2b"] == 1


# ── Test 7: Seven-feature baseline remains unchanged ─────────────────────────

def test_is_b2b_preserves_seven_feature_baseline():
    """
    Adding is_b2b must not alter any of the 7 accepted features:
    last_game, last3_avg, last5_avg, last7_avg, games_played_before_target,
    expand_mean, days_since_prev_game.
    """
    dates = ["2026-03-01", "2026-03-02", "2026-03-03", "2026-03-04",
             "2026-03-05", "2026-03-06", "2026-03-07", "2026-03-08"]
    ratings = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    df = _make_player_df(8010, ratings, dates)

    feat = construct_prediction_features(df, target_date="2026-03-09")
    assert len(feat) == 1
    row = feat.iloc[0]

    assert row["last_game"] == pytest.approx(8.0)
    assert row["last3_avg"] == pytest.approx((6.0 + 7.0 + 8.0) / 3)
    assert row["last5_avg"] == pytest.approx((4.0 + 5.0 + 6.0 + 7.0 + 8.0) / 5)
    assert row["last7_avg"] == pytest.approx((2.0 + 3.0 + 4.0 + 5.0 + 6.0 + 7.0 + 8.0) / 7)
    assert row["games_played_before_target"] == 8
    assert row["expand_mean"] == pytest.approx(sum(ratings) / len(ratings))
    assert row["days_since_prev_game"] == 1
    assert row["is_b2b"] == 1

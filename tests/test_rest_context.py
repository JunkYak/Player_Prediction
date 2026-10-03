"""
Phase 2D.3 Experiment 1A — Rest Context (days_since_prev_game) tests.
Tests the correctness, leakage safety, contamination safety, short-history handling,
and train/inference semantic consistency of the days_since_prev_game feature.
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


# ── Test 1: Exact chronological rest calculation ─────────────────────────────

def test_days_since_prev_game_exact_calculation():
    """
    If a player's last completed game was 2026-03-01 and the upcoming target
    game is 2026-03-04:
    days_since_prev_game must equal 3.
    """
    df = _make_player_df(7001, [5.0], ["2026-03-01"])
    feat = construct_prediction_features(df, target_date="2026-03-04")
    assert len(feat) == 1
    row = feat.iloc[0]
    assert row["days_since_prev_game"] == 3


# ── Test 2: Back-to-back calculation (1 day rest) ────────────────────────────

def test_days_since_prev_game_back_to_back():
    """
    If a player's last completed game was 2026-03-05 and the upcoming target
    game is 2026-03-06:
    days_since_prev_game must equal 1 (back-to-back).
    """
    df = _make_player_df(7002, [4.0, 6.0], ["2026-03-03", "2026-03-05"])
    feat = construct_prediction_features(df, target_date="2026-03-06")
    assert len(feat) == 1
    row = feat.iloc[0]
    assert row["days_since_prev_game"] == 1


# ── Test 3: Multiple completed games uses the MOST RECENT game ───────────────

def test_days_since_prev_game_uses_most_recent_game():
    """
    For a player with games on [2026-03-01, 2026-03-05, 2026-03-08] and target
    game on 2026-03-10:
    days_since_prev_game must equal 2 (2026-03-10 - 2026-03-08),
    NOT any older game date.
    """
    df = _make_player_df(7003, [1.0, 2.0, 3.0], ["2026-03-01", "2026-03-05", "2026-03-08"])
    feat = construct_prediction_features(df, target_date="2026-03-10")
    assert len(feat) == 1
    row = feat.iloc[0]
    assert row["days_since_prev_game"] == 2


# ── Test 4: Target-game leakage prevention ───────────────────────────────────

def test_days_since_prev_game_no_target_leakage():
    """
    The target game's own date must only be used as the destination reference date.
    A hypothetical future game date or target rating must never leak into the history.
    """
    history_dates = ["2026-03-01", "2026-03-04"]
    history_ratings = [3.0, 5.0]
    df = _make_player_df(7004, history_ratings, history_dates)

    target_date = "2026-03-07"
    feat = construct_prediction_features(df, target_date=target_date)
    assert len(feat) == 1
    row = feat.iloc[0]
    assert row["days_since_prev_game"] == 3  # 2026-03-07 - 2026-03-04


# ── Test 5: No cross-player contamination ────────────────────────────────────

def test_days_since_prev_game_no_cross_player_contamination():
    """
    Each player's rest must be computed strictly from their own most recent game date.
    Player A last played 2026-03-02 -> rest before 2026-03-05 = 3 days.
    Player B last played 2026-03-04 -> rest before 2026-03-05 = 1 day.
    """
    df_p1 = _make_player_df(7005, [2.0], ["2026-03-02"])
    df_p2 = _make_player_df(7006, [7.0], ["2026-03-04"])

    combined = pd.concat([df_p1, df_p2], ignore_index=True)
    feat = construct_prediction_features(combined, target_date="2026-03-05")
    assert len(feat) == 2

    p1_row = feat[feat["personId"] == 7005].iloc[0]
    p2_row = feat[feat["personId"] == 7006].iloc[0]

    assert p1_row["days_since_prev_game"] == 3
    assert p2_row["days_since_prev_game"] == 1


# ── Test 6: Short-history single-game player ─────────────────────────────────

def test_days_since_prev_game_single_prior_game():
    """
    A player with exactly one historical game on 2026-03-01 predicting for 2026-03-03:
    days_since_prev_game is well-defined and equals 2.
    No NaN, no imputation required.
    """
    df = _make_player_df(7007, [4.5], ["2026-03-01"])
    feat = construct_prediction_features(df, target_date="2026-03-03")
    assert len(feat) == 1
    row = feat.iloc[0]
    assert row["days_since_prev_game"] == 2
    assert not pd.isna(row["days_since_prev_game"])


# ── Test 7: Train/Inference semantic consistency ─────────────────────────────

def test_days_since_prev_game_train_inference_consistency():
    """
    Verify that build_features.py logic (target_game_date - prev_game_date)
    and predict_ratings.py logic (target_date - last_game_date) yield identical
    values for the same sequence of dates.
    """
    dates = ["2026-03-01", "2026-03-03", "2026-03-07"]
    ratings = [2.0, 4.0, 6.0]

    # Training logic on game 3 (date 2026-03-07):
    # rest before game 3 = 2026-03-07 - 2026-03-03 = 4 days
    df_raw = _make_player_df(7008, ratings, dates)
    df_raw["days_since_prev_game"] = (
        df_raw["game_date"] - df_raw.groupby("personId")["game_date"].shift(1)
    ).dt.days
    train_rest_val = df_raw.iloc[2]["days_since_prev_game"]

    # Inference logic when history has games 1 & 2, and target is game 3:
    df_history = df_raw.iloc[:2].copy()
    feat = construct_prediction_features(df_history, target_date="2026-03-07")
    inf_rest_val = feat.iloc[0]["days_since_prev_game"]

    assert train_rest_val == 4
    assert inf_rest_val == 4
    assert train_rest_val == inf_rest_val


# ── Test 8: Existing six features remain unchanged ───────────────────────────

def test_days_since_prev_game_preserves_existing_features():
    """
    Adding days_since_prev_game must not alter any of the six accepted features:
    last_game, last3_avg, last5_avg, last7_avg, games_played_before_target, expand_mean.
    """
    dates = ["2026-03-01", "2026-03-02", "2026-03-03", "2026-03-04",
             "2026-03-05", "2026-03-06", "2026-03-07", "2026-03-08"]
    ratings = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    df = _make_player_df(7009, ratings, dates)

    feat = construct_prediction_features(df, target_date="2026-03-10")
    assert len(feat) == 1
    row = feat.iloc[0]

    assert row["last_game"] == pytest.approx(8.0)
    assert row["last3_avg"] == pytest.approx((6.0 + 7.0 + 8.0) / 3)
    assert row["last5_avg"] == pytest.approx((4.0 + 5.0 + 6.0 + 7.0 + 8.0) / 5)
    assert row["last7_avg"] == pytest.approx((2.0 + 3.0 + 4.0 + 5.0 + 6.0 + 7.0 + 8.0) / 7)
    assert row["games_played_before_target"] == 8
    assert row["expand_mean"] == pytest.approx(sum(ratings) / len(ratings))
    assert row["days_since_prev_game"] == 2  # 2026-03-10 - 2026-03-08

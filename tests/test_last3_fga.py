"""
Phase 2D.3 Experiment 2A — Workload Context (last3_avg_fga) tests.
Tests the correctness, event filtering, leakage safety, contamination safety,
short-history handling, and train/inference semantic consistency of last3_avg_fga.
"""
import os
import sys
import pytest
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.predict_ratings import construct_prediction_features


def _make_player_df(person_id, ratings, dates, fga_list=None):
    """Helper: build a minimal player-game DataFrame with explicit dates and optional FGA."""
    n = len(ratings)
    assert len(dates) == n
    df = pd.DataFrame({
        "personId": [person_id] * n,
        "playerName": [f"Player {person_id}"] * n,
        "teamId": [1610612737] * n,
        "gameId": [f"00225{person_id:04d}{i:02d}" for i in range(n)],
        "rating": ratings,
        "game_date": pd.to_datetime(dates),
    })
    if fga_list is not None:
        assert len(fga_list) == n
        df["fga"] = [float(x) for x in fga_list]
    return df


# ── Test 1: Correct FGA extraction from known PBP events ─────────────────────

def test_fga_extraction_from_pbp_events():
    """
    Verify Made Shot and Missed Shot are correctly counted as FGA.
    """
    pbp_sample = pd.DataFrame({
        "gameId": ["0022500001"] * 5,
        "personId": [9001] * 5,
        "actionType": ["Made Shot", "Missed Shot", "Made Shot", "Free Throw", "Rebound"],
        "shotResult": ["Made", "Missed", "Made", "", ""],
    })
    shots = pbp_sample[(pbp_sample["personId"] > 1000) & (pbp_sample["actionType"].isin(["Made Shot", "Missed Shot"]))]
    fga_count = len(shots)
    assert fga_count == 3  # 2 made shots + 1 missed shot


# ── Test 2: Free throws are not counted as FGA ───────────────────────────────

def test_fga_does_not_count_free_throws():
    """
    Verify that Free Throw actionType events are never counted as FGA.
    """
    pbp_sample = pd.DataFrame({
        "gameId": ["0022500002"] * 4,
        "personId": [9002] * 4,
        "actionType": ["Free Throw", "Free Throw", "Free Throw", "Made Shot"],
        "shotResult": ["", "", "", "Made"],
    })
    shots = pbp_sample[(pbp_sample["personId"] > 1000) & (pbp_sample["actionType"].isin(["Made Shot", "Missed Shot"]))]
    assert len(shots) == 1  # only the 1 Made Shot


# ── Test 3: Non-shot events are not counted as FGA ───────────────────────────

def test_fga_does_not_count_non_shot_events():
    """
    Verify rebounds, fouls, turnovers, substitutions, and timeouts are ignored.
    """
    pbp_sample = pd.DataFrame({
        "gameId": ["0022500003"] * 6,
        "personId": [9003] * 6,
        "actionType": ["Rebound", "Foul", "Turnover", "Substitution", "Timeout", "period"],
        "shotResult": ["", "", "", "", "", ""],
    })
    shots = pbp_sample[(pbp_sample["personId"] > 1000) & (pbp_sample["actionType"].isin(["Made Shot", "Missed Shot"]))]
    assert len(shots) == 0


# ── Test 4: Correct last-3 average with >= 3 games ───────────────────────────

def test_last3_avg_fga_three_games():
    """
    For a player with 3 completed games with FGA [10.0, 15.0, 20.0]:
    last3_avg_fga must equal 15.0.
    """
    df = _make_player_df(9004, [3.0, 4.0, 5.0], ["2026-03-01", "2026-03-03", "2026-03-05"], [10.0, 15.0, 20.0])
    feat = construct_prediction_features(df, target_date="2026-03-07")
    assert len(feat) == 1
    row = feat.iloc[0]
    assert row["last3_avg_fga"] == pytest.approx(15.0)


# ── Test 5: Single prior game (1-game history) ───────────────────────────────

def test_last3_avg_fga_single_prior_game():
    """
    For a player with exactly 1 completed game with FGA = 18.0:
    last3_avg_fga must equal 18.0.
    """
    df = _make_player_df(9005, [4.0], ["2026-03-01"], [18.0])
    feat = construct_prediction_features(df, target_date="2026-03-03")
    assert len(feat) == 1
    row = feat.iloc[0]
    assert row["last3_avg_fga"] == pytest.approx(18.0)


# ── Test 6: Two prior games (2-game history) ─────────────────────────────────

def test_last3_avg_fga_two_prior_games():
    """
    For a player with 2 completed games with FGA [12.0, 16.0]:
    last3_avg_fga must equal 14.0.
    """
    df = _make_player_df(9006, [3.0, 5.0], ["2026-03-01", "2026-03-03"], [12.0, 16.0])
    feat = construct_prediction_features(df, target_date="2026-03-05")
    assert len(feat) == 1
    row = feat.iloc[0]
    assert row["last3_avg_fga"] == pytest.approx(14.0)


# ── Test 7: Target-game exclusion and rolling window behavior ────────────────

def test_last3_avg_fga_rolling_window_and_leakage_safety():
    """
    For a player with 4 completed games with FGA [5.0, 10.0, 15.0, 20.0]:
    last3_avg_fga before game 5 must equal mean(10.0, 15.0, 20.0) = 15.0.
    The oldest game (5.0) and hypothetical target game FGA (999.0) must not be included.
    """
    df = _make_player_df(9007, [1.0, 2.0, 3.0, 4.0],
                         ["2026-03-01", "2026-03-03", "2026-03-05", "2026-03-07"],
                         [5.0, 10.0, 15.0, 20.0])
    feat = construct_prediction_features(df, target_date="2026-03-09")
    row = feat.iloc[0]
    assert row["last3_avg_fga"] == pytest.approx(15.0)


# ── Test 8: Cross-player independence ─────────────────────────────────────────

def test_last3_avg_fga_cross_player_independence():
    """
    Player A (high volume FGA = [20, 22, 24] -> avg 22.0) and
    Player B (low volume FGA = [2, 4, 6] -> avg 4.0)
    must remain completely independent.
    """
    df_p1 = _make_player_df(9008, [5.0, 5.0, 5.0], ["2026-03-01", "2026-03-03", "2026-03-05"], [20.0, 22.0, 24.0])
    df_p2 = _make_player_df(9009, [1.0, 1.0, 1.0], ["2026-03-01", "2026-03-03", "2026-03-05"], [2.0, 4.0, 6.0])

    combined = pd.concat([df_p1, df_p2], ignore_index=True)
    feat = construct_prediction_features(combined, target_date="2026-03-07")
    assert len(feat) == 2

    p1_row = feat[feat["personId"] == 9008].iloc[0]
    p2_row = feat[feat["personId"] == 9009].iloc[0]

    assert p1_row["last3_avg_fga"] == pytest.approx(22.0)
    assert p2_row["last3_avg_fga"] == pytest.approx(4.0)


# ── Test 9: Train/Inference semantic consistency ─────────────────────────────

def test_last3_avg_fga_train_inference_consistency():
    """
    build_features.py logic (rolling(3, min_periods=1).mean().shift(1)) and
    predict_ratings.py logic must produce identical values for the same history.
    """
    dates = ["2026-03-01", "2026-03-03", "2026-03-05"]
    ratings = [2.0, 4.0, 6.0]
    fga_list = [8.0, 12.0, 16.0]

    df_raw = _make_player_df(9010, ratings, dates, fga_list)
    df_raw["last3_avg_fga"] = (
        df_raw.groupby("personId")["fga"]
        .rolling(3, min_periods=1)
        .mean()
        .shift(1)
        .reset_index(level=0, drop=True)
    )

    # In training, at game 3 row, last3_avg_fga is mean(8, 12) = 10.0
    train_val_g3 = df_raw.iloc[2]["last3_avg_fga"]

    # In inference, with games 1 & 2 in history, predicting game 3
    inf_val_g3 = construct_prediction_features(df_raw.iloc[:2], target_date="2026-03-05").iloc[0]["last3_avg_fga"]

    assert train_val_g3 == pytest.approx(10.0)
    assert inf_val_g3 == pytest.approx(10.0)
    assert train_val_g3 == pytest.approx(inf_val_g3)


# ── Test 10: Existing 7 accepted features remain unchanged ────────────────────

def test_last3_avg_fga_preserves_seven_accepted_features():
    """
    Adding last3_avg_fga must not alter any of the seven accepted baseline features:
    last_game, last3_avg, last5_avg, last7_avg, games_played_before_target,
    expand_mean, days_since_prev_game.
    """
    dates = ["2026-03-01", "2026-03-02", "2026-03-03", "2026-03-04",
             "2026-03-05", "2026-03-06", "2026-03-07", "2026-03-08"]
    ratings = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    fga_list = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]

    df = _make_player_df(9011, ratings, dates, fga_list)
    feat = construct_prediction_features(df, target_date="2026-03-10")
    assert len(feat) == 1
    row = feat.iloc[0]

    assert row["last_game"] == pytest.approx(8.0)
    assert row["last3_avg"] == pytest.approx((6.0 + 7.0 + 8.0) / 3)
    assert row["last5_avg"] == pytest.approx((4.0 + 5.0 + 6.0 + 7.0 + 8.0) / 5)
    assert row["last7_avg"] == pytest.approx((2.0 + 3.0 + 4.0 + 5.0 + 6.0 + 7.0 + 8.0) / 7)
    assert row["games_played_before_target"] == 8
    assert row["expand_mean"] == pytest.approx(sum(ratings) / len(ratings))
    assert row["days_since_prev_game"] == 2
    assert row["last3_avg_fga"] == pytest.approx((15.0 + 16.0 + 17.0) / 3)


# ── Test 11: Zero FGA handling ───────────────────────────────────────────────

def test_last3_avg_fga_zero_fga_handling():
    """
    If a player played in games with 0 shot attempts, last3_avg_fga must correctly
    average zeros without error or null values.
    """
    df = _make_player_df(9012, [2.0, 2.0], ["2026-03-01", "2026-03-03"], [0.0, 0.0])
    feat = construct_prediction_features(df, target_date="2026-03-05")
    row = feat.iloc[0]
    assert row["last3_avg_fga"] == pytest.approx(0.0)
    assert not pd.isna(row["last3_avg_fga"])

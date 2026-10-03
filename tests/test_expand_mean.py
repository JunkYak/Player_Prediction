"""
Phase 2D.2C Experiment 1 — expand_mean feature tests.
Tests the correctness, leakage safety, contamination safety, and
train/inference semantic consistency of the expand_mean feature.
"""
import os
import sys
import pytest
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.predict_ratings import construct_prediction_features


def _make_player_df(person_id, ratings, start_date="2026-03-01"):
    """Helper: build a minimal player-game DataFrame."""
    n = len(ratings)
    return pd.DataFrame({
        "personId": [person_id] * n,
        "playerName": [f"Player {person_id}"] * n,
        "teamId": [1610612737] * n,
        "gameId": [f"00225{person_id:04d}{i:02d}" for i in range(n)],
        "rating": ratings,
        "game_date": pd.date_range(start_date, periods=n, freq="D"),
    })


# ── Test 1: Single-player correctness ────────────────────────────────────────

def test_expand_mean_single_player_correct():
    """
    expand_mean for a player with N completed games before the target must equal
    the mean of all N prior ratings.
    Concretely: 5 completed games → expand_mean = mean(g1..g5) = 3.0.
    """
    ratings = [1.0, 2.0, 3.0, 4.0, 5.0]
    df = _make_player_df(5001, ratings)
    feat = construct_prediction_features(df, target_date="2026-03-06")
    assert len(feat) == 1
    row = feat.iloc[0]
    expected = sum(ratings) / len(ratings)  # 3.0
    assert row["expand_mean"] == pytest.approx(expected)


# ── Test 2: Multiple non-trivial completed games ──────────────────────────────

def test_expand_mean_multiple_completed_games():
    """
    Verify expand_mean matches the exact arithmetic mean over all prior games
    for the example ratings from the Phase 2D.2C spec.
    """
    ratings = [10.418, 7.036, 6.342, 6.546, 7.884, 7.329, 1.404]
    df = _make_player_df(5002, ratings)
    feat = construct_prediction_features(df, target_date="2026-03-08")
    assert len(feat) == 1
    row = feat.iloc[0]
    expected = sum(ratings) / len(ratings)
    assert row["expand_mean"] == pytest.approx(expected, rel=1e-9)


# ── Test 3: No target-game leakage ───────────────────────────────────────────

def test_expand_mean_no_target_leakage():
    """
    The target game's rating must NEVER enter expand_mean.
    Supply 7 completed games; expand_mean must equal the mean of those 7 only.
    A hypothetical future rating of 99.0 must not influence the result.
    """
    completed = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    future_rating = 99.0  # must not appear

    df = _make_player_df(5003, completed)
    feat = construct_prediction_features(df, target_date="2026-03-08")
    assert len(feat) == 1
    row = feat.iloc[0]
    expected = sum(completed) / len(completed)
    assert row["expand_mean"] == pytest.approx(expected)
    # Confirm future_rating would materially change the result if it had leaked
    assert abs(expected - future_rating) > 1.0
    assert row["expand_mean"] != pytest.approx(future_rating)


# ── Test 4: No cross-player contamination ────────────────────────────────────

def test_expand_mean_no_cross_player_contamination():
    """
    expand_mean for each player must be computed independently.
    Two players with different rating histories must produce different
    expand_means, each reflecting only their own history.
    """
    df_p1 = _make_player_df(5004, [1.0, 1.0, 1.0, 1.0])  # mean = 1.0
    df_p2 = _make_player_df(5005, [9.0, 9.0, 9.0, 9.0])  # mean = 9.0

    combined = pd.concat([df_p1, df_p2], ignore_index=True)
    feat = construct_prediction_features(combined, target_date="2026-03-05")
    assert len(feat) == 2

    feat_p1 = feat[feat["personId"] == 5004].iloc[0]
    feat_p2 = feat[feat["personId"] == 5005].iloc[0]

    assert feat_p1["expand_mean"] == pytest.approx(1.0)
    assert feat_p2["expand_mean"] == pytest.approx(9.0)
    # Confirm no bleed-over
    assert feat_p1["expand_mean"] != pytest.approx(9.0)
    assert feat_p2["expand_mean"] != pytest.approx(1.0)


# ── Test 5: Train/inference semantic consistency ──────────────────────────────

def test_expand_mean_train_inference_semantic_consistency():
    """
    Training semantics (build_features.py: expanding().mean().shift(1)) and
    inference semantics (construct_prediction_features: mean of all ratings)
    must produce identical values for the same player-game history.

    For a player with history [2, 4, 6] predicting game 4:
      Training expand_mean at game-4 row = mean(g1,g2,g3) = 4.0
      Inference expand_mean                = mean([2, 4, 6]) = 4.0
    Both paths must return 4.0.
    """
    ratings = [2.0, 4.0, 6.0]
    df = _make_player_df(5006, ratings)

    # Inference path
    feat = construct_prediction_features(df, target_date="2026-03-04")
    assert len(feat) == 1
    row = feat.iloc[0]
    inference_expand_mean = row["expand_mean"]

    # Training path value: mean(2, 4, 6) = 4.0
    training_expand_mean = sum(ratings) / len(ratings)

    assert inference_expand_mean == pytest.approx(training_expand_mean)
    assert inference_expand_mean == pytest.approx(4.0)


# ── Test 6: One prior game ────────────────────────────────────────────────────

def test_expand_mean_one_prior_game():
    """
    A player with exactly one completed game: expand_mean = that game's rating.
    The mean of a single value equals the value itself.
    """
    df = _make_player_df(5007, [3.75])
    feat = construct_prediction_features(df, target_date="2026-03-02")
    assert len(feat) == 1
    row = feat.iloc[0]
    assert row["expand_mean"] == pytest.approx(3.75)
    assert row["games_played_before_target"] == 1


# ── Test 7: Existing features unchanged ──────────────────────────────────────

def test_expand_mean_does_not_change_existing_features():
    """
    Adding expand_mean must not alter the values of any existing feature.
    Verify last_game, last3_avg, last5_avg, last7_avg, and
    games_played_before_target retain their Phase 2D.2B semantics.
    """
    ratings = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    df = _make_player_df(5008, ratings)
    feat = construct_prediction_features(df, target_date="2026-03-09")
    assert len(feat) == 1
    row = feat.iloc[0]

    # Existing features - unchanged from Phase 2D.2B
    assert row["last_game"] == pytest.approx(8.0)
    assert row["last3_avg"] == pytest.approx((6.0 + 7.0 + 8.0) / 3)
    assert row["last5_avg"] == pytest.approx((4.0 + 5.0 + 6.0 + 7.0 + 8.0) / 5)
    assert row["last7_avg"] == pytest.approx((2.0 + 3.0 + 4.0 + 5.0 + 6.0 + 7.0 + 8.0) / 7)
    assert row["games_played_before_target"] == 8

    # New feature: mean of all 8 completed games = 4.5
    expected_expand = sum(ratings) / len(ratings)
    assert row["expand_mean"] == pytest.approx(expected_expand)
    assert row["expand_mean"] == pytest.approx(4.5)

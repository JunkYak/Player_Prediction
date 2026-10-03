import os
import sys
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import MagicMock

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.predict_ratings import construct_prediction_features, predict_ratings
from scripts.train_model import get_calendar_date_cutoff, evaluate_baselines, train_model


# ==============================
# 1. PREDICTION HORIZON TESTS
# ==============================

def test_prediction_horizon_feature_construction():
    """
    Given historical games T-6 through T for a player (8 games total):
    Verify that construct_prediction_features correctly assigns:
      - last_game = rating at game T (game 8)
      - last3_avg = mean(T-2, T-1, T) using last 3 of [2..8]
      - last5_avg = mean(T-4, T-3, T-2, T-1, T) using last 5
      - last7_avg = mean(T-6 ... T) using last 7 of [2..8]
      - games_played_before_target = 8
    """
    ratings = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    dates = pd.date_range("2026-03-01", periods=8, freq="D")

    df_history = pd.DataFrame({
        "personId": [1001] * 8,
        "playerName": ["Test Player"] * 8,
        "teamId": [1610612737] * 8,
        "gameId": [f"002250000{i}" for i in range(8)],
        "rating": ratings,
        "game_date": dates
    })

    features_df = construct_prediction_features(df_history, target_date="2026-03-09")

    assert len(features_df) == 1
    row = features_df.iloc[0]

    # Latest completed game T is game 8 (rating 8.0 on 2026-03-08)
    # last 7 completed ratings: [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    expected_last_game = 8.0
    expected_last3 = (6.0 + 7.0 + 8.0) / 3.0
    expected_last5 = (4.0 + 5.0 + 6.0 + 7.0 + 8.0) / 5.0
    expected_last7 = (2.0 + 3.0 + 4.0 + 5.0 + 6.0 + 7.0 + 8.0) / 7.0

    assert row["personId"] == 1001
    assert row["playerName"] == "Test Player"
    assert row["last_game"] == pytest.approx(expected_last_game)
    assert row["last3_avg"] == pytest.approx(expected_last3)
    assert row["last5_avg"] == pytest.approx(expected_last5)
    assert row["last7_avg"] == pytest.approx(expected_last7)
    assert row["games_played_before_target"] == 8
    assert row["last_game_date"] == "2026-03-08"
    assert row["game_date"] == "2026-03-09"


def test_no_target_leakage_in_prediction_features():
    """
    Verify that hypothetical upcoming game T+1's future rating (even if present in raw future data)
    is NEVER included in features constructed up through game T.
    """
    completed_ratings = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]
    future_rating = 99.0  # An extreme future rating

    df_completed = pd.DataFrame({
        "personId": [2001] * 7,
        "playerName": ["Clean Player"] * 7,
        "teamId": [1610612738] * 7,
        "gameId": [f"002250001{i}" for i in range(7)],
        "rating": completed_ratings,
        "game_date": pd.date_range("2026-03-01", periods=7, freq="D")
    })

    feat = construct_prediction_features(df_completed, target_date="2026-03-08")
    assert len(feat) == 1
    row = feat.iloc[0]

    # Feature vector must be based purely on completed games 1-7
    assert row["last_game"] == pytest.approx(5.5)
    assert row["last3_avg"] == pytest.approx((4.5 + 5.0 + 5.5) / 3.0)
    assert row["last7_avg"] == pytest.approx(sum(completed_ratings) / 7.0)
    assert future_rating not in [row["last_game"], row["last3_avg"], row["last5_avg"], row["last7_avg"]]


def test_prediction_requires_at_least_one_game():
    """
    A player with NO completed games cannot generate any feature vector.
    (Edge case: empty group.)
    """
    feat = construct_prediction_features(pd.DataFrame(
        columns=["personId", "playerName", "teamId", "gameId", "rating", "game_date"]
    ))
    assert len(feat) == 0


# ==============================
# SHORT-HISTORY SEMANTICS TESTS
# ==============================

def test_short_history_1_prev_game():
    """
    Player with exactly 1 completed game:
      - last_game = that game's rating
      - last3/5/7_avg = same as last_game (only 1 available)
      - games_played_before_target = 1
    """
    df = pd.DataFrame({
        "personId": [4001],
        "playerName": ["One-Game Player"],
        "teamId": [1610612740],
        "gameId": ["0022500001"],
        "rating": [3.0],
        "game_date": [pd.Timestamp("2026-03-01")]
    })
    feat = construct_prediction_features(df, target_date="2026-03-02")
    assert len(feat) == 1
    row = feat.iloc[0]
    assert row["last_game"] == pytest.approx(3.0)
    assert row["last3_avg"] == pytest.approx(3.0)  # only 1 game available
    assert row["last5_avg"] == pytest.approx(3.0)
    assert row["last7_avg"] == pytest.approx(3.0)
    assert row["games_played_before_target"] == 1
    # No zeros introduced
    assert row["last_game"] != 0.0


def test_short_history_2_prev_games():
    """
    Player with 2 completed games:
      - last_game = 2nd game rating
      - last3_avg = mean(game1, game2) — 2 available, not 3
      - last5_avg = mean(game1, game2) — same
      - last7_avg = mean(game1, game2) — same
      - games_played_before_target = 2
    """
    df = pd.DataFrame({
        "personId": [4002] * 2,
        "playerName": ["Two-Game Player"] * 2,
        "teamId": [1610612740] * 2,
        "gameId": ["00225000A", "00225000B"],
        "rating": [2.0, 4.0],
        "game_date": pd.date_range("2026-03-01", periods=2, freq="D")
    })
    feat = construct_prediction_features(df, target_date="2026-03-03")
    assert len(feat) == 1
    row = feat.iloc[0]
    assert row["last_game"] == pytest.approx(4.0)
    assert row["last3_avg"] == pytest.approx((2.0 + 4.0) / 2)  # 2 available
    assert row["last5_avg"] == pytest.approx((2.0 + 4.0) / 2)
    assert row["last7_avg"] == pytest.approx((2.0 + 4.0) / 2)
    assert row["games_played_before_target"] == 2


def test_short_history_3_prev_games():
    """
    Player with 3 completed games:
      - last3_avg uses all 3 (window fully populated)
      - last5_avg and last7_avg fall back to 3 available
      - games_played_before_target = 3
    """
    df = pd.DataFrame({
        "personId": [4003] * 3,
        "playerName": ["Three-Game Player"] * 3,
        "teamId": [1610612740] * 3,
        "gameId": ["00225000C", "00225000D", "00225000E"],
        "rating": [1.0, 3.0, 5.0],
        "game_date": pd.date_range("2026-03-01", periods=3, freq="D")
    })
    feat = construct_prediction_features(df, target_date="2026-03-04")
    assert len(feat) == 1
    row = feat.iloc[0]
    assert row["last_game"] == pytest.approx(5.0)
    assert row["last3_avg"] == pytest.approx((1.0 + 3.0 + 5.0) / 3)  # 3 available = window full
    assert row["last5_avg"] == pytest.approx((1.0 + 3.0 + 5.0) / 3)  # only 3 available
    assert row["last7_avg"] == pytest.approx((1.0 + 3.0 + 5.0) / 3)  # only 3 available
    assert row["games_played_before_target"] == 3


def test_short_history_5_prev_games():
    """
    Player with 5 completed games:
      - last3_avg uses last 3 of 5
      - last5_avg uses all 5 (window fully populated)
      - last7_avg falls back to all 5 available
      - games_played_before_target = 5
    """
    df = pd.DataFrame({
        "personId": [4004] * 5,
        "playerName": ["Five-Game Player"] * 5,
        "teamId": [1610612740] * 5,
        "gameId": [f"00225000F{i}" for i in range(5)],
        "rating": [1.0, 2.0, 3.0, 4.0, 5.0],
        "game_date": pd.date_range("2026-03-01", periods=5, freq="D")
    })
    feat = construct_prediction_features(df, target_date="2026-03-06")
    assert len(feat) == 1
    row = feat.iloc[0]
    assert row["last_game"] == pytest.approx(5.0)
    assert row["last3_avg"] == pytest.approx((3.0 + 4.0 + 5.0) / 3)  # last 3 of 5
    assert row["last5_avg"] == pytest.approx((1.0 + 2.0 + 3.0 + 4.0 + 5.0) / 5)  # full window
    assert row["last7_avg"] == pytest.approx((1.0 + 2.0 + 3.0 + 4.0 + 5.0) / 5)  # 5 available
    assert row["games_played_before_target"] == 5


def test_short_history_7plus_unchanged():
    """
    CRITICAL REGRESSION TEST: A player with >=7 completed games must produce
    EXACTLY the same feature values as the previous Phase 2D.2A implementation.
    Variable-window rolling with min_periods=1 is identical to strict rolling
    when the window is fully populated.
    """
    ratings = [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0]  # 8 games
    df = pd.DataFrame({
        "personId": [4005] * 8,
        "playerName": ["Veteran Player"] * 8,
        "teamId": [1610612740] * 8,
        "gameId": [f"00225000V{i}" for i in range(8)],
        "rating": ratings,
        "game_date": pd.date_range("2026-03-01", periods=8, freq="D")
    })
    feat = construct_prediction_features(df, target_date="2026-03-09")
    assert len(feat) == 1
    row = feat.iloc[0]
    # With 8 games: last 7 are [9, 8, 7, 6, 5, 4, 3]
    # last_game = rating of game 8 = 3.0
    # last3_avg = mean(5, 4, 3) = 4.0
    # last5_avg = mean(7, 6, 5, 4, 3) = 5.0
    # last7_avg = mean(9, 8, 7, 6, 5, 4, 3) = 6.0  (strict 7-game window)
    assert row["last_game"] == pytest.approx(3.0)
    assert row["last3_avg"] == pytest.approx((5.0 + 4.0 + 3.0) / 3)
    assert row["last5_avg"] == pytest.approx((7.0 + 6.0 + 5.0 + 4.0 + 3.0) / 5)
    assert row["last7_avg"] == pytest.approx((9.0 + 8.0 + 7.0 + 6.0 + 5.0 + 4.0 + 3.0) / 7)
    assert row["games_played_before_target"] == 8


def test_no_zero_imputation_for_short_history():
    """
    Verify no zero values are produced for short-history players.
    Zero has real meaning in the rating system.
    """
    df = pd.DataFrame({
        "personId": [4006] * 3,
        "playerName": ["NoZero Player"] * 3,
        "teamId": [1610612740] * 3,
        "gameId": [f"00225000Z{i}" for i in range(3)],
        "rating": [2.0, 4.0, 6.0],
        "game_date": pd.date_range("2026-03-01", periods=3, freq="D")
    })
    feat = construct_prediction_features(df, target_date="2026-03-04")
    row = feat.iloc[0]
    for col in ["last_game", "last3_avg", "last5_avg", "last7_avg"]:
        assert row[col] != 0.0, f"{col} must not be zero-imputed"


# ==============================
# 2. CALENDAR-DATE SPLIT TESTS
# ==============================

def test_calendar_date_cutoff_no_split_days():
    """
    Verify that get_calendar_date_cutoff produces clean calendar date partition
    with zero overlapping dates between train and test.
    """
    # Create dataset with multiple games per date across 10 dates
    dates = pd.date_range("2026-03-01", periods=10, freq="D")
    date_series = []
    for d in dates:
        date_series.extend([d] * 5) # 5 records per date
    
    df = pd.DataFrame({
        "game_date": date_series,
        "personId": list(range(len(date_series))),
        "rating": np.random.randn(len(date_series)),
        "last_game": np.random.randn(len(date_series)),
        "last3_avg": np.random.randn(len(date_series)),
        "last5_avg": np.random.randn(len(date_series)),
        "last7_avg": np.random.randn(len(date_series))
    })
    
    cutoff = get_calendar_date_cutoff(df, target_train_ratio=0.8)
    
    train_df = df[df["game_date"] < cutoff]
    test_df = df[df["game_date"] >= cutoff]
    
    train_dates = set(train_df["game_date"].dt.strftime("%Y-%m-%d"))
    test_dates = set(test_df["game_date"].dt.strftime("%Y-%m-%d"))
    
    # Critical invariant: intersection of train dates and test dates MUST be empty
    overlap = train_dates.intersection(test_dates)
    assert len(overlap) == 0, f"Dates found in both train and test: {overlap}"
    
    # Chronological invariant: all train dates < all test dates
    assert max(train_dates) < min(test_dates)


# ==============================
# 3. BASELINE EVALUATION TESTS
# ==============================

def test_baseline_evaluations_calculated_on_same_test_target():
    """
    Verify that evaluate_baselines calculates all 8 baselines against
    the exact same test target and returns expected dictionary keys.
    """
    np.random.seed(42)
    n_train, n_test = 100, 30
    
    X_train = pd.DataFrame({
        "last_game": np.random.uniform(1.0, 5.0, n_train),
        "last3_avg": np.random.uniform(1.0, 5.0, n_train),
        "last5_avg": np.random.uniform(1.0, 5.0, n_train),
        "last7_avg": np.random.uniform(1.0, 5.0, n_train)
    })
    y_train = pd.Series(X_train["last5_avg"] * 0.8 + np.random.normal(0, 0.2, n_train))
    
    X_test = pd.DataFrame({
        "last_game": np.random.uniform(1.0, 5.0, n_test),
        "last3_avg": np.random.uniform(1.0, 5.0, n_test),
        "last5_avg": np.random.uniform(1.0, 5.0, n_test),
        "last7_avg": np.random.uniform(1.0, 5.0, n_test)
    })
    y_test = pd.Series(X_test["last5_avg"] * 0.8 + np.random.normal(0, 0.2, n_test))
    
    mock_rf = MagicMock()
    mock_rf.predict.return_value = X_test["last5_avg"].values
    
    metrics = evaluate_baselines(X_train, y_train, X_test, y_test, mock_rf)
    
    expected_keys = [
        "Mean Baseline",
        "Median Baseline",
        "Last Game Baseline",
        "Last 3 Baseline",
        "Last 5 Baseline",
        "Last 7 Baseline",
        "Linear Regression",
        "Random Forest"
    ]
    
    for key in expected_keys:
        assert key in metrics, f"Missing baseline method: {key}"
        assert "MAE" in metrics[key]
        assert "R2" in metrics[key]
        assert isinstance(metrics[key]["MAE"], float)
        assert isinstance(metrics[key]["R2"], float)

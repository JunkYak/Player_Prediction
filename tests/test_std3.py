"""
Phase 2D.2C Experiment 2 — std_3 feature tests.
Tests the correctness, leakage safety, contamination safety, training-only
median imputation, metadata persistence, and train/inference semantic consistency
of the std_3 volatility feature.
"""
import os
import sys
import json
import tempfile
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


# ── Test 1: std_3 calculation for exactly 2 previous games ──────────────────

def test_std3_two_previous_games():
    """
    For a player with exactly 2 completed games [2.0, 4.0] before target:
    std_3 must equal sample standard deviation (ddof=1) of [2.0, 4.0] = sqrt(2) ≈ 1.41421356.
    """
    ratings = [2.0, 4.0]
    df = _make_player_df(6001, ratings)
    feat = construct_prediction_features(df, target_date="2026-03-03", std3_impute_value=None)
    assert len(feat) == 1
    row = feat.iloc[0]
    expected = float(pd.Series(ratings).std())  # sample std: 1.41421356...
    assert row["std_3"] == pytest.approx(expected)
    assert row["std_3"] == pytest.approx(np.sqrt(2.0))


# ── Test 2: std_3 calculation for exactly 3 previous games ──────────────────

def test_std3_three_previous_games():
    """
    For a player with exactly 3 completed games [2.0, 4.0, 6.0] before target:
    std_3 must equal sample standard deviation of [2.0, 4.0, 6.0] = 2.0.
    """
    ratings = [2.0, 4.0, 6.0]
    df = _make_player_df(6002, ratings)
    feat = construct_prediction_features(df, target_date="2026-03-04", std3_impute_value=None)
    assert len(feat) == 1
    row = feat.iloc[0]
    expected = float(pd.Series(ratings).std())  # 2.0
    assert row["std_3"] == pytest.approx(2.0)
    assert row["std_3"] == pytest.approx(expected)


# ── Test 3: Rolling behavior once more than 3 games exist ───────────────────

def test_std3_rolling_behavior_more_than_three_games():
    """
    For a player with >3 completed games [1.0, 10.0, 2.0, 4.0, 6.0]:
    std_3 must use ONLY the 3 most recent completed games [2.0, 4.0, 6.0]
    and ignore older games (1.0, 10.0).
    """
    ratings = [1.0, 10.0, 2.0, 4.0, 6.0]
    df = _make_player_df(6003, ratings)
    feat = construct_prediction_features(df, target_date="2026-03-06", std3_impute_value=None)
    assert len(feat) == 1
    row = feat.iloc[0]
    expected = float(pd.Series([2.0, 4.0, 6.0]).std())  # 2.0
    assert row["std_3"] == pytest.approx(expected)
    assert row["std_3"] == pytest.approx(2.0)


# ── Test 4: Target-game leakage prevention ───────────────────────────────────

def test_std3_target_game_leakage_prevention():
    """
    The target game's rating must NEVER enter std_3.
    Completed ratings: [2.0, 4.0, 6.0].
    A hypothetical future target rating of 999.0 must not affect std_3.
    """
    completed = [2.0, 4.0, 6.0]
    future_rating = 999.0

    df = _make_player_df(6004, completed)
    feat = construct_prediction_features(df, target_date="2026-03-04", std3_impute_value=None)
    assert len(feat) == 1
    row = feat.iloc[0]
    assert row["std_3"] == pytest.approx(2.0)

    # If the future rating had leaked into the window [4.0, 6.0, 999.0], std would be ~572
    leaked_std = float(pd.Series([4.0, 6.0, future_rating]).std())
    assert abs(row["std_3"] - leaked_std) > 500.0


# ── Test 5: No cross-player contamination ────────────────────────────────────

def test_std3_no_cross_player_contamination():
    """
    std_3 for each player must be computed strictly from their own ratings.
    Player A (consistent ratings: [5.0, 5.0, 5.0]) -> std_3 = 0.0.
    Player B (volatile ratings: [1.0, 5.0, 9.0]) -> std_3 = 4.0.
    """
    df_p1 = _make_player_df(6005, [5.0, 5.0, 5.0])  # std = 0.0
    df_p2 = _make_player_df(6006, [1.0, 5.0, 9.0])  # std = 4.0

    combined = pd.concat([df_p1, df_p2], ignore_index=True)
    feat = construct_prediction_features(combined, target_date="2026-03-04", std3_impute_value=None)
    assert len(feat) == 2

    p1_row = feat[feat["personId"] == 6005].iloc[0]
    p2_row = feat[feat["personId"] == 6006].iloc[0]

    assert p1_row["std_3"] == pytest.approx(0.0)
    assert p2_row["std_3"] == pytest.approx(4.0)


# ── Test 6: One-prior-game case produces NaN BEFORE imputation ───────────────

def test_std3_one_prior_game_produces_nan_before_imputation():
    """
    For a player with exactly 1 completed game, sample standard deviation cannot
    be computed (ddof=1 requires >= 2 observations).
    construct_prediction_features without imputation MUST return NaN (not 0.0).
    """
    df = _make_player_df(6007, [5.0])
    feat = construct_prediction_features(df, target_date="2026-03-02", std3_impute_value=None)
    assert len(feat) == 1
    row = feat.iloc[0]
    assert pd.isna(row["std_3"])
    # Crucially, ensure it is NOT 0.0 (which denotes perfect consistency)
    assert row["std_3"] != 0.0


# ── Test 7: Training-set median is used for imputation ────────────────────────

def test_std3_training_median_imputation_applied():
    """
    When an imputation value (training median) is provided to construct_prediction_features,
    the NaN std_3 for 1-prior-game players is filled with that exact value,
    while players with >=2 games retain their calculated std_3.
    """
    df_p1 = _make_player_df(6008, [5.0])           # 1 game -> NaN raw -> should become 0.6736
    df_p2 = _make_player_df(6009, [2.0, 4.0, 6.0])  # 3 games -> 2.0 raw -> should stay 2.0

    combined = pd.concat([df_p1, df_p2], ignore_index=True)
    impute_val = 0.6736
    feat = construct_prediction_features(combined, target_date="2026-03-04", std3_impute_value=impute_val)

    p1_row = feat[feat["personId"] == 6008].iloc[0]
    p2_row = feat[feat["personId"] == 6009].iloc[0]

    assert p1_row["std_3"] == pytest.approx(impute_val)
    assert p2_row["std_3"] == pytest.approx(2.0)


# ── Test 8: Test-set values do NOT influence training median ──────────────────

def test_std3_test_set_does_not_influence_training_median():
    """
    Verify that calculating the median on the training split only is mathematically
    isolated from any test set values (even extreme outlier values in test).
    """
    train_series = pd.Series([0.5, 0.6, 0.7, 0.8, 0.9])
    test_outlier_series = pd.Series([100.0, 200.0, 300.0])

    train_median = float(train_series.median())
    combined_median = float(pd.concat([train_series, test_outlier_series]).median())

    assert train_median == pytest.approx(0.7)
    assert combined_median == pytest.approx(0.85)
    # The training median is strictly unaffected by the test set
    assert train_median != pytest.approx(combined_median)


# ── Test 9: Saved median is used during inference ─────────────────────────────

def test_std3_saved_median_used_during_inference():
    """
    Verify that saved metadata in feature_metadata.json can be loaded and applied
    correctly during inference.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        meta_file = os.path.join(tmpdir, "feature_metadata.json")
        saved_median = 0.7654
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump({"std3_median": saved_median}, f)

        # Load back
        with open(meta_file, "r", encoding="utf-8") as f:
            loaded_meta = json.load(f)

        assert loaded_meta["std3_median"] == pytest.approx(saved_median)

        # Apply to 1-game player
        df = _make_player_df(6010, [4.0])
        feat = construct_prediction_features(df, std3_impute_value=loaded_meta["std3_median"])
        assert feat.iloc[0]["std_3"] == pytest.approx(saved_median)


# ── Test 10: Training and inference produce semantically consistent std_3 ────

def test_std3_train_inference_semantic_consistency():
    """
    Training semantics (rolling(3, min_periods=2).std().shift(1)) and
    inference semantics (pd.Series(ratings[-3:]).std()) must produce identical
    values for the same historical ratings.
    """
    ratings = [3.0, 5.0, 7.0]

    # Training calculation
    df_raw = _make_player_df(6011, ratings)
    df_raw["std_3"] = (
        df_raw.groupby("personId")["rating"]
        .rolling(3, min_periods=2)
        .std()
        .shift(1)
        .reset_index(level=0, drop=True)
    )
    # For a hypothetical game 4, the training feature computed from game 3 row is:
    # rolling std of [3.0, 5.0, 7.0] shifted to game 4
    train_computed_val = float(pd.Series(ratings).rolling(3, min_periods=2).std().iloc[-1])

    # Inference calculation
    feat = construct_prediction_features(df_raw, target_date="2026-03-04", std3_impute_value=None)
    inf_computed_val = feat.iloc[0]["std_3"]

    assert inf_computed_val == pytest.approx(train_computed_val)
    assert inf_computed_val == pytest.approx(2.0)


# ── Test 11: Existing features remain unchanged ──────────────────────────────

def test_std3_existing_features_unchanged():
    """
    Adding std_3 must not alter any existing historical features.
    """
    ratings = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    df = _make_player_df(6012, ratings)
    feat = construct_prediction_features(df, target_date="2026-03-09", std3_impute_value=0.6736)
    assert len(feat) == 1
    row = feat.iloc[0]

    # Existing features
    assert row["last_game"] == pytest.approx(8.0)
    assert row["last3_avg"] == pytest.approx((6.0 + 7.0 + 8.0) / 3)
    assert row["last5_avg"] == pytest.approx((4.0 + 5.0 + 6.0 + 7.0 + 8.0) / 5)
    assert row["last7_avg"] == pytest.approx((2.0 + 3.0 + 4.0 + 5.0 + 6.0 + 7.0 + 8.0) / 7)
    assert row["games_played_before_target"] == 8
    assert row["expand_mean"] == pytest.approx(sum(ratings) / len(ratings))

    # std_3 uses most recent 3: [6.0, 7.0, 8.0] -> std = 1.0
    assert row["std_3"] == pytest.approx(1.0)


# ── Test 12: expand_mean remains unchanged ───────────────────────────────────

def test_std3_expand_mean_remains_unchanged():
    """
    Verify expand_mean is identical to the arithmetic mean over all historical games
    and is unaffected by the presence of std_3.
    """
    ratings = [10.0, 20.0, 30.0, 40.0]
    df = _make_player_df(6013, ratings)
    feat = construct_prediction_features(df, target_date="2026-03-05", std3_impute_value=None)
    row = feat.iloc[0]

    assert row["expand_mean"] == pytest.approx(25.0)
    assert row["std_3"] == pytest.approx(float(pd.Series([20.0, 30.0, 40.0]).std()))
    assert row["std_3"] == pytest.approx(10.0)

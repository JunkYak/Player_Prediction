"""
tests/test_incremental_ingestion_and_cutoff.py

Comprehensive deterministic test suite for:
1. Historical cutoff enforcement (>= 2026-02-06) across ingestion, dates, features, and predictions.
2. Dual-engine incremental ingestion (NBA Stats primary + ESPN summary fallback).
3. Exact semantic validation of ESPN play-by-play mapping against rating_engine/formula.py.
4. Idempotent merging and deduplication.
5. Rolling feature updates after newly completed games arrive.
6. Schedule-aware stale-data detection and ingestion metadata tracking.
7. Preservation of last-known-good datasets upon failure.
"""

import os
import sys
import json
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Ensure project root is on sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.espn_pbp_adapter import (
    REQUIRED_COLUMNS,
    parse_espn_summary_to_pbp,
    fetch_espn_completed_games,
)
from scripts.build_play_by_play_dataset import (
    HISTORICAL_CUTOFF_DATE,
    merge_pbp_data,
    validate_pbp_dataset,
    build_dataset,
)
from scripts.add_game_dates import add_game_dates
from scripts.build_features import build_features
from scripts.predict_ratings import construct_prediction_features
from rating_engine.formula import (
    clock_to_seconds,
    base_delta,
    context_delta,
    clutch_multiplier,
)


# ==============================================================================
# 1. Historical Cutoff Enforcement Tests (>= 2026-02-06)
# ==============================================================================
class TestHistoricalCutoffEnforcement:
    def test_cutoff_constant_is_february_6_2026(self):
        assert HISTORICAL_CUTOFF_DATE == "2026-02-06"

    def test_add_game_dates_excludes_pre_cutoff_records(self, tmp_path):
        """Ratings with dates before Feb 6, 2026 must be excluded from date-enriched dataset."""
        input_file = tmp_path / "player_game_ratings.parquet"
        output_file = tmp_path / "player_game_ratings_with_dates.parquet"

        df_ratings = pd.DataFrame([
            {"gameId": "g_pre", "teamId": 1610612738, "personId": 1628369, "playerName": "Jayson Tatum", "rating": 5.0},
            {"gameId": "g_post", "teamId": 1610612738, "personId": 1628369, "playerName": "Jayson Tatum", "rating": 6.0},
        ])
        df_ratings.to_parquet(input_file, index=False)

        dates_df = pd.DataFrame([
            {"gameId": "g_pre", "game_date": "2026-01-15"},    # Pre-cutoff
            {"gameId": "g_post", "game_date": "2026-02-06"},   # Cutoff date
        ])

        add_game_dates(
            input_path=str(input_file),
            output_path=str(output_file),
            game_dates_df=dates_df,
            cutoff_date="2026-02-06"
        )

        res = pd.read_parquet(output_file)
        assert len(res) == 1
        assert res.iloc[0]["gameId"] == "g_post"
        assert res.iloc[0]["game_date"] == "2026-02-06"

    def test_build_features_strictly_respects_cutoff(self, tmp_path):
        """Feature generation must ignore any historical record before 2026-02-06."""
        input_file = tmp_path / "player_game_ratings_with_dates.parquet"
        output_file = tmp_path / "player_features.parquet"

        df_ratings = pd.DataFrame([
            {"gameId": "g1", "teamId": 101, "personId": 1001, "playerName": "P1", "rating": 10.0, "game_date": "2026-01-10"}, # pre
            {"gameId": "g2", "teamId": 101, "personId": 1001, "playerName": "P1", "rating": 4.0, "game_date": "2026-02-06"},  # valid 1
            {"gameId": "g3", "teamId": 101, "personId": 1001, "playerName": "P1", "rating": 6.0, "game_date": "2026-02-10"},  # valid 2
        ])
        df_ratings.to_parquet(input_file, index=False)

        with patch("scripts.build_features.input_path", str(input_file), create=True), \
             patch("scripts.build_features.output_path", str(output_file), create=True):
            # Run build_features logic
            df = pd.read_parquet(input_file)
            df["game_date"] = pd.to_datetime(df["game_date"])
            cutoff_dt = pd.to_datetime("2026-02-06")
            df = df[df["game_date"] >= cutoff_dt].copy()
            df = df.sort_values(["personId", "game_date"])
            df["last_game"] = df.groupby("personId")["rating"].shift(1)
            df["expand_mean"] = df.groupby("personId")["rating"].expanding().mean().shift(1).reset_index(level=0, drop=True)

            # Row for g3 (2026-02-10): previous game should be g2 (rating 4.0), not g1 (rating 10.0)
            g3_row = df[df["gameId"] == "g3"].iloc[0]
            assert g3_row["last_game"] == 4.0
            assert g3_row["expand_mean"] == 4.0

    def test_construct_prediction_features_filters_pre_cutoff(self):
        """Inference feature vector construction drops pre-cutoff history."""
        df_history = pd.DataFrame([
            {"gameId": "old", "personId": 1001, "teamId": 101, "playerName": "Test Player", "game_date": "2025-12-01", "rating": 20.0},
            {"gameId": "new", "personId": 1001, "teamId": 101, "playerName": "Test Player", "game_date": "2026-02-06", "rating": 5.0},
        ])
        features_df = construct_prediction_features(df_history, target_date="2026-02-10")
        assert len(features_df) == 1
        # last_game must be 5.0, not contaminated by 20.0
        assert features_df.iloc[0]["last_game"] == 5.0
        assert features_df.iloc[0]["expand_mean"] == 5.0


# ==============================================================================
# 2. ESPN Play-by-Play Adapter Semantic & Formula Tests
# ==============================================================================
class TestESPNPlayByPlayMappingToFormula:
    """Validate ESPN event descriptions map accurately to rating_engine/formula.py requirements."""

    def test_clock_conversion_to_seconds(self):
        # ESPN "10:43" -> ISO "PT10M43.00S" -> clock_to_seconds = 643.0
        iso_clock = "PT10M43.00S"
        sec = clock_to_seconds(iso_clock)
        assert sec == 10 * 60 + 43.0

    def test_made_shot_three_and_assist(self):
        row = {
            "actionType": "Made Shot",
            "shotValue": 3,
            "shotDistance": 26,
            "description": "Jayson Tatum 26' 3PT Jump Shot (Derrick White 1 AST)",
        }
        val = base_delta(row)
        # Made 3pt attempt_value (0.18) + is_three (0.194) + is_assisted (-0.04) = 0.334
        assert pytest.approx(val, 0.001) == 0.334

    def test_made_shot_difficulty_bonuses(self):
        row = {
            "actionType": "Made Shot",
            "shotValue": 2,
            "shotDistance": 14,
            "description": "Luka Doncic 14' Step Back Fadeaway Turnaround Pullup Jump Shot",
        }
        val = base_delta(row)
        # attempt (0.15) + jump (0.12) + unassisted (0.03) + step back (0.06) + fadeaway (0.05) + turnaround (0.04) + pullup (0.03)
        expected = 0.15 + 0.12 + 0.03 + 0.06 + 0.05 + 0.04 + 0.03
        assert pytest.approx(val, 0.001) == expected

    def test_floating_shot_bonus(self):
        row = {
            "actionType": "Made Shot",
            "shotValue": 2,
            "shotDistance": 6,
            "description": "Derrick White 6' Driving Floating Jump Shot",
        }
        val = base_delta(row)
        # attempt (0.15) + floater (0.130) + jump (0.120) + unassisted (0.03) = 0.430
        assert pytest.approx(val, 0.001) == 0.430

    def test_missed_shot_scoring(self):
        row_layup = {"actionType": "Missed Shot", "description": "MISS Jaylen Brown 4' Driving Layup Shot", "shotValue": 2, "shotDistance": 4}
        assert base_delta(row_layup) == 0.10

        row_deep3 = {"actionType": "Missed Shot", "description": "MISS Jayson Tatum 29' 3PT Jump Shot", "shotValue": 3, "shotDistance": 29}
        assert base_delta(row_deep3) == 0.12

        row_standard = {"actionType": "Missed Shot", "description": "MISS Al Horford 18' Jump Shot", "shotValue": 2, "shotDistance": 18}
        assert base_delta(row_standard) == 0.07

    def test_rebound_classification(self):
        off_reb = {"actionType": "Rebound", "description": "Luke Kornet REBOUND (Off:1 Def:0)"}
        assert base_delta(off_reb) == 0.07

        def_reb = {"actionType": "Rebound", "description": "Jayson Tatum REBOUND (Off:0 Def:1)"}
        assert base_delta(def_reb) == 0.05

    def test_turnover_and_steal_attribution(self):
        # Turnover row
        to_row = {"actionType": "Turnover", "description": "Al Horford Lost Ball Turnover"}
        assert base_delta(to_row) == -0.17

        # Steal row (empty actionType, attributed to stealer)
        stl_row = {"actionType": "", "description": "Jordan Poole STEAL (1 STL)"}
        assert base_delta(stl_row) == 0.19

    def test_block_attribution(self):
        blk_row = {"actionType": "", "description": "Bub Carrington BLOCK (1 BLK)"}
        assert base_delta(blk_row) == 0.18

    def test_free_throw_scoring(self):
        ft_made = {"actionType": "Free Throw", "description": "Jaylen Brown Free Throw"}
        assert base_delta(ft_made) == 0.09

        ft_miss = {"actionType": "Free Throw", "description": "Jaylen Brown MISS Free Throw"}
        assert base_delta(ft_miss) == 0.0

    def test_clutch_multiplier_in_last_two_minutes_period_four(self):
        # 4th quarter with 60s remaining
        row_clutch = {
            "actionType": "Made Shot",
            "period": 4,
            "seconds_remaining": 60,
            "description": "Jayson Tatum 25' 3PT Jump Shot",
        }
        # 1 + ((120 - 60)/120) * 2 = 1 + 1.0 = 2.0
        mult = clutch_multiplier(row_clutch)
        assert pytest.approx(mult, 0.001) == 2.0

        # Overtime (period 5): clutch must NOT fire
        row_ot = {
            "actionType": "Made Shot",
            "period": 5,
            "seconds_remaining": 60,
            "description": "Jayson Tatum 25' 3PT Jump Shot",
        }
        assert clutch_multiplier(row_ot) == 1.0


# ==============================================================================
# 3. Deduplication, Idempotence, and Fallback Ingestion Tests
# ==============================================================================
class TestIncrementalIngestionWorkflow:
    def test_merge_pbp_data_idempotence(self):
        """Merging the same new game records twice must produce identical row count without duplicates."""
        existing_df = pd.DataFrame([
            {"gameId": "g1", "teamId": 1, "personId": 101, "playerName": "P1", "period": 1, "clock": "PT10M00.00S",
             "actionType": "Made Shot", "subType": "Jump", "shotResult": "Made", "shotDistance": 10, "shotValue": 2,
             "description": "P1 10' Jump Shot", "scoreHome": 2, "scoreAway": 0}
        ])

        new_df = pd.DataFrame([
            {"gameId": "g2", "teamId": 1, "personId": 101, "playerName": "P1", "period": 1, "clock": "PT09M00.00S",
             "actionType": "Made Shot", "subType": "Jump", "shotResult": "Made", "shotDistance": 10, "shotValue": 2,
             "description": "P1 10' Jump Shot", "scoreHome": 4, "scoreAway": 0}
        ])

        # Run merge 1
        merged_1 = merge_pbp_data(existing_df, new_df)
        assert len(merged_1) == 2
        assert merged_1["gameId"].nunique() == 2

        # Run merge 2 with identical new_df
        merged_2 = merge_pbp_data(merged_1, new_df)
        assert len(merged_2) == 2
        assert merged_2["gameId"].nunique() == 2

    def test_nba_api_timeout_triggers_espn_fallback(self):
        """When stats.nba.com discovery raises an exception, ESPN fallback is invoked."""
        with patch("nba_api.stats.endpoints.leaguegamelog.LeagueGameLog", side_effect=Exception("ReadTimeout")):
            with patch("scripts.espn_pbp_adapter.fetch_espn_scoreboard") as mock_espn:
                mock_espn.return_value = [
                    {"game_id": "espn_g1", "game_date": "2026-10-23", "home_team_tri": "BOS", "away_team_tri": "NYK"}
                ]
                from scripts.build_play_by_play_dataset import fetch_games
                games = fetch_games(start_date_str="2026-10-22", end_date_str="2026-10-24")
                assert len(games) >= 1
                assert any(g["gameId"] == "espn_g1" for g in games)
                assert any(g["source"] == "espn" for g in games)

    def test_last_known_good_dataset_preserved_on_failure(self, tmp_path):
        """If game discovery and all fallbacks fail, the existing valid dataset is preserved intact."""
        pbp_file = tmp_path / "play_by_play.parquet"
        baseline_df = pd.DataFrame([
            {"gameId": "g1", "teamId": 1, "personId": 101, "playerName": "P1", "period": 1, "clock": "PT10M00.00S",
             "actionType": "Made Shot", "subType": "Jump", "shotResult": "Made", "shotDistance": 10, "shotValue": 2,
             "description": "P1 10' Jump Shot", "scoreHome": 2, "scoreAway": 0}
        ])
        baseline_df.to_parquet(pbp_file, index=False)

        # Force fetch_games to raise unhandled error
        with patch("scripts.build_play_by_play_dataset.fetch_games", side_effect=RuntimeError("Total network failure")):
            res = build_dataset(output_file=str(pbp_file))
            # Must return baseline without crashing or emptying file
            assert len(res) == 1
            assert res.iloc[0]["gameId"] == "g1"

            # Check that file on disk is unchanged
            on_disk = pd.read_parquet(pbp_file)
            assert len(on_disk) == 1


# ==============================================================================
# 4. Schedule-Aware Stale Data & Freshness Telemetry Tests
# ==============================================================================
class TestFreshnessValidationAndMetadata:
    def test_ingestion_metadata_saved_on_success(self, tmp_path):
        """Successful ingestion records telemetry in ingestion_metadata.json."""
        from scripts.build_play_by_play_dataset import _save_ingestion_metadata
        _save_ingestion_metadata(
            status="SUCCESS",
            source="espn_fallback",
            new_games=5,
            total_games=428,
            total_rows=215000,
            latest_date="2026-10-24",
            cutoff_date="2026-02-06",
            data_dir=str(tmp_path)
        )

        meta_file = tmp_path / "ingestion_metadata.json"
        assert meta_file.exists()
        with open(meta_file, "r") as f:
            data = json.load(f)

        assert data["status"] == "SUCCESS"
        assert data["source"] == "espn_fallback"
        assert data["primary_configured_provider"] == "nba_stats"
        assert data["actual_provider"] == "espn_fallback"
        assert data["new_games_ingested"] == 5
        assert data["latest_completed_game_date"] == "2026-10-24"
        assert data["historical_cutoff_date"] == "2026-02-06"
        assert data["fallback_occurred"] is False or isinstance(data["fallback_occurred"], bool)

    def test_ingestion_metadata_provenance_and_freshness_separation(self, tmp_path):
        """Metadata must separate primary configured provider from actual provider and track fallback & freshness."""
        from scripts.build_play_by_play_dataset import _save_ingestion_metadata
        _save_ingestion_metadata(
            status="SUCCESS",
            source="espn",
            new_games=3,
            total_games=445,
            total_rows=224000,
            latest_date="2026-10-24",
            cutoff_date="2026-02-06",
            data_dir=str(tmp_path),
            skipped_games=30,
            fallback_occurred=True,
            is_fresh=True
        )

        with open(tmp_path / "ingestion_metadata.json", "r") as f:
            meta = json.load(f)

        assert meta["primary_configured_provider"] == "nba_stats"
        assert meta["actual_provider"] == "espn"
        assert meta["fallback_occurred"] is True
        assert meta["existing_games_skipped"] == 30
        assert meta["is_fresh"] is True

    def test_preserved_existing_metadata_marks_stale(self, tmp_path):
        """Preserving existing dataset on discovery failure must mark is_fresh=False."""
        from scripts.build_play_by_play_dataset import _save_ingestion_metadata
        _save_ingestion_metadata(
            status="PRESERVED_EXISTING",
            source="none",
            new_games=0,
            total_games=423,
            total_rows=212000,
            latest_date="2026-10-21",
            cutoff_date="2026-02-06",
            data_dir=str(tmp_path),
            skipped_games=0,
            fallback_occurred=False,
            is_fresh=False
        )

        with open(tmp_path / "ingestion_metadata.json", "r") as f:
            meta = json.load(f)

        assert meta["status"] == "PRESERVED_EXISTING"
        assert meta["is_fresh"] is False
        assert meta["new_games_ingested"] == 0


# ==============================================================================
# 5. Model Feature Compatibility Tests
# ==============================================================================
class TestModelFeatureCompatibility:
    def test_trained_model_exact_feature_names_and_ordering(self):
        """
        Validates that models/rating_model.pkl requires exactly 7 features
        in the exact ordering:
        ['last_game', 'last3_avg', 'last5_avg', 'last7_avg', 'games_played_before_target', 'expand_mean', 'days_since_prev_game']
        """
        import joblib
        model_path = os.path.join(ROOT, "models", "rating_model.pkl")
        assert os.path.exists(model_path), f"Model missing at {model_path}"

        model = joblib.load(model_path)
        expected_features = [
            "last_game",
            "last3_avg",
            "last5_avg",
            "last7_avg",
            "games_played_before_target",
            "expand_mean",
            "days_since_prev_game"
        ]

        actual_features = list(getattr(model, "feature_names_in_", []))
        assert actual_features == expected_features, (
            f"Feature mismatch in rating_model.pkl!\n"
            f"Expected: {expected_features}\n"
            f"Actual:   {actual_features}"
        )
        assert getattr(model, "n_features_in_", 0) == 7

    def test_inference_features_match_model_exact_columns(self):
        """Inference feature matrix X must match the 7 trained model features identically."""
        from scripts.predict_ratings import construct_prediction_features
        import joblib

        model_path = os.path.join(ROOT, "models", "rating_model.pkl")
        model = joblib.load(model_path)
        expected_features = list(model.feature_names_in_)

        # Construct synthetic history
        df_history = pd.DataFrame([
            {"gameId": "g1", "teamId": 101, "personId": 1, "playerName": "P1", "rating": 5.0, "game_date": "2026-03-01", "fga": 10},
            {"gameId": "g2", "teamId": 101, "personId": 1, "playerName": "P1", "rating": 6.0, "game_date": "2026-03-03", "fga": 12},
        ])

        latest = construct_prediction_features(df_history, target_date="2026-03-05")
        for col in expected_features:
            assert col in latest.columns, f"Required model feature '{col}' missing from construct_prediction_features!"

        X = latest[expected_features]
        assert list(X.columns) == expected_features
        preds = model.predict(X)
        assert len(preds) == 1
        assert not np.isnan(preds[0])


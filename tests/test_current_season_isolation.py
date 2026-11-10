"""
tests/test_current_season_isolation.py

Comprehensive test suite verifying Phase 2.9 Part 2:
Strict isolation of model training data and prediction feature history to the current NBA season.
Zero leakage from previous season (e.g. Jan/Feb/March 2026) into:
- Training examples.
- Rolling averages.
- Last-N-game features.
- Player form features.
- Prediction inference history.
"""

import os
import tempfile
import pandas as pd
import numpy as np
import pytest

from scripts.build_features import build_features
from scripts.predict_ratings import construct_prediction_features
from scripts.compute_season_multipliers import (
    determine_season_boundary,
    compute_season_multipliers,
)


class TestCurrentSeasonIsolation:
    """Verifies complete temporal isolation between previous and current NBA seasons."""

    def test_no_model_training_row_uses_previous_season_game(self):
        """No training example or target label in build_features uses a pre-season game."""
        df_ratings = pd.DataFrame([
            # Player A: previous season games (2025-26)
            {"gameId": "prev_1", "personId": 101, "teamId": 1, "playerName": "Player A", "game_date": "2026-03-10", "rating": 15.0},
            {"gameId": "prev_2", "personId": 101, "teamId": 1, "playerName": "Player A", "game_date": "2026-03-12", "rating": 18.0},
            {"gameId": "prev_3", "personId": 101, "teamId": 1, "playerName": "Player A", "game_date": "2026-03-15", "rating": 14.0},
            # Player A: current season games (2026-27, boundary 2026-10-20)
            {"gameId": "curr_1", "personId": 101, "teamId": 1, "playerName": "Player A", "game_date": "2026-10-21", "rating": 22.0},
            {"gameId": "curr_2", "personId": 101, "teamId": 1, "playerName": "Player A", "game_date": "2026-10-23", "rating": 24.0},
            {"gameId": "curr_3", "personId": 101, "teamId": 1, "playerName": "Player A", "game_date": "2026-10-25", "rating": 20.0},
            # Player B: only previous season games
            {"gameId": "prev_b1", "personId": 102, "teamId": 2, "playerName": "Player B", "game_date": "2026-03-10", "rating": 10.0},
            {"gameId": "prev_b2", "personId": 102, "teamId": 2, "playerName": "Player B", "game_date": "2026-03-12", "rating": 12.0},
        ])

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "ratings.parquet")
            output_path = os.path.join(tmpdir, "features.parquet")
            df_ratings.to_parquet(input_path, index=False)

            build_features(input_path=input_path, output_path=output_path, reference_date="2026-10-26")

            features_df = pd.read_parquet(output_path)

            # Must only contain rows for games in current season (>= 2026-10-20)
            assert len(features_df) > 0
            min_date = pd.to_datetime(features_df["game_date"]).min()
            assert min_date >= pd.to_datetime("2026-10-20")

            # Player B had 0 current-season games, so must have 0 training rows
            assert 102 not in features_df["personId"].values

            # Player A's training rows must not include any prev_* games
            game_ids = set(features_df["gameId"].values)
            assert "prev_1" not in game_ids
            assert "prev_2" not in game_ids
            assert "prev_3" not in game_ids

    def test_no_rolling_feature_includes_previous_season_observation(self):
        """Rolling features (last_game, last3_avg, expand_mean) do not leak previous-season ratings."""
        df_ratings = pd.DataFrame([
            # Prior season game with high rating
            {"gameId": "g_old", "personId": 101, "teamId": 1, "playerName": "Player A", "game_date": "2026-03-20", "rating": 50.0},
            # Current season game 1
            {"gameId": "g_curr1", "personId": 101, "teamId": 1, "playerName": "Player A", "game_date": "2026-10-21", "rating": 10.0},
            # Current season game 2
            {"gameId": "g_curr2", "personId": 101, "teamId": 1, "playerName": "Player A", "game_date": "2026-10-23", "rating": 14.0},
        ])

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "ratings.parquet")
            output_path = os.path.join(tmpdir, "features.parquet")
            df_ratings.to_parquet(input_path, index=False)

            build_features(input_path=input_path, output_path=output_path, reference_date="2026-10-24")

            features_df = pd.read_parquet(output_path)

            # Game 2 (g_curr2) is predicted using prior game(s) in current season
            g2_row = features_df[features_df["gameId"] == "g_curr2"].iloc[0]

            # Prior game must be g_curr1 (rating 10.0), NOT contaminated by g_old (50.0)
            assert g2_row["last_game"] == 10.0
            assert g2_row["expand_mean"] == 10.0
            assert g2_row["games_played_before_target"] == 1

    def test_player_with_three_current_and_ten_previous_season_games_uses_only_three(self):
        """Inference feature vector uses only the 3 current-season games for a player with 10 prior-season games."""
        prior_games = [
            {"gameId": f"prev_{i}", "personId": 201, "teamId": 10, "playerName": "Vet Player", "game_date": f"2026-02-{10+i:02d}", "rating": 30.0}
            for i in range(10)
        ]
        curr_games = [
            {"gameId": "c1", "personId": 201, "teamId": 10, "playerName": "Vet Player", "game_date": "2026-10-20", "rating": 10.0},
            {"gameId": "c2", "personId": 201, "teamId": 10, "playerName": "Vet Player", "game_date": "2026-10-22", "rating": 12.0},
            {"gameId": "c3", "personId": 201, "teamId": 10, "playerName": "Vet Player", "game_date": "2026-10-24", "rating": 14.0},
        ]
        history_df = pd.DataFrame(prior_games + curr_games)

        upcoming = pd.DataFrame([{"gameId": "target_1", "homeTeamId": 10, "awayTeamId": 20, "nba_game_date": "2026-10-26"}])
        roster_map = {201: {"teamId": 10, "teamName": "Team X", "playerName": "Vet Player"}}

        features = construct_prediction_features(
            history_df,
            target_date="2026-10-26",
            upcoming_games=upcoming,
            roster_map=roster_map,
        )

        assert len(features) == 1
        row = features.iloc[0]

        # Must strictly reflect only the 3 current-season games: 10.0, 12.0, 14.0
        assert row["games_played_before_target"] == 3
        assert row["last_game"] == 14.0
        assert row["last3_avg"] == (10.0 + 12.0 + 14.0) / 3.0  # 12.0
        assert row["expand_mean"] == (10.0 + 12.0 + 14.0) / 3.0  # 12.0, NOT 12 * 3 + 30 * 10 / 13

    def test_insufficient_current_season_history_does_not_fallback_to_previous_season(self):
        """Player with 0 current-season games does not get inference features formed from prior season."""
        prior_games = [
            {"gameId": f"prev_{i}", "personId": 301, "teamId": 5, "playerName": "Inactive Player", "game_date": f"2026-03-{10+i:02d}", "rating": 25.0}
            for i in range(5)
        ]
        history_df = pd.DataFrame(prior_games)

        upcoming = pd.DataFrame([{"gameId": "target_2", "homeTeamId": 5, "awayTeamId": 6, "nba_game_date": "2026-10-26"}])

        # Case 1: Without roster_map, player is not in current season -> 0 rows generated
        features_no_roster = construct_prediction_features(
            history_df,
            target_date="2026-10-26",
            upcoming_games=upcoming,
            roster_map=None,
        )
        assert len(features_no_roster) == 0

        # Case 2: With roster_map (active on roster but 0 current-season games), gets baseline 0s, NEVER prior 25.0
        roster_map = {301: {"teamId": 5, "teamName": "Team Y", "playerName": "Inactive Player"}}
        features_with_roster = construct_prediction_features(
            history_df,
            target_date="2026-10-26",
            upcoming_games=upcoming,
            roster_map=roster_map,
        )
        assert len(features_with_roster) == 1
        row = features_with_roster.iloc[0]
        assert row["games_played_before_target"] == 0
        assert row["last_game"] == 0.0
        assert row["expand_mean"] == 0.0
        assert row["rating"] == 0.0

    def test_season_transition_dynamically_resolves_future_seasons(self):
        """Season boundary resolves dynamically for 2026, 2027, 2028 and beyond."""
        # 1. During early 2026-27 season (Oct 2026)
        start, end = determine_season_boundary(None, reference_date="2026-10-25")
        assert start == "2026-10-20"
        assert end == "2027-10-19"

        # 2. During mid 2026-27 season (Jan 2027)
        start, end = determine_season_boundary(None, reference_date="2027-01-15")
        assert start == "2026-10-20"
        assert end == "2027-10-19"

        # 3. Next season rollover (Oct 2027)
        start, end = determine_season_boundary(None, reference_date="2027-10-25")
        assert start == "2027-10-20"
        assert end == "2028-10-19"

        # 4. Future season rollover (Nov 2028)
        start, end = determine_season_boundary(None, reference_date="2028-11-01")
        assert start == "2028-10-20"
        assert end == "2029-10-19"

    def test_candidate_d_independent_of_training_dataset(self):
        """Candidate D season bonus calculations remain independent of model training data."""
        ratings = []
        for pid in range(1, 21):
            for g in range(4):
                ratings.append({
                    "personId": pid,
                    "playerName": f"Player {pid}",
                    "teamId": 100,
                    "gameId": f"g_{pid}_{g}",
                    "game_date": f"2026-10-{20+g}",
                    "rating": float(pid * 2),
                })
        ratings_df = pd.DataFrame(ratings)

        mults = compute_season_multipliers(ratings_df, reference_date="2026-10-25")

        # Candidate D requirements:
        assert all(mults["season_game_count"] == 4)
        assert all(mults["season_multiplier_eligible"])
        eligible = mults[mults["season_performance_bonus"] > 0]
        assert len(eligible) > 0
        assert "season_performance_bonus" in mults.columns
        assert "season_avg_rating" in mults.columns

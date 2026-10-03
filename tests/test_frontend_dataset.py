"""
Phase 2E.2B — Regression tests for offseason / no-games schedule behavior.
Covers:
1. Schedule contains multiple games -> only participating teams appear in draft pool.
2. Schedule contains single game -> only its two teams appear.
3. Schedule is empty -> frontend_home dataset contains exactly zero players.
4. Empty schedule -> no player receives opponentName = 'Unknown' as an artificial fallback.
5. Existing prediction dataset remains completely unchanged when schedule is empty.
6. Injury statuses and predicted ratings are preserved intact when games exist.
7. frontend/generate_json.py handles empty parquet files by producing valid empty JSON array.
"""
import os
import sys
import json
import pytest
import tempfile
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.build_frontend_dataset import build_frontend_dataset
from frontend.generate_json import convert_parquet_to_json


@pytest.fixture
def mock_predictions_df():
    """Returns sample predictions across multiple teams with various injury statuses."""
    return pd.DataFrame({
        "personId": [101, 102, 201, 202, 301, 302, 401, 402],
        "playerName": [
            "Celtics Star", "Celtics Guard",
            "Heat Star", "Heat Guard",
            "Lakers Star", "Lakers Guard",
            "Warriors Star", "Warriors Guard"
        ],
        "teamId": [
            1610612738, 1610612738,  # Celtics
            1610612748, 1610612748,  # Heat
            1610612747, 1610612747,  # Lakers
            1610612744, 1610612744   # Warriors
        ],
        "rating": [5.5, 4.2, 5.0, 3.8, 6.1, 4.5, 5.8, 4.0],
        "predicted_rating": [5.6, 4.3, 5.1, 3.9, 6.0, 4.4, 5.9, 4.1],
        "status": ["Active", "Probable", "Active", "Questionable", "Active", "Out", "Active", "Available"]
    })


# ── Test 1: Multiple games schedule ──────────────────────────────────────────

def test_multiple_games_schedule_filters_participating_teams_only(mock_predictions_df):
    """When Celtics vs Heat and Lakers vs Warriors play, all 4 teams appear."""
    games_df = pd.DataFrame([
        {"gameId": "0022501001", "homeTeamId": 1610612738, "awayTeamId": 1610612748},
        {"gameId": "0022501002", "homeTeamId": 1610612747, "awayTeamId": 1610612744}
    ])

    with tempfile.TemporaryDirectory() as tmpdir:
        home_out = os.path.join(tmpdir, "frontend_home.parquet")
        all_out = os.path.join(tmpdir, "frontend_all.parquet")

        build_frontend_dataset(
            pred_df=mock_predictions_df,
            games_df=games_df,
            output_home_path=home_out,
            output_all_path=all_out
        )

        home_res = pd.read_parquet(home_out)
        all_res = pd.read_parquet(all_out)

        # All 4 teams playing appear in home_res
        assert set(home_res["teamId"].unique()) == {1610612738, 1610612748, 1610612747, 1610612744}
        assert len(all_res) == len(mock_predictions_df)

        # Matchups are correctly assigned
        celtics_player = home_res[home_res["teamId"] == 1610612738].iloc[0]
        assert celtics_player["opponentName"] == "Heat"


# ── Test 2: Single game schedule ─────────────────────────────────────────────

def test_single_game_schedule_filters_only_two_teams(mock_predictions_df):
    """When only Celtics vs Heat play, Lakers and Warriors must NOT appear in frontend_home."""
    games_df = pd.DataFrame([
        {"gameId": "0022501001", "homeTeamId": 1610612738, "awayTeamId": 1610612748}
    ])

    with tempfile.TemporaryDirectory() as tmpdir:
        home_out = os.path.join(tmpdir, "frontend_home.parquet")
        all_out = os.path.join(tmpdir, "frontend_all.parquet")

        build_frontend_dataset(
            pred_df=mock_predictions_df,
            games_df=games_df,
            output_home_path=home_out,
            output_all_path=all_out
        )

        home_res = pd.read_parquet(home_out)
        assert set(home_res["teamId"].unique()) == {1610612738, 1610612748}
        assert 1610612747 not in home_res["teamId"].values
        assert 1610612744 not in home_res["teamId"].values


# ── Test 3: Empty schedule produces zero draft players ────────────────────────

def test_empty_schedule_produces_zero_draft_players(mock_predictions_df):
    """When next_day_games is empty, frontend_home must be an empty dataset (0 rows)."""
    empty_games_df = pd.DataFrame(columns=["gameId", "homeTeamId", "awayTeamId"])

    with tempfile.TemporaryDirectory() as tmpdir:
        home_out = os.path.join(tmpdir, "frontend_home.parquet")
        all_out = os.path.join(tmpdir, "frontend_all.parquet")

        build_frontend_dataset(
            pred_df=mock_predictions_df,
            games_df=empty_games_df,
            output_home_path=home_out,
            output_all_path=all_out
        )

        home_res = pd.read_parquet(home_out)
        all_res = pd.read_parquet(all_out)

        # frontend_home must contain 0 players (no fallback to all 30 teams)
        assert len(home_res) == 0
        # frontend_all still preserves complete league predictions
        assert len(all_res) == len(mock_predictions_df)


# ── Test 4: Empty schedule does not assign 'Unknown' opponent ─────────────────

def test_empty_schedule_does_not_assign_unknown_opponent(mock_predictions_df):
    """When schedule is empty, players must receive empty string (''), not 'Unknown'."""
    empty_games_df = pd.DataFrame(columns=["gameId", "homeTeamId", "awayTeamId"])

    with tempfile.TemporaryDirectory() as tmpdir:
        home_out = os.path.join(tmpdir, "frontend_home.parquet")
        all_out = os.path.join(tmpdir, "frontend_all.parquet")

        build_frontend_dataset(
            pred_df=mock_predictions_df,
            games_df=empty_games_df,
            output_home_path=home_out,
            output_all_path=all_out
        )

        all_res = pd.read_parquet(all_out)
        # Ensure opponentName is empty string, not 'Unknown'
        assert (all_res["opponentName"] == "Unknown").sum() == 0
        assert (all_res["opponentName"] == "").sum() == len(all_res)


# ── Test 5: Existing prediction dataset remains unchanged ─────────────────────

def test_empty_schedule_preserves_prediction_values(mock_predictions_df):
    """Empty schedule generation does not alter underlying predicted ratings or personIds."""
    empty_games_df = pd.DataFrame(columns=["gameId", "homeTeamId", "awayTeamId"])

    with tempfile.TemporaryDirectory() as tmpdir:
        home_out = os.path.join(tmpdir, "frontend_home.parquet")
        all_out = os.path.join(tmpdir, "frontend_all.parquet")

        build_frontend_dataset(
            pred_df=mock_predictions_df,
            games_df=empty_games_df,
            output_home_path=home_out,
            output_all_path=all_out
        )

        all_res = pd.read_parquet(all_out)
        left_series = mock_predictions_df.sort_values("personId")["predicted_rating"].reset_index(drop=True)
        right_series = all_res.sort_values("personId")["predicted_rating"].reset_index(drop=True)
        pd.testing.assert_series_equal(left_series, right_series, check_names=False)


# ── Test 6: Injury status and rating integrity with games ────────────────────

def test_injury_status_preserved_in_frontend_datasets(mock_predictions_df):
    """Status rankings and predicted ratings flow accurately into frontend artifacts."""
    games_df = pd.DataFrame([
        {"gameId": "0022501001", "homeTeamId": 1610612738, "awayTeamId": 1610612748}
    ])

    with tempfile.TemporaryDirectory() as tmpdir:
        home_out = os.path.join(tmpdir, "frontend_home.parquet")
        all_out = os.path.join(tmpdir, "frontend_all.parquet")

        build_frontend_dataset(
            pred_df=mock_predictions_df,
            games_df=games_df,
            output_home_path=home_out,
            output_all_path=all_out
        )

        home_res = pd.read_parquet(home_out)
        celtics_star = home_res[home_res["playerName"] == "Celtics Star"].iloc[0]
        assert celtics_star["status"] == "Active"
        assert celtics_star["predicted_rating"] == pytest.approx(5.6)


# ── Test 7: generate_json handles empty parquet gracefully ───────────────────

def test_generate_json_handles_empty_parquet():
    """convert_parquet_to_json writes [] to JSON when parquet is empty, without raising exceptions."""
    empty_df = pd.DataFrame(columns=["personId", "playerName", "predicted_rating"])

    with tempfile.TemporaryDirectory() as tmpdir:
        parquet_path = os.path.join(tmpdir, "empty.parquet")
        json_path = os.path.join(tmpdir, "empty.json")

        empty_df.to_parquet(parquet_path, index=False)

        # Must convert cleanly without raising
        convert_parquet_to_json(parquet_path, json_path)

        assert os.path.exists(json_path)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data == []

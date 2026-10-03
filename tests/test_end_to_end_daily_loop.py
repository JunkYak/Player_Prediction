"""
tests/test_end_to_end_daily_loop.py

End-to-End verification of the complete daily product loop:
1. Batch pipeline execution (via Orchestrator)
2. Atomic publication of fresh serving artifacts
3. FastAPI serving layer detection and reload
4. Client consumption matching React src/api.js requirements
5. Critical failure safety (failed runs preserve existing serving artifacts)
"""

import os
import sys
import time
import pytest
import pandas as pd
from unittest.mock import patch
from fastapi.testclient import TestClient

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import create_app
from api.config import Settings
from scripts.orchestrator import execute_daily_pipeline, PipelineLock
from scripts.build_frontend_dataset import build_frontend_dataset


@pytest.fixture
def mock_pipeline_environment(tmp_path):
    """Sets up an isolated data directory with canonical datasets and mock runner."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Base test player record
    player_tatum = {
        "gameId": "00223001",
        "teamId": 1610612738,
        "personId": 1628369,
        "playerName": "Jayson Tatum",
        "rating": 91.5,
        "last_game_id": "00223000",
        "last_game_date": "2026-04-01",
        "game_date": "2026-04-02",
        "last_game": 91.5,
        "last3_avg": 90.0,
        "last5_avg": 89.0,
        "last7_avg": 88.0,
        "predicted_rating": 93.0,
        "status": "Active",
        "injury_reason": "",
        "injury_date": "",
        "injury_source": "",
        "status_rank": 0,
        "teamName": "Celtics",
        "opponentId": 1610612748,
        "opponentName": "Heat",
        "headshot": "https://cdn.nba.com/headshots/nba/latest/1040x760/1628369.png",
    }

    # Write initial serving artifacts
    df_init = pd.DataFrame([player_tatum])
    df_init.to_parquet(data_dir / "frontend_all.parquet")
    df_init.to_parquet(data_dir / "frontend_home.parquet")

    settings = Settings(
        data_dir=data_dir,
        allowed_origins=["http://localhost:5173"],
        env="testing",
        reload=False,
    )
    return {
        "data_dir": data_dir,
        "settings": settings,
        "status_file": str(data_dir / "pipeline_status.json"),
        "lock_file": str(data_dir / "pipeline.lock"),
        "initial_player": player_tatum,
    }


def test_complete_daily_product_loop(mock_pipeline_environment):
    env = mock_pipeline_environment
    data_dir = env["data_dir"]
    settings = env["settings"]

    # 1. Start FastAPI application
    app = create_app(settings)
    with TestClient(app) as client:
        # Step A: Verify initial FastAPI state
        res_players_1 = client.get("/api/v1/players")
        assert res_players_1.status_code == 200
        data1 = res_players_1.json()
        assert data1["games_scheduled"] is True
        assert len(data1["players"]) == 1
        assert data1["players"][0]["playerName"] == "Jayson Tatum"
        assert data1["players"][0]["predicted_rating"] == 93.0

        # Step B: New NBA games complete → Orchestrator executes daily update
        def mock_daily_update_work(*args, **kwargs):
            # Simulate pipeline generating new predictions with Jaylen Brown added
            time.sleep(0.05)  # Ensure timestamp advances
            new_player = dict(
                env["initial_player"],
                personId=1627759,
                playerName="Jaylen Brown",
                predicted_rating=88.5,
            )
            df_updated = pd.DataFrame([env["initial_player"], new_player])
            build_frontend_dataset(
                output_home_path=str(data_dir / "frontend_home.parquet"),
                output_all_path=str(data_dir / "frontend_all.parquet"),
                pred_df=df_updated,
                games_df=pd.DataFrame([{"homeTeamId": 1610612738, "awayTeamId": 1610612748}]),
            )

        with patch("scripts.orchestrator.run_full_pipeline", side_effect=mock_daily_update_work):
            success = execute_daily_pipeline(
                mode="full",
                skip_fetch=True,
                skip_train=True,
                status_file=env["status_file"],
                lock_file=env["lock_file"],
            )
            assert success is True

        # Step C: FastAPI serves fresh rankings immediately on next request
        res_players_2 = client.get("/api/v1/players")
        assert res_players_2.status_code == 200
        data2 = res_players_2.json()
        assert data2["games_scheduled"] is True
        assert len(data2["players"]) == 2
        player_names = [p["playerName"] for p in data2["players"]]
        assert "Jayson Tatum" in player_names
        assert "Jaylen Brown" in player_names

        # Step D: Verify React-consumed contracts (/api/v1/teams and /api/v1/players)
        res_teams = client.get("/api/v1/teams")
        assert res_teams.status_code == 200
        teams_data = res_teams.json()
        assert "total_players" in teams_data
        assert teams_data["total_players"] == 2
        assert len(teams_data["players"]) == 2


def test_failed_daily_run_preserves_valid_serving_artifact(mock_pipeline_environment):
    env = mock_pipeline_environment
    data_dir = env["data_dir"]
    settings = env["settings"]

    app = create_app(settings)
    with TestClient(app) as client:
        # Step A: Initial valid state
        res_init = client.get("/api/v1/players")
        assert res_init.status_code == 200
        assert len(res_init.json()["players"]) == 1

        # Step B: Pipeline execution fails during external API or feature stage
        def mock_failed_work(*args, **kwargs):
            raise ConnectionError("NBA API gateway timeout 504")

        with patch("scripts.orchestrator.run_full_pipeline", side_effect=mock_failed_work):
            success = execute_daily_pipeline(
                mode="full",
                status_file=env["status_file"],
                lock_file=env["lock_file"],
            )
            assert success is False

        # Step C: Serving artifacts MUST remain valid and FastAPI continues serving existing data
        res_after_fail = client.get("/api/v1/players")
        assert res_after_fail.status_code == 200
        assert len(res_after_fail.json()["players"]) == 1
        assert res_after_fail.json()["players"][0]["playerName"] == "Jayson Tatum"


def test_offseason_pipeline_run_produces_clean_empty_slate(mock_pipeline_environment):
    env = mock_pipeline_environment
    data_dir = env["data_dir"]
    settings = env["settings"]

    app = create_app(settings)
    with TestClient(app) as client:
        # Simulate offseason update: 0 games scheduled
        def mock_offseason_work(*args, **kwargs):
            time.sleep(0.05)
            build_frontend_dataset(
                output_home_path=str(data_dir / "frontend_home.parquet"),
                output_all_path=str(data_dir / "frontend_all.parquet"),
                pred_df=pd.DataFrame([env["initial_player"]]),
                games_df=pd.DataFrame(),  # 0 games scheduled
            )

        with patch("scripts.orchestrator.run_full_pipeline", side_effect=mock_offseason_work):
            success = execute_daily_pipeline(
                mode="full",
                status_file=env["status_file"],
                lock_file=env["lock_file"],
            )
            assert success is True

        # FastAPI returns games_scheduled=False, players=[]
        res_offseason = client.get("/api/v1/players")
        assert res_offseason.status_code == 200
        offseason_data = res_offseason.json()
        assert offseason_data["games_scheduled"] is False
        assert offseason_data["players"] == []
        assert offseason_data["game_date"] is None

        # /api/v1/teams remains fully available
        res_teams = client.get("/api/v1/teams")
        assert res_teams.status_code == 200
        assert res_teams.json()["total_players"] == 1

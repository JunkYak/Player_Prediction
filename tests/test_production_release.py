"""
tests/test_production_release.py

Phase 2E.9 Production Release & Cloud Deployment Verification Suite.

Validates:
1. Production FastAPI server contract (/health, /ready, /status, /api/v1/players, /api/v1/teams).
2. Autonomous pipeline update & mtime reload without server restart.
3. Persistent volume simulation & artifact flow.
4. Failure safety under external API outages.
5. Production CORS configuration.
6. Unified frontend static serving integration.
7. Offseason / zero-games handling.
"""

import os
import sys
import json
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
def production_env(tmp_path):
    """Creates a production-like directory layout with persistent data and static build assets."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    static_dir = tmp_path / "static"
    static_dir.mkdir()

    # Create dummy index.html in static_dir
    index_html = static_dir / "index.html"
    index_html.write_text("<!DOCTYPE html><html><head><title>NBA ProPredict</title></head><body><div id='root'></div></body></html>")

    # Sample canonical player record
    player_data = {
        "gameId": "0022501001",
        "teamId": 1610612738,
        "personId": 1628369,
        "playerName": "Jayson Tatum",
        "rating": 92.0,
        "last_game_id": "0022501000",
        "last_game_date": "2026-04-06",
        "game_date": "2026-04-07",
        "last_game": 92.0,
        "last3_avg": 91.0,
        "last5_avg": 90.5,
        "last7_avg": 89.0,
        "predicted_rating": 93.5,
        "status": "Active",
        "injury_reason": "Available",
        "injury_date": "2026-04-07",
        "injury_source": "Official NBA Injury Report",
        "status_rank": 0,
        "teamName": "Celtics",
        "opponentId": 1610612748,
        "opponentName": "Heat",
        "headshot": "https://cdn.nba.com/headshots/nba/latest/1040x760/1628369.png",
    }

    df = pd.DataFrame([player_data])
    df.to_parquet(data_dir / "frontend_all.parquet")
    df.to_parquet(data_dir / "frontend_home.parquet")

    # Write initial pipeline status
    status_file = data_dir / "pipeline_status.json"
    with open(status_file, "w") as f:
        json.dump({
            "status": "SUCCESS",
            "mode": "full",
            "start_time": "2026-10-02T10:00:00",
            "end_time": "2026-10-02T10:02:15",
            "duration_seconds": 135.0,
            "error": None,
            "artifacts_updated": True,
        }, f)

    settings = Settings(
        data_dir=data_dir,
        allowed_origins=["https://propredict.nba.com", "http://localhost:5173"],
        env="production",
        reload=False,
        static_dir=static_dir,
    )

    return {
        "data_dir": data_dir,
        "static_dir": static_dir,
        "settings": settings,
        "player": player_data,
        "status_file": str(status_file),
        "lock_file": str(data_dir / "pipeline.lock"),
    }


# ==============================================================================
# 1. Production API & Monitoring Endpoints
# ==============================================================================
def test_production_health_and_monitoring_endpoints(production_env):
    app = create_app(production_env["settings"])
    with TestClient(app) as client:
        # 1. Health liveness probe
        r_health = client.get("/health")
        assert r_health.status_code == 200
        assert r_health.json() == {"status": "ok"}

        # 2. Ready probe with pipeline diagnostics
        r_ready = client.get("/ready")
        assert r_ready.status_code == 200
        data_ready = r_ready.json()
        assert data_ready["status"] == "ready"
        assert data_ready["slate_count"] == 1
        assert data_ready["all_count"] == 1
        assert "pipeline" in data_ready
        assert data_ready["pipeline"]["last_status"] == "SUCCESS"
        assert data_ready["pipeline"]["last_duration_seconds"] == 135.0

        # 3. Operational status endpoint
        r_status = client.get("/status")
        assert r_status.status_code == 200
        data_status = r_status.json()
        assert data_status["status"] == "ok"
        assert data_status["artifacts_loaded"] is True
        assert data_status["slate_count"] == 1
        assert data_status["all_count"] == 1


# ==============================================================================
# 2. Unified Static Frontend Serving
# ==============================================================================
def test_production_static_frontend_serving(production_env):
    app = create_app(production_env["settings"])
    with TestClient(app) as client:
        # GET / must serve the React SPA index.html
        resp = client.get("/")
        assert resp.status_code == 200
        assert "NBA ProPredict" in resp.text
        assert "<div id='root'></div>" in resp.text


# ==============================================================================
# 3. Production CORS Restrictions
# ==============================================================================
def test_production_cors_restricted(production_env):
    app = create_app(production_env["settings"])
    with TestClient(app) as client:
        # Origin in allowed list: accepted
        resp_allowed = client.get(
            "/api/v1/players",
            headers={"Origin": "https://propredict.nba.com"},
        )
        assert resp_allowed.status_code == 200
        assert resp_allowed.headers.get("access-control-allow-origin") == "https://propredict.nba.com"

        # Unauthorized origin: rejected
        resp_blocked = client.get(
            "/api/v1/players",
            headers={"Origin": "https://malicious-site.com"},
        )
        assert resp_blocked.status_code == 200
        assert "access-control-allow-origin" not in resp_blocked.headers


# ==============================================================================
# 4. Autonomous Daily Update Flow (Without Server Restart)
# ==============================================================================
def test_autonomous_daily_update_and_mtime_reload(production_env):
    app = create_app(production_env["settings"])
    with TestClient(app) as client:
        # Verify initial serving state
        init_players = client.get("/api/v1/players").json()["players"]
        assert len(init_players) == 1
        assert init_players[0]["playerName"] == "Jayson Tatum"

        # Simulate scheduled orchestrator run executing in background
        def mock_daily_run(*args, **kwargs):
            time.sleep(0.05)  # Ensure mtime advances
            new_player = dict(
                production_env["player"],
                personId=1627759,
                playerName="Jaylen Brown",
                predicted_rating=89.0,
            )
            build_frontend_dataset(
                output_home_path=str(production_env["data_dir"] / "frontend_home.parquet"),
                output_all_path=str(production_env["data_dir"] / "frontend_all.parquet"),
                pred_df=pd.DataFrame([production_env["player"], new_player]),
                games_df=pd.DataFrame([{"homeTeamId": 1610612738, "awayTeamId": 1610612748}]),
            )

        with patch("scripts.orchestrator.run_full_pipeline", side_effect=mock_daily_run):
            success = execute_daily_pipeline(
                mode="full",
                skip_fetch=True,
                skip_train=True,
                status_file=production_env["status_file"],
                lock_file=production_env["lock_file"],
            )
            assert success is True

        # Next incoming request to FastAPI MUST see updated rankings with zero server restart
        updated_res = client.get("/api/v1/players")
        assert updated_res.status_code == 200
        updated_players = updated_res.json()["players"]
        assert len(updated_players) == 2
        names = [p["playerName"] for p in updated_players]
        assert "Jayson Tatum" in names
        assert "Jaylen Brown" in names


# ==============================================================================
# 5. Production Failure Safety Under External API Failure
# ==============================================================================
def test_production_failure_safety_under_api_outage(production_env):
    app = create_app(production_env["settings"])
    with TestClient(app) as client:
        # Record valid state
        valid_res = client.get("/api/v1/players")
        assert len(valid_res.json()["players"]) == 1

        # Simulate NBA API 503 outage during scheduled pipeline run
        with patch("scripts.orchestrator.run_full_pipeline", side_effect=ConnectionError("NBA API 503 Service Unavailable")):
            success = execute_daily_pipeline(
                mode="full",
                status_file=production_env["status_file"],
                lock_file=production_env["lock_file"],
            )
            assert success is False

        # Status file recorded the failure
        with open(production_env["status_file"], "r") as f:
            status_data = json.load(f)
            assert status_data["status"] == "FAILED"
            assert "NBA API 503" in status_data["error"]

        # FastAPI continues serving original valid data seamlessly
        client_res = client.get("/api/v1/players")
        assert client_res.status_code == 200
        assert len(client_res.json()["players"]) == 1
        assert client_res.json()["players"][0]["playerName"] == "Jayson Tatum"

        # Concurrency lock is cleanly freed
        assert not os.path.exists(production_env["lock_file"])


# ==============================================================================
# 6. Offseason Safe Slate Handling
# ==============================================================================
def test_production_offseason_handling(production_env):
    app = create_app(production_env["settings"])
    with TestClient(app) as client:
        # Simulate offseason update with 0 games scheduled
        def mock_offseason_run(*args, **kwargs):
            time.sleep(0.05)
            build_frontend_dataset(
                output_home_path=str(production_env["data_dir"] / "frontend_home.parquet"),
                output_all_path=str(production_env["data_dir"] / "frontend_all.parquet"),
                pred_df=pd.DataFrame([production_env["player"]]),
                games_df=pd.DataFrame(),  # 0 games
            )

        with patch("scripts.orchestrator.run_full_pipeline", side_effect=mock_offseason_run):
            success = execute_daily_pipeline(
                mode="full",
                status_file=production_env["status_file"],
                lock_file=production_env["lock_file"],
            )
            assert success is True

        res_offseason = client.get("/api/v1/players")
        assert res_offseason.status_code == 200
        body = res_offseason.json()
        assert body["games_scheduled"] is False
        assert body["players"] == []

        # Teams endpoint remains fully available
        res_teams = client.get("/api/v1/teams")
        assert res_teams.status_code == 200
        assert res_teams.json()["total_players"] == 1

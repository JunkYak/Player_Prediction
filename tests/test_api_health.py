import os
import sys
import pytest
from fastapi.testclient import TestClient
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.config import Settings
from api.main import create_app
from api.loader import REQUIRED_COLUMNS


def make_test_df(num_rows=2, **overrides):
    """Create a minimal valid DataFrame with all 22 required columns."""
    data = {
        "gameId": [f"002250100{i}" for i in range(num_rows)],
        "teamId": [1610612737] * num_rows,
        "personId": [1000 + i for i in range(num_rows)],
        "playerName": [f"Player {i}" for i in range(num_rows)],
        "rating": [5.0 + i for i in range(num_rows)],
        "last_game_id": [f"002250100{i}" for i in range(num_rows)],
        "last_game_date": ["2026-04-06"] * num_rows,
        "game_date": ["2026-04-07"] * num_rows,
        "last_game": [5.0 + i for i in range(num_rows)],
        "last3_avg": [4.5 + i for i in range(num_rows)],
        "last5_avg": [4.2 + i for i in range(num_rows)],
        "last7_avg": [4.0 + i for i in range(num_rows)],
        "predicted_rating": [4.8 + i for i in range(num_rows)],
        "status": ["Active"] * num_rows,
        "injury_reason": ["Available"] * num_rows,
        "injury_date": ["2026-04-07"] * num_rows,
        "injury_source": ["test_source"] * num_rows,
        "status_rank": [0] * num_rows,
        "teamName": ["Hawks"] * num_rows,
        "opponentId": [1610612738.0] * num_rows,
        "opponentName": ["Celtics"] * num_rows,
        "headshot": ["https://cdn.nba.com/headshots/test.png"] * num_rows,
    }
    data.update(overrides)
    return pd.DataFrame(data)


def test_health_returns_200_and_ok():
    """Verify /health returns HTTP 200 with status ok."""
    app = create_app()
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


def test_health_independent_of_artifact_availability(tmp_path):
    """Verify /health succeeds even when data directory is empty or missing."""
    empty_dir = tmp_path / "non_existent_data"
    settings = Settings(
        data_dir=empty_dir,
        allowed_origins=["http://localhost:5173"],
        env="testing",
        reload=False,
    )
    app = create_app(settings)
    with TestClient(app) as client:
        # /ready should fail with 503
        r_ready = client.get("/ready")
        assert r_ready.status_code == 503

        # /health must still succeed with 200
        r_health = client.get("/health")
        assert r_health.status_code == 200
        assert r_health.json() == {"status": "ok"}


def test_ready_endpoint_success(tmp_path):
    """Verify /ready returns 200 and expected metadata when artifacts are present."""
    df_home = make_test_df(num_rows=2, game_date=["2026-04-07", "2026-04-07"])
    df_all = make_test_df(num_rows=5)

    df_home.to_parquet(tmp_path / "frontend_home.parquet")
    df_all.to_parquet(tmp_path / "frontend_all.parquet")

    settings = Settings(
        data_dir=tmp_path,
        allowed_origins=["http://localhost:5173"],
        env="testing",
        reload=False,
    )
    app = create_app(settings)
    with TestClient(app) as client:
        resp = client.get("/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"
        assert data["slate_count"] == 2
        assert data["all_count"] == 5
        assert data["game_date"] == "2026-04-07"
        assert "generated_at" in data
        assert data["generated_at"] is not None


def test_ready_endpoint_failure_when_artifacts_missing(tmp_path):
    """Verify /ready returns 503 with machine-readable code when artifacts missing."""
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()

    settings = Settings(
        data_dir=empty_dir,
        allowed_origins=["http://localhost:5173"],
        env="testing",
        reload=False,
    )
    app = create_app(settings)
    with TestClient(app) as client:
        resp = client.get("/ready")
        assert resp.status_code == 503
        data = resp.json()
        assert "detail" in data
        assert data["code"] == "ARTIFACT_MISSING"

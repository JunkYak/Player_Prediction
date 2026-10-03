import os
import sys
import pytest
from fastapi.testclient import TestClient
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.config import Settings
from api.main import create_app
from api.loader import CANONICAL_PLAYER_FIELDS


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


def test_players_endpoint_valid_response(tmp_path):
    """Verify valid player response, games_scheduled=true, and canonical fields."""
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
        resp = client.get("/api/v1/players")
        assert resp.status_code == 200
        data = resp.json()

        assert data["game_date"] == "2026-04-07"
        assert data["games_scheduled"] is True
        assert len(data["players"]) == 2
        assert "generated_at" in data

        first_player = data["players"][0]
        # Verify all canonical fields are present
        for field in CANONICAL_PLAYER_FIELDS:
            assert field in first_player, f"Missing canonical field {field}"

        # Verify no unnecessary columns are exposed
        assert "injury_reason" not in first_player
        assert "injury_source" not in first_player
        assert "gameId" not in first_player


def test_players_endpoint_empty_home_dataset(tmp_path):
    """Verify empty home dataset returns HTTP 200 with players=[] and games_scheduled=false."""
    # Empty DataFrame with full required schema
    df_home_empty = make_test_df(num_rows=0)
    df_all = make_test_df(num_rows=5)

    df_home_empty.to_parquet(tmp_path / "frontend_home.parquet")
    df_all.to_parquet(tmp_path / "frontend_all.parquet")

    settings = Settings(
        data_dir=tmp_path,
        allowed_origins=["http://localhost:5173"],
        env="testing",
        reload=False,
    )
    app = create_app(settings)
    with TestClient(app) as client:
        resp = client.get("/api/v1/players")
        assert resp.status_code == 200
        data = resp.json()

        assert data["game_date"] is None
        assert data["games_scheduled"] is False
        assert data["players"] == []
        assert "generated_at" in data


def test_players_endpoint_artifact_failure_returns_503(tmp_path):
    """Verify artifact failure returns HTTP 503 with machine-readable code."""
    empty_dir = tmp_path / "missing_data"
    empty_dir.mkdir()

    settings = Settings(
        data_dir=empty_dir,
        allowed_origins=["http://localhost:5173"],
        env="testing",
        reload=False,
    )
    app = create_app(settings)
    with TestClient(app) as client:
        resp = client.get("/api/v1/players")
        assert resp.status_code == 503
        data = resp.json()
        assert "detail" in data
        assert data["code"] in ("ARTIFACT_MISSING", "ARTIFACT_NOT_LOADED", "ARTIFACT_UNREADABLE")

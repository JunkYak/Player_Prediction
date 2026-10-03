import os
import sys
import json
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.config import Settings
from api.loader import (
    REQUIRED_COLUMNS,
    CANONICAL_PLAYER_FIELDS,
    clean_record_value,
    dataframe_to_canonical_records,
    validate_schema,
)
from api.main import create_app


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


def test_valid_schema_accepted():
    """Verify that a DataFrame containing all 22 required columns passes validation."""
    df = make_test_df(num_rows=3)
    validate_schema(df, "test_artifact.parquet")


def test_missing_required_columns_rejected():
    """Verify that omitting any required column raises ValueError during validation."""
    df = make_test_df(num_rows=3)
    for col in ["predicted_rating", "personId", "opponentName", "headshot"]:
        corrupted = df.drop(columns=[col])
        with pytest.raises(ValueError, match="missing required columns"):
            validate_schema(corrupted, "corrupted.parquet")


def test_missing_required_columns_fails_api_readiness(tmp_path):
    """Verify that an artifact missing required columns causes HTTP 503 readiness failure."""
    df_home = make_test_df(num_rows=2)
    # df_all missing 'predicted_rating'
    df_all_bad = make_test_df(num_rows=5).drop(columns=["predicted_rating"])

    df_home.to_parquet(tmp_path / "frontend_home.parquet")
    df_all_bad.to_parquet(tmp_path / "frontend_all.parquet")

    settings = Settings(
        data_dir=tmp_path,
        allowed_origins=["http://localhost:5173"],
        env="testing",
        reload=False,
    )
    app = create_app(settings)
    with TestClient(app) as client:
        resp = client.get("/ready")
        assert resp.status_code == 503
        data = resp.json()
        assert data["code"] == "ARTIFACT_UNREADABLE"


def test_nan_and_null_values_are_safely_converted():
    """Verify that NaN and null values are converted to Python None and valid JSON null."""
    df = make_test_df(
        num_rows=2,
        opponentName=[np.nan, "Celtics"],
        last_game=[np.nan, 4.5],
        last3_avg=[None, np.nan],
        headshot=[np.nan, None],
    )
    records = dataframe_to_canonical_records(df)

    assert records[0]["opponentName"] is None
    assert records[0]["last_game"] is None
    assert records[0]["last3_avg"] is None
    assert records[0]["headshot"] is None

    # Serialization with allow_nan=False must succeed without ValueError
    serialized = json.dumps(records, allow_nan=False)
    parsed = json.loads(serialized)
    assert parsed[0]["opponentName"] is None
    assert parsed[0]["last_game"] is None
    assert parsed[1]["opponentName"] == "Celtics"
    assert parsed[1]["last_game"] == 4.5


def test_empty_home_parquet_is_accepted_for_readiness(tmp_path):
    """Verify empty home parquet (off-day) is accepted and API is ready."""
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
        resp = client.get("/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"
        assert data["slate_count"] == 0
        assert data["all_count"] == 5
        assert data["game_date"] is None


def test_empty_all_players_parquet_is_not_ready(tmp_path):
    """Verify empty all-player parquet fails readiness with HTTP 503."""
    df_home = make_test_df(num_rows=2)
    df_all_empty = make_test_df(num_rows=0)

    df_home.to_parquet(tmp_path / "frontend_home.parquet")
    df_all_empty.to_parquet(tmp_path / "frontend_all.parquet")

    settings = Settings(
        data_dir=tmp_path,
        allowed_origins=["http://localhost:5173"],
        env="testing",
        reload=False,
    )
    app = create_app(settings)
    with TestClient(app) as client:
        resp = client.get("/ready")
        assert resp.status_code == 503
        data = resp.json()
        assert data["code"] == "ARTIFACT_UNREADABLE"

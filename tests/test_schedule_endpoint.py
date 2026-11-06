import os
import sys
import json
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.config import Settings
from api.main import create_app
from tests.test_api_players import make_test_df


def test_schedule_endpoint_confirmed_games(tmp_path):
    """Verify /api/v1/schedule/tomorrow returns 200 with CONFIRMED status and matchups."""
    now_et = datetime.now(ZoneInfo("America/New_York"))
    tomorrow_et = (now_et + timedelta(days=1)).strftime("%Y-%m-%d")

    # Serving artifacts
    df_home = make_test_df(num_rows=2, game_date=[tomorrow_et, tomorrow_et])
    df_all = make_test_df(num_rows=5)
    df_home.to_parquet(tmp_path / "frontend_home.parquet")
    df_all.to_parquet(tmp_path / "frontend_all.parquet")

    # Tomorrow schedule artifact
    sched_df = pd.DataFrame([
        {
            "game_id": "0022600101",
            "game_date": tomorrow_et,
            "schedule_date": tomorrow_et,
            "home_team_id": 1610612738,
            "home_team_name": "Boston Celtics",
            "away_team_id": 1610612747,
            "away_team_name": "Los Angeles Lakers",
            "home_team_tricode": "BOS",
            "away_team_tricode": "LAL",
            "status": "Scheduled",
            "game_status": 1,
            "game_time_utc": f"{tomorrow_et}T23:30:00+00:00",
            "is_postponed": False,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
    ])
    sched_df.to_parquet(tmp_path / "schedule_tomorrow.parquet", index=False)

    with open(tmp_path / "schedule_tomorrow_metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "status": "CONFIRMED",
            "schedule_date": tomorrow_et,
            "target_date": tomorrow_et,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "games_scheduled": True,
            "games_count": 1,
            "is_stale": False,
            "error": None,
        }, f)

    settings = Settings(
        data_dir=tmp_path,
        allowed_origins=["http://localhost:5173"],
        env="testing",
        reload=False,
    )
    app = create_app(settings)
    with TestClient(app) as client:
        resp = client.get("/api/v1/schedule/tomorrow")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "CONFIRMED"
        assert data["games_scheduled"] is True
        assert data["games_count"] == 1
        assert len(data["matchups"]) == 1
        assert 1610612738 in data["team_ids"]
        assert 1610612747 in data["team_ids"]
        assert data["schedule_date"] == tomorrow_et


def test_schedule_endpoint_off_day(tmp_path):
    """Verify /api/v1/schedule/tomorrow returns 200 with OFF_DAY and games_scheduled=False."""
    now_et = datetime.now(ZoneInfo("America/New_York"))
    tomorrow_et = (now_et + timedelta(days=1)).strftime("%Y-%m-%d")

    df_home = make_test_df(num_rows=0)
    df_all = make_test_df(num_rows=5)
    df_home.to_parquet(tmp_path / "frontend_home.parquet")
    df_all.to_parquet(tmp_path / "frontend_all.parquet")

    # Empty schedule
    empty_df = pd.DataFrame(columns=[
        "game_id", "game_date", "schedule_date", "home_team_id", "home_team_name",
        "away_team_id", "away_team_name", "home_team_tricode", "away_team_tricode",
        "status", "game_status", "game_time_utc", "is_postponed", "generated_at",
    ])
    empty_df.to_parquet(tmp_path / "schedule_tomorrow.parquet", index=False)

    with open(tmp_path / "schedule_tomorrow_metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "status": "OFF_DAY",
            "schedule_date": tomorrow_et,
            "target_date": tomorrow_et,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "games_scheduled": False,
            "games_count": 0,
            "is_stale": False,
            "error": None,
        }, f)

    settings = Settings(
        data_dir=tmp_path,
        allowed_origins=["http://localhost:5173"],
        env="testing",
        reload=False,
    )
    app = create_app(settings)
    with TestClient(app) as client:
        resp = client.get("/api/v1/schedule/tomorrow")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "OFF_DAY"
        assert data["games_scheduled"] is False
        assert data["games_count"] == 0
        assert data["matchups"] == []
        assert data["team_ids"] == []


def test_schedule_endpoint_stale(tmp_path):
    """Verify /api/v1/schedule/tomorrow detects stale dates."""
    yesterday_date = "2026-01-01"

    df_home = make_test_df(num_rows=2)
    df_all = make_test_df(num_rows=5)
    df_home.to_parquet(tmp_path / "frontend_home.parquet")
    df_all.to_parquet(tmp_path / "frontend_all.parquet")

    sched_df = pd.DataFrame([
        {
            "game_id": "0022600101",
            "game_date": yesterday_date,
            "schedule_date": yesterday_date,
            "home_team_id": 1610612738,
            "home_team_name": "Boston Celtics",
            "away_team_id": 1610612747,
            "away_team_name": "Los Angeles Lakers",
            "home_team_tricode": "BOS",
            "away_team_tricode": "LAL",
            "status": "Scheduled",
            "game_status": 1,
            "game_time_utc": f"{yesterday_date}T23:30:00+00:00",
            "is_postponed": False,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
    ])
    sched_df.to_parquet(tmp_path / "schedule_tomorrow.parquet", index=False)

    with open(tmp_path / "schedule_tomorrow_metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "status": "CONFIRMED",
            "schedule_date": yesterday_date,
            "target_date": yesterday_date,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "games_scheduled": True,
            "games_count": 1,
            "is_stale": True,
            "error": None,
        }, f)

    settings = Settings(
        data_dir=tmp_path,
        allowed_origins=["http://localhost:5173"],
        env="testing",
        reload=False,
    )
    app = create_app(settings)
    with TestClient(app) as client:
        resp = client.get("/api/v1/schedule/tomorrow")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "STALE"
        assert data["games_scheduled"] is False


def test_schedule_endpoint_missing_artifact_raises_503(tmp_path):
    """Verify HTTP 503 is returned when schedule artifact does not exist."""
    df_home = make_test_df(num_rows=2)
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
    app.state.tomorrow_schedule = None

    with TestClient(app) as client:
        resp = client.get("/api/v1/schedule/tomorrow")
        assert resp.status_code == 503
        data = resp.json()
        assert data["code"] == "SCHEDULE_UNAVAILABLE"


def test_schedule_endpoint_future_dated_raises_503(tmp_path):
    """Verify future-dated schedule (> tomorrow ET) is rejected with 503."""
    now_et = datetime.now(ZoneInfo("America/New_York"))
    day_after_tomorrow_et = (now_et + timedelta(days=2)).strftime("%Y-%m-%d")

    df_home = make_test_df(num_rows=2)
    df_all = make_test_df(num_rows=5)
    df_home.to_parquet(tmp_path / "frontend_home.parquet")
    df_all.to_parquet(tmp_path / "frontend_all.parquet")

    sched_df = pd.DataFrame([{
        "game_id": "0022600101",
        "game_date": day_after_tomorrow_et,
        "schedule_date": day_after_tomorrow_et,
        "home_team_id": 1610612738,
        "home_team_name": "Boston Celtics",
        "away_team_id": 1610612747,
        "away_team_name": "Los Angeles Lakers",
        "home_team_tricode": "BOS",
        "away_team_tricode": "LAL",
        "status": "Scheduled",
        "game_status": 1,
        "game_time_utc": f"{day_after_tomorrow_et}T23:30:00+00:00",
        "is_postponed": False,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }])
    sched_df.to_parquet(tmp_path / "schedule_tomorrow.parquet", index=False)

    with open(tmp_path / "schedule_tomorrow_metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "status": "CONFIRMED",
            "schedule_date": day_after_tomorrow_et,
            "target_date": day_after_tomorrow_et,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "games_scheduled": True,
            "games_count": 1,
            "is_stale": False,
            "error": None,
        }, f)

    settings = Settings(data_dir=tmp_path, allowed_origins=["http://localhost:5173"], env="testing", reload=False)
    app = create_app(settings)
    with TestClient(app) as client:
        resp = client.get("/api/v1/schedule/tomorrow")
        assert resp.status_code == 503
        assert resp.json()["code"] == "SCHEDULE_UNAVAILABLE"


def test_schedule_endpoint_missing_schedule_date_raises_503(tmp_path):
    """Missing schedule_date in metadata must fail closed with 503."""
    df_home = make_test_df(num_rows=2)
    df_all = make_test_df(num_rows=5)
    df_home.to_parquet(tmp_path / "frontend_home.parquet")
    df_all.to_parquet(tmp_path / "frontend_all.parquet")

    sched_df = pd.DataFrame([{
        "game_id": "0022600101",
        "home_team_id": 1610612738,
        "away_team_id": 1610612747,
    }])
    sched_df.to_parquet(tmp_path / "schedule_tomorrow.parquet", index=False)

    with open(tmp_path / "schedule_tomorrow_metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "status": "CONFIRMED",
            "games_scheduled": True,
            "games_count": 1,
        }, f)

    settings = Settings(data_dir=tmp_path, allowed_origins=["http://localhost:5173"], env="testing", reload=False)
    app = create_app(settings)
    with TestClient(app) as client:
        resp = client.get("/api/v1/schedule/tomorrow")
        assert resp.status_code == 503
        assert resp.json()["code"] == "SCHEDULE_UNAVAILABLE"


def test_schedule_endpoint_malformed_metadata_raises_503(tmp_path):
    """Malformed metadata JSON must fail closed with 503."""
    df_home = make_test_df(num_rows=2)
    df_all = make_test_df(num_rows=5)
    df_home.to_parquet(tmp_path / "frontend_home.parquet")
    df_all.to_parquet(tmp_path / "frontend_all.parquet")

    sched_df = pd.DataFrame([{"game_id": "0022600101"}])
    sched_df.to_parquet(tmp_path / "schedule_tomorrow.parquet", index=False)

    with open(tmp_path / "schedule_tomorrow_metadata.json", "w", encoding="utf-8") as f:
        f.write("{invalid_json: true")

    settings = Settings(data_dir=tmp_path, allowed_origins=["http://localhost:5173"], env="testing", reload=False)
    app = create_app(settings)
    with TestClient(app) as client:
        resp = client.get("/api/v1/schedule/tomorrow")
        assert resp.status_code == 503
        assert resp.json()["code"] == "SCHEDULE_UNAVAILABLE"


def test_schedule_endpoint_fetch_failed_status_raises_503(tmp_path):
    """Metadata marked FETCH_FAILED must return HTTP 503, never HTTP 200."""
    now_et = datetime.now(ZoneInfo("America/New_York"))
    tomorrow_et = (now_et + timedelta(days=1)).strftime("%Y-%m-%d")

    df_home = make_test_df(num_rows=2)
    df_all = make_test_df(num_rows=5)
    df_home.to_parquet(tmp_path / "frontend_home.parquet")
    df_all.to_parquet(tmp_path / "frontend_all.parquet")

    sched_df = pd.DataFrame(columns=["game_id"])
    sched_df.to_parquet(tmp_path / "schedule_tomorrow.parquet", index=False)

    with open(tmp_path / "schedule_tomorrow_metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "status": "FETCH_FAILED",
            "schedule_date": tomorrow_et,
            "target_date": tomorrow_et,
            "games_count": 0,
            "error": "Upstream timeout",
        }, f)

    settings = Settings(data_dir=tmp_path, allowed_origins=["http://localhost:5173"], env="testing", reload=False)
    app = create_app(settings)
    with TestClient(app) as client:
        resp = client.get("/api/v1/schedule/tomorrow")
        assert resp.status_code == 503
        assert resp.json()["code"] == "SCHEDULE_UNAVAILABLE"


def test_schedule_endpoint_empty_parquet_without_off_day_metadata_raises_503(tmp_path):
    """Empty parquet cannot be inferred as OFF_DAY without explicit upstream metadata."""
    now_et = datetime.now(ZoneInfo("America/New_York"))
    tomorrow_et = (now_et + timedelta(days=1)).strftime("%Y-%m-%d")

    df_home = make_test_df(num_rows=2)
    df_all = make_test_df(num_rows=5)
    df_home.to_parquet(tmp_path / "frontend_home.parquet")
    df_all.to_parquet(tmp_path / "frontend_all.parquet")

    # Empty parquet
    empty_df = pd.DataFrame(columns=["game_id"])
    empty_df.to_parquet(tmp_path / "schedule_tomorrow.parquet", index=False)

    # Metadata incorrectly claims CONFIRMED or lacks OFF_DAY
    with open(tmp_path / "schedule_tomorrow_metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "status": "CONFIRMED",
            "schedule_date": tomorrow_et,
            "target_date": tomorrow_et,
            "games_count": 0,
            "games_scheduled": False,
        }, f)

    settings = Settings(data_dir=tmp_path, allowed_origins=["http://localhost:5173"], env="testing", reload=False)
    app = create_app(settings)
    with TestClient(app) as client:
        resp = client.get("/api/v1/schedule/tomorrow")
        assert resp.status_code == 503
        assert resp.json()["code"] == "SCHEDULE_UNAVAILABLE"


def test_schedule_endpoint_row_count_mismatch_raises_503(tmp_path):
    """Discrepancy between Parquet rows and metadata games_count must raise 503."""
    now_et = datetime.now(ZoneInfo("America/New_York"))
    tomorrow_et = (now_et + timedelta(days=1)).strftime("%Y-%m-%d")

    df_home = make_test_df(num_rows=2)
    df_all = make_test_df(num_rows=5)
    df_home.to_parquet(tmp_path / "frontend_home.parquet")
    df_all.to_parquet(tmp_path / "frontend_all.parquet")

    sched_df = pd.DataFrame([{
        "game_id": "0022600101",
        "home_team_id": 1610612738,
        "away_team_id": 1610612747,
    }])
    sched_df.to_parquet(tmp_path / "schedule_tomorrow.parquet", index=False)

    # Metadata claims games_count = 5 when only 1 is in parquet
    with open(tmp_path / "schedule_tomorrow_metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "status": "CONFIRMED",
            "schedule_date": tomorrow_et,
            "target_date": tomorrow_et,
            "games_scheduled": True,
            "games_count": 5,
        }, f)

    settings = Settings(data_dir=tmp_path, allowed_origins=["http://localhost:5173"], env="testing", reload=False)
    app = create_app(settings)
    with TestClient(app) as client:
        resp = client.get("/api/v1/schedule/tomorrow")
        assert resp.status_code == 503
        assert resp.json()["code"] == "SCHEDULE_UNAVAILABLE"

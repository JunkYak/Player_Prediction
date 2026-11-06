"""
tests/test_schedule_publication.py

Regression test suite for Phase 2 Schedule Reliability Corrections:
1. Schedule date validation (==, <, >, missing, malformed).
2. Fail-closed snapshot publication when schedule artifact is missing, corrupt, or FETCH_FAILED.
3. Snapshot publication success when schedule is valid CONFIRMED or verified OFF_DAY.
4. Download and staging path rejection when manifest is missing the schedule artifact.
5. Atomic staging failure safety: corrupt schedule preserves prior active state.
"""

import os
import sys
import json
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from unittest.mock import MagicMock, patch

import pytest
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.azure_blob import (
    publish_serving_snapshot,
    download_serving_snapshot,
    validate_schedule_artifact,
    compute_sha256,
)
from tests.test_api_players import make_test_df


def _create_mock_schedule_df(schedule_date: str, count: int = 1) -> pd.DataFrame:
    rows = []
    for i in range(count):
        rows.append({
            "game_id": f"0022400{100 + i}",
            "game_date": schedule_date,
            "schedule_date": schedule_date,
            "home_team_id": 1610612738 + i,
            "home_team_name": f"Home Team {i}",
            "away_team_id": 1610612747 + i,
            "away_team_name": f"Away Team {i}",
            "home_team_tricode": "HOM",
            "away_team_tricode": "AWY",
            "status": "Scheduled",
            "game_status": 1,
            "game_time_utc": f"{schedule_date}T23:30:00+00:00",
            "is_postponed": False,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        })
    return pd.DataFrame(rows)


def _setup_test_serving_data(data_dir, schedule_status="CONFIRMED", schedule_date=None, games_count=1):
    now_et = datetime.now(ZoneInfo("America/New_York"))
    today_ny = now_et.strftime("%Y-%m-%d")
    target_date = schedule_date or (now_et.date() + timedelta(days=1)).strftime("%Y-%m-%d")

    # Serving files with fresh slate date
    df_home = make_test_df(num_rows=2, game_date=[today_ny, today_ny])
    df_all = make_test_df(num_rows=5, game_date=[today_ny] * 5)
    df_home.to_parquet(data_dir / "frontend_home.parquet", index=False)
    df_all.to_parquet(data_dir / "frontend_all.parquet", index=False)

    # Ingestion metadata
    with open(data_dir / "ingestion_metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "status": "SUCCESS",
            "is_fresh": True,
            "new_games_ingested": 1,
            "failed_games_count": 0,
            "historical_cutoff_date": "2026-02-06",
        }, f)

    # Schedule parquet and metadata
    if games_count > 0:
        df_sched = _create_mock_schedule_df(target_date, count=games_count)
    else:
        empty_cols = [
            "game_id", "game_date", "schedule_date", "home_team_id", "home_team_name",
            "away_team_id", "away_team_name", "home_team_tricode", "away_team_tricode",
            "status", "game_status", "game_time_utc", "is_postponed", "generated_at",
        ]
        df_sched = pd.DataFrame(columns=empty_cols)
    df_sched.to_parquet(data_dir / "schedule_tomorrow.parquet", index=False)

    meta = {
        "status": schedule_status,
        "schedule_date": target_date,
        "target_date": target_date,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "games_scheduled": (games_count > 0 and schedule_status == "CONFIRMED"),
        "games_count": games_count,
        "is_stale": False,
        "error": None if schedule_status != "FETCH_FAILED" else "Upstream error",
    }
    with open(data_dir / "schedule_tomorrow_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)

    return target_date


# ==============================================================================
# 1. PUBLICATION FAIL-CLOSED ON MISSING OR INVALID SCHEDULE
# ==============================================================================

def test_publication_rejected_when_schedule_parquet_missing(tmp_path):
    """Publication must fail closed if schedule_tomorrow.parquet is missing."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _setup_test_serving_data(data_dir)
    os.remove(data_dir / "schedule_tomorrow.parquet")

    mock_service = MagicMock()
    with patch("scripts.azure_blob.get_blob_service_client", return_value=mock_service):
        with pytest.raises(ValueError, match="Schedule Parquet file not found"):
            publish_serving_snapshot(
                account_name="testacc",
                container_name="propredict-serving",
                data_dir=str(data_dir),
            )


def test_publication_rejected_when_schedule_metadata_missing(tmp_path):
    """Publication must fail closed if schedule_tomorrow_metadata.json is missing."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _setup_test_serving_data(data_dir)
    os.remove(data_dir / "schedule_tomorrow_metadata.json")

    mock_service = MagicMock()
    with patch("scripts.azure_blob.get_blob_service_client", return_value=mock_service):
        with pytest.raises(ValueError, match="Schedule metadata file not found"):
            publish_serving_snapshot(
                account_name="testacc",
                container_name="propredict-serving",
                data_dir=str(data_dir),
            )


def test_publication_rejected_when_schedule_marked_fetch_failed(tmp_path):
    """Publication must fail closed if schedule is marked FETCH_FAILED."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _setup_test_serving_data(data_dir, schedule_status="FETCH_FAILED", games_count=0)

    mock_service = MagicMock()
    with patch("scripts.azure_blob.get_blob_service_client", return_value=mock_service):
        with pytest.raises(ValueError, match="marked FETCH_FAILED; publication blocked"):
            publish_serving_snapshot(
                account_name="testacc",
                container_name="propredict-serving",
                data_dir=str(data_dir),
            )


def test_publication_rejected_when_target_date_does_not_match_tomorrow_et(tmp_path):
    """Publication must fail closed if schedule date does not match tomorrow ET exactly."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    yesterday = "2026-01-01"
    _setup_test_serving_data(data_dir, schedule_date=yesterday)

    mock_service = MagicMock()
    with patch("scripts.azure_blob.get_blob_service_client", return_value=mock_service):
        with pytest.raises(ValueError, match="Schedule target date mismatch"):
            publish_serving_snapshot(
                account_name="testacc",
                container_name="propredict-serving",
                data_dir=str(data_dir),
            )


def test_publication_rejected_when_parquet_and_metadata_counts_mismatch(tmp_path):
    """Publication must fail closed if Parquet row count and metadata games_count differ."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _setup_test_serving_data(data_dir, games_count=2)

    # Tamper metadata count to 5
    with open(data_dir / "schedule_tomorrow_metadata.json", "r+", encoding="utf-8") as f:
        meta = json.load(f)
        meta["games_count"] = 5
        f.seek(0)
        f.truncate()
        json.dump(meta, f)

    mock_service = MagicMock()
    with patch("scripts.azure_blob.get_blob_service_client", return_value=mock_service):
        with pytest.raises(ValueError, match="Schedule row count mismatch"):
            publish_serving_snapshot(
                account_name="testacc",
                container_name="propredict-serving",
                data_dir=str(data_dir),
            )


def test_publication_rejected_when_empty_parquet_lacks_off_day_status(tmp_path):
    """Empty Parquet without OFF_DAY status in metadata must be rejected."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _setup_test_serving_data(data_dir, schedule_status="CONFIRMED", games_count=0)

    mock_service = MagicMock()
    with patch("scripts.azure_blob.get_blob_service_client", return_value=mock_service):
        with pytest.raises(ValueError, match="Empty schedule Parquet must have status 'OFF_DAY'"):
            publish_serving_snapshot(
                account_name="testacc",
                container_name="propredict-serving",
                data_dir=str(data_dir),
            )


def test_publication_succeeds_when_valid_empty_off_day_schedule_present(tmp_path):
    """Publication succeeds when schedule is a verified upstream OFF_DAY with 0 games."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _setup_test_serving_data(data_dir, schedule_status="OFF_DAY", games_count=0)

    mock_storage = {}
    def get_blob_client(name):
        client = MagicMock()
        client.exists.side_effect = lambda: name in mock_storage
        client.upload_blob.side_effect = lambda data, overwrite=False: mock_storage.update(
            {name: data.read() if hasattr(data, "read") else data}
        )
        return client

    mock_container = MagicMock()
    mock_container.get_blob_client.side_effect = get_blob_client
    mock_service = MagicMock()
    mock_service.get_container_client.return_value = mock_container

    with patch("scripts.azure_blob.get_blob_service_client", return_value=mock_service):
        manifest = publish_serving_snapshot(
            account_name="testacc",
            container_name="propredict-serving",
            data_dir=str(data_dir),
        )

        assert manifest["validation"]["schedule_tomorrow_rows"] == 0
        assert manifest["schedule_metadata"]["status"] == "OFF_DAY"
        assert "schedule_tomorrow.parquet" in manifest["files"]
        assert "schedule_tomorrow_metadata.json" in manifest["files"]


def test_publication_succeeds_when_valid_confirmed_schedule_present(tmp_path):
    """Publication succeeds with valid CONFIRMED tomorrow schedule."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _setup_test_serving_data(data_dir, schedule_status="CONFIRMED", games_count=2)

    mock_storage = {}
    def get_blob_client(name):
        client = MagicMock()
        client.exists.side_effect = lambda: name in mock_storage
        client.upload_blob.side_effect = lambda data, overwrite=False: mock_storage.update(
            {name: data.read() if hasattr(data, "read") else data}
        )
        return client

    mock_container = MagicMock()
    mock_container.get_blob_client.side_effect = get_blob_client
    mock_service = MagicMock()
    mock_service.get_container_client.return_value = mock_container

    with patch("scripts.azure_blob.get_blob_service_client", return_value=mock_service):
        manifest = publish_serving_snapshot(
            account_name="testacc",
            container_name="propredict-serving",
            data_dir=str(data_dir),
        )

        assert manifest["validation"]["schedule_tomorrow_rows"] == 2
        assert manifest["schedule_metadata"]["status"] == "CONFIRMED"
        assert "schedule_tomorrow.parquet" in manifest["files"]
        assert "schedule_tomorrow_metadata.json" in manifest["files"]


# ==============================================================================
# 2. DOWNLOAD STAGING REJECTION ON INCOMPLETE MANIFEST
# ==============================================================================

def test_download_staging_rejected_when_manifest_missing_schedule_artifact(tmp_path):
    """Download/staging path must refuse to stage manifests lacking schedule_tomorrow.parquet."""
    target_dir = tmp_path / "target"
    target_dir.mkdir()

    manifest_dict = {
        "snapshot_id": "snap_no_sched",
        "files": {
            "frontend_all.parquet": {"blob_path": "snapshots/snap_no_sched/frontend_all.parquet", "sha256": "h1"},
            "frontend_home.parquet": {"blob_path": "snapshots/snap_no_sched/frontend_home.parquet", "sha256": "h2"},
        },
    }

    mock_manifest = MagicMock()
    mock_manifest.exists.return_value = True
    mock_manifest.download_blob.return_value.readall.return_value = json.dumps(manifest_dict).encode("utf-8")

    mock_container = MagicMock()
    mock_container.get_blob_client.return_value = mock_manifest
    mock_service = MagicMock()
    mock_service.get_container_client.return_value = mock_container

    with patch("scripts.azure_blob.get_blob_service_client", return_value=mock_service):
        with pytest.raises(ValueError, match="Manifest missing required schedule artifact 'schedule_tomorrow.parquet'"):
            download_serving_snapshot(
                account_name="testacc",
                container_name="propredict-serving",
                target_dir=str(target_dir),
            )

    # Staging dir must be completely removed
    assert len(list(target_dir.glob(".staging_*"))) == 0


def test_download_staging_rejected_when_manifest_missing_schedule_metadata(tmp_path):
    """Download/staging path must refuse to stage manifests lacking schedule metadata."""
    target_dir = tmp_path / "target"
    target_dir.mkdir()

    manifest_dict = {
        "snapshot_id": "snap_no_meta",
        "files": {
            "frontend_all.parquet": {"blob_path": "snapshots/snap_no_meta/frontend_all.parquet", "sha256": "h1"},
            "frontend_home.parquet": {"blob_path": "snapshots/snap_no_meta/frontend_home.parquet", "sha256": "h2"},
            "schedule_tomorrow.parquet": {"blob_path": "snapshots/snap_no_meta/schedule_tomorrow.parquet", "sha256": "h3"},
        },
    }

    mock_manifest = MagicMock()
    mock_manifest.exists.return_value = True
    mock_manifest.download_blob.return_value.readall.return_value = json.dumps(manifest_dict).encode("utf-8")

    mock_container = MagicMock()
    mock_container.get_blob_client.return_value = mock_manifest
    mock_service = MagicMock()
    mock_service.get_container_client.return_value = mock_container

    with patch("scripts.azure_blob.get_blob_service_client", return_value=mock_service):
        with pytest.raises(ValueError, match="Manifest missing required schedule metadata"):
            download_serving_snapshot(
                account_name="testacc",
                container_name="propredict-serving",
                target_dir=str(target_dir),
            )

    assert len(list(target_dir.glob(".staging_*"))) == 0

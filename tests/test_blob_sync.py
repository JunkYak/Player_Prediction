"""
tests/test_blob_sync.py

Comprehensive regression test suite for:
1. Schedule correctness and daily idempotency (EDT vs EST, scheduled triggers, runner delays, duplicate blocks).
2. Atomic serving snapshot staging, directory promotion, checksum verification, and failure safety.
3. Historical data fail-closed protections, schema validation, and justified continuity rules.
4. FastAPI serving layer integration, offline compatibility, startup readiness, and atomic in-memory state swaps.

All Azure SDK network calls and credentials are mock-tested (no real credentials or cloud access required).
"""

import os
import sys
import json
import hashlib
import io
import shutil
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import pandas as pd
from fastapi.testclient import TestClient

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from api.config import Settings
from api.main import create_app
from api.loader import load_serving_state, reload_serving_state_if_stale, load_snapshot_state
from scripts.check_schedule import is_scheduled_slot_active
from scripts.verify_baseline import verify_baseline
from scripts.azure_blob import (
    compute_sha256,
    validate_play_by_play_file,
    validate_serving_parquet,
    validate_artifacts,
    restore_historical_data,
    publish_historical_data,
    publish_serving_snapshot,
    download_serving_snapshot,
    check_daily_publication_idempotency,
    acquire_daily_claim,
    complete_daily_claim,
    fail_daily_claim,
    get_daily_claim_blob_name,
)
from scripts.build_play_by_play_dataset import REQUIRED_COLUMNS as PBP_COLS
from api.loader import REQUIRED_COLUMNS as SERVING_COLS


@pytest.fixture
def sample_serving_dfs():
    """Generates valid sample serving DataFrames."""
    all_data = {col: [f"val_{i}"] for i, col in enumerate(SERVING_COLS)}
    all_data["rating"] = [85.0]
    all_data["predicted_rating"] = [88.5]
    all_data["personId"] = [1628369]
    all_data["teamId"] = [1610612738]
    all_data["opponentId"] = [1610612748]
    all_data["status_rank"] = [0]
    all_data["playerName"] = ["Jayson Tatum"]
    all_data["teamName"] = ["Celtics"]
    all_data["opponentName"] = ["Heat"]
    all_data["status"] = ["Active"]
    all_data["headshot"] = ["https://example.com/headshot.png"]
    all_data["last_game"] = [85.0]
    all_data["last3_avg"] = [84.0]
    all_data["last7_avg"] = [83.0]

    df_all = pd.DataFrame(all_data)
    df_home = df_all.copy()
    return df_all, df_home


@pytest.fixture
def sample_pbp_df():
    """Generates valid play-by-play DataFrame."""
    data = {col: [1] for col in PBP_COLS}
    data["gameId"] = ["0022400001"]
    data["playerName"] = ["Player One"]
    return pd.DataFrame(data)


# ==============================================================================
# 1. Schedule Correctness & Daily Idempotency Tests
# ==============================================================================
def test_schedule_edt_0900_matches():
    """During EDT (summer, UTC-4), 09:00 UTC is 05:00 EDT (INTENDED RUN SLOT)."""
    # July 15, 2026 09:00 UTC -> 05:00 EDT
    edt_time = datetime(2026, 7, 15, 9, 0, tzinfo=timezone.utc)
    assert is_scheduled_slot_active(current_time=edt_time) is True


def test_schedule_edt_0945_runner_delay_matches():
    """During EDT, runner queue delay of 45 minutes (09:45 UTC = 05:45 EDT) still matches the 05:xx window."""
    edt_delay_time = datetime(2026, 7, 15, 9, 45, tzinfo=timezone.utc)
    assert is_scheduled_slot_active(current_time=edt_delay_time) is True


def test_schedule_edt_1000_strictly_skips():
    """During EDT, the second cron trigger (10:00 UTC = 06:00 EDT) MUST strictly skip to prevent dual execution."""
    edt_second_cron = datetime(2026, 7, 15, 10, 0, tzinfo=timezone.utc)
    assert is_scheduled_slot_active(current_time=edt_second_cron) is False


def test_schedule_edt_1030_strictly_skips():
    """During EDT, 10:30 UTC = 06:30 EDT is outside the 05:xx window and must skip."""
    edt_late_time = datetime(2026, 7, 15, 10, 30, tzinfo=timezone.utc)
    assert is_scheduled_slot_active(current_time=edt_late_time) is False


def test_schedule_est_0900_strictly_skips():
    """During EST (winter, UTC-5), the first cron trigger (09:00 UTC = 04:00 EST) MUST strictly skip."""
    est_first_cron = datetime(2026, 1, 15, 9, 0, tzinfo=timezone.utc)
    assert is_scheduled_slot_active(current_time=est_first_cron) is False


def test_schedule_est_0945_strictly_skips():
    """During EST, 09:45 UTC = 04:45 EST is still in the 04:xx hour and must skip."""
    est_early_delay = datetime(2026, 1, 15, 9, 45, tzinfo=timezone.utc)
    assert is_scheduled_slot_active(current_time=est_early_delay) is False


def test_schedule_est_1000_matches():
    """During EST, the second cron trigger (10:00 UTC = 05:00 EST) is the INTENDED RUN SLOT."""
    est_time = datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc)
    assert is_scheduled_slot_active(current_time=est_time) is True


def test_schedule_est_1045_runner_delay_matches():
    """During EST, runner queue delay of 45 minutes (10:45 UTC = 05:45 EST) still matches the 05:xx window."""
    est_delay_time = datetime(2026, 1, 15, 10, 45, tzinfo=timezone.utc)
    assert is_scheduled_slot_active(current_time=est_delay_time) is True


def test_daily_publication_idempotency_blocks_duplicate(monkeypatch):
    """If current_manifest.json published today in America/New_York, idempotency guard returns True (skip)."""
    now_ny = datetime.now(timezone.utc).astimezone(ZoneInfo("America/New_York"))
    published_today_iso = now_ny.replace(hour=5, minute=15).isoformat()

    manifest_content = json.dumps({
        "snapshot_id": "snap_today",
        "published_at": published_today_iso,
    }).encode("utf-8")

    mock_blob = MagicMock()
    mock_blob.exists.return_value = True
    mock_blob.download_blob.return_value.readall.return_value = manifest_content

    mock_container = MagicMock()
    mock_container.get_blob_client.return_value = mock_blob
    mock_service = MagicMock()
    mock_service.get_container_client.return_value = mock_container

    with patch("scripts.azure_blob.get_blob_service_client", return_value=mock_service):
        already_run = check_daily_publication_idempotency(
            account_name="testacc",
            container_name="propredict-serving",
            force=False,
        )
        assert already_run is True


def test_daily_publication_idempotency_permits_next_day(monkeypatch):
    """If current_manifest.json was published yesterday, idempotency guard returns False (proceed)."""
    # Manifest published 24 hours ago
    yesterday_utc = datetime.now(timezone.utc) - pd.Timedelta(days=1)
    published_yesterday_iso = yesterday_utc.isoformat()

    manifest_content = json.dumps({
        "snapshot_id": "snap_yesterday",
        "published_at": published_yesterday_iso,
    }).encode("utf-8")

    mock_blob = MagicMock()
    mock_blob.exists.return_value = True
    mock_blob.download_blob.return_value.readall.return_value = manifest_content

    mock_container = MagicMock()
    mock_container.get_blob_client.return_value = mock_blob
    mock_service = MagicMock()
    mock_service.get_container_client.return_value = mock_container

    with patch("scripts.azure_blob.get_blob_service_client", return_value=mock_service):
        already_run = check_daily_publication_idempotency(
            account_name="testacc",
            container_name="propredict-serving",
            force=False,
        )
        assert already_run is False


def test_daily_publication_idempotency_force_bypass():
    """When force=True (e.g. manual dispatch), idempotency guard always permits execution."""
    already_run = check_daily_publication_idempotency(
        account_name="testacc",
        container_name="propredict-serving",
        force=True,
    )
    assert already_run is False


# ==============================================================================
# 2. Atomic Serving Snapshot Staging, Download, and Failure Safety Tests
# ==============================================================================
def test_publish_and_download_serving_snapshot_staging_flow(tmp_path, sample_serving_dfs):
    """Verifies complete publication and staging download with atomic directory promotion."""
    df_all, df_home = sample_serving_dfs
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    df_all.to_parquet(data_dir / "frontend_all.parquet")
    df_home.to_parquet(data_dir / "frontend_home.parquet")

    now_et = datetime.now(ZoneInfo("America/New_York"))
    tomorrow_et = now_et.date().strftime("%Y-%m-%d")
    df_sched = pd.DataFrame([{
        "game_id": "0022400999",
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
    }])
    df_sched.to_parquet(data_dir / "schedule_tomorrow.parquet", index=False)
    with open(data_dir / "schedule_tomorrow_metadata.json", "w", encoding="utf-8") as f:
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

    with open(data_dir / "ingestion_metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "status": "SUCCESS",
            "is_fresh": True,
            "new_games_ingested": 1,
            "failed_games_count": 0,
            "historical_cutoff_date": "2026-02-06",
        }, f)

    mock_storage = {}

    def get_blob_client(name):
        client = MagicMock()
        client.exists.side_effect = lambda: name in mock_storage
        client.download_blob.return_value.readall.side_effect = lambda: mock_storage.get(name, b"")
        client.download_blob.return_value.readinto.side_effect = lambda f: f.write(mock_storage.get(name, b""))
        client.upload_blob.side_effect = lambda data, overwrite=False: mock_storage.update(
            {name: data.read() if hasattr(data, "read") else data}
        )
        return client

    mock_container = MagicMock()
    mock_container.get_blob_client.side_effect = get_blob_client
    mock_service = MagicMock()
    mock_service.get_container_client.return_value = mock_container

    with patch("scripts.azure_blob.get_blob_service_client", return_value=mock_service):
        # 1. Publish snapshot
        manifest = publish_serving_snapshot(
            account_name="testacc",
            container_name="propredict-serving",
            data_dir=str(data_dir),
            pipeline_run_id="run_100",
            git_sha="git_sha_abc",
        )
        snap_id = manifest["snapshot_id"]

        # 2. Download snapshot into target_dir
        target_dir = tmp_path / "target"
        updated, downloaded_snap, snap_dir = download_serving_snapshot(
            account_name="testacc",
            container_name="propredict-serving",
            target_dir=str(target_dir),
        )

        assert updated is True
        assert downloaded_snap == snap_id
        assert snap_dir.exists()
        assert (snap_dir / "frontend_all.parquet").exists()
        assert (snap_dir / "frontend_home.parquet").exists()
        assert (snap_dir / "manifest.json").exists()
        assert (target_dir / "current_manifest.json").exists()
        # Staging directory must be cleaned up
        staging_dirs = list(target_dir.glob(".staging_*"))
        assert len(staging_dirs) == 0


def test_download_second_artifact_failure_preserves_serving_state(tmp_path, sample_serving_dfs):
    """
    If the second artifact download fails, staging is completely cleaned up,
    and existing files / in-memory serving state remain untouched.
    """
    df_all, df_home = sample_serving_dfs
    target_dir = tmp_path / "target"
    target_dir.mkdir()

    # Pre-existing valid serving files
    orig_all = target_dir / "frontend_all.parquet"
    orig_home = target_dir / "frontend_home.parquet"
    df_all.to_parquet(orig_all)
    df_home.to_parquet(orig_home)
    orig_all_sha = compute_sha256(str(orig_all))

    manifest_dict = {
        "snapshot_id": "snap_broken_pair",
        "files": {
            "frontend_all.parquet": {
                "blob_path": "snapshots/snap_broken_pair/frontend_all.parquet",
                "sha256": "hash_all",
                "size_bytes": 100,
                "row_count": 1,
            },
            "frontend_home.parquet": {
                "blob_path": "snapshots/snap_broken_pair/frontend_home.parquet",
                "sha256": "hash_home",
                "size_bytes": 100,
                "row_count": 1,
            },
            "schedule_tomorrow.parquet": {
                "blob_path": "snapshots/snap_broken_pair/schedule_tomorrow.parquet",
                "sha256": "hash_sched",
                "size_bytes": 100,
                "row_count": 1,
            },
            "schedule_tomorrow_metadata.json": {
                "blob_path": "snapshots/snap_broken_pair/schedule_tomorrow_metadata.json",
                "sha256": "hash_sched_meta",
                "size_bytes": 100,
            },
        },
    }

    mock_manifest_blob = MagicMock()
    mock_manifest_blob.exists.return_value = True
    mock_manifest_blob.download_blob.return_value.readall.return_value = json.dumps(manifest_dict).encode("utf-8")

    mock_all_blob = MagicMock()
    mock_all_blob.download_blob.return_value.readinto.side_effect = lambda f: f.write(b"PAR1_valid_all_bytes")

    # Second artifact fails (simulating connection break or missing remote blob)
    mock_home_blob = MagicMock()
    mock_home_blob.download_blob.side_effect = ConnectionResetError("Connection lost during second file download")

    def get_blob_mock(name):
        if name == "current_manifest.json":
            return mock_manifest_blob
        elif "frontend_all.parquet" in name:
            return mock_all_blob
        elif "frontend_home.parquet" in name:
            return mock_home_blob
        return MagicMock()

    mock_container = MagicMock()
    mock_container.get_blob_client.side_effect = get_blob_mock
    mock_service = MagicMock()
    mock_service.get_container_client.return_value = mock_container

    with patch("scripts.azure_blob.get_blob_service_client", return_value=mock_service):
        with pytest.raises(ConnectionResetError, match="Connection lost during second file download"):
            download_serving_snapshot(
                account_name="testacc",
                container_name="propredict-serving",
                target_dir=str(target_dir),
            )

    # 1. Existing live serving files must be completely untouched
    assert orig_all.exists()
    assert orig_home.exists()
    assert compute_sha256(str(orig_all)) == orig_all_sha

    # 2. Staging directories must be completely removed
    assert len(list(target_dir.glob(".staging_*"))) == 0


def test_download_second_artifact_checksum_mismatch_preserves_state(tmp_path, sample_serving_dfs):
    """If the second artifact fails checksum validation, staging is cleaned up and state is preserved."""
    df_all, df_home = sample_serving_dfs
    target_dir = tmp_path / "target"
    target_dir.mkdir()

    orig_all = target_dir / "frontend_all.parquet"
    orig_home = target_dir / "frontend_home.parquet"
    df_all.to_parquet(orig_all)
    df_home.to_parquet(orig_home)
    sha_all = compute_sha256(str(orig_all))

    manifest_dict = {
        "snapshot_id": "snap_corrupt_home",
        "files": {
            "frontend_all.parquet": {
                "blob_path": "snapshots/snap_corrupt_home/frontend_all.parquet",
                "sha256": sha_all,
                "size_bytes": os.path.getsize(orig_all),
                "row_count": len(df_all),
            },
            "frontend_home.parquet": {
                "blob_path": "snapshots/snap_corrupt_home/frontend_home.parquet",
                "sha256": "EXPECTED_VALID_HASH_12345",
                "size_bytes": 100,
                "row_count": 1,
            },
            "schedule_tomorrow.parquet": {
                "blob_path": "snapshots/snap_corrupt_home/schedule_tomorrow.parquet",
                "sha256": "hash_sched",
                "size_bytes": 100,
                "row_count": 1,
            },
            "schedule_tomorrow_metadata.json": {
                "blob_path": "snapshots/snap_corrupt_home/schedule_tomorrow_metadata.json",
                "sha256": "hash_sched_meta",
                "size_bytes": 100,
            },
        },
    }

    mock_manifest = MagicMock()
    mock_manifest.exists.return_value = True
    mock_manifest.download_blob.return_value.readall.return_value = json.dumps(manifest_dict).encode("utf-8")

    mock_all = MagicMock()
    mock_all.download_blob.return_value.readinto.side_effect = lambda f: f.write(orig_all.read_bytes())

    # Second file returns corrupted bytes
    mock_home = MagicMock()
    mock_home.download_blob.return_value.readinto.side_effect = lambda f: f.write(b"PAR1_corrupted_payload")

    def mock_blob_resolver(name):
        if name == "current_manifest.json":
            return mock_manifest
        elif "frontend_all.parquet" in name:
            return mock_all
        elif "frontend_home.parquet" in name:
            return mock_home
        return MagicMock()

    mock_container = MagicMock()
    mock_container.get_blob_client.side_effect = mock_blob_resolver
    mock_service = MagicMock()
    mock_service.get_container_client.return_value = mock_container

    with patch("scripts.azure_blob.get_blob_service_client", return_value=mock_service):
        with pytest.raises(ValueError, match="Checksum mismatch on frontend_home.parquet"):
            download_serving_snapshot(
                account_name="testacc",
                container_name="propredict-serving",
                target_dir=str(target_dir),
            )

    # Live files remain untouched, staging removed
    assert orig_all.exists()
    assert orig_home.exists()
    assert len(list(target_dir.glob(".staging_*"))) == 0


# ==============================================================================
# 3. Historical Data Safety & Continuity Tests
# ==============================================================================
def test_restore_historical_data_missing_blob_fails_closed(tmp_path):
    """If play_by_play.parquet is missing remotely, must fail closed immediately."""
    mock_blob_client = MagicMock()
    mock_blob_client.exists.return_value = False

    mock_container_client = MagicMock()
    mock_container_client.get_blob_client.return_value = mock_blob_client
    mock_service_client = MagicMock()
    mock_service_client.get_container_client.return_value = mock_container_client

    with patch("scripts.azure_blob.get_blob_service_client", return_value=mock_service_client):
        with pytest.raises(FileNotFoundError, match="CRITICAL: Canonical historical blob"):
            restore_historical_data(
                account_name="testacc",
                container_name="propredict-history",
                target_path=str(tmp_path / "play_by_play.parquet"),
            )


def test_restore_historical_data_corrupt_blob_fails_closed(tmp_path):
    """If remote blob is corrupt (e.g. SSO login page), fails closed and cleans up temporary file."""
    mock_blob_client = MagicMock()
    mock_blob_client.exists.return_value = True

    download_stream = MagicMock()
    download_stream.readinto.side_effect = lambda f: f.write(b"<!-- Azure Sign In HTML -->")
    mock_blob_client.download_blob.return_value = download_stream

    mock_container_client = MagicMock()
    mock_container_client.get_blob_client.return_value = mock_blob_client
    mock_service_client = MagicMock()
    mock_service_client.get_container_client.return_value = mock_container_client

    target_file = tmp_path / "play_by_play.parquet"

    with patch("scripts.azure_blob.get_blob_service_client", return_value=mock_service_client):
        with pytest.raises(RuntimeError, match="Restored historical blob is corrupt or invalid"):
            restore_historical_data(
                account_name="testacc",
                container_name="propredict-history",
                target_path=str(target_file),
            )

    assert not target_file.exists()
    assert not (tmp_path / "play_by_play.parquet.tmp").exists()


def test_publish_historical_data_game_count_decrease_rejection(tmp_path, sample_pbp_df):
    """Continuity check: if unique game count decreases, publication is rejected."""
    pbp_file = tmp_path / "play_by_play.parquet"
    sample_pbp_df.to_parquet(pbp_file)

    meta_file = tmp_path / "play_by_play.parquet.meta.json"
    with open(meta_file, "w") as f:
        # Baseline had 409 games, new file only has 1 game (e.g. truncated 60-day run)
        json.dump({"row_count": 204575, "unique_games": 409}, f)

    with patch("scripts.azure_blob.get_blob_service_client"):
        with pytest.raises(ValueError, match="unique game count decreased from 409 to 1"):
            publish_historical_data(
                account_name="testacc",
                container_name="propredict-history",
                source_path=str(pbp_file),
            )


def test_publish_historical_data_row_count_truncation_rejection(tmp_path, sample_pbp_df):
    """Continuity check: if row count drops by more than baseline duplicate slack (>100 plays), rejected."""
    pbp_file = tmp_path / "play_by_play.parquet"
    # Create dataset with 100 rows across 5 games
    rows = []
    for g in range(5):
        for _ in range(20):
            rows.append(dict(sample_pbp_df.iloc[0], gameId=f"002240000{g}"))
    df = pd.DataFrame(rows)
    df.to_parquet(pbp_file)

    meta_file = tmp_path / "play_by_play.parquet.meta.json"
    with open(meta_file, "w") as f:
        # Baseline had 5 games but 500 rows (lost 400 rows!)
        json.dump({"row_count": 500, "unique_games": 5}, f)

    with patch("scripts.azure_blob.get_blob_service_client"):
        with pytest.raises(ValueError, match="exceeding allowable deduplication slack"):
            publish_historical_data(
                account_name="testacc",
                container_name="propredict-history",
                source_path=str(pbp_file),
            )


def test_publish_historical_data_offseason_zero_games_success(tmp_path, sample_pbp_df):
    """Continuity check: offseason run with 0 new games and deduplication is accepted."""
    pbp_file = tmp_path / "play_by_play.parquet"
    sample_pbp_df.to_parquet(pbp_file)

    meta_file = tmp_path / "play_by_play.parquet.meta.json"
    with open(meta_file, "w") as f:
        # Baseline had 1 game and 1 row (identical)
        json.dump({"row_count": 1, "unique_games": 1}, f)

    mock_blob = MagicMock()
    mock_container = MagicMock()
    mock_container.get_blob_client.return_value = mock_blob
    mock_service = MagicMock()
    mock_service.get_container_client.return_value = mock_container

    with patch("scripts.azure_blob.get_blob_service_client", return_value=mock_service):
        meta = publish_historical_data(
            account_name="testacc",
            container_name="propredict-history",
            source_path=str(pbp_file),
        )
        assert meta["unique_games"] == 1
        assert meta["row_count"] == 1
        assert mock_blob.upload_blob.called


def test_baseline_verification_is_read_only(tmp_path, sample_pbp_df):
    """verify_baseline must inspect play_by_play.parquet without modifying or overwriting it."""
    pbp_file = tmp_path / "play_by_play.parquet"
    sample_pbp_df.to_parquet(pbp_file)

    orig_mtime = os.path.getmtime(pbp_file)
    orig_sha = compute_sha256(str(pbp_file))

    report = verify_baseline(str(pbp_file))
    assert report["is_valid"] is True
    assert report["num_rows"] == 1
    assert report["unique_games"] == 1

    # Verify file was NOT modified
    assert os.path.getmtime(pbp_file) == orig_mtime
    assert compute_sha256(str(pbp_file)) == orig_sha


# ==============================================================================
# 4. FastAPI Integration & Atomic In-Memory State Swap Tests
# ==============================================================================
def test_fastapi_offline_operation_preserves_local_flow(tmp_path, sample_serving_dfs):
    """When azure_storage_account is None, FastAPI runs 100% locally with zero cloud dependencies."""
    df_all, df_home = sample_serving_dfs
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    df_all.to_parquet(data_dir / "frontend_all.parquet")
    df_home.to_parquet(data_dir / "frontend_home.parquet")

    settings = Settings(
        data_dir=data_dir,
        allowed_origins=["http://localhost:5173"],
        env="development",
        reload=False,
        azure_storage_account=None,
    )

    app = create_app(settings)
    with TestClient(app) as client:
        r_health = client.get("/health")
        assert r_health.status_code == 200
        assert r_health.json() == {"status": "ok"}

        r_ready = client.get("/ready")
        assert r_ready.status_code == 200
        assert r_ready.json()["status"] == "ready"

        r_players = client.get("/api/v1/players")
        assert r_players.status_code == 200
        assert len(r_players.json()["players"]) == 1


def test_fastapi_startup_readiness_fails_when_blob_unavailable(tmp_path):
    """When Azure Blob mode is configured and remote snapshot cannot be retrieved, /ready returns 503."""
    empty_data_dir = tmp_path / "empty_data"
    empty_data_dir.mkdir()

    settings = Settings(
        data_dir=empty_data_dir,
        allowed_origins=["http://localhost:5173"],
        env="production",
        reload=False,
        azure_storage_account="mockstorageacc",
    )

    with patch("scripts.azure_blob.download_serving_snapshot", side_effect=ConnectionError("Blob service unreachable")):
        app = create_app(settings)
        with TestClient(app) as client:
            r_health = client.get("/health")
            assert r_health.status_code == 200  # Liveness passes

            r_ready = client.get("/ready")
            assert r_ready.status_code == 503  # Readiness probe fails closed
            assert r_ready.json()["code"] == "BLOB_SNAPSHOT_UNAVAILABLE"


def test_fastapi_refresh_second_artifact_failure_retains_prior_state(tmp_path, sample_serving_dfs):
    """
    When periodic refresh encounters an error downloading the second artifact of a new snapshot,
    FastAPI continues serving the previous valid snapshot seamlessly and never mixes states.
    """
    df_all, df_home = sample_serving_dfs
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    df_all.to_parquet(data_dir / "frontend_all.parquet")
    df_home.to_parquet(data_dir / "frontend_home.parquet")

    settings = Settings(
        data_dir=data_dir,
        allowed_origins=["http://localhost:5173"],
        env="production",
        reload=False,
        azure_storage_account="mockstorageacc",
        azure_blob_check_interval=0,  # Immediate check
    )

    # Initial state: snapshot 1 is loaded
    with patch("scripts.azure_blob.download_serving_snapshot", return_value=(False, "snap_1", None)):
        app = create_app(settings)
        with TestClient(app) as client:
            r1 = client.get("/api/v1/players")
            assert r1.status_code == 200
            assert len(r1.json()["players"]) == 1
            assert r1.json()["players"][0]["playerName"] == "Jayson Tatum"

            # Background refresh fails while downloading second artifact of snapshot 2
            with patch(
                "scripts.azure_blob.download_serving_snapshot",
                side_effect=RuntimeError("Corrupt second artifact in snapshot 2"),
            ):
                r2 = client.get("/api/v1/players")
                assert r2.status_code == 200
                # Continues serving snapshot 1 seamlessly
                assert len(r2.json()["players"]) == 1
                assert r2.json()["players"][0]["playerName"] == "Jayson Tatum"


# ==============================================================================
# 5. Atomic Daily Claim & Failure Safety Tests
# ==============================================================================
class MockBlobClient:
    def __init__(self, store, blob_name):
        self.store = store
        self.blob_name = blob_name

    def exists(self):
        return self.blob_name in self.store

    def upload_blob(self, data, overwrite=False):
        from azure.core.exceptions import ResourceExistsError
        if isinstance(data, (bytes, bytearray)):
            content = bytes(data)
        elif hasattr(data, "read"):
            content = data.read()
            if isinstance(content, str):
                content = content.encode("utf-8")
        else:
            content = str(data).encode("utf-8")

        if not overwrite and self.blob_name in self.store:
            raise ResourceExistsError(f"Blob {self.blob_name} already exists (conditional write failed).")
        self.store[self.blob_name] = content

    def download_blob(self):
        if self.blob_name not in self.store:
            from azure.core.exceptions import ResourceNotFoundError
            raise ResourceNotFoundError(f"Blob {self.blob_name} does not exist.")
        mock_download = MagicMock()
        mock_download.readall.return_value = self.store[self.blob_name]
        return mock_download


class MockContainerClient:
    def __init__(self, store):
        self.store = store

    def get_blob_client(self, blob_name):
        return MockBlobClient(self.store, blob_name)


class MockBlobServiceClient:
    def __init__(self, store):
        self.store = store

    def get_container_client(self, container_name):
        return MockContainerClient(self.store)


def test_concurrent_daily_claims_mutual_exclusion():
    """
    Simulates two workflow invocations checking and claiming at the exact same millisecond.
    Demonstrates that Azure Blob's conditional write (overwrite=False / If-None-Match: *)
    enforces mutual exclusion: only ONE writer succeeds, and the other is strictly rejected.
    """
    store = {}
    mock_service = MockBlobServiceClient(store)

    with patch("scripts.azure_blob.get_blob_service_client", return_value=mock_service):
        # Worker 1 attempts to claim
        worker1_claimed = acquire_daily_claim(
            account_name="mockacc",
            container_name="propredict-serving",
            run_id="run_101",
            git_sha="sha_aaa",
            force=False,
        )
        assert worker1_claimed is True, "Worker 1 must successfully acquire the unheld claim."

        # Worker 2 attempts to claim at the exact same time for the same date
        worker2_claimed = acquire_daily_claim(
            account_name="mockacc",
            container_name="propredict-serving",
            run_id="run_102",
            git_sha="sha_bbb",
            force=False,
        )
        assert worker2_claimed is False, "Worker 2 must be rejected due to active lease by Worker 1."

        # Verify claim contents reflect Worker 1
        today_ny = datetime.now(timezone.utc).astimezone(ZoneInfo("America/New_York")).date().isoformat()
        claim_blob = get_daily_claim_blob_name(today_ny)
        claim_data = json.loads(store[claim_blob].decode("utf-8"))
        assert claim_data["run_id"] == "run_101"
        assert claim_data["status"] == "IN_PROGRESS"


def test_completed_claim_blocks_subsequent_runs():
    """Once a run marks the claim COMPLETED, subsequent runs on that Eastern date are rejected."""
    store = {}
    mock_service = MockBlobServiceClient(store)

    with patch("scripts.azure_blob.get_blob_service_client", return_value=mock_service):
        acquire_daily_claim(account_name="mockacc", container_name="propredict-serving", run_id="run_201")
        complete_daily_claim(account_name="mockacc", container_name="propredict-serving", run_id="run_201")

        # Second run arrives after completion
        subsequent_claimed = acquire_daily_claim(
            account_name="mockacc",
            container_name="propredict-serving",
            run_id="run_202",
            force=False,
        )
        assert subsequent_claimed is False


def test_failed_claim_permits_immediate_retry():
    """A failed run marks status as FAILED, allowing immediate safe retry without waiting for timeout."""
    store = {}
    mock_service = MockBlobServiceClient(store)

    with patch("scripts.azure_blob.get_blob_service_client", return_value=mock_service):
        acquire_daily_claim(account_name="mockacc", container_name="propredict-serving", run_id="run_301")
        fail_daily_claim(
            account_name="mockacc",
            container_name="propredict-serving",
            run_id="run_301",
            error="Transient NBA API rate limit",
        )

        # Retry run arrives immediately
        retry_claimed = acquire_daily_claim(
            account_name="mockacc",
            container_name="propredict-serving",
            run_id="run_302",
            force=False,
        )
        assert retry_claimed is True, "Retry run must be allowed to reclaim a FAILED claim slot."

        today_ny = datetime.now(timezone.utc).astimezone(ZoneInfo("America/New_York")).date().isoformat()
        claim_blob = get_daily_claim_blob_name(today_ny)
        claim_data = json.loads(store[claim_blob].decode("utf-8"))
        assert claim_data["run_id"] == "run_302"
        assert claim_data["status"] == "IN_PROGRESS"


def test_abandoned_claim_timeout_reclamation():
    """An abandoned run (e.g. runner VM terminated unexpectedly) times out after 75m and is reclaimed."""
    today_ny = datetime.now(timezone.utc).astimezone(ZoneInfo("America/New_York")).date().isoformat()
    claim_blob = get_daily_claim_blob_name(today_ny)

    # Stale claim from 90 minutes ago
    from datetime import timedelta
    old_time = (datetime.now(timezone.utc) - timedelta(minutes=90)).isoformat()
    stale_payload = {
        "date_ny": today_ny,
        "status": "IN_PROGRESS",
        "claimed_at": old_time,
        "run_id": "run_stale",
        "git_sha": "sha_old",
    }
    store = {claim_blob: json.dumps(stale_payload).encode("utf-8")}
    mock_service = MockBlobServiceClient(store)

    with patch("scripts.azure_blob.get_blob_service_client", return_value=mock_service):
        reclaimed = acquire_daily_claim(
            account_name="mockacc",
            container_name="propredict-serving",
            run_id="run_fresh",
            force=False,
            timeout_minutes=75,
        )
        assert reclaimed is True, "Stale claim older than timeout_minutes must be reclaimed."
        claim_data = json.loads(store[claim_blob].decode("utf-8"))
        assert claim_data["run_id"] == "run_fresh"


def test_force_claim_bypasses_completed_status():
    """A manual dispatch with force=True overwrites any existing claim status immediately."""
    today_ny = datetime.now(timezone.utc).astimezone(ZoneInfo("America/New_York")).date().isoformat()
    claim_blob = get_daily_claim_blob_name(today_ny)

    completed_payload = {
        "date_ny": today_ny,
        "status": "COMPLETED",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "run_id": "run_done",
    }
    store = {claim_blob: json.dumps(completed_payload).encode("utf-8")}
    mock_service = MockBlobServiceClient(store)

    with patch("scripts.azure_blob.get_blob_service_client", return_value=mock_service):
        forced = acquire_daily_claim(
            account_name="mockacc",
            container_name="propredict-serving",
            run_id="run_force",
            force=True,
        )
        assert forced is True, "Manual run with force=True must bypass existing completed claim."
        claim_data = json.loads(store[claim_blob].decode("utf-8"))
        assert claim_data["run_id"] == "run_force"
        assert claim_data["status"] == "IN_PROGRESS"


def test_validate_artifacts_catches_empty_and_corrupt_files(tmp_path, sample_serving_dfs, sample_pbp_df):
    """Pre-publication validation fails closed if serving artifacts are empty or history is degraded."""
    df_all, df_home = sample_serving_dfs
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    pbp_file = data_dir / "play_by_play.parquet"
    all_file = data_dir / "frontend_all.parquet"
    home_file = data_dir / "frontend_home.parquet"

    sample_pbp_df.to_parquet(pbp_file)
    df_all.to_parquet(all_file)
    df_home.to_parquet(home_file)

    # Baseline metadata matching current data
    meta_path = str(pbp_file) + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"row_count": len(sample_pbp_df), "unique_games": 1}, f)

    # Valid artifacts pass
    res = validate_artifacts(str(data_dir))
    assert res["frontend_all"]["row_count"] == len(df_all)
    assert res["pbp"]["unique_games"] == 1

    # Empty frontend_all must raise ValueError
    df_empty = df_all.iloc[0:0]
    df_empty.to_parquet(all_file)
    with pytest.raises(ValueError, match="0 rows"):
        validate_artifacts(str(data_dir))

    # Restore frontend_all, test game decrease in play_by_play
    df_all.to_parquet(all_file)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"row_count": len(sample_pbp_df), "unique_games": 10}, f)  # previous had 10 games
    with pytest.raises(ValueError, match="unique game count decreased"):
        validate_artifacts(str(data_dir))


def test_serving_publication_failure_leaves_previous_manifest_intact(tmp_path, sample_serving_dfs):
    """
    If any error occurs while publishing snapshot parquet files,
    root current_manifest.json is NOT updated and the previous manifest remains intact.
    """
    df_all, df_home = sample_serving_dfs
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    df_all.to_parquet(data_dir / "frontend_all.parquet")
    df_home.to_parquet(data_dir / "frontend_home.parquet")

    now_et = datetime.now(ZoneInfo("America/New_York"))
    tomorrow_et = now_et.date().strftime("%Y-%m-%d")
    df_sched = pd.DataFrame([{
        "game_id": "0022400999",
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
    }])
    df_sched.to_parquet(data_dir / "schedule_tomorrow.parquet", index=False)
    with open(data_dir / "schedule_tomorrow_metadata.json", "w", encoding="utf-8") as f:
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

    with open(data_dir / "ingestion_metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "status": "SUCCESS",
            "is_fresh": True,
            "new_games_ingested": 1,
            "failed_games_count": 0,
            "historical_cutoff_date": "2026-02-06",
        }, f)

    # Initial manifest in store
    initial_manifest = {
        "snapshot_id": "snap_initial",
        "published_at": "2026-10-17T09:00:00+00:00",
        "files": {"frontend_all.parquet": {}, "frontend_home.parquet": {}},
    }
    initial_bytes = json.dumps(initial_manifest).encode("utf-8")
    store = {"current_manifest.json": initial_bytes}

    mock_service = MockBlobServiceClient(store)

    # Patch container client so home.parquet upload fails
    original_get_container = mock_service.get_container_client

    class FaultyContainerClient:
        def __init__(self, store):
            self.store = store

        def get_blob_client(self, blob_name):
            if "frontend_home.parquet" in blob_name:
                mock_client = MagicMock()
                mock_client.upload_blob.side_effect = RuntimeError("Network partition during parquet upload")
                return mock_client
            return MockBlobClient(self.store, blob_name)

    with patch("scripts.azure_blob.get_blob_service_client", return_value=mock_service):
        with patch.object(mock_service, "get_container_client", return_value=FaultyContainerClient(store)):
            with pytest.raises(RuntimeError, match="Network partition during parquet upload"):
                publish_serving_snapshot(
                    account_name="mockacc",
                    container_name="propredict-serving",
                    data_dir=str(data_dir),
                )

    # Root current_manifest.json must be untouched
    assert store["current_manifest.json"] == initial_bytes
    manifest_after = json.loads(store["current_manifest.json"].decode("utf-8"))
    assert manifest_after["snapshot_id"] == "snap_initial"


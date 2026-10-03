"""
tests/test_orchestrator.py

Comprehensive tests for Phase 2E.8 Deployment Hardening:
- Concurrency locking (PipelineLock)
- Pipeline status recording and history tracking
- Failure isolation and safe execution
- Atomic publication and corruption safety
- FastAPI dynamic mtime-based reload
"""

import os
import sys
import json
import time
import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock
import pandas as pd
from fastapi.testclient import TestClient

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.orchestrator import (
    PipelineLock,
    record_status,
    execute_daily_pipeline,
    parse_args,
)
from scripts.build_frontend_dataset import build_frontend_dataset
from api.main import create_app
from api.config import Settings
from api.loader import reload_serving_state_if_stale, load_serving_state


# ==============================================================================
# 1. PipelineLock Concurrency Tests
# ==============================================================================
class TestPipelineLock:
    def test_acquire_and_release(self, tmp_path):
        lock_file = str(tmp_path / "pipeline.lock")
        lock = PipelineLock(lock_file)

        assert lock.acquire() is True
        assert os.path.exists(lock_file)

        with open(lock_file, "r") as f:
            data = json.load(f)
            assert data["pid"] == os.getpid()

        lock.release()
        assert not os.path.exists(lock_file)

    def test_lock_context_manager(self, tmp_path):
        lock_file = str(tmp_path / "pipeline.lock")
        with PipelineLock(lock_file):
            assert os.path.exists(lock_file)
        assert not os.path.exists(lock_file)

    def test_prevent_concurrent_execution(self, tmp_path):
        lock_file = str(tmp_path / "pipeline.lock")
        lock1 = PipelineLock(lock_file)
        lock2 = PipelineLock(lock_file)

        assert lock1.acquire() is True
        # Second acquire while first is held must return False
        assert lock2.acquire() is False

        lock1.release()
        # After release, lock2 can acquire
        assert lock2.acquire() is True
        lock2.release()

    def test_stale_lock_recovery(self, tmp_path):
        lock_file = str(tmp_path / "pipeline.lock")
        # Write a lock file with an impossible/dead PID
        with open(lock_file, "w") as f:
            json.dump({"pid": 99999999, "acquired_at": "2026-01-01T00:00:00"}, f)

        lock = PipelineLock(lock_file)
        with patch.object(PipelineLock, "_is_process_alive", return_value=False):
            assert lock.acquire() is True
        lock.release()


# ==============================================================================
# 2. Status Recording Tests
# ==============================================================================
class TestStatusRecording:
    def test_record_status_lifecycle(self, tmp_path):
        status_file = str(tmp_path / "pipeline_status.json")
        runs_log = str(tmp_path / "pipeline_runs.jsonl")

        start = datetime.now()
        # Record RUNNING
        record_status(
            status="RUNNING",
            start_time=start,
            mode="full",
            status_file=status_file,
            runs_log=runs_log,
        )
        with open(status_file, "r") as f:
            s1 = json.load(f)
            assert s1["status"] == "RUNNING"
            assert s1["end_time"] is None

        # Record SUCCESS
        end = datetime.now()
        record_status(
            status="SUCCESS",
            start_time=start,
            end_time=end,
            mode="full",
            artifacts_updated=True,
            status_file=status_file,
            runs_log=runs_log,
        )
        with open(status_file, "r") as f:
            s2 = json.load(f)
            assert s2["status"] == "SUCCESS"
            assert s2["duration_seconds"] is not None
            assert s2["artifacts_updated"] is True

        # Check runs_log has one completed entry
        with open(runs_log, "r") as f:
            lines = f.readlines()
            assert len(lines) == 1
            entry = json.loads(lines[0])
            assert entry["status"] == "SUCCESS"


# ==============================================================================
# 3. Orchestrator Execution Tests
# ==============================================================================
class TestExecuteDailyPipeline:
    def test_successful_run(self, tmp_path):
        status_file = str(tmp_path / "status.json")
        lock_file = str(tmp_path / "test.lock")

        with patch("scripts.orchestrator.run_full_pipeline") as mock_run:
            success = execute_daily_pipeline(
                mode="full",
                skip_fetch=True,
                skip_train=True,
                status_file=status_file,
                lock_file=lock_file,
            )
            assert success is True
            mock_run.assert_called_once_with(skip_fetch=True, skip_train=True, export_json=False)

        with open(status_file, "r") as f:
            status = json.load(f)
            assert status["status"] == "SUCCESS"
            assert status["artifacts_updated"] is True

        # Lock must be released
        assert not os.path.exists(lock_file)

    def test_failed_run_isolates_error(self, tmp_path):
        status_file = str(tmp_path / "status.json")
        lock_file = str(tmp_path / "test.lock")

        with patch("scripts.orchestrator.run_full_pipeline", side_effect=RuntimeError("NBA API 503 Timeout")):
            success = execute_daily_pipeline(
                mode="full",
                skip_fetch=False,
                skip_train=True,
                status_file=status_file,
                lock_file=lock_file,
            )
            assert success is False

        with open(status_file, "r") as f:
            status = json.load(f)
            assert status["status"] == "FAILED"
            assert "NBA API 503 Timeout" in status["error"]
            assert status["artifacts_updated"] is False

        # Lock must still be cleanly released
        assert not os.path.exists(lock_file)

    def test_locked_execution_skips_and_returns_false(self, tmp_path):
        status_file = str(tmp_path / "status.json")
        lock_file = str(tmp_path / "test.lock")

        # Manually hold lock
        lock = PipelineLock(lock_file)
        assert lock.acquire() is True

        with patch("scripts.orchestrator.run_full_pipeline") as mock_run:
            success = execute_daily_pipeline(
                mode="full",
                status_file=status_file,
                lock_file=lock_file,
            )
            assert success is False
            mock_run.assert_not_called()

        lock.release()


# ==============================================================================
# 4. Atomic Publication and Corruption Safety
# ==============================================================================
class TestAtomicArtifactSafety:
    def test_atomic_publication_preserves_valid_on_write_failure(self, tmp_path):
        """If writing or validating temporary artifacts fails, existing files are untouched."""
        home_path = str(tmp_path / "frontend_home.parquet")
        all_path = str(tmp_path / "frontend_all.parquet")

        # Create valid existing artifacts
        original_home = pd.DataFrame([{"dummy": 1}])
        original_all = pd.DataFrame([{"dummy": 2}])
        original_home.to_parquet(home_path)
        original_all.to_parquet(all_path)

        orig_mtime_all = os.path.getmtime(all_path)

        # Mock pd.DataFrame.to_parquet to simulate a mid-write disk/network failure on home
        with patch.object(pd.DataFrame, "to_parquet", side_effect=IOError("Disk quota exceeded")):
            with pytest.raises(IOError):
                build_frontend_dataset(
                    output_home_path=home_path,
                    output_all_path=all_path,
                    pred_df=pd.DataFrame([{"teamId": 1, "personId": 10, "status": "Active", "predicted_rating": 85.0}]),
                    games_df=pd.DataFrame(),
                )

        # Existing files must be untouched
        assert os.path.getmtime(all_path) == orig_mtime_all
        pd.testing.assert_frame_equal(pd.read_parquet(home_path), original_home)
        pd.testing.assert_frame_equal(pd.read_parquet(all_path), original_all)

        # No dangling temporary files left behind
        assert not os.path.exists(f"{home_path}.tmp")
        assert not os.path.exists(f"{all_path}.tmp")


# ==============================================================================
# 5. FastAPI Dynamic mtime Reload Tests
# ==============================================================================
class TestFastAPIDynamicReload:
    def test_fastapi_reloads_when_artifact_mtime_advances(self, tmp_path):
        """FastAPI automatically detects newly written parquet and serves fresh data."""
        # Setup settings pointing to tmp_path
        settings = Settings(
            data_dir=tmp_path,
            allowed_origins=["*"],
            env="testing",
            reload=False,
        )

        # Create initial valid parquet artifacts (1 player)
        initial_player = {
            "gameId": "1",
            "teamId": 1610612738,
            "personId": 101,
            "playerName": "Jayson Tatum",
            "rating": 90.0,
            "last_game_id": "0",
            "last_game_date": "2026-04-01",
            "game_date": "2026-04-02",
            "last_game": 90.0,
            "last3_avg": 90.0,
            "last5_avg": 90.0,
            "last7_avg": 90.0,
            "predicted_rating": 92.5,
            "status": "Active",
            "injury_reason": "",
            "injury_date": "",
            "injury_source": "",
            "status_rank": 0,
            "teamName": "Celtics",
            "opponentId": 1610612748,
            "opponentName": "Heat",
            "headshot": "",
        }
        df_all = pd.DataFrame([initial_player])
        df_home = pd.DataFrame([initial_player])

        df_all.to_parquet(settings.frontend_all_path)
        df_home.to_parquet(settings.frontend_home_path)

        app = create_app(settings)
        with TestClient(app) as client:
            resp1 = client.get("/api/v1/players")
            assert resp1.status_code == 200
            data1 = resp1.json()
            assert len(data1["players"]) == 1
            assert data1["players"][0]["playerName"] == "Jayson Tatum"

            # Simulate pipeline updating artifacts with a new player
            time.sleep(0.05)  # Ensure distinct mtime
            updated_player = dict(initial_player, playerName="Jaylen Brown", personId=102)
            df_updated_all = pd.DataFrame([updated_player])
            df_updated_home = pd.DataFrame([updated_player])

            df_updated_all.to_parquet(settings.frontend_all_path)
            df_updated_home.to_parquet(settings.frontend_home_path)

            # Subsequent request to FastAPI must automatically detect new mtime and reload
            resp2 = client.get("/api/v1/players")
            assert resp2.status_code == 200
            data2 = resp2.json()
            assert len(data2["players"]) == 1
            assert data2["players"][0]["playerName"] == "Jaylen Brown"

    def test_failed_reload_preserves_in_memory_state(self, tmp_path):
        """If on-disk file becomes corrupted after startup, FastAPI continues serving in-memory data."""
        settings = Settings(
            data_dir=tmp_path,
            allowed_origins=["*"],
            env="testing",
            reload=False,
        )

        valid_player = {
            "gameId": "1",
            "teamId": 1610612738,
            "personId": 101,
            "playerName": "Jayson Tatum",
            "rating": 90.0,
            "last_game_id": "0",
            "last_game_date": "2026-04-01",
            "game_date": "2026-04-02",
            "last_game": 90.0,
            "last3_avg": 90.0,
            "last5_avg": 90.0,
            "last7_avg": 90.0,
            "predicted_rating": 92.5,
            "status": "Active",
            "injury_reason": "",
            "injury_date": "",
            "injury_source": "",
            "status_rank": 0,
            "teamName": "Celtics",
            "opponentId": 1610612748,
            "opponentName": "Heat",
            "headshot": "",
        }
        pd.DataFrame([valid_player]).to_parquet(settings.frontend_all_path)
        pd.DataFrame([valid_player]).to_parquet(settings.frontend_home_path)

        app = create_app(settings)
        with TestClient(app) as client:
            resp1 = client.get("/api/v1/players")
            assert resp1.status_code == 200
            assert resp1.json()["players"][0]["playerName"] == "Jayson Tatum"

            # Corrupt the file on disk
            time.sleep(0.05)
            with open(settings.frontend_all_path, "wb") as f:
                f.write(b"not a valid parquet file")

            # Client request must not crash; preserves existing in-memory data
            resp2 = client.get("/api/v1/players")
            assert resp2.status_code == 200
            assert resp2.json()["players"][0]["playerName"] == "Jayson Tatum"

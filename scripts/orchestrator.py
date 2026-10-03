"""
scripts/orchestrator.py

Production Daily Pipeline Orchestrator and Scheduler for the NBA Player Prediction System.

Responsibilities:
- Invokes the existing canonical production pipeline without duplicating logic.
- Completely decoupled from FastAPI (does not run inside FastAPI).
- Concurrency-safe: uses a process lockfile to prevent overlapping runs.
- Tracks execution status, timings, and errors in data/pipeline_status.json.
- Provides two execution modes:
    1. One-shot (--once, default): suitable for cron, Windows Task Scheduler, GitHub Actions, AWS EventBridge.
    2. Daemon mode (--daemon): runs continuously in background and triggers execution daily at a scheduled time.
"""

import os
import sys
import time
import json
import logging
import argparse
import traceback
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from run_full_pipeline import run_full_pipeline, run_prediction_pipeline

# ==============================
# LOGGING SETUP
# ==============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("propredict.orchestrator")

DEFAULT_STATUS_FILE = os.path.join(BASE_DIR, "data", "pipeline_status.json")
DEFAULT_RUNS_LOG = os.path.join(BASE_DIR, "data", "pipeline_runs.jsonl")
DEFAULT_LOCK_FILE = os.path.join(BASE_DIR, "data", "pipeline.lock")


# ==============================
# CONCURRENCY LOCK
# ==============================
class PipelineLock:
    """File-based process lock to guarantee single-flight pipeline execution."""

    def __init__(self, lock_file: str = DEFAULT_LOCK_FILE):
        self.lock_file = lock_file
        self.acquired = False

    def acquire(self) -> bool:
        os.makedirs(os.path.dirname(self.lock_file) or ".", exist_ok=True)
        if os.path.exists(self.lock_file):
            try:
                with open(self.lock_file, "r") as f:
                    lock_info = json.load(f)
                pid = lock_info.get("pid")
                # Check if process is still alive
                if pid and self._is_process_alive(pid):
                    logger.warning("Another pipeline execution is active (PID %s). Aborting run.", pid)
                    return False
                else:
                    logger.info("Found stale lockfile from PID %s. Overwriting.", pid)
            except Exception:
                pass

        try:
            lock_payload = {
                "pid": os.getpid(),
                "acquired_at": datetime.now().isoformat(),
            }
            with open(self.lock_file, "w") as f:
                json.dump(lock_payload, f)
            self.acquired = True
            return True
        except Exception as e:
            logger.error("Failed to acquire lock: %s", e)
            return False

    def release(self):
        if self.acquired and os.path.exists(self.lock_file):
            try:
                os.remove(self.lock_file)
            except OSError:
                pass
            self.acquired = False

    def __enter__(self):
        if not self.acquire():
            raise RuntimeError("Pipeline lock could not be acquired: another run is in progress.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    @staticmethod
    def _is_process_alive(pid: int) -> bool:
        if sys.platform == "win32":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            SYNCHRONIZE = 0x00100000
            process = kernel32.OpenProcess(SYNCHRONIZE, False, pid)
            if process != 0:
                kernel32.CloseHandle(process)
                return True
            return False
        else:
            try:
                os.kill(pid, 0)
                return True
            except OSError:
                return False


# ==============================
# STATUS RECORDING
# ==============================
def record_status(
    status: str,
    start_time: datetime,
    end_time: Optional[datetime] = None,
    mode: str = "full",
    skip_fetch: bool = False,
    skip_train: bool = True,
    error: Optional[str] = None,
    artifacts_updated: bool = False,
    status_file: str = DEFAULT_STATUS_FILE,
    runs_log: str = DEFAULT_RUNS_LOG,
) -> Dict[str, Any]:
    """Records pipeline execution status to status JSON and appends to runs JSONL."""
    os.makedirs(os.path.dirname(status_file) or ".", exist_ok=True)

    duration = round((end_time - start_time).total_seconds(), 2) if end_time else None
    record = {
        "status": status,
        "mode": mode,
        "skip_fetch": skip_fetch,
        "skip_train": skip_train,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat() if end_time else None,
        "duration_seconds": duration,
        "error": error,
        "artifacts_updated": artifacts_updated,
    }

    try:
        # Atomic status write
        temp_status = f"{status_file}.tmp"
        with open(temp_status, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        os.replace(temp_status, status_file)
    except Exception as e:
        logger.error("Failed to write status file: %s", e)

    # Append completed runs to history log
    if status in ("SUCCESS", "FAILED"):
        try:
            with open(runs_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.error("Failed to append to runs log: %s", e)

    return record


# ==============================
# ORCHESTRATED RUNNER
# ==============================
def execute_daily_pipeline(
    mode: str = "full",
    skip_fetch: bool = False,
    skip_train: bool = True,
    export_json: bool = False,
    status_file: str = DEFAULT_STATUS_FILE,
    lock_file: str = DEFAULT_LOCK_FILE,
) -> bool:
    """Executes the daily prediction update with process locking and status tracking.

    Returns:
        True if the pipeline completed successfully, False on error or lock contention.
    """
    lock = PipelineLock(lock_file=lock_file)
    if not lock.acquire():
        logger.warning("Aborting execution: could not acquire pipeline lock.")
        return False

    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("ORCHESTRATOR: Starting daily pipeline (mode=%s, skip_fetch=%s, skip_train=%s)", mode, skip_fetch, skip_train)
    logger.info("=" * 60)

    record_status(
        status="RUNNING",
        start_time=start_time,
        mode=mode,
        skip_fetch=skip_fetch,
        skip_train=skip_train,
        status_file=status_file,
    )

    try:
        if mode == "full":
            run_full_pipeline(skip_fetch=skip_fetch, skip_train=skip_train, export_json=export_json)
        elif mode == "predict":
            run_prediction_pipeline(export_json=export_json)
        else:
            raise ValueError(f"Unsupported orchestrator mode: {mode}")

        end_time = datetime.now()
        duration = round((end_time - start_time).total_seconds(), 2)
        logger.info("=" * 60)
        logger.info("ORCHESTRATOR: Pipeline finished SUCCESS in %.2fs", duration)
        logger.info("=" * 60)

        record_status(
            status="SUCCESS",
            start_time=start_time,
            end_time=end_time,
            mode=mode,
            skip_fetch=skip_fetch,
            skip_train=skip_train,
            error=None,
            artifacts_updated=True,
            status_file=status_file,
        )
        return True

    except Exception as e:
        end_time = datetime.now()
        duration = round((end_time - start_time).total_seconds(), 2)
        err_msg = str(e)
        logger.error("=" * 60)
        logger.error("ORCHESTRATOR: Pipeline FAILED after %.2fs: %s", duration, err_msg)
        logger.error(traceback.format_exc())
        logger.error("=" * 60)

        record_status(
            status="FAILED",
            start_time=start_time,
            end_time=end_time,
            mode=mode,
            skip_fetch=skip_fetch,
            skip_train=skip_train,
            error=err_msg,
            artifacts_updated=False,
            status_file=status_file,
        )
        return False

    finally:
        lock.release()


# ==============================
# DAEMON SCHEDULER LOOP
# ==============================
def run_daemon(
    schedule_time_str: str = "06:00",
    mode: str = "full",
    skip_fetch: bool = False,
    skip_train: bool = True,
    export_json: bool = False,
    status_file: str = DEFAULT_STATUS_FILE,
    lock_file: str = DEFAULT_LOCK_FILE,
    max_iterations: Optional[int] = None,
):
    """Runs the scheduler continuously in the background, firing daily at schedule_time_str (HH:MM)."""
    target_hour, target_minute = map(int, schedule_time_str.split(":"))
    logger.info("Starting Orchestrator daemon. Target daily update time: %02d:%02d local time.", target_hour, target_minute)

    iteration = 0
    while True:
        now = datetime.now()
        target = now.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)

        wait_seconds = (target - now).total_seconds()
        logger.info("Next daily pipeline run scheduled for %s (in %.1f hours)", target.strftime("%Y-%m-%d %H:%M:%S"), wait_seconds / 3600)

        # Sleep until scheduled time
        time.sleep(wait_seconds)

        logger.info("Scheduled time reached (%s). Launching pipeline update.", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        execute_daily_pipeline(
            mode=mode,
            skip_fetch=skip_fetch,
            skip_train=skip_train,
            export_json=export_json,
            status_file=status_file,
            lock_file=lock_file,
        )

        iteration += 1
        if max_iterations is not None and iteration >= max_iterations:
            logger.info("Reached maximum iterations (%d). Stopping daemon.", max_iterations)
            break


# ==============================
# CLI PARSER & MAIN
# ==============================
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="NBA Player Prediction Daily Pipeline Orchestrator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--once",
        action="store_true",
        default=True,
        help="Execute the pipeline once and exit immediately (default)",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run continuously as a daily background scheduler",
    )
    parser.add_argument(
        "--time",
        type=str,
        default=os.environ.get("PROPREDICT_SCHEDULE_TIME", "06:00"),
        help="Daily scheduled run time in HH:MM format for --daemon mode",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "predict"],
        default="full",
        help="Pipeline mode to execute",
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip historical play-by-play API fetch (applies to 'full' mode)",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Retrain the prediction model (default skips training during daily runs)",
    )
    parser.add_argument(
        "--export-json",
        action="store_true",
        help="Also export legacy static JSON files to lineup-builder/public",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    skip_train = not args.force_retrain

    if args.daemon:
        run_daemon(
            schedule_time_str=args.time,
            mode=args.mode,
            skip_fetch=args.skip_fetch,
            skip_train=skip_train,
            export_json=args.export_json,
        )
    else:
        success = execute_daily_pipeline(
            mode=args.mode,
            skip_fetch=args.skip_fetch,
            skip_train=skip_train,
            export_json=args.export_json,
        )
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

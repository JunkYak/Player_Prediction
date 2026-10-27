"""
scripts/run_aca_job.py

Dedicated execution runner for Azure Container Apps (ACA) scheduled batch jobs.
Executes the production NBA prediction pipeline in batch mode:
1. Acquire daily claim (single-writer mutual exclusion via Azure Blob CAS).
2. Restore historical baseline from propredict-history.
3. Run existing pipeline with skip_train=True (model retraining disabled).
4. Validate generated pipeline and serving artifacts locally.
5. Publish updated historical data to propredict-history.
6. Publish serving snapshot to propredict-serving (atomically updating current_manifest.json).
7. Mark daily claim as COMPLETED.

Failure handling:
If any step fails, marks the daily claim as FAILED to allow immediate safe retries,
and exits with code 1. Never marks claim complete after a partial publication.
"""

import os
import sys
import time
import logging
import traceback
import argparse
from typing import Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from scripts.azure_blob import (
    acquire_daily_claim,
    restore_historical_data,
    validate_artifacts,
    publish_historical_data,
    publish_serving_snapshot,
    complete_daily_claim,
    fail_daily_claim,
    DEFAULT_STORAGE_ACCOUNT,
    DEFAULT_HISTORY_CONTAINER,
    DEFAULT_SERVING_CONTAINER,
)
from scripts.orchestrator import execute_daily_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("propredict.aca_job")


def run_job(
    account_name: Optional[str] = None,
    history_container: Optional[str] = None,
    serving_container: Optional[str] = None,
    data_dir: Optional[str] = None,
    force: bool = False,
    skip_fetch: bool = False,
    run_id: Optional[str] = None,
    git_sha: Optional[str] = None,
) -> bool:
    account = account_name or os.environ.get("AZURE_STORAGE_ACCOUNT", DEFAULT_STORAGE_ACCOUNT)
    hist_cont = history_container or os.environ.get("AZURE_HISTORY_CONTAINER", DEFAULT_HISTORY_CONTAINER)
    srv_cont = serving_container or os.environ.get("AZURE_SERVING_CONTAINER", DEFAULT_SERVING_CONTAINER)
    data_path = data_dir or os.environ.get("PROPREDICT_DATA_DIR", os.path.join(BASE_DIR, "data"))
    os.makedirs(data_path, exist_ok=True)

    job_run_id = (
        run_id
        or os.environ.get("CONTAINER_APP_JOB_EXECUTION_NAME")
        or os.environ.get("ACA_JOB_EXECUTION_NAME")
        or os.environ.get("GITHUB_RUN_ID")
        or f"aca_run_{int(time.time())}"
    )
    commit_sha = git_sha or os.environ.get("GIT_SHA") or os.environ.get("GITHUB_SHA") or "unknown"

    logger.info("============================================================")
    logger.info("PROPREDICT ACA BATCH PIPELINE JOB STARTING")
    logger.info("Run ID:            %s", job_run_id)
    logger.info("Storage Account:   %s", account)
    logger.info("History Container: %s", hist_cont)
    logger.info("Serving Container: %s", srv_cont)
    logger.info("Data Directory:    %s", data_path)
    logger.info("Force Execution:   %s", force)
    logger.info("Skip NBA Fetch:    %s", skip_fetch)
    logger.info("============================================================")

    # Step 1: Acquire daily claim
    logger.info("[Step 1/7] Acquiring atomic daily single-writer claim...")
    claim_acquired = acquire_daily_claim(
        account_name=account,
        container_name=srv_cont,
        run_id=job_run_id,
        git_sha=commit_sha,
        force=force,
    )
    if not claim_acquired:
        logger.info(
            "Daily claim was not acquired (today's run already completed or active concurrent run detected). "
            "Exiting gracefully."
        )
        return True

    try:
        # Step 2: Restore historical baseline
        target_pbp = os.path.join(data_path, "play_by_play.parquet")
        logger.info("[Step 2/7] Restoring canonical historical baseline to '%s'...", target_pbp)
        restore_historical_data(
            account_name=account,
            container_name=hist_cont,
            target_path=target_pbp,
        )

        # Step 3: Run existing pipeline with skip_train=True
        logger.info("[Step 3/7] Running daily prediction pipeline (model training skipped)...")
        status_file = os.path.join(data_path, "pipeline_status.json")
        lock_file = os.path.join(data_path, "pipeline.lock")
        success = execute_daily_pipeline(
            mode="full",
            skip_fetch=skip_fetch,
            skip_train=True,
            export_json=False,
            status_file=status_file,
            lock_file=lock_file,
        )
        if not success:
            raise RuntimeError("execute_daily_pipeline returned False; inspect logs for step failure.")

        # Step 4: Validate generated artifacts
        logger.info("[Step 4/7] Validating pipeline artifacts prior to cloud publication...")
        validate_artifacts(data_dir=data_path)

        # Step 5: Publish updated historical data
        logger.info("[Step 5/7] Publishing validated historical data to '%s'...", hist_cont)
        publish_historical_data(
            account_name=account,
            container_name=hist_cont,
            source_path=target_pbp,
        )

        # Step 6: Publish serving snapshot (atomically updating current_manifest.json only after upload)
        logger.info("[Step 6/7] Publishing serving snapshot to '%s'...", srv_cont)
        publish_serving_snapshot(
            account_name=account,
            container_name=srv_cont,
            data_dir=data_path,
            pipeline_run_id=job_run_id,
            git_sha=commit_sha,
        )

        # Step 7: Mark claim COMPLETED
        logger.info("[Step 7/7] Marking daily claim as COMPLETED...")
        complete_daily_claim(
            account_name=account,
            container_name=srv_cont,
            run_id=job_run_id,
        )

        logger.info("============================================================")
        logger.info("PROPREDICT ACA BATCH PIPELINE JOB COMPLETED SUCCESSFULLY")
        logger.info("============================================================")
        return True

    except Exception as e:
        logger.error("ACA Pipeline Job encountered a fatal error: %s", e)
        logger.error(traceback.format_exc())
        try:
            logger.info("Marking daily claim as FAILED to allow immediate retries...")
            fail_daily_claim(
                account_name=account,
                container_name=srv_cont,
                run_id=job_run_id,
                error=str(e),
            )
        except Exception as fail_err:
            logger.warning("Could not record claim as FAILED: %s", fail_err)
        return False


def main():
    parser = argparse.ArgumentParser(description="ProPredict ACA Job Pipeline Runner")
    parser.add_argument("--account", default=None, help="Azure Storage account name")
    parser.add_argument("--history-container", default=None, help="History container name")
    parser.add_argument("--serving-container", default=None, help="Serving container name")
    parser.add_argument("--data-dir", default=None, help="Directory for pipeline data")
    parser.add_argument("--force", action="store_true", help="Bypass idempotency guard and force execution")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip NBA API play-by-play fetch")
    parser.add_argument("--run-id", default=None, help="Job run identifier")
    parser.add_argument("--git-sha", default=None, help="Git commit SHA")
    args = parser.parse_args()

    force = args.force or os.environ.get("FORCE_RUN", "").lower() in ("1", "true", "yes")
    skip_fetch = args.skip_fetch or os.environ.get("SKIP_FETCH", "").lower() in ("1", "true", "yes")

    success = run_job(
        account_name=args.account,
        history_container=args.history_container,
        serving_container=args.serving_container,
        data_dir=args.data_dir,
        force=force,
        skip_fetch=skip_fetch,
        run_id=args.run_id,
        git_sha=args.git_sha,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

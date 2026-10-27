import os
import sys
import json
import shutil
import logging
import datetime
from zoneinfo import ZoneInfo
import azure.functions as func

# Ensure function app directory is on sys.path
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("propredict.azure_function")

app = func.FunctionApp()


def setup_working_environment():
    """
    On Azure Functions Linux, the application root (/home/site/wwwroot) is mounted read-only.
    Sets up a fully writable scratch directory in /tmp/propredict with symlinks to models
    and a writable data/ directory initialized with baseline files.
    """
    if sys.platform != "win32":
        work_dir = "/tmp/propredict"
        os.makedirs(work_dir, exist_ok=True)

        # 1. Setup models directory
        models_src = os.path.join(APP_DIR, "models")
        models_dst = os.path.join(work_dir, "models")
        if not os.path.exists(models_dst) and os.path.exists(models_src):
            try:
                os.symlink(models_src, models_dst)
            except OSError:
                if not os.path.exists(models_dst):
                    shutil.copytree(models_src, models_dst)

        # 2. Setup writable data directory
        data_work_dir = os.path.join(work_dir, "data")
        os.makedirs(data_work_dir, exist_ok=True)
        os.environ["PROPREDICT_DATA_DIR"] = data_work_dir
        os.environ["INJURY_CACHE_FALLBACK"] = "1"

        # 3. Copy baseline seed data files if not already in scratch
        seed_data_dir = os.path.join(APP_DIR, "data")
        if os.path.exists(seed_data_dir):
            for fname in os.listdir(seed_data_dir):
                src_path = os.path.join(seed_data_dir, fname)
                dst_path = os.path.join(data_work_dir, fname)
                if os.path.isfile(src_path) and not os.path.exists(dst_path):
                    shutil.copy2(src_path, dst_path)

        os.chdir(work_dir)
        logger.info("Configured writable working directory at: %s", work_dir)


def run_pipeline_task(mode: str = "full", skip_fetch: bool = False) -> dict:
    """
    Executes the NBA data & prediction pipeline and publishes serving artifacts to Azure Blob Storage.
    """
    start_time = datetime.datetime.now(datetime.timezone.utc)
    ist_time = start_time.astimezone(ZoneInfo("Asia/Kolkata"))
    logger.info("=" * 60)
    logger.info("AZURE FUNCTION: Starting NBA Daily Pipeline at %s (IST: %s)", start_time.isoformat(), ist_time.isoformat())
    logger.info("Configuration: mode=%s, skip_fetch=%s, publish_azure=True", mode, skip_fetch)
    logger.info("=" * 60)

    try:
        # Setup scratch workspace in /tmp
        setup_working_environment()

        from scripts.azure_blob import restore_historical_data
        try:
            logger.info("Checking & restoring canonical history from propredict-history...")
            restore_historical_data()
        except Exception as e:
            logger.warning("Could not restore canonical history from blob (using local scratch copy): %s", e)

        from run_full_pipeline import run_full_pipeline, run_prediction_pipeline
        if mode == "full":
            run_full_pipeline(skip_fetch=skip_fetch, skip_train=True, publish_azure=True)
        else:
            run_prediction_pipeline(publish_azure=True)

        end_time = datetime.datetime.now(datetime.timezone.utc)
        duration = round((end_time - start_time).total_seconds(), 2)
        logger.info("AZURE FUNCTION: Pipeline completed SUCCESS in %.2fs", duration)

        return {
            "status": "SUCCESS",
            "start_time_utc": start_time.isoformat(),
            "start_time_ist": ist_time.isoformat(),
            "duration_seconds": duration,
            "mode": mode,
            "skip_fetch": skip_fetch,
            "error": None,
        }

    except Exception as exc:
        end_time = datetime.datetime.now(datetime.timezone.utc)
        duration = round((end_time - start_time).total_seconds(), 2)
        err_msg = str(exc)
        logger.error("AZURE FUNCTION: Pipeline FAILED after %.2fs: %s", duration, err_msg, exc_info=True)

        return {
            "status": "FAILED",
            "start_time_utc": start_time.isoformat(),
            "start_time_ist": ist_time.isoformat(),
            "duration_seconds": duration,
            "mode": mode,
            "skip_fetch": skip_fetch,
            "error": err_msg,
        }


# ==============================================================================
# 1. Scheduled Timer Trigger (Fixed Daily Run at 13:00 IST / 07:30 UTC)
# ==============================================================================
# Azure Functions 6-field CRON: {second} {minute} {hour} {day} {month} {day-of-week}
# 0 30 7 * * * = Every day at 07:30:00 UTC (13:00:00 IST)
@app.timer_trigger(schedule="0 30 7 * * *", arg_name="myTimer", run_on_startup=False, use_monitor=True)
def daily_nba_pipeline_timer(myTimer: func.TimerRequest) -> None:
    if myTimer.past_due:
        logger.info("Timer trigger is running past due.")

    logger.info("⏰ Scheduled Timer triggered daily NBA prediction pipeline.")
    result = run_pipeline_task(mode="full", skip_fetch=False)
    logger.info("Daily pipeline result: %s", json.dumps(result))


# ==============================================================================
# 1b. One-Time Test Timer (Targeted for October 27, 2026 at 15:30 IST / 10:00 UTC)
# ==============================================================================
# 0 0 10 27 10 * = Exactly at 10:00:00 UTC on October 27th (3:30:00 PM IST)
@app.timer_trigger(schedule="0 0 10 27 10 *", arg_name="testTimer", run_on_startup=False, use_monitor=True)
def test_pipeline_timer_oct27(testTimer: func.TimerRequest) -> None:
    if testTimer.past_due:
        logger.info("One-time test timer is running past due.")

    logger.info("⏰ One-time test timer triggered NBA prediction pipeline for Oct 27 15:30 IST.")
    result = run_pipeline_task(mode="full", skip_fetch=False)
    logger.info("One-time test pipeline result: %s", json.dumps(result))


# ==============================================================================
# 2. HTTP Route for Manual / Automated Verification Triggers
# ==============================================================================
@app.route(route="trigger_refresh", methods=["POST", "GET"], auth_level=func.AuthLevel.FUNCTION)
def trigger_refresh_http(req: func.HttpRequest) -> func.HttpResponse:
    logger.info("🌐 HTTP request received to trigger NBA pipeline refresh.")

    mode = req.params.get("mode", "full")
    skip_fetch_param = req.params.get("skip_fetch", "false").lower() in ("1", "true", "yes")
    is_async = req.params.get("async", "false").lower() in ("1", "true", "yes")

    if is_async:
        import threading
        thread = threading.Thread(target=run_pipeline_task, kwargs={"mode": mode, "skip_fetch": skip_fetch_param})
        thread.daemon = True
        thread.start()
        return func.HttpResponse(
            body=json.dumps({"status": "ACCEPTED", "message": "Pipeline execution started asynchronously in background thread"}),
            mimetype="application/json",
            status_code=202,
        )

    result = run_pipeline_task(mode=mode, skip_fetch=skip_fetch_param)
    status_code = 200 if result["status"] == "SUCCESS" else 500

    return func.HttpResponse(
        body=json.dumps(result, indent=2),
        mimetype="application/json",
        status_code=status_code,
    )

"""
tests/test_aca_job.py

Focused tests for scripts/run_aca_job.py:
1. Exact execution sequence in success path (claim -> restore -> pipeline -> validate -> pub history -> pub serving -> complete).
2. Clean graceful exit when claim is not acquired.
3. Proper fail-closed behavior and fail_daily_claim invocation on restore failure.
4. Proper fail-closed behavior on pipeline failure.
5. Proper fail-closed behavior on artifact validation failure.
6. Partial publication prevention: failure during publish_historical_data blocks publish_serving_snapshot and claim completion.
7. Partial publication prevention: failure during publish_serving_snapshot blocks claim completion.
"""

import os
import sys
from unittest.mock import MagicMock, patch, call
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from scripts.run_aca_job import run_job


def test_run_job_successful_execution_order(tmp_path):
    """
    Verifies that run_job executes the exact required 7-step sequence:
    1. acquire_daily_claim
    2. restore_historical_data
    3. execute_daily_pipeline (skip_train=True)
    4. validate_artifacts
    5. publish_historical_data
    6. publish_serving_snapshot
    7. complete_daily_claim
    """
    call_order = []

    def mock_acquire(*args, **kwargs):
        call_order.append("acquire_daily_claim")
        return True

    def mock_restore(*args, **kwargs):
        call_order.append("restore_historical_data")
        return {"row_count": 100}

    def mock_pipeline(*args, **kwargs):
        call_order.append("execute_daily_pipeline")
        assert kwargs.get("skip_train") is True
        return True

    def mock_validate(*args, **kwargs):
        call_order.append("validate_artifacts")
        return {"status": "ok"}

    def mock_pub_history(*args, **kwargs):
        call_order.append("publish_historical_data")
        return {"published": True}

    def mock_pub_serving(*args, **kwargs):
        call_order.append("publish_serving_snapshot")
        return {"snapshot_id": "snap_123"}

    def mock_complete(*args, **kwargs):
        call_order.append("complete_daily_claim")

    with patch("scripts.run_aca_job.acquire_daily_claim", side_effect=mock_acquire) as p_acq, \
         patch("scripts.run_aca_job.restore_historical_data", side_effect=mock_restore) as p_rst, \
         patch("scripts.run_aca_job.execute_daily_pipeline", side_effect=mock_pipeline) as p_pip, \
         patch("scripts.run_aca_job.validate_artifacts", side_effect=mock_validate) as p_val, \
         patch("scripts.run_aca_job.publish_historical_data", side_effect=mock_pub_history) as p_ph, \
         patch("scripts.run_aca_job.publish_serving_snapshot", side_effect=mock_pub_serving) as p_ps, \
         patch("scripts.run_aca_job.complete_daily_claim", side_effect=mock_complete) as p_comp, \
         patch("scripts.run_aca_job.fail_daily_claim") as p_fail:

        success = run_job(
            account_name="testacc",
            history_container="test-hist",
            serving_container="test-srv",
            data_dir=str(tmp_path),
            run_id="job_run_1",
            git_sha="git_sha_abc",
        )

        assert success is True
        assert call_order == [
            "acquire_daily_claim",
            "restore_historical_data",
            "execute_daily_pipeline",
            "validate_artifacts",
            "publish_historical_data",
            "publish_serving_snapshot",
            "complete_daily_claim",
        ]
        p_fail.assert_not_called()


def test_run_job_claim_not_acquired_exits_gracefully(tmp_path):
    """When claim is not acquired (already ran today), exits cleanly without executing pipeline."""
    with patch("scripts.run_aca_job.acquire_daily_claim", return_value=False) as p_acq, \
         patch("scripts.run_aca_job.restore_historical_data") as p_rst, \
         patch("scripts.run_aca_job.execute_daily_pipeline") as p_pip, \
         patch("scripts.run_aca_job.validate_artifacts") as p_val, \
         patch("scripts.run_aca_job.complete_daily_claim") as p_comp, \
         patch("scripts.run_aca_job.fail_daily_claim") as p_fail:

        success = run_job(data_dir=str(tmp_path))

        assert success is True
        p_rst.assert_not_called()
        p_pip.assert_not_called()
        p_val.assert_not_called()
        p_comp.assert_not_called()
        p_fail.assert_not_called()


def test_run_job_restore_failure_marks_failed(tmp_path):
    """If restore_historical_data fails, pipeline does not run and claim is marked FAILED."""
    with patch("scripts.run_aca_job.acquire_daily_claim", return_value=True), \
         patch("scripts.run_aca_job.restore_historical_data", side_effect=FileNotFoundError("Blob not found")), \
         patch("scripts.run_aca_job.execute_daily_pipeline") as p_pip, \
         patch("scripts.run_aca_job.complete_daily_claim") as p_comp, \
         patch("scripts.run_aca_job.fail_daily_claim") as p_fail:

        success = run_job(data_dir=str(tmp_path), run_id="run_fail_restore")

        assert success is False
        p_pip.assert_not_called()
        p_comp.assert_not_called()
        p_fail.assert_called_once()
        assert "Blob not found" in p_fail.call_args[1]["error"]


def test_run_job_pipeline_failure_marks_failed(tmp_path):
    """If execute_daily_pipeline returns False, publication is aborted and claim is marked FAILED."""
    with patch("scripts.run_aca_job.acquire_daily_claim", return_value=True), \
         patch("scripts.run_aca_job.restore_historical_data", return_value={}), \
         patch("scripts.run_aca_job.execute_daily_pipeline", return_value=False), \
         patch("scripts.run_aca_job.validate_artifacts") as p_val, \
         patch("scripts.run_aca_job.publish_historical_data") as p_ph, \
         patch("scripts.run_aca_job.complete_daily_claim") as p_comp, \
         patch("scripts.run_aca_job.fail_daily_claim") as p_fail:

        success = run_job(data_dir=str(tmp_path), run_id="run_fail_pipeline")

        assert success is False
        p_val.assert_not_called()
        p_ph.assert_not_called()
        p_comp.assert_not_called()
        p_fail.assert_called_once()


def test_run_job_validation_failure_blocks_all_publishing(tmp_path):
    """If validate_artifacts fails, neither history nor serving are published."""
    with patch("scripts.run_aca_job.acquire_daily_claim", return_value=True), \
         patch("scripts.run_aca_job.restore_historical_data", return_value={}), \
         patch("scripts.run_aca_job.execute_daily_pipeline", return_value=True), \
         patch("scripts.run_aca_job.validate_artifacts", side_effect=ValueError("frontend_all has 0 rows")), \
         patch("scripts.run_aca_job.publish_historical_data") as p_ph, \
         patch("scripts.run_aca_job.publish_serving_snapshot") as p_ps, \
         patch("scripts.run_aca_job.complete_daily_claim") as p_comp, \
         patch("scripts.run_aca_job.fail_daily_claim") as p_fail:

        success = run_job(data_dir=str(tmp_path), run_id="run_fail_val")

        assert success is False
        p_ph.assert_not_called()
        p_ps.assert_not_called()
        p_comp.assert_not_called()
        p_fail.assert_called_once()
        assert "0 rows" in p_fail.call_args[1]["error"]


def test_run_job_history_publish_failure_blocks_serving_and_complete(tmp_path):
    """If publish_historical_data fails, serving snapshot is not published and claim is not completed."""
    with patch("scripts.run_aca_job.acquire_daily_claim", return_value=True), \
         patch("scripts.run_aca_job.restore_historical_data", return_value={}), \
         patch("scripts.run_aca_job.execute_daily_pipeline", return_value=True), \
         patch("scripts.run_aca_job.validate_artifacts", return_value={}), \
         patch("scripts.run_aca_job.publish_historical_data", side_effect=RuntimeError("History upload timed out")), \
         patch("scripts.run_aca_job.publish_serving_snapshot") as p_ps, \
         patch("scripts.run_aca_job.complete_daily_claim") as p_comp, \
         patch("scripts.run_aca_job.fail_daily_claim") as p_fail:

        success = run_job(data_dir=str(tmp_path), run_id="run_fail_pub_hist")

        assert success is False
        p_ps.assert_not_called()
        p_comp.assert_not_called()
        p_fail.assert_called_once()
        assert "History upload timed out" in p_fail.call_args[1]["error"]


def test_run_job_serving_publish_failure_blocks_complete(tmp_path):
    """If publish_serving_snapshot fails, claim is never marked complete."""
    with patch("scripts.run_aca_job.acquire_daily_claim", return_value=True), \
         patch("scripts.run_aca_job.restore_historical_data", return_value={}), \
         patch("scripts.run_aca_job.execute_daily_pipeline", return_value=True), \
         patch("scripts.run_aca_job.validate_artifacts", return_value={}), \
         patch("scripts.run_aca_job.publish_historical_data", return_value={}), \
         patch("scripts.run_aca_job.publish_serving_snapshot", side_effect=RuntimeError("Serving upload error")), \
         patch("scripts.run_aca_job.complete_daily_claim") as p_comp, \
         patch("scripts.run_aca_job.fail_daily_claim") as p_fail:

        success = run_job(data_dir=str(tmp_path), run_id="run_fail_pub_serv")

        assert success is False
        p_comp.assert_not_called()
        p_fail.assert_called_once()
        assert "Serving upload error" in p_fail.call_args[1]["error"]

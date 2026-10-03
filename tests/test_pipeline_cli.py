"""
Phase 2E.2A — Regression tests for non-interactive pipeline CLI behavior.
Tests command-line parsing, mode selection, skip flags, and delegation.
"""
import sys
import os
import pytest
from unittest.mock import patch, MagicMock

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import run_full_pipeline
import scripts.run_full_pipeline as script_runner


def test_cli_default_is_full_mode_non_interactive():
    """Default invocation with no arguments must parse to mode='full' non-interactively."""
    args = run_full_pipeline.parse_args([])
    assert args.mode == "full"
    assert args.skip_fetch is False
    assert args.skip_train is False


def test_cli_explicit_modes():
    """Explicit mode flags must be recognized cleanly."""
    args_train = run_full_pipeline.parse_args(["--mode", "train"])
    assert args_train.mode == "train"

    args_predict = run_full_pipeline.parse_args(["--mode", "predict"])
    assert args_predict.mode == "predict"

    args_full = run_full_pipeline.parse_args(["--mode", "full"])
    assert args_full.mode == "full"


def test_cli_skip_flags():
    """Flags for skipping fetch or train must toggle boolean properties."""
    args = run_full_pipeline.parse_args(["--mode", "full", "--skip-fetch", "--skip-train"])
    assert args.mode == "full"
    assert args.skip_fetch is True
    assert args.skip_train is True


def test_cli_invalid_mode_rejected():
    """Invalid modes must trigger SystemExit error from argparse."""
    with pytest.raises(SystemExit):
        run_full_pipeline.parse_args(["--mode", "invalid_mode"])


def test_main_dispatches_correct_mode_function():
    """Verifies that main() dispatches to the corresponding mode function."""
    with patch("run_full_pipeline.run_full_pipeline") as mock_full, \
         patch("run_full_pipeline.run_training_pipeline") as mock_train, \
         patch("run_full_pipeline.run_prediction_pipeline") as mock_predict:

        # Test full mode dispatch
        run_full_pipeline.main(["--mode", "full", "--skip-fetch"])
        mock_full.assert_called_once_with(skip_fetch=True, skip_train=False)

        # Test train mode dispatch
        run_full_pipeline.main(["--mode", "train"])
        mock_train.assert_called_once()

        # Test predict mode dispatch
        run_full_pipeline.main(["--mode", "predict"])
        mock_predict.assert_called_once()


def test_script_runner_delegation():
    """Verifies that scripts/run_full_pipeline.py delegates to root run_full_pipeline.py."""
    with patch("run_full_pipeline.main") as mock_root_main:
        script_runner.main()
        mock_root_main.assert_called_once()

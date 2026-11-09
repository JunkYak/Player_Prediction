"""
tests/test_azure_function_parity.py

Comprehensive local-only unit and integration tests verifying Azure Function package
parity, pipeline execution sequence, and module functionality without network access
or Azure cloud credentials.
"""

import os
import sys
import json
import inspect
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

AZURE_FUNCTION_DIR = ROOT_DIR / "azure_function"


def load_az_module(module_name: str, rel_path: str):
    """Dynamically load a module from the azure_function directory with isolated sys.path."""
    file_path = AZURE_FUNCTION_DIR / rel_path
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    orig_path = list(sys.path)
    try:
        sys.path.insert(0, str(AZURE_FUNCTION_DIR))
        spec.loader.exec_module(mod)
    finally:
        sys.path = orig_path
    return mod


# ==============================================================================
# 1. Package File Parity Tests
# ==============================================================================

def test_azure_function_scripts_exist():
    """Verify that all canonical pipeline scripts exist in azure_function/scripts/."""
    expected_scripts = [
        "add_game_dates.py",
        "apply_injury_status.py",
        "azure_blob.py",
        "build_features.py",
        "build_frontend_dataset.py",
        "build_play_by_play_dataset.py",
        "compute_player_ratings.py",
        "compute_season_multipliers.py",
        "get_next_day_games.py",
        "injury_provider.py",
        "predict_ratings.py",
        "roster_provider.py",
        "team_rankings.py",
        "train_model.py",
    ]
    scripts_dir = AZURE_FUNCTION_DIR / "scripts"
    assert scripts_dir.is_dir(), f"Missing azure_function/scripts directory: {scripts_dir}"
    for script_name in expected_scripts:
        script_file = scripts_dir / script_name
        assert script_file.is_file(), f"Missing {script_name} in azure_function/scripts/"


def test_azure_function_exports_and_signatures():
    """Verify that azure_function modules export the required functions and constants."""
    # 1. get_next_day_games parity
    az_gndg = load_az_module("az_gndg", "scripts/get_next_day_games.py")
    import scripts.get_next_day_games as root_gndg

    assert hasattr(az_gndg, "get_tomorrow_schedule")
    assert hasattr(az_gndg, "SCHEDULE_COLUMNS")
    assert hasattr(az_gndg, "TEAM_ID_TO_NAME")
    assert az_gndg.SCHEDULE_COLUMNS == root_gndg.SCHEDULE_COLUMNS
    assert az_gndg.TEAM_ID_TO_NAME == root_gndg.TEAM_ID_TO_NAME

    # 2. compute_season_multipliers parity
    az_csm = load_az_module("az_csm", "scripts/compute_season_multipliers.py")
    import scripts.compute_season_multipliers as root_csm

    assert hasattr(az_csm, "compute_season_multipliers")
    assert hasattr(az_csm, "apply_multiplier_curve")
    assert hasattr(az_csm, "calculate_percentiles")
    assert hasattr(az_csm, "determine_season_boundary")
    assert az_csm.DEFAULT_MIN_GAMES == root_csm.DEFAULT_MIN_GAMES
    assert az_csm.DEFAULT_CUTOFF_PERCENTILE == root_csm.DEFAULT_CUTOFF_PERCENTILE

    # 3. azure_blob schedule validation parity
    az_blob = load_az_module("az_blob", "scripts/azure_blob.py")
    import scripts.azure_blob as root_blob

    assert hasattr(az_blob, "validate_schedule_artifact")
    assert hasattr(az_blob, "SCHEDULE_REQUIRED_COLUMNS")
    assert az_blob.SCHEDULE_REQUIRED_COLUMNS == root_blob.SCHEDULE_REQUIRED_COLUMNS

    # 4. run_full_pipeline parity
    az_rfp = load_az_module("az_rfp", "run_full_pipeline.py")
    import run_full_pipeline as root_rfp

    assert hasattr(az_rfp, "run_full_pipeline")
    assert hasattr(az_rfp, "run_prediction_pipeline")
    assert inspect.signature(az_rfp.run_full_pipeline) == inspect.signature(root_rfp.run_full_pipeline)
    assert inspect.signature(az_rfp.run_prediction_pipeline) == inspect.signature(root_rfp.run_prediction_pipeline)


# ==============================================================================
# 2. Pipeline Execution Order Parity Tests
# ==============================================================================

def test_full_pipeline_step_order_parity():
    """Verify that azure_function and root run_full_pipeline invoke identical steps in identical order."""
    az_rfp = load_az_module("az_rfp_full", "run_full_pipeline.py")
    import run_full_pipeline as root_rfp

    root_steps = []
    az_steps = []

    def mock_root_run_step(name, func):
        root_steps.append(name)

    def mock_az_run_step(name, func):
        az_steps.append(name)

    with patch("run_full_pipeline.run_step", side_effect=mock_root_run_step):
        root_rfp.run_full_pipeline(skip_fetch=False, skip_train=False, export_json=False, publish_azure=False)

    with patch.object(az_rfp, "run_step", side_effect=mock_az_run_step):
        az_rfp.run_full_pipeline(skip_fetch=False, skip_train=False, export_json=False, publish_azure=False)

    expected_order = [
        "Build Play-by-Play Dataset",
        "Compute Player Ratings",
        "Add Game Dates",
        "Build Features",
        "Train Model",
        "Refresh Current Rosters",
        "Get Next Day Games",
        "Get Tomorrow Schedule",
        "Predict Ratings",
        "Apply Injury Status",
        "Compute Season Multipliers",
        "Build Frontend Dataset",
        "Build Team Rankings",
    ]

    assert root_steps == expected_order, f"Root steps mismatch: {root_steps}"
    assert az_steps == expected_order, f"Azure Function steps mismatch: {az_steps}"
    assert root_steps == az_steps, "Step execution sequence diverges between root and azure_function"


def test_prediction_pipeline_step_order_parity():
    """Verify that azure_function and root run_prediction_pipeline invoke identical steps in identical order."""
    az_rfp = load_az_module("az_rfp_pred", "run_full_pipeline.py")
    import run_full_pipeline as root_rfp

    root_steps = []
    az_steps = []

    def mock_root_run_step(name, func):
        root_steps.append(name)

    def mock_az_run_step(name, func):
        az_steps.append(name)

    with patch("run_full_pipeline.run_step", side_effect=mock_root_run_step):
        root_rfp.run_prediction_pipeline(export_json=False, publish_azure=False)

    with patch.object(az_rfp, "run_step", side_effect=mock_az_run_step):
        az_rfp.run_prediction_pipeline(export_json=False, publish_azure=False)

    expected_prediction_order = [
        "Refresh Current Rosters",
        "Get Next Day Games",
        "Get Tomorrow Schedule",
        "Predict Ratings",
        "Apply Injury Status",
        "Compute Season Multipliers",
        "Build Frontend Dataset",
        "Build Team Rankings",
    ]

    assert root_steps == expected_prediction_order, f"Root prediction steps mismatch: {root_steps}"
    assert az_steps == expected_prediction_order, f"Azure Function prediction steps mismatch: {az_steps}"
    assert root_steps == az_steps


# ==============================================================================
# 3. Local Functional Execution Tests (No Network / Cloud Calls)
# ==============================================================================

def test_azure_function_schedule_generator_local(tmp_path):
    """Verify azure_function.scripts.get_next_day_games.get_tomorrow_schedule creates valid artifacts."""
    az_gndg = load_az_module("az_gndg_exec", "scripts/get_next_day_games.py")
    az_blob = load_az_module("az_blob_exec", "scripts/azure_blob.py")

    out_parquet = str(tmp_path / "schedule_tomorrow.parquet")
    out_meta = str(tmp_path / "schedule_tomorrow_metadata.json")

    # Mock fetcher returning 1 game for tomorrow
    def mock_fetcher(date_str):
        return [
            {
                "gameId": "0022409999",
                "gameDate": date_str,
                "homeTeam": {"teamId": 1610612738, "teamName": "Celtics", "teamTricode": "BOS"},
                "awayTeam": {"teamId": 1610612752, "teamName": "Knicks", "teamTricode": "NYK"},
                "gameStatusText": "7:30 pm ET",
                "gameStatus": 1,
            }
        ]

    az_gndg.get_tomorrow_schedule(
        reference_time=None,
        schedule_fetcher=mock_fetcher,
        output_path=out_parquet,
        metadata_path=out_meta,
    )

    # Validate output using azure_function.scripts.azure_blob validator
    val_result = az_blob.validate_schedule_artifact(out_parquet, out_meta, enforce_tomorrow_date=True)
    assert val_result["metadata"]["status"] == "CONFIRMED"
    assert val_result["metadata"]["games_count"] == 1

    # Verify Parquet columns
    df = pd.read_parquet(out_parquet)
    assert list(df.columns) == az_gndg.SCHEDULE_COLUMNS
    assert len(df) == 1
    assert df["home_team_tricode"].iloc[0] == "BOS"
    assert df["away_team_tricode"].iloc[0] == "NYK"


def test_azure_function_season_multipliers_computation_local(tmp_path):
    """Verify azure_function.scripts.compute_season_multipliers computes bonuses on mock game ratings."""
    az_csm = load_az_module("az_csm_exec", "scripts/compute_season_multipliers.py")

    mock_ratings_df = pd.DataFrame([
        {"gameId": "G1", "personId": 101, "game_date": "2026-10-25", "rating": 3.0},
        {"gameId": "G2", "personId": 101, "game_date": "2026-10-27", "rating": 3.5},
        {"gameId": "G3", "personId": 101, "game_date": "2026-10-29", "rating": 4.0},
        {"gameId": "G4", "personId": 102, "game_date": "2026-10-25", "rating": 1.0},
        {"gameId": "G5", "personId": 102, "game_date": "2026-10-27", "rating": 1.2},
        # Person 102 only has 2 games -> ineligible
    ])

    result_df = az_csm.compute_season_multipliers(
        ratings_df=mock_ratings_df,
        min_games_threshold=3,
        curve="candidate_d",
        cutoff_percentile=95.0,
        reference_date="2026-11-01",
    )

    assert not result_df.empty
    row_101 = result_df[result_df["personId"] == 101].iloc[0]
    row_102 = result_df[result_df["personId"] == 102].iloc[0]

    assert row_101["season_games_played"] == 3
    assert row_101["season_multiplier_eligible"] is True or row_101["season_multiplier_eligible"] == 1
    assert row_101["season_avg_rating"] == pytest.approx(3.5, rel=1e-3)

    assert row_102["season_games_played"] == 2
    assert not bool(row_102["season_multiplier_eligible"])
    assert row_102["season_performance_bonus"] == 0.0

"""Artifact loader, schema validation, and serving state management."""

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
import numpy as np
import pandas as pd

from api.config import Settings

logger = logging.getLogger("propredict.api")

REQUIRED_COLUMNS = [
    "gameId",
    "teamId",
    "personId",
    "playerName",
    "rating",
    "last_game_id",
    "last_game_date",
    "game_date",
    "last_game",
    "last3_avg",
    "last5_avg",
    "last7_avg",
    "predicted_rating",
    "status",
    "injury_reason",
    "injury_date",
    "injury_source",
    "status_rank",
    "teamName",
    "opponentId",
    "opponentName",
    "headshot",
]

CANONICAL_PLAYER_FIELDS = [
    "personId",
    "playerName",
    "teamId",
    "teamName",
    "opponentName",
    "predicted_rating",
    "status",
    "status_rank",
    "headshot",
    "last_game",
    "last3_avg",
    "last7_avg",
]


class APIException(HTTPException):
    """Custom HTTP exception with machine-readable error code."""

    def __init__(self, status_code: int, detail: str, code: str):
        super().__init__(status_code=status_code, detail=detail)
        self.code = code


@dataclass
class ServingState:
    home_players: Optional[List[Dict[str, Any]]] = None
    all_players: Optional[List[Dict[str, Any]]] = None
    generated_at: Optional[str] = None
    game_date: Optional[str] = None
    artifacts_loaded: bool = False
    load_error: Optional[str] = None
    load_error_code: Optional[str] = None
    last_mtime: Optional[float] = None


def validate_schema(df: pd.DataFrame, artifact_name: str) -> None:
    """Validate that the required frontend columns exist in the DataFrame."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        error_msg = f"{artifact_name} missing required columns: {', '.join(missing)}"
        logger.error("Schema validation mismatch: %s", error_msg)
        raise ValueError(error_msg)


def clean_record_value(val: Any) -> Any:
    """Convert pandas/numpy values to JSON-safe Python primitives.

    Ensures NaN / NA becomes None (JSON null) and numpy types become standard Python types.
    """
    if pd.isna(val):
        return None
    if isinstance(val, (np.integer, int)):
        return int(val)
    if isinstance(val, (np.floating, float)):
        return float(val)
    if isinstance(val, (np.bool_, bool)):
        return bool(val)
    return str(val) if not isinstance(val, (str, int, float, bool)) else val


def dataframe_to_canonical_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert a DataFrame to canonical player records with clean JSON-safe values."""
    if df.empty:
        return []
    subset = df[CANONICAL_PLAYER_FIELDS]
    records = []
    for row in subset.to_dict(orient="records"):
        clean_row = {k: clean_record_value(v) for k, v in row.items()}
        records.append(clean_row)
    return records


def load_serving_state(settings: Settings) -> ServingState:
    """Load parquet artifacts from disk into in-memory serving state.

    Never exposes full filesystem paths in user-facing error messages.
    """
    state = ServingState()

    # 1. Verify and read frontend_all.parquet
    all_path = settings.frontend_all_path
    if not all_path.exists():
        logger.error("Missing artifact: frontend_all.parquet does not exist")
        state.load_error = "frontend_all.parquet is missing"
        state.load_error_code = "ARTIFACT_MISSING"
        return state

    try:
        df_all = pd.read_parquet(all_path)
    except Exception as e:
        logger.error("Unreadable artifact frontend_all.parquet: %s", e)
        state.load_error = "frontend_all.parquet is unreadable"
        state.load_error_code = "ARTIFACT_UNREADABLE"
        return state

    try:
        validate_schema(df_all, "frontend_all.parquet")
    except ValueError:
        state.load_error = "frontend_all.parquet schema validation failed"
        state.load_error_code = "ARTIFACT_UNREADABLE"
        return state

    if len(df_all) == 0:
        logger.error("Invalid artifact: frontend_all.parquet contains 0 rows")
        state.load_error = "frontend_all.parquet contains zero records"
        state.load_error_code = "ARTIFACT_UNREADABLE"
        return state

    # 2. Verify and read frontend_home.parquet
    home_path = settings.frontend_home_path
    if not home_path.exists():
        logger.error("Missing artifact: frontend_home.parquet does not exist")
        state.load_error = "frontend_home.parquet is missing"
        state.load_error_code = "ARTIFACT_MISSING"
        return state

    try:
        df_home = pd.read_parquet(home_path)
    except Exception as e:
        logger.error("Unreadable artifact frontend_home.parquet: %s", e)
        state.load_error = "frontend_home.parquet is unreadable"
        state.load_error_code = "ARTIFACT_UNREADABLE"
        return state

    try:
        validate_schema(df_home, "frontend_home.parquet")
    except ValueError:
        state.load_error = "frontend_home.parquet schema validation failed"
        state.load_error_code = "ARTIFACT_UNREADABLE"
        return state

    # 3. Derive generated_at timestamp from frontend_all.parquet modification time
    try:
        mtime = os.path.getmtime(all_path)
        dt = datetime.fromtimestamp(mtime)
        state.generated_at = dt.replace(microsecond=0).isoformat()
        state.last_mtime = mtime
    except Exception as e:
        logger.error("Failed to read timestamp for frontend_all.parquet: %s", e)
        state.load_error = "Failed to determine artifact timestamp"
        state.load_error_code = "ARTIFACT_UNREADABLE"
        return state

    # 4. Canonicalize records for all_players
    state.all_players = dataframe_to_canonical_records(df_all)

    # 5. Canonicalize records for home_players and derive game_date
    if len(df_home) == 0:
        state.home_players = []
        state.game_date = None
    else:
        state.home_players = dataframe_to_canonical_records(df_home)
        valid_dates = df_home["game_date"].dropna()
        if len(valid_dates) > 0:
            state.game_date = str(valid_dates.iloc[0])
        else:
            state.game_date = None

    state.artifacts_loaded = True
    logger.info("Loaded frontend_home.parquet: %d players", len(state.home_players))
    logger.info("Loaded frontend_all.parquet: %d players", len(state.all_players))
    logger.info("Serving data generated at: %s", state.generated_at)
    return state


def reload_serving_state_if_stale(app: Any) -> bool:
    """Check if on-disk artifacts have been updated and reload them into app.state.

    Returns True if state was reloaded, False otherwise.
    Never raises exceptions; preserves existing valid state if reload fails.
    """
    settings: Settings = getattr(app.state, "settings", None)
    if settings is None:
        return False

    all_path = settings.frontend_all_path
    if not all_path.exists():
        return False

    try:
        current_mtime = os.path.getmtime(all_path)
    except OSError:
        return False

    last_mtime = getattr(app.state, "last_mtime", None)
    if last_mtime is not None and current_mtime <= last_mtime:
        return False

    logger.info("Newer serving artifact detected on disk (mtime %s > %s). Reloading...", current_mtime, last_mtime)
    new_state = load_serving_state(settings)
    if new_state.artifacts_loaded:
        app.state.home_players = new_state.home_players
        app.state.all_players = new_state.all_players
        app.state.generated_at = new_state.generated_at
        app.state.game_date = new_state.game_date
        app.state.artifacts_loaded = True
        app.state.load_error = None
        app.state.load_error_code = None
        app.state.last_mtime = new_state.last_mtime
        logger.info("Serving state refreshed successfully: %s", new_state.generated_at)
        return True
    else:
        logger.warning("Artifact reload failed; retaining previous serving state: %s", new_state.load_error)
        return False


def get_pipeline_diagnostics(settings: Settings) -> Dict[str, Any]:
    """Read the latest pipeline status and run history to evaluate freshness and operational health."""
    status_path = settings.pipeline_status_path
    runs_path = settings.pipeline_runs_path

    last_status = None
    if status_path.exists():
        try:
            with open(status_path, "r", encoding="utf-8") as f:
                last_status = json.load(f)
        except Exception as e:
            logger.warning("Could not read pipeline status file: %s", e)

    # Calculate staleness: if generated_at or last successful run is older than 36h
    is_stale = False
    last_success_time = None
    if last_status and last_status.get("status") == "SUCCESS":
        end_time_str = last_status.get("end_time") or last_status.get("start_time")
        if end_time_str:
            try:
                dt = datetime.fromisoformat(end_time_str)
                last_success_time = dt.isoformat()
                if (datetime.now() - dt).total_seconds() > 36 * 3600:
                    is_stale = True
            except Exception:
                pass

    recent_runs = []
    if runs_path.exists():
        try:
            with open(runs_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines[-5:]:
                line = line.strip()
                if line:
                    try:
                        recent_runs.append(json.loads(line))
                    except Exception:
                        pass
        except Exception as e:
            logger.warning("Could not read pipeline runs log: %s", e)

    return {
        "last_status": last_status.get("status") if last_status else "UNKNOWN",
        "last_start_time": last_status.get("start_time") if last_status else None,
        "last_end_time": last_status.get("end_time") if last_status else None,
        "last_duration_seconds": last_status.get("duration_seconds") if last_status else None,
        "last_error": last_status.get("error") if last_status else None,
        "last_success_time": last_success_time,
        "is_stale": is_stale,
        "recent_runs_count": len(recent_runs),
    }

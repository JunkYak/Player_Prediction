"""Artifact loader, schema validation, and serving state management."""

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
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
    "season_game_count",
    "season_average_rating",
    "season_percentile",
    "season_performance_bonus",
    "adjusted_predicted_rating",
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
    "season_game_count",
    "season_games_played",
    "season_average_rating",
    "season_avg_rating",
    "season_percentile",
    "season_performance_bonus",
    "season_multiplier_eligible",
    "ranking_score",
    "adjusted_predicted_rating",
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
    tomorrow_schedule: Optional[Dict[str, Any]] = None
    generated_at: Optional[str] = None
    game_date: Optional[str] = None
    artifacts_loaded: bool = False
    load_error: Optional[str] = None
    load_error_code: Optional[str] = None
    last_mtime: Optional[float] = None
    current_snapshot_id: Optional[str] = None
    last_blob_check: Optional[float] = None


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
    df_copy = df.copy()

    # Synchronize synonymous fields if one is present
    if "season_game_count" in df_copy.columns and "season_games_played" not in df_copy.columns:
        df_copy["season_games_played"] = df_copy["season_game_count"]
    elif "season_games_played" in df_copy.columns and "season_game_count" not in df_copy.columns:
        df_copy["season_game_count"] = df_copy["season_games_played"]

    if "season_average_rating" in df_copy.columns and "season_avg_rating" not in df_copy.columns:
        df_copy["season_avg_rating"] = df_copy["season_average_rating"]
    elif "season_avg_rating" in df_copy.columns and "season_average_rating" not in df_copy.columns:
        df_copy["season_average_rating"] = df_copy["season_avg_rating"]

    for col in CANONICAL_PLAYER_FIELDS:
        if col not in df_copy.columns:
            if col == "season_performance_bonus":
                df_copy[col] = 0.0
            elif col == "ranking_score":
                if "adjusted_predicted_rating" in df_copy.columns:
                    df_copy[col] = df_copy["adjusted_predicted_rating"]
                else:
                    pred_series = df_copy.get("predicted_rating", 0.0)
                    bonus_series = df_copy.get("season_performance_bonus", 0.0)
                    df_copy[col] = (pred_series * (1.0 + bonus_series)).round(2)
            elif col == "adjusted_predicted_rating":
                df_copy[col] = df_copy.get("ranking_score", df_copy.get("predicted_rating", 0.0))
            elif col in ("season_game_count", "season_games_played", "season_percentile", "season_average_rating", "season_avg_rating"):
                df_copy[col] = 0.0
            elif col == "season_multiplier_eligible":
                df_copy[col] = False
            else:
                df_copy[col] = None
    subset = df_copy[CANONICAL_PLAYER_FIELDS]
    records = []
    for row in subset.to_dict(orient="records"):
        clean_row = {k: clean_record_value(v) for k, v in row.items()}
        records.append(clean_row)
    return records


def load_tomorrow_schedule(base_dir: Path) -> Optional[Dict[str, Any]]:
    """Load and format tomorrow's schedule artifact from base_dir.

    Fails closed (returns None) if:
    - schedule_tomorrow.parquet is missing or unreadable
    - schedule_tomorrow_metadata.json is missing or unreadable
    - schedule status is FETCH_FAILED or UNAVAILABLE
    - schedule date is missing, malformed, or future-dated (> tomorrow_et_date)
    - row count and metadata games_count mismatch
    - empty schedule without explicit OFF_DAY status
    """
    parquet_path = base_dir / "schedule_tomorrow.parquet"
    meta_path = base_dir / "schedule_tomorrow_metadata.json"

    if not parquet_path.exists() or not meta_path.exists():
        return None

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if not isinstance(meta, dict):
            logger.warning("schedule_tomorrow_metadata.json is not a valid JSON dict")
            return None
    except Exception as e:
        logger.warning("Failed reading schedule_tomorrow_metadata.json: %s", e)
        return None

    meta_status = meta.get("status")
    if meta_status in ("FETCH_FAILED", "UNAVAILABLE"):
        logger.warning("Schedule metadata status is %s; artifact marked unavailable", meta_status)
        return None

    schedule_date_raw = meta.get("schedule_date")
    if not schedule_date_raw or not isinstance(schedule_date_raw, str):
        logger.warning("Missing or non-string schedule_date in metadata")
        return None

    try:
        schedule_date = datetime.strptime(schedule_date_raw[:10], "%Y-%m-%d").date().strftime("%Y-%m-%d")
    except ValueError:
        logger.warning("Malformed schedule_date in metadata: %s", schedule_date_raw)
        return None

    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        logger.warning("Failed to read schedule_tomorrow.parquet: %s", e)
        return None

    now_et = datetime.now(ZoneInfo("America/New_York"))
    tomorrow_et_date = (now_et.date() + timedelta(days=1)).strftime("%Y-%m-%d")

    # Future-dated artifacts are invalid/unavailable
    if schedule_date > tomorrow_et_date:
        logger.warning("Future-dated schedule detected (%s > %s); unavailable", schedule_date, tomorrow_et_date)
        return None

    is_stale = meta.get("is_stale", False) or (schedule_date < tomorrow_et_date)
    row_count = len(df)
    meta_games_count = meta.get("games_count")

    if is_stale:
        status = "STALE"
    else:
        # Enforce consistency between Parquet row count and metadata
        if meta_games_count is None or meta_games_count != row_count:
            logger.warning(
                "Schedule row count mismatch: parquet has %d rows, metadata reports %s",
                row_count,
                meta_games_count,
            )
            return None

        if row_count == 0:
            if meta_status != "OFF_DAY" or meta.get("games_scheduled") is not False:
                logger.warning("Empty schedule rejected: metadata does not establish OFF_DAY")
                return None
            status = "OFF_DAY"
        else:
            if meta_status != "CONFIRMED" or meta.get("games_scheduled") is not True:
                logger.warning("Schedule with games rejected: metadata does not establish CONFIRMED")
                return None
            status = "CONFIRMED"

    matchups = []
    team_ids = set()
    for _, row in df.iterrows():
        try:
            home_id = int(row.get("home_team_id", 0))
        except (ValueError, TypeError):
            home_id = 0
        try:
            away_id = int(row.get("away_team_id", 0))
        except (ValueError, TypeError):
            away_id = 0
        if home_id:
            team_ids.add(home_id)
        if away_id:
            team_ids.add(away_id)
        matchups.append({
            "game_id": str(row.get("game_id", "")),
            "home_team_id": home_id,
            "home_team_name": row.get("home_team_name"),
            "home_team_tricode": row.get("home_team_tricode"),
            "away_team_id": away_id,
            "away_team_name": row.get("away_team_name"),
            "away_team_tricode": row.get("away_team_tricode"),
            "game_time_utc": row.get("game_time_utc"),
            "status": row.get("status", "Scheduled"),
            "is_postponed": bool(row.get("is_postponed", False)),
        })

    generated_at = meta.get("generated_at")
    if not generated_at:
        try:
            mtime = os.path.getmtime(parquet_path)
            generated_at = datetime.fromtimestamp(mtime, timezone.utc).isoformat()
        except Exception:
            generated_at = datetime.now(timezone.utc).isoformat()

    return {
        "schedule_date": schedule_date,
        "generated_at": generated_at,
        "status": status,
        "games_scheduled": len(matchups) > 0 and status == "CONFIRMED",
        "games_count": len(matchups),
        "matchups": matchups,
        "team_ids": sorted(list(team_ids)),
    }


def load_snapshot_state(
    all_path: Path,
    home_path: Path,
    snapshot_id: Optional[str] = None,
) -> ServingState:
    """Load and validate both serving parquet artifacts together from a single snapshot.

    Guarantees that both artifacts are validated together before constructing ServingState.
    Never exposes full filesystem paths in user-facing error messages.
    """
    state = ServingState(current_snapshot_id=snapshot_id)

    # 1. Verify and read frontend_all.parquet
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

    # 4. Canonicalize records for both files together
    state.all_players = dataframe_to_canonical_records(df_all)

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

    state.tomorrow_schedule = load_tomorrow_schedule(all_path.parent)

    state.artifacts_loaded = True
    logger.info(
        "Serving state loaded successfully (%s): %d all, %d slate",
        snapshot_id or "local",
        len(state.all_players),
        len(state.home_players),
    )
    return state


def load_serving_state(settings: Settings) -> ServingState:
    """Load parquet artifacts from disk into in-memory serving state.

    If azure_storage_account is configured, attempts to synchronize the latest
    serving snapshot from propredict-serving first.
    Never exposes full filesystem paths in user-facing error messages.
    """
    snapshot_dir = None
    snap_id = None

    # 0. Remote Azure Blob synchronization (if configured)
    if settings.azure_storage_account:
        try:
            from scripts.azure_blob import download_serving_snapshot
            downloaded, snap_id, snapshot_dir = download_serving_snapshot(
                account_name=settings.azure_storage_account,
                container_name=settings.azure_serving_container,
                target_dir=str(settings.data_dir),
                known_snapshot_id=None,
            )
        except Exception as e:
            logger.error("Failed to sync initial serving snapshot from Azure Blob: %s", e)
            if not settings.frontend_all_path.exists():
                return ServingState(
                    artifacts_loaded=False,
                    load_error="Serving snapshot could not be loaded from Azure Blob Storage",
                    load_error_code="BLOB_SNAPSHOT_UNAVAILABLE",
                )

    # If snapshot directory exists and has files, load from it; otherwise load from data_dir
    if snapshot_dir and (snapshot_dir / "frontend_all.parquet").exists():
        state = load_snapshot_state(
            snapshot_dir / "frontend_all.parquet",
            snapshot_dir / "frontend_home.parquet",
            snapshot_id=snap_id,
        )
    else:
        state = load_snapshot_state(
            settings.frontend_all_path,
            settings.frontend_home_path,
            snapshot_id=snap_id,
        )

    if settings.azure_storage_account and state.artifacts_loaded:
        state.last_blob_check = time.time()

    return state


def reload_serving_state_if_stale(app: Any) -> bool:
    """Check if on-disk artifacts or remote Blob snapshots have been updated.

    If azure_storage_account is configured, throttles remote manifest checks
    using azure_blob_check_interval.
    Atomically updates app.state when a verified snapshot is available.
    Returns True if state was reloaded, False otherwise.
    Never raises exceptions; preserves existing valid state if reload fails.
    """
    settings: Settings = getattr(app.state, "settings", None)
    if settings is None:
        return False

    # 1. Periodic Azure Blob snapshot check (if configured)
    if settings.azure_storage_account:
        now = time.time()
        last_check = getattr(app.state, "last_blob_check", 0) or 0
        if now - last_check >= settings.azure_blob_check_interval:
            app.state.last_blob_check = now
            try:
                from scripts.azure_blob import download_serving_snapshot
                curr_snap = getattr(app.state, "current_snapshot_id", None)
                updated, new_snap, snap_dir = download_serving_snapshot(
                    account_name=settings.azure_storage_account,
                    container_name=settings.azure_serving_container,
                    target_dir=str(settings.data_dir),
                    known_snapshot_id=curr_snap,
                )
                if updated and snap_dir:
                    new_state = load_snapshot_state(
                        snap_dir / "frontend_all.parquet",
                        snap_dir / "frontend_home.parquet",
                        snapshot_id=new_snap,
                    )
                    if new_state.artifacts_loaded:
                        # Atomic in-memory state swap
                        app.state.serving_state = new_state
                        app.state.home_players = new_state.home_players
                        app.state.all_players = new_state.all_players
                        app.state.tomorrow_schedule = new_state.tomorrow_schedule
                        app.state.generated_at = new_state.generated_at
                        app.state.game_date = new_state.game_date
                        app.state.current_snapshot_id = new_state.current_snapshot_id
                        app.state.artifacts_loaded = True
                        app.state.load_error = None
                        app.state.load_error_code = None
                        app.state.last_mtime = new_state.last_mtime
                        logger.info("Serving state atomically refreshed from snapshot %s: %s", new_snap, new_state.generated_at)
                        return True
                    else:
                        logger.warning("Downloaded snapshot %s failed validation; retaining prior serving state", new_snap)
                        return False
            except Exception as e:
                logger.warning("Periodic Azure Blob snapshot check failed: %s; retaining in-memory state", e)
                return False

    # 2. Local filesystem mtime check (for offline/local dev or non-Azure runs)
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
    new_state = load_snapshot_state(
        settings.frontend_all_path,
        settings.frontend_home_path,
        snapshot_id=getattr(app.state, "current_snapshot_id", None),
    )
    if new_state.artifacts_loaded:
        app.state.serving_state = new_state
        app.state.home_players = new_state.home_players
        app.state.all_players = new_state.all_players
        app.state.tomorrow_schedule = new_state.tomorrow_schedule
        app.state.generated_at = new_state.generated_at
        app.state.game_date = new_state.game_date
        app.state.artifacts_loaded = True
        app.state.load_error = None
        app.state.load_error_code = None
        app.state.last_mtime = new_state.last_mtime
        if new_state.current_snapshot_id:
            app.state.current_snapshot_id = new_state.current_snapshot_id
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

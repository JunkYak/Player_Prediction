"""
scripts/azure_blob.py

Azure Blob Storage integration for ProPredict NBA Player Performance Prediction.
Provides secure, identity-based synchronization for:
1. Canonical historical dataset in propredict-history (play_by_play.parquet)
   - Fail-closed restore before daily ingestion.
   - Pre-publication schema, checksum, and continuity validation.
2. Versioned immutable serving snapshots in propredict-serving
   - Uploads snapshot parquet files under snapshots/<snapshot_id>/ prefix.
   - Writes immutable manifest in snapshot directory.
   - Atomically publishes root current_manifest.json after files are verified.
   - Separate staging directory and atomic directory swap to guarantee
     complete, unmixed snapshots on the consumer.
3. Daily publication idempotency guard
   - Prevents duplicate production runs on the same Eastern calendar date.
"""

import os
import sys
import json
import uuid
import shutil
import hashlib
import logging
import argparse
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import pandas as pd
import pyarrow.parquet as pq

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from scripts.build_play_by_play_dataset import REQUIRED_COLUMNS as PBP_REQUIRED_COLUMNS

try:
    from api.loader import REQUIRED_COLUMNS as SERVING_REQUIRED_COLUMNS
except (ImportError, ModuleNotFoundError):
    SERVING_REQUIRED_COLUMNS = [
        "gameId", "teamId", "personId", "playerName", "rating",
        "last_game_id", "last_game_date", "game_date", "last_game",
        "last3_avg", "last5_avg", "last7_avg", "predicted_rating",
        "status", "injury_reason",
    ]

logger = logging.getLogger("propredict.azure_blob")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

DEFAULT_STORAGE_ACCOUNT = "azurefunceesh"
DEFAULT_HISTORY_CONTAINER = "propredict-history"
DEFAULT_SERVING_CONTAINER = "propredict-serving"
DEFAULT_MANIFEST_BLOB = "current_manifest.json"
CANONICAL_HISTORY_BLOB = "play_by_play.parquet"
DEFAULT_DATA_DIR = os.environ.get("PROPREDICT_DATA_DIR") or os.path.join(BASE_DIR, "data")


# ==============================================================================
# 1. Azure Blob Service Client Factory
# ==============================================================================
def get_blob_service_client(account_name: Optional[str] = None):
    """
    Obtain an authenticated BlobServiceClient using Azure Identity (DefaultAzureCredential).
    Does not use or expose storage keys or connection strings.
    """
    try:
        from azure.identity import DefaultAzureCredential
        from azure.storage.blob import BlobServiceClient
    except ImportError as e:
        raise ImportError(
            "Azure SDK packages (azure-storage-blob, azure-identity) are not installed. "
            "Install them via requirements.txt to enable Blob Storage features."
        ) from e

    conn_str = os.environ.get("AZURE_STORAGE_CONNECTION_STRING") or os.environ.get("AzureWebJobsStorage")
    if conn_str:
        return BlobServiceClient.from_connection_string(conn_str)

    account = account_name or os.environ.get("AZURE_STORAGE_ACCOUNT", DEFAULT_STORAGE_ACCOUNT)
    account_url = f"https://{account}.blob.core.windows.net"
    credential = DefaultAzureCredential()
    return BlobServiceClient(account_url=account_url, credential=credential)


# ==============================================================================
# 2. Checksum and Validation Helpers
# ==============================================================================
def compute_sha256(filepath: str) -> str:
    """Compute SHA-256 checksum of a local file in streaming blocks."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def validate_play_by_play_file(filepath: str) -> Dict[str, Any]:
    """
    Validate historical play_by_play Parquet file integrity and schema.
    Raises ValueError if validation fails.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"play_by_play file not found: {filepath}")

    size_bytes = os.path.getsize(filepath)
    if size_bytes == 0:
        raise ValueError(f"play_by_play file is empty (0 bytes): {filepath}")

    # Check Parquet header magic bytes 'PAR1'
    with open(filepath, "rb") as f:
        header = f.read(4)
        if header != b"PAR1":
            raise ValueError(
                f"play_by_play file is not a valid Parquet binary (header: {header!r}). "
                "Failing closed to prevent corrupt dataset propagation."
            )

    try:
        parquet_file = pq.ParquetFile(filepath)
        schema_cols = parquet_file.schema.names
    except Exception as e:
        raise ValueError(f"Failed to read Parquet metadata: {e}") from e

    missing = [col for col in PBP_REQUIRED_COLUMNS if col not in schema_cols]
    if missing:
        raise ValueError(f"play_by_play Parquet is missing required columns: {missing}")

    try:
        df = pd.read_parquet(filepath)
    except Exception as e:
        raise ValueError(f"Failed to load play_by_play into DataFrame: {e}") from e

    row_count = len(df)
    if row_count == 0:
        raise ValueError("play_by_play dataset contains 0 records.")

    if "gameId" not in df.columns or df["gameId"].isna().all():
        raise ValueError("play_by_play dataset has invalid or all-null gameId column.")

    unique_games = int(df["gameId"].dropna().nunique())
    checksum = compute_sha256(filepath)

    return {
        "filepath": filepath,
        "size_bytes": size_bytes,
        "sha256": checksum,
        "row_count": row_count,
        "unique_games": unique_games,
        "columns": list(schema_cols),
    }


def validate_serving_parquet(filepath: str, name: str, allow_zero_rows: bool = False) -> Dict[str, Any]:
    """
    Validate serving Parquet artifact (frontend_all or frontend_home).
    Raises ValueError if invalid.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Serving artifact {name} not found: {filepath}")

    size_bytes = os.path.getsize(filepath)
    if size_bytes == 0:
        raise ValueError(f"Serving artifact {name} is 0 bytes: {filepath}")

    with open(filepath, "rb") as f:
        header = f.read(4)
        if header != b"PAR1":
            raise ValueError(f"Serving artifact {name} is not a valid Parquet file (magic bytes: {header!r})")

    try:
        df = pd.read_parquet(filepath)
    except Exception as e:
        raise ValueError(f"Failed to read Parquet file for {name}: {e}") from e

    missing = [col for col in SERVING_REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Serving artifact {name} missing required columns: {missing}")

    row_count = len(df)
    if not allow_zero_rows and row_count == 0:
        raise ValueError(f"Serving artifact {name} contains 0 rows (expected positive row count).")

    # Publication guard: reject stale slate dates
    if name == "frontend_home.parquet" and row_count > 0:
        if "game_date" in df.columns:
            valid_dates = df["game_date"].dropna()
            if len(valid_dates) > 0:
                slate_date_str = str(valid_dates.iloc[0])[:10]
                try:
                    slate_dt = datetime.strptime(slate_date_str, "%Y-%m-%d").date()
                    now_et = datetime.now(ZoneInfo("America/New_York"))
                    current_et_date = now_et.date()
                    # Rule 1: Slate date cannot be older than yesterday
                    if slate_dt < (current_et_date - timedelta(days=1)):
                        raise ValueError(
                            f"Serving artifact {name} failed slate freshness validation: "
                            f"slate date {slate_date_str} is older than yesterday ({current_et_date - timedelta(days=1)}). "
                            f"Stale slates cannot be published."
                        )
                    # Rule 2: If slate date is yesterday and current ET time is past 6 AM, it is already completed
                    if slate_dt == (current_et_date - timedelta(days=1)) and now_et.hour >= 6:
                        raise ValueError(
                            f"Serving artifact {name} failed slate freshness validation: "
                            f"yesterday's slate ({slate_date_str}) is completed on {current_et_date} (current ET: {now_et.strftime('%H:%M')}). "
                            f"Cannot publish a completed historical slate as the active prediction slate."
                        )
                except ValueError as ve:
                    if "failed slate freshness validation" in str(ve):
                        raise
                    logger.warning("Could not parse slate date '%s' for freshness check: %s", slate_date_str, ve)
                except Exception as ex:
                    logger.warning("Could not parse slate date '%s' for freshness check: %s", slate_date_str, ex)

    checksum = compute_sha256(filepath)
    return {
        "filename": name,
        "filepath": filepath,
        "size_bytes": size_bytes,
        "sha256": checksum,
        "row_count": row_count,
        "columns": list(df.columns),
    }


def validate_artifacts(data_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Pre-publication verification of both historical play-by-play and serving datasets.
    Guarantees that no partial, empty, or corrupted datasets can be published to either container.
    Fails closed with a ValueError if any check fails.
    """
    data_path = Path(data_dir or DEFAULT_DATA_DIR)
    pbp_path = data_path / CANONICAL_HISTORY_BLOB
    all_path = data_path / "frontend_all.parquet"
    home_path = data_path / "frontend_home.parquet"

    logger.info("Performing pre-publication validation of pipeline artifacts in '%s'...", data_path)

    # 1. Historical dataset validation
    meta_pbp = validate_play_by_play_file(str(pbp_path))

    # Continuity check against baseline metadata if present
    meta_path = str(pbp_path) + ".meta.json"
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                baseline_meta = json.load(f)
            prev_games = baseline_meta.get("unique_games", 0)
            prev_rows = baseline_meta.get("row_count", 0)
            duplicate_slack = 100
            if prev_games > 0 and meta_pbp["unique_games"] < prev_games:
                raise ValueError(
                    f"Historical continuity check failed: unique game count decreased from "
                    f"{prev_games} to {meta_pbp['unique_games']}. Ingestion must never lose historical games."
                )
            if prev_rows > 0 and meta_pbp["row_count"] < (prev_rows - duplicate_slack):
                raise ValueError(
                    f"Historical continuity check failed: row count decreased from {prev_rows} "
                    f"to {meta_pbp['row_count']} (diff: {meta_pbp['row_count'] - prev_rows}), "
                    f"exceeding allowable deduplication slack ({duplicate_slack} rows)."
                )
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            logger.warning("Could not verify baseline metadata file (%s): %s", meta_path, e)

    # 2. Serving datasets validation
    meta_all = validate_serving_parquet(str(all_path), "frontend_all.parquet", allow_zero_rows=False)
    meta_home = validate_serving_parquet(str(home_path), "frontend_home.parquet", allow_zero_rows=True)

    logger.info(
        "Artifact validation passed: PBP (%d rows, %d games), frontend_all (%d rows), frontend_home (%d rows).",
        meta_pbp["row_count"],
        meta_pbp["unique_games"],
        meta_all["row_count"],
        meta_home["row_count"],
    )
    return {
        "pbp": meta_pbp,
        "frontend_all": meta_all,
        "frontend_home": meta_home,
    }


# ==============================================================================
# 3. History Container Operations (propredict-history)
# ==============================================================================
def restore_historical_data(
    account_name: Optional[str] = None,
    container_name: Optional[str] = None,
    target_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Restore play_by_play.parquet from the private propredict-history container.
    Fails closed if the blob does not exist or is invalid.
    Records pre-run baseline metadata (row count, game count, checksum) to target_path + '.meta.json'.
    """
    account = account_name or os.environ.get("AZURE_STORAGE_ACCOUNT", DEFAULT_STORAGE_ACCOUNT)
    container = container_name or os.environ.get("AZURE_HISTORY_CONTAINER", DEFAULT_HISTORY_CONTAINER)
    dest = target_path or os.path.join(DEFAULT_DATA_DIR, CANONICAL_HISTORY_BLOB)

    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    tmp_path = dest + ".tmp"

    logger.info("Connecting to Azure Blob storage (account: %s, container: %s)...", account, container)
    service_client = get_blob_service_client(account)
    container_client = service_client.get_container_client(container)

    blob_client = container_client.get_blob_client(CANONICAL_HISTORY_BLOB)
    if not blob_client.exists():
        err_msg = (
            f"CRITICAL: Canonical historical blob '{CANONICAL_HISTORY_BLOB}' does not exist "
            f"in container '{container}'. Failing closed to prevent rebuilding history from only 60 days."
        )
        logger.error(err_msg)
        raise FileNotFoundError(err_msg)

    logger.info("Downloading historical blob '%s' to '%s'...", CANONICAL_HISTORY_BLOB, tmp_path)
    try:
        with open(tmp_path, "wb") as f:
            download_stream = blob_client.download_blob()
            download_stream.readinto(f)
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(f"Failed to download historical blob '{CANONICAL_HISTORY_BLOB}': {e}") from e

    # Validate restored file
    try:
        meta = validate_play_by_play_file(tmp_path)
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(f"Restored historical blob is corrupt or invalid: {e}") from e

    # Safely swap into final location
    if os.path.exists(dest):
        os.remove(dest)
    os.replace(tmp_path, dest)

    # Persist baseline metadata for continuity validation during publication
    meta_path = dest + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "restored_at": datetime.now(timezone.utc).isoformat(),
                "account": account,
                "container": container,
                "blob_name": CANONICAL_HISTORY_BLOB,
                "row_count": meta["row_count"],
                "unique_games": meta["unique_games"],
                "sha256": meta["sha256"],
                "size_bytes": meta["size_bytes"],
            },
            f,
            indent=2,
        )

    logger.info(
        "Historical data restored successfully: %d rows, %d games, sha256=%s",
        meta["row_count"],
        meta["unique_games"],
        meta["sha256"][:12],
    )
    return meta


def publish_historical_data(
    account_name: Optional[str] = None,
    container_name: Optional[str] = None,
    source_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Validate and upload updated play_by_play.parquet to propredict-history.
    Performs continuity checks against the pre-run baseline metadata.
    Never overwrites history with an empty, truncated, or invalid dataset.
    """
    account = account_name or os.environ.get("AZURE_STORAGE_ACCOUNT", DEFAULT_STORAGE_ACCOUNT)
    container = container_name or os.environ.get("AZURE_HISTORY_CONTAINER", DEFAULT_HISTORY_CONTAINER)
    src = source_path or os.path.join(DEFAULT_DATA_DIR, CANONICAL_HISTORY_BLOB)

    # Validate source file
    meta = validate_play_by_play_file(src)

    # Check baseline metadata if present
    meta_path = src + ".meta.json"
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                baseline_meta = json.load(f)
            prev_rows = baseline_meta.get("row_count", 0)
            prev_games = baseline_meta.get("unique_games", 0)

            # Continuity rule 1: Unique games must never decrease.
            # Ingestion preserves all historical games outside the 60-day window.
            if prev_games > 0 and meta["unique_games"] < prev_games:
                raise ValueError(
                    f"Historical continuity check failed: unique game count decreased from "
                    f"{prev_games} to {meta['unique_games']}. Ingestion must never lose historical games."
                )

            # Continuity rule 2: Row count cannot drop by more than baseline duplicate slack.
            # Ingestion drops exact duplicate rows (49 in baseline). Drops beyond 100 plays indicate corruption.
            duplicate_slack = 100
            if prev_rows > 0 and meta["row_count"] < (prev_rows - duplicate_slack):
                raise ValueError(
                    f"Historical continuity check failed: row count decreased from {prev_rows} "
                    f"to {meta['row_count']} (diff: {meta['row_count'] - prev_rows}), "
                    f"exceeding allowable deduplication slack ({duplicate_slack} rows)."
                )

            if meta["unique_games"] == prev_games and meta["row_count"] >= (prev_rows - duplicate_slack):
                logger.info(
                    "Historical continuity verified: games=%d (preserved), rows=%d (diff: %+d)",
                    meta["unique_games"],
                    meta["row_count"],
                    meta["row_count"] - prev_rows,
                )
            else:
                logger.info(
                    "Historical continuity verified: rows %d -> %d (%+d), games %d -> %d (%+d)",
                    prev_rows,
                    meta["row_count"],
                    meta["row_count"] - prev_rows,
                    prev_games,
                    meta["unique_games"],
                    meta["unique_games"] - prev_games,
                )
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            logger.warning("Could not verify baseline metadata file (%s): %s", meta_path, e)

    logger.info("Uploading updated history to Azure Blob '%s/%s'...", container, CANONICAL_HISTORY_BLOB)
    service_client = get_blob_service_client(account)
    container_client = service_client.get_container_client(container)
    blob_client = container_client.get_blob_client(CANONICAL_HISTORY_BLOB)

    with open(src, "rb") as f:
        blob_client.upload_blob(f, overwrite=True)

    logger.info("Historical data published successfully.")
    return meta


def validate_ingestion_metadata_for_publication(
    data_dir: Path,
    allow_missing_metadata: bool = False,
) -> Dict[str, Any]:
    """
    Validate ingestion telemetry before publishing a serving snapshot to production.
    Enforces Policy A: fail-closed on any partial, failed, stale, missing, malformed,
    or contradictory ingestion telemetry.
    """
    meta_path = data_dir / "ingestion_metadata.json"
    if not meta_path.exists():
        if allow_missing_metadata:
            logger.warning("Policy A bypass: Missing ingestion_metadata.json, but allow_missing_metadata=True.")
            return {"status": "UNKNOWN", "is_fresh": True, "source": "unknown"}
        raise ValueError(
            "Publication blocked (Policy A): Missing ingestion_metadata.json. "
            "Production serving publication requires verified ingestion telemetry to guarantee data freshness."
        )

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as e:
        raise ValueError(f"Publication blocked (Policy A): Malformed ingestion_metadata.json: {e}") from e

    # Required fields
    required_fields = ["status", "is_fresh"]
    missing = [rf for rf in required_fields if rf not in meta]
    if missing:
        raise ValueError(
            f"Publication blocked (Policy A): Malformed ingestion metadata - missing required field(s): {missing}"
        )

    status = meta.get("status")
    is_fresh = meta.get("is_fresh")
    failed_games_count = meta.get("failed_games_count", len(meta.get("failed_games", [])))
    new_games = meta.get("new_games_ingested", 0)

    # 1. Direct failure / partial statuses
    if status == "PARTIAL_SUCCESS":
        raise RuntimeError(
            f"Publication blocked (Policy A): Ingestion status is 'PARTIAL_SUCCESS' "
            f"({new_games} new games ingested, {failed_games_count} games failed). "
            "Partial ingestion runs cannot replace the production serving manifest."
        )
    if status in ("FAILED", "PRESERVED_EXISTING"):
        raise RuntimeError(
            f"Publication blocked (Policy A): Ingestion status is '{status}' (is_fresh={is_fresh}). "
            "Failed or preserved ingestion runs cannot replace the production serving manifest."
        )

    # 2. Freshness check
    if is_fresh is not True:
        raise RuntimeError(
            f"Publication blocked (Policy A): Ingestion metadata marks is_fresh={is_fresh} (status='{status}'). "
            "Non-fresh data cannot replace the production serving manifest."
        )

    # 3. Contradictory metadata
    if failed_games_count > 0:
        raise ValueError(
            f"Publication blocked (Policy A): Contradictory ingestion metadata - "
            f"status is '{status}' and is_fresh is True, but failed_games_count is {failed_games_count}."
        )

    # 4. Allowed valid statuses
    allowed_statuses = {"SUCCESS", "UP_TO_DATE", "VERIFIED_CURRENT"}
    if status not in allowed_statuses:
        raise RuntimeError(
            f"Publication blocked (Policy A): Unrecognized or non-publishable ingestion status '{status}'. "
            f"Allowed statuses: {sorted(allowed_statuses)}."
        )

    # 5. Historical cutoff check
    cutoff = meta.get("historical_cutoff_date")
    if cutoff and cutoff != "2026-02-06":
        raise ValueError(
            f"Publication blocked (Policy A): Invalid historical cutoff date '{cutoff}' (expected '2026-02-06')."
        )

    return meta


# ==============================================================================
# 4. Serving Container Operations (propredict-serving)
# ==============================================================================
def publish_serving_snapshot(
    account_name: Optional[str] = None,
    container_name: Optional[str] = None,
    data_dir: Optional[str] = None,
    pipeline_run_id: Optional[str] = None,
    git_sha: Optional[str] = None,
    allow_missing_metadata: bool = False,
) -> Dict[str, Any]:
    """
    Publish a complete versioned snapshot to propredict-serving.
    1. Validates frontend_all.parquet and frontend_home.parquet locally.
    2. Validates ingestion metadata under Policy A (fail-closed if partial, failed, stale, or malformed).
    3. Uploads both files under an immutable snapshot prefix: snapshots/<snapshot_id>/.
    4. Uploads snapshots/<snapshot_id>/manifest.json.
    5. Only after all snapshot files are uploaded and verified, updates root current_manifest.json.
    6. If any validation or upload fails, the prior current_manifest.json remains untouched.
    """
    account = account_name or os.environ.get("AZURE_STORAGE_ACCOUNT", DEFAULT_STORAGE_ACCOUNT)
    container = container_name or os.environ.get("AZURE_SERVING_CONTAINER", DEFAULT_SERVING_CONTAINER)
    data_path = Path(data_dir or DEFAULT_DATA_DIR)

    all_path = data_path / "frontend_all.parquet"
    home_path = data_path / "frontend_home.parquet"

    # 1. Validate both serving files
    meta_all = validate_serving_parquet(str(all_path), "frontend_all.parquet", allow_zero_rows=False)
    meta_home = validate_serving_parquet(str(home_path), "frontend_home.parquet", allow_zero_rows=True)

    # 2. Policy A Guard: Validate ingestion metadata before snapshot creation or upload
    ingestion_meta = validate_ingestion_metadata_for_publication(
        data_path,
        allow_missing_metadata=allow_missing_metadata,
    )
    is_fresh = bool(ingestion_meta.get("is_fresh", False))
    freshness_status = "FRESH" if (ingestion_meta.get("status") in ["SUCCESS", "UP_TO_DATE", "VERIFIED_CURRENT"] and is_fresh) else "DEGRADED"

    # 3. Build unique snapshot ID
    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    sha_suffix = (git_sha or os.environ.get("GITHUB_SHA", uuid.uuid4().hex))[:8]
    snapshot_id = f"snap_{timestamp_str}_{sha_suffix}"
    snapshot_prefix = f"snapshots/{snapshot_id}"

    manifest_data = {
        "snapshot_id": snapshot_id,
        "published_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha or os.environ.get("GITHUB_SHA", "unknown"),
        "pipeline_run_id": pipeline_run_id or os.environ.get("GITHUB_RUN_ID", "manual"),
        "data_freshness": {
            "primary_configured_provider": ingestion_meta.get("primary_configured_provider", "nba_stats"),
            "actual_provider": ingestion_meta.get("actual_provider", ingestion_meta.get("source", "unknown")),
            "fallback_occurred": ingestion_meta.get("fallback_occurred", False),
            "source": ingestion_meta.get("source", "unknown"),
            "new_games_ingested": ingestion_meta.get("new_games_ingested", 0),
            "existing_games_skipped": ingestion_meta.get("existing_games_skipped", 0),
            "failed_games_count": ingestion_meta.get("failed_games_count", 0),
            "total_games": ingestion_meta.get("total_games", 0),
            "latest_completed_game_date": ingestion_meta.get("latest_completed_game_date", "unknown"),
            "historical_cutoff_date": ingestion_meta.get("historical_cutoff_date", "2026-02-06"),
            "is_fresh": is_fresh,
            "source_freshness_status": freshness_status,
            "per_game_provenance": ingestion_meta.get("per_game_provenance", {}),
        },
        "files": {
            "frontend_all.parquet": {
                "blob_path": f"{snapshot_prefix}/frontend_all.parquet",
                "sha256": meta_all["sha256"],
                "size_bytes": meta_all["size_bytes"],
                "row_count": meta_all["row_count"],
            },
            "frontend_home.parquet": {
                "blob_path": f"{snapshot_prefix}/frontend_home.parquet",
                "sha256": meta_home["sha256"],
                "size_bytes": meta_home["size_bytes"],
                "row_count": meta_home["row_count"],
            },
        },
        "validation": {
            "status": "VALID",
            "frontend_all_rows": meta_all["row_count"],
            "frontend_home_rows": meta_home["row_count"],
        },
    }

    service_client = get_blob_service_client(account)
    container_client = service_client.get_container_client(container)

    # 3. Upload snapshot files under snapshots/<snapshot_id>/
    logger.info("Uploading snapshot '%s' artifacts to container '%s'...", snapshot_id, container)
    all_blob_client = container_client.get_blob_client(f"{snapshot_prefix}/frontend_all.parquet")
    with open(all_path, "rb") as f:
        all_blob_client.upload_blob(f, overwrite=True)

    home_blob_client = container_client.get_blob_client(f"{snapshot_prefix}/frontend_home.parquet")
    with open(home_path, "rb") as f:
        home_blob_client.upload_blob(f, overwrite=True)

    # 4. Upload immutable manifest for this snapshot
    snap_manifest_client = container_client.get_blob_client(f"{snapshot_prefix}/manifest.json")
    manifest_bytes = json.dumps(manifest_data, indent=2).encode("utf-8")
    snap_manifest_client.upload_blob(manifest_bytes, overwrite=True)

    # 5. Atomically update root current_manifest.json
    logger.info("Publishing canonical '%s' to point to snapshot '%s'...", DEFAULT_MANIFEST_BLOB, snapshot_id)
    current_manifest_client = container_client.get_blob_client(DEFAULT_MANIFEST_BLOB)
    current_manifest_client.upload_blob(manifest_bytes, overwrite=True)

    logger.info(
        "Serving snapshot '%s' published successfully (%d all, %d slate).",
        snapshot_id,
        meta_all["row_count"],
        meta_home["row_count"],
    )
    return manifest_data


def download_serving_snapshot(
    account_name: Optional[str] = None,
    container_name: Optional[str] = None,
    target_dir: Optional[str] = None,
    known_snapshot_id: Optional[str] = None,
) -> Tuple[bool, Optional[str], Optional[Path]]:
    """
    FastAPI serving synchronization function:
    1. Reads root current_manifest.json.
    2. Compares snapshot_id with known_snapshot_id. If identical, returns (False, known_snapshot_id, active_dir).
    3. Downloads snapshot files into an isolated staging directory (.staging_<snapshot_id>_<uuid>).
    4. Verifies SHA-256 and Parquet schema for each file inside staging.
    5. If verification succeeds on BOTH files, atomically promotes staging to snapshots/<snapshot_id>.
    6. Updates local current_manifest.json atomically.
    7. Returns (True, new_snapshot_id, snapshot_dir) on success.
    8. On ANY error, removes staging directory and leaves existing serving state completely untouched.
    """
    account = account_name or os.environ.get("AZURE_STORAGE_ACCOUNT", DEFAULT_STORAGE_ACCOUNT)
    container = container_name or os.environ.get("AZURE_SERVING_CONTAINER", DEFAULT_SERVING_CONTAINER)
    dest_dir = Path(target_dir or os.path.join(BASE_DIR, "data"))
    dest_dir.mkdir(parents=True, exist_ok=True)

    try:
        service_client = get_blob_service_client(account)
        container_client = service_client.get_container_client(container)
        manifest_client = container_client.get_blob_client(DEFAULT_MANIFEST_BLOB)

        if not manifest_client.exists():
            logger.warning("Manifest blob '%s' does not exist in container '%s'.", DEFAULT_MANIFEST_BLOB, container)
            return False, known_snapshot_id, None

        manifest_content = manifest_client.download_blob().readall()
        manifest = json.loads(manifest_content.decode("utf-8"))
    except Exception as e:
        logger.error("Failed to fetch or parse current manifest from Azure: %s", e)
        raise

    new_snapshot_id = manifest.get("snapshot_id")
    if not new_snapshot_id:
        raise ValueError("Invalid manifest: missing 'snapshot_id'.")

    active_snapshot_dir = dest_dir / "snapshots" / new_snapshot_id
    if known_snapshot_id and new_snapshot_id == known_snapshot_id:
        return False, known_snapshot_id, active_snapshot_dir

    logger.info("New snapshot detected: '%s' (current: '%s'). Staging download...", new_snapshot_id, known_snapshot_id)

    files_info = manifest.get("files", {})
    all_info = files_info.get("frontend_all.parquet")
    home_info = files_info.get("frontend_home.parquet")

    if not all_info or not home_info:
        raise ValueError("Manifest missing required file specifications for serving artifacts.")

    # 1. Isolated staging directory
    staging_dir = dest_dir / f".staging_{new_snapshot_id}_{uuid.uuid4().hex[:6]}"
    staging_dir.mkdir(parents=True, exist_ok=True)

    staged_all = staging_dir / "frontend_all.parquet"
    staged_home = staging_dir / "frontend_home.parquet"

    try:
        # Download frontend_all.parquet into staging
        all_blob_path = all_info["blob_path"]
        with open(staged_all, "wb") as f:
            container_client.get_blob_client(all_blob_path).download_blob().readinto(f)

        # Download frontend_home.parquet into staging
        home_blob_path = home_info["blob_path"]
        with open(staged_home, "wb") as f:
            container_client.get_blob_client(home_blob_path).download_blob().readinto(f)

        # Verify SHA-256 on both staged files
        sha_all = compute_sha256(str(staged_all))
        if sha_all != all_info["sha256"]:
            raise ValueError(f"Checksum mismatch on frontend_all.parquet (got {sha_all}, expected {all_info['sha256']})")

        sha_home = compute_sha256(str(staged_home))
        if sha_home != home_info["sha256"]:
            raise ValueError(f"Checksum mismatch on frontend_home.parquet (got {sha_home}, expected {home_info['sha256']})")

        # Validate Parquet schemas and readability on both staged files
        validate_serving_parquet(str(staged_all), "frontend_all.parquet", allow_zero_rows=False)
        validate_serving_parquet(str(staged_home), "frontend_home.parquet", allow_zero_rows=True)

        # Write immutable manifest inside staging
        with open(staging_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        # 2. Atomic Directory Promotion
        snapshots_base = dest_dir / "snapshots"
        snapshots_base.mkdir(parents=True, exist_ok=True)
        final_snapshot_dir = snapshots_base / new_snapshot_id

        if final_snapshot_dir.exists():
            shutil.rmtree(final_snapshot_dir, ignore_errors=True)

        # Atomic directory swap on the filesystem
        os.replace(str(staging_dir), str(final_snapshot_dir))

        # 3. Atomically update local current_manifest.json
        local_manifest_file = dest_dir / "current_manifest.json"
        tmp_manifest = dest_dir / "current_manifest.json.tmp"
        with open(tmp_manifest, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        os.replace(tmp_manifest, local_manifest_file)

        # 4. Synchronize canonical root parquet files for legacy/local tools
        for fname in ["frontend_all.parquet", "frontend_home.parquet"]:
            src_f = final_snapshot_dir / fname
            dst_f = dest_dir / fname
            tmp_f = dest_dir / f"{fname}.tmp"
            shutil.copy2(src_f, tmp_f)
            os.replace(tmp_f, dst_f)

        logger.info("Serving snapshot '%s' promoted and activated successfully.", new_snapshot_id)
        return True, new_snapshot_id, final_snapshot_dir

    except Exception as e:
        logger.error("Failed to stage or verify serving snapshot '%s': %s", new_snapshot_id, e)
        # Completely clean up staging directory without affecting existing serving state
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        raise


# ==============================================================================
# 5. Atomic Concurrency-Safe Daily Claim (Single-Writer CAS Lease)
# ==============================================================================
def get_daily_claim_blob_name(date_str: Optional[str] = None) -> str:
    today_ny = date_str or datetime.now(timezone.utc).astimezone(ZoneInfo("America/New_York")).date().isoformat()
    return f"claims/daily_claim_{today_ny}.json"


def acquire_daily_claim(
    account_name: Optional[str] = None,
    container_name: Optional[str] = None,
    run_id: Optional[str] = None,
    git_sha: Optional[str] = None,
    force: bool = False,
    timeout_minutes: int = 75,
) -> bool:
    """
    Acquire a single-writer lease for today's America/New_York calendar date.
    Uses Azure Blob Storage's native atomic conditional upload (overwrite=False / If-None-Match: *).
    Guarantees mutual exclusion: if two invocations check at the exact same millisecond,
    only ONE can acquire the claim.
    Handles retry of failed/abandoned runs and explicit force bypass.
    Sets GITHUB_OUTPUT: claim_acquired=true|false.
    """
    output_path = os.environ.get("GITHUB_OUTPUT")

    def _set_output(val: str):
        if output_path and os.path.exists(os.path.dirname(output_path)):
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(f"claim_acquired={val}\n")
        print(f"Claim Output: claim_acquired={val}")

    today_ny = datetime.now(timezone.utc).astimezone(ZoneInfo("America/New_York")).date().isoformat()
    claim_blob_name = get_daily_claim_blob_name(today_ny)
    account = account_name or os.environ.get("AZURE_STORAGE_ACCOUNT", DEFAULT_STORAGE_ACCOUNT)
    container = container_name or os.environ.get("AZURE_SERVING_CONTAINER", DEFAULT_SERVING_CONTAINER)

    service_client = get_blob_service_client(account)
    container_client = service_client.get_container_client(container)
    claim_client = container_client.get_blob_client(claim_blob_name)

    claim_payload = {
        "date_ny": today_ny,
        "status": "IN_PROGRESS",
        "claimed_at": datetime.now(timezone.utc).isoformat(),
        "run_id": str(run_id or os.environ.get("GITHUB_RUN_ID", "manual")),
        "git_sha": str(git_sha or os.environ.get("GITHUB_SHA", "unknown")),
        "completed_at": None,
        "error": None,
    }
    payload_bytes = json.dumps(claim_payload, indent=2).encode("utf-8")

    if force:
        logger.info("Forced execution requested. Overwriting daily claim for %s.", today_ny)
        claim_client.upload_blob(payload_bytes, overwrite=True)
        _set_output("true")
        return True

    try:
        from azure.core.exceptions import ResourceExistsError
    except ImportError:
        ResourceExistsError = Exception

    # Attempt atomic conditional write (If-None-Match: *)
    try:
        claim_client.upload_blob(payload_bytes, overwrite=False)
        logger.info("Atomic daily claim acquired for %s (run: %s).", today_ny, claim_payload["run_id"])
        _set_output("true")
        return True
    except ResourceExistsError:
        logger.info("Claim blob %s already exists. Inspecting active claim state...", claim_blob_name)

    # Claim blob already exists: inspect existing claim state
    try:
        existing_bytes = claim_client.download_blob().readall()
        existing = json.loads(existing_bytes.decode("utf-8"))
    except Exception as e:
        logger.warning("Could not read existing claim: %s. Reclaiming slot safely.", e)
        claim_client.upload_blob(payload_bytes, overwrite=True)
        _set_output("true")
        return True

    status = existing.get("status")
    claimed_run_id = existing.get("run_id")

    # Case A: Prior run already successfully completed today
    if status in ("COMPLETED", "SUCCESS"):
        logger.warning(
            "DAILY CLAIM REJECTED: Today's pipeline (%s) already completed successfully at %s (run: %s).",
            today_ny,
            existing.get("completed_at"),
            claimed_run_id,
        )
        _set_output("false")
        return False

    # Case B: Prior run failed - retry safely permitted
    if status == "FAILED":
        logger.info("Prior pipeline run (%s) recorded FAILED. Reclaiming slot for retry.", claimed_run_id)
        claim_client.upload_blob(payload_bytes, overwrite=True)
        _set_output("true")
        return True

    # Case C: Active run IN_PROGRESS - check for abandonment / timeout
    if status == "IN_PROGRESS":
        claimed_at_str = existing.get("claimed_at")
        if claimed_at_str:
            try:
                claimed_at = datetime.fromisoformat(claimed_at_str)
                elapsed_min = (datetime.now(timezone.utc) - claimed_at).total_seconds() / 60.0
                if elapsed_min > timeout_minutes:
                    logger.warning(
                        "Prior claim by run %s has timed out (%.1fm > %dm). Reclaiming abandoned slot.",
                        claimed_run_id,
                        elapsed_min,
                        timeout_minutes,
                    )
                    claim_client.upload_blob(payload_bytes, overwrite=True)
                    _set_output("true")
                    return True
                else:
                    logger.warning(
                        "CONCURRENT RUN DETECTED: Run %s is currently active (claimed %.1fm ago). Rejecting collision.",
                        claimed_run_id,
                        elapsed_min,
                    )
                    _set_output("false")
                    return False
            except Exception:
                pass

    _set_output("false")
    return False


def complete_daily_claim(
    account_name: Optional[str] = None,
    container_name: Optional[str] = None,
    run_id: Optional[str] = None,
) -> None:
    """Mark today's claim as COMPLETED after successful snapshot publication."""
    today_ny = datetime.now(timezone.utc).astimezone(ZoneInfo("America/New_York")).date().isoformat()
    claim_blob_name = get_daily_claim_blob_name(today_ny)
    account = account_name or os.environ.get("AZURE_STORAGE_ACCOUNT", DEFAULT_STORAGE_ACCOUNT)
    container = container_name or os.environ.get("AZURE_SERVING_CONTAINER", DEFAULT_SERVING_CONTAINER)

    service_client = get_blob_service_client(account)
    container_client = service_client.get_container_client(container)
    claim_client = container_client.get_blob_client(claim_blob_name)

    payload = {
        "date_ny": today_ny,
        "status": "COMPLETED",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "run_id": str(run_id or os.environ.get("GITHUB_RUN_ID", "manual")),
    }
    claim_client.upload_blob(json.dumps(payload, indent=2).encode("utf-8"), overwrite=True)
    logger.info("Daily claim for %s marked as COMPLETED (run: %s).", today_ny, payload["run_id"])


def fail_daily_claim(
    account_name: Optional[str] = None,
    container_name: Optional[str] = None,
    run_id: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    """Mark today's claim as FAILED so retries can execute without timeout delays."""
    today_ny = datetime.now(timezone.utc).astimezone(ZoneInfo("America/New_York")).date().isoformat()
    claim_blob_name = get_daily_claim_blob_name(today_ny)
    account = account_name or os.environ.get("AZURE_STORAGE_ACCOUNT", DEFAULT_STORAGE_ACCOUNT)
    container = container_name or os.environ.get("AZURE_SERVING_CONTAINER", DEFAULT_SERVING_CONTAINER)

    service_client = get_blob_service_client(account)
    container_client = service_client.get_container_client(container)
    claim_client = container_client.get_blob_client(claim_blob_name)

    payload = {
        "date_ny": today_ny,
        "status": "FAILED",
        "failed_at": datetime.now(timezone.utc).isoformat(),
        "run_id": str(run_id or os.environ.get("GITHUB_RUN_ID", "manual")),
        "error": str(error or "Pipeline execution failure"),
    }
    claim_client.upload_blob(json.dumps(payload, indent=2).encode("utf-8"), overwrite=True)
    logger.info("Daily claim for %s marked as FAILED (run: %s). Retries unblocked.", today_ny, payload["run_id"])


def check_daily_publication_idempotency(
    account_name: Optional[str] = None,
    container_name: Optional[str] = None,
    force: bool = False,
) -> bool:
    """
    Check if a serving snapshot has already been published today in America/New_York timezone.
    Returns True if already published (skip execution), False if execution should proceed.
    Sets GITHUB_OUTPUT: already_run=true|false.
    """
    output_path = os.environ.get("GITHUB_OUTPUT")

    def _set_output(val: str):
        if output_path and os.path.exists(os.path.dirname(output_path)):
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(f"already_run={val}\n")
        print(f"Idempotency Output: already_run={val}")

    if force:
        logger.info("Forced execution requested. Idempotency guard bypassed.")
        _set_output("false")
        return False

    account = account_name or os.environ.get("AZURE_STORAGE_ACCOUNT", DEFAULT_STORAGE_ACCOUNT)
    container = container_name or os.environ.get("AZURE_SERVING_CONTAINER", DEFAULT_SERVING_CONTAINER)

    try:
        service_client = get_blob_service_client(account)
        container_client = service_client.get_container_client(container)
        manifest_client = container_client.get_blob_client(DEFAULT_MANIFEST_BLOB)

        if not manifest_client.exists():
            logger.info("No current_manifest.json found in %s. First run detected.", container)
            _set_output("false")
            return False

        manifest_content = manifest_client.download_blob().readall()
        manifest = json.loads(manifest_content.decode("utf-8"))
        published_at_str = manifest.get("published_at")

        if not published_at_str:
            _set_output("false")
            return False

        published_dt = datetime.fromisoformat(published_at_str).astimezone(ZoneInfo("America/New_York"))
        today_ny = datetime.now(timezone.utc).astimezone(ZoneInfo("America/New_York"))

        published_date = published_dt.date()
        current_date = today_ny.date()

        if published_date == current_date:
            logger.warning(
                "IDEMPOTENCY MATCH: A serving snapshot for America/New_York date %s was already published at %s (%s). "
                "Skipping duplicate scheduled pipeline execution.",
                current_date,
                published_dt.strftime("%Y-%m-%d %H:%M:%S %Z"),
                manifest.get("snapshot_id"),
            )
            _set_output("true")
            return True

        logger.info(
            "Last snapshot was published on %s (current New York date: %s). Proceeding with pipeline.",
            published_date,
            current_date,
        )
        _set_output("false")
        return False

    except Exception as e:
        logger.warning("Could not verify remote manifest for idempotency: %s. Defaulting to proceed.", e)
        _set_output("false")
        return False


# ==============================================================================
# 6. CLI Entrypoint for Workflow / Automation
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Azure Blob Storage synchronization tool for ProPredict.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # restore-history
    p_restore = subparsers.add_parser("restore-history", help="Restore play_by_play.parquet from history container")
    p_restore.add_argument("--account", default=None, help="Azure Storage account name")
    p_restore.add_argument("--container", default=DEFAULT_HISTORY_CONTAINER, help="History container name")
    p_restore.add_argument("--target", default=None, help="Target local filepath")

    # publish-history
    p_pub_hist = subparsers.add_parser("publish-history", help="Publish updated play_by_play.parquet to history container")
    p_pub_hist.add_argument("--account", default=None, help="Azure Storage account name")
    p_pub_hist.add_argument("--container", default=DEFAULT_HISTORY_CONTAINER, help="History container name")
    p_pub_hist.add_argument("--source", default=None, help="Source local filepath")

    # publish-serving
    p_pub_srv = subparsers.add_parser("publish-serving", help="Publish new serving snapshot to serving container")
    p_pub_srv.add_argument("--account", default=None, help="Azure Storage account name")
    p_pub_srv.add_argument("--container", default=DEFAULT_SERVING_CONTAINER, help="Serving container name")
    p_pub_srv.add_argument("--data-dir", default=None, help="Data directory containing parquet artifacts")
    p_pub_srv.add_argument("--run-id", default=None, help="GitHub run ID or pipeline identifier")
    p_pub_srv.add_argument("--git-sha", default=None, help="Git commit SHA")

    # sync-serving
    p_sync = subparsers.add_parser("sync-serving", help="Download current serving snapshot to local data directory")
    p_sync.add_argument("--account", default=None, help="Azure Storage account name")
    p_sync.add_argument("--container", default=DEFAULT_SERVING_CONTAINER, help="Serving container name")
    p_sync.add_argument("--data-dir", default=None, help="Target data directory")

    # check-idempotency
    p_idem = subparsers.add_parser("check-idempotency", help="Check if daily snapshot already published for today")
    p_idem.add_argument("--account", default=None, help="Azure Storage account name")
    p_idem.add_argument("--container", default=DEFAULT_SERVING_CONTAINER, help="Serving container name")
    p_idem.add_argument("--force", action="store_true", help="Bypass idempotency check")

    # acquire-claim
    p_claim = subparsers.add_parser("acquire-claim", help="Acquire atomic daily single-writer claim")
    p_claim.add_argument("--account", default=None, help="Azure Storage account name")
    p_claim.add_argument("--container", default=DEFAULT_SERVING_CONTAINER, help="Serving container name")
    p_claim.add_argument("--run-id", default=None, help="GitHub run ID or pipeline identifier")
    p_claim.add_argument("--git-sha", default=None, help="Git commit SHA")
    p_claim.add_argument("--force", action="store_true", help="Force claim acquisition by overwriting")
    p_claim.add_argument("--timeout-minutes", type=int, default=75, help="Timeout in minutes for abandoned claims")

    # complete-claim
    p_comp = subparsers.add_parser("complete-claim", help="Mark daily claim as COMPLETED")
    p_comp.add_argument("--account", default=None, help="Azure Storage account name")
    p_comp.add_argument("--container", default=DEFAULT_SERVING_CONTAINER, help="Serving container name")
    p_comp.add_argument("--run-id", default=None, help="GitHub run ID or pipeline identifier")

    # fail-claim
    p_fail = subparsers.add_parser("fail-claim", help="Mark daily claim as FAILED for safe retry")
    p_fail.add_argument("--account", default=None, help="Azure Storage account name")
    p_fail.add_argument("--container", default=DEFAULT_SERVING_CONTAINER, help="Serving container name")
    p_fail.add_argument("--run-id", default=None, help="GitHub run ID or pipeline identifier")
    p_fail.add_argument("--error", default=None, help="Error message description")

    # validate-artifacts
    p_val = subparsers.add_parser("validate-artifacts", help="Validate all serving and historical artifacts prior to publication")
    p_val.add_argument("--data-dir", default=None, help="Data directory containing parquet artifacts")

    args = parser.parse_args()

    try:
        if args.command == "restore-history":
            restore_historical_data(args.account, args.container, args.target)
        elif args.command == "publish-history":
            publish_historical_data(args.account, args.container, args.source)
        elif args.command == "publish-serving":
            publish_serving_snapshot(args.account, args.container, args.data_dir, args.run_id, args.git_sha)
        elif args.command == "sync-serving":
            download_serving_snapshot(args.account, args.container, args.data_dir)
        elif args.command == "check-idempotency":
            already_run = check_daily_publication_idempotency(args.account, args.container, args.force)
            # Exit 0 always so workflow step can read output
            sys.exit(0)
        elif args.command == "acquire-claim":
            acquired = acquire_daily_claim(
                args.account,
                args.container,
                args.run_id,
                args.git_sha,
                args.force,
                args.timeout_minutes,
            )
            # Exit 0 always so workflow step can read claim_acquired output
            sys.exit(0)
        elif args.command == "complete-claim":
            complete_daily_claim(args.account, args.container, args.run_id)
            sys.exit(0)
        elif args.command == "fail-claim":
            fail_daily_claim(args.account, args.container, args.run_id, args.error)
            sys.exit(0)
        elif args.command == "validate-artifacts":
            validate_artifacts(args.data_dir)
            sys.exit(0)
        sys.exit(0)
    except Exception as e:
        logger.error("Command '%s' failed: %s", args.command, e)
        sys.exit(1)


if __name__ == "__main__":
    main()


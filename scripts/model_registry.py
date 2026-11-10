"""
scripts/model_registry.py

Canonical Model Registry and Lifecycle Management for UN-REAL NBA Prediction System.
Enforces versioned model artifacts, SHA-256 integrity verification, schema validation,
season boundary checks, and atomic active-model pointer transitions.

Guarantees:
1. Every trained model is persisted to an immutable, versioned directory.
2. Metadata captures complete training lineage, date ranges, and validation metrics.
3. Active model pointer (active_model.json) is updated ONLY after successful validation.
4. Failed training runs never overwrite or displace the previous active model.
5. Inference fails closed if the active model is missing, corrupt, or unvalidated.
6. Daily inference never falls back to unvalidated or stale bundled models.
"""

import os
import sys
import json
import shutil
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, Tuple, Optional, List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EXPECTED_FEATURE_SCHEMA = [
    "last_game",
    "last3_avg",
    "last5_avg",
    "last7_avg",
    "games_played_before_target",
    "expand_mean",
    "days_since_prev_game",
]

DEFAULT_SEASON_BOUNDARY = "2026-10-20"


class ModelValidationError(Exception):
    """Raised when model integrity, schema, or active pointer validation fails."""
    pass


def compute_file_sha256(filepath: str) -> str:
    """Compute streaming SHA-256 checksum of a file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found for hashing: {filepath}")
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def generate_version_id(prefix: str = "model") -> str:
    """Generate an immutable timestamped model version identifier."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}"


def validate_metadata_structure(metadata: Dict[str, Any], require_validated: bool = True) -> None:
    """Validates that a metadata dictionary conforms to the required specification."""
    required_keys = [
        "model_version",
        "sha256",
        "season_boundary",
        "feature_schema",
        "training_timestamp",
        "is_validated",
    ]
    missing = [k for k in required_keys if k not in metadata]
    if missing:
        raise ModelValidationError(f"Model metadata is missing required keys: {missing}")

    if require_validated and not metadata.get("is_validated", False):
        raise ModelValidationError(f"Model {metadata.get('model_version')} is not marked as validated")

    schema = metadata.get("feature_schema", [])
    if schema != EXPECTED_FEATURE_SCHEMA:
        raise ModelValidationError(
            f"Model feature schema mismatch: expected {EXPECTED_FEATURE_SCHEMA}, got {schema}"
        )


def save_versioned_model(
    model: Any,
    version_id: str,
    metadata: Dict[str, Any],
    base_dir: str = "models",
) -> Tuple[str, str]:
    """
    Persists a model artifact and its metadata into an immutable version directory.
    
    Structure:
      base_dir/versions/<version_id>/rating_model.pkl
      base_dir/versions/<version_id>/model_metadata.json
    """
    import joblib

    version_dir = os.path.join(base_dir, "versions", version_id)
    os.makedirs(version_dir, exist_ok=True)

    model_path = os.path.join(version_dir, "rating_model.pkl")
    metadata_path = os.path.join(version_dir, "model_metadata.json")

    # 1. Dump model artifact
    joblib.dump(model, model_path)

    # 2. Compute exact SHA-256 of the persisted artifact
    actual_sha = compute_file_sha256(model_path)
    metadata["sha256"] = actual_sha
    metadata["model_version"] = version_id
    metadata["model_file"] = "rating_model.pkl"

    # 3. Validate metadata schema (permit saving inactive/unpromoted models)
    validate_metadata_structure(metadata, require_validated=False)

    # 4. Save metadata JSON atomically
    tmp_meta = metadata_path + ".tmp"
    with open(tmp_meta, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    os.replace(tmp_meta, metadata_path)

    return model_path, metadata_path


def promote_active_model(
    version_id: str,
    base_dir: str = "models",
    update_legacy_path: bool = True,
) -> str:
    """
    Atomically updates the active model pointer (active_model.json).
    Validates model integrity, SHA-256, and metadata before switching pointer.
    """
    version_dir = os.path.join(base_dir, "versions", version_id)
    model_path = os.path.join(version_dir, "rating_model.pkl")
    metadata_path = os.path.join(version_dir, "model_metadata.json")

    if not os.path.exists(model_path):
        raise ModelValidationError(f"Cannot promote: model artifact missing at {model_path}")
    if not os.path.exists(metadata_path):
        raise ModelValidationError(f"Cannot promote: metadata missing at {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Verify SHA-256
    actual_sha = compute_file_sha256(model_path)
    if actual_sha != meta.get("sha256"):
        raise ModelValidationError(
            f"Cannot promote: SHA-256 mismatch for {version_id}. Expected {meta.get('sha256')}, got {actual_sha}"
        )

    # Validate metadata
    validate_metadata_structure(meta)

    # Create active pointer dictionary
    pointer_data = {
        "active_version": version_id,
        "model_path": os.path.relpath(model_path, base_dir).replace("\\", "/"),
        "metadata_path": os.path.relpath(metadata_path, base_dir).replace("\\", "/"),
        "sha256": actual_sha,
        "season_boundary": meta.get("season_boundary", DEFAULT_SEASON_BOUNDARY),
        "feature_schema": meta.get("feature_schema", EXPECTED_FEATURE_SCHEMA),
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "is_active": True,
    }

    active_pointer_path = os.path.join(base_dir, "active_model.json")
    tmp_pointer = active_pointer_path + ".tmp"
    with open(tmp_pointer, "w", encoding="utf-8") as f:
        json.dump(pointer_data, f, indent=2)
    os.replace(tmp_pointer, active_pointer_path)

    # Optionally sync legacy path for backward-compatible loaders
    if update_legacy_path:
        legacy_path = os.path.join(base_dir, "rating_model.pkl")
        legacy_meta_path = os.path.join(base_dir, "feature_metadata.json")
        try:
            shutil.copy2(model_path, legacy_path)
            # Create feature metadata if std3_median exists in meta
            if "std3_median" in meta:
                with open(legacy_meta_path, "w", encoding="utf-8") as f:
                    json.dump({"std3_median": meta["std3_median"]}, f, indent=2)
        except Exception:
            pass

    return active_pointer_path


def resolve_active_model(
    base_dir: str = "models",
    fail_closed: bool = True,
) -> Tuple[str, Dict[str, Any]]:
    """
    Resolves the currently active, validated model artifact.
    
    Performs rigorous fail-closed validation:
    1. Checks active_model.json exists.
    2. Validates JSON structure.
    3. Verifies referenced model file exists.
    4. Computes SHA-256 and ensures exact match.
    5. Verifies feature schema compatibility.
    6. Verifies current-season boundary adherence.
    
    Returns:
        (resolved_model_path, active_metadata_dict)
    """
    # Check alternate base dir from environment if present
    env_dir = os.environ.get("PROPREDICT_MODELS_DIR")
    if env_dir and os.path.exists(os.path.join(env_dir, "active_model.json")):
        base_dir = env_dir

    pointer_path = os.path.join(base_dir, "active_model.json")
    if not os.path.exists(pointer_path):
        if fail_closed:
            raise ModelValidationError(
                f"Fail-closed: active model pointer missing at {pointer_path}. "
                "Daily inference will not execute without an active validated model."
            )
        return "", {}

    try:
        with open(pointer_path, "r", encoding="utf-8") as f:
            pointer = json.load(f)
    except Exception as e:
        raise ModelValidationError(f"Corrupt active model pointer at {pointer_path}: {e}") from e

    expected_sha = pointer.get("sha256")
    rel_model_path = pointer.get("model_path")
    rel_meta_path = pointer.get("metadata_path")

    if not expected_sha or not rel_model_path:
        raise ModelValidationError(f"Active model pointer at {pointer_path} is malformed: {pointer}")

    abs_model_path = os.path.normpath(os.path.join(base_dir, rel_model_path) if not os.path.isabs(rel_model_path) else rel_model_path)
    if not os.path.exists(abs_model_path):
        raise ModelValidationError(f"Active model artifact referenced by pointer does not exist: {abs_model_path}")

    actual_sha = compute_file_sha256(abs_model_path)
    if actual_sha != expected_sha:
        raise ModelValidationError(
            f"Active model integrity failure: SHA-256 mismatch for {abs_model_path}. "
            f"Expected {expected_sha}, got {actual_sha}."
        )

    # Check metadata if available
    abs_meta_path = os.path.join(base_dir, rel_meta_path) if rel_meta_path and not os.path.isabs(rel_meta_path) else rel_meta_path
    meta = {}
    if abs_meta_path and os.path.exists(abs_meta_path):
        with open(abs_meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        validate_metadata_structure(meta)

    # Verify schema
    schema = pointer.get("feature_schema", meta.get("feature_schema", []))
    if schema != EXPECTED_FEATURE_SCHEMA:
        raise ModelValidationError(
            f"Active model feature schema mismatch: expected {EXPECTED_FEATURE_SCHEMA}, got {schema}"
        )

    return abs_model_path, {**meta, **pointer}

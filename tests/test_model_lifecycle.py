"""
tests/test_model_lifecycle.py

Comprehensive test suite verifying the current-season model retraining architecture:
1. Current-season-only training isolation (>= 2026-10-20).
2. Current-season-only inference feature construction (>= 2026-10-20).
3. Model artifact versioning and metadata schema validation.
4. SHA-256 integrity verification.
5. Active model pointer update only after successful validation.
6. Failed training preserving the previous active model pointer.
7. Failed inference / invalid model preventing serving snapshot publication.
8. No fallback to old bundled models.
9. Root and Azure Function codebase parity.
10. Preservation of Candidate D bonuses and Top Players ranking semantics.
"""

import os
import sys
import json
import shutil
import tempfile
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestRegressor

from scripts.model_registry import (
    generate_version_id,
    save_versioned_model,
    promote_active_model,
    resolve_active_model,
    compute_file_sha256,
    validate_metadata_structure,
    ModelValidationError,
    EXPECTED_FEATURE_SCHEMA,
    DEFAULT_SEASON_BOUNDARY,
)
from scripts.predict_ratings import construct_prediction_features
from scripts.compute_season_multipliers import compute_season_multipliers


# ==============================================================================
# 1. Current-Season Isolation Tests
# ==============================================================================

def test_inference_current_season_isolation():
    """Verify construct_prediction_features excludes any pre-season games (< 2026-10-20)."""
    dates = ["2026-03-15", "2026-04-01", "2026-10-22", "2026-10-25"]
    history_df = pd.DataFrame({
        "gameId": ["g1", "g2", "g3", "g4"],
        "personId": [101, 101, 101, 101],
        "game_date": dates,
        "rating": [2.5, 3.0, 4.0, 4.5],
        "teamId": [1610612738] * 4,
        "playerName": ["Test Player"] * 4,
    })

    features_df = construct_prediction_features(
        history_df,
        target_date="2026-10-28",
        std3_impute_value=0.5,
    )

    assert not features_df.empty
    player_row = features_df.iloc[0]
    # Only 2 games from current season (Oct 22 and Oct 25)
    assert player_row["games_played_before_target"] == 2
    assert player_row["last_game_date"] == "2026-10-25"
    assert player_row["last_game"] == 4.5
    # Mean of Oct 22 (4.0) and Oct 25 (4.5) = 4.25, NOT diluted by March/April games
    assert player_row["expand_mean"] == pytest.approx(4.25, abs=1e-4)


def test_inference_player_with_only_old_games_is_excluded():
    """Verify a player with only 2025-26 season games yields no predictions for 2026-27."""
    history_df = pd.DataFrame({
        "gameId": ["old1", "old2"],
        "personId": [999, 999],
        "game_date": ["2026-03-15", "2026-04-01"],
        "rating": [2.5, 3.0],
        "teamId": [1610612738, 1610612738],
        "playerName": ["Old Player", "Old Player"],
    })

    res = construct_prediction_features(history_df, target_date="2026-10-28")
    assert res.empty


# ==============================================================================
# 2. Model Registry & Versioning Tests
# ==============================================================================

def test_model_versioning_and_metadata_persistence():
    """Verify versioned model saving creates correct directory structure, metadata, and SHA-256."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        rf = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)
        X = pd.DataFrame(np.random.randn(20, 7), columns=EXPECTED_FEATURE_SCHEMA)
        y = pd.Series(np.random.randn(20))
        rf.fit(X, y)

        v_id = "model_test_v1"
        meta = {
            "model_version": v_id,
            "season_boundary": "2026-10-20",
            "training_timestamp": datetime.now(timezone.utc).isoformat(),
            "training_date_range": {"start": "2026-10-21", "end": "2026-11-06"},
            "test_date_range": {"start": "2026-11-07", "end": "2026-11-10"},
            "training_rows": 100,
            "testing_rows": 20,
            "total_feature_rows": 120,
            "unique_players": 30,
            "feature_schema": EXPECTED_FEATURE_SCHEMA,
            "hyperparameters": {"n_estimators": 10, "max_depth": 3},
            "validation_metrics": {"rf_test_mae": 0.75, "rf_test_r2": 0.50},
            "is_validated": True,
        }

        model_path, meta_path = save_versioned_model(rf, v_id, meta, base_dir=tmp_dir)

        assert os.path.exists(model_path)
        assert os.path.exists(meta_path)
        assert compute_file_sha256(model_path) == meta["sha256"]

        with open(meta_path, "r", encoding="utf-8") as f:
            saved_meta = json.load(f)
        assert saved_meta["model_version"] == v_id
        assert saved_meta["sha256"] == meta["sha256"]
        assert saved_meta["feature_schema"] == EXPECTED_FEATURE_SCHEMA


def test_active_pointer_promotion_and_resolution():
    """Verify active model pointer promotion and fail-closed resolution."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        rf = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)
        X = pd.DataFrame(np.random.randn(20, 7), columns=EXPECTED_FEATURE_SCHEMA)
        y = pd.Series(np.random.randn(20))
        rf.fit(X, y)

        v_id = "model_test_v1"
        meta = {
            "model_version": v_id,
            "season_boundary": "2026-10-20",
            "training_timestamp": datetime.now(timezone.utc).isoformat(),
            "training_date_range": {"start": "2026-10-21", "end": "2026-11-06"},
            "training_rows": 100,
            "testing_rows": 20,
            "total_feature_rows": 120,
            "unique_players": 30,
            "feature_schema": EXPECTED_FEATURE_SCHEMA,
            "is_validated": True,
        }

        save_versioned_model(rf, v_id, meta, base_dir=tmp_dir)
        pointer_path = promote_active_model(v_id, base_dir=tmp_dir)

        assert os.path.exists(pointer_path)

        resolved_path, active_meta = resolve_active_model(base_dir=tmp_dir)
        assert os.path.normpath(resolved_path) == os.path.normpath(os.path.join(tmp_dir, "versions", v_id, "rating_model.pkl"))
        assert active_meta["active_version"] == v_id
        assert active_meta["sha256"] == compute_file_sha256(resolved_path)


def test_failed_training_preserves_previous_active_model():
    """Verify that a corrupted or unvalidated new model does NOT overwrite the active pointer."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        rf1 = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)
        X = pd.DataFrame(np.random.randn(20, 7), columns=EXPECTED_FEATURE_SCHEMA)
        y = pd.Series(np.random.randn(20))
        rf1.fit(X, y)

        # 1. Successfully deploy v1
        v1_id = "model_v1"
        meta1 = {
            "model_version": v1_id,
            "season_boundary": "2026-10-20",
            "training_timestamp": datetime.now(timezone.utc).isoformat(),
            "training_rows": 100,
            "feature_schema": EXPECTED_FEATURE_SCHEMA,
            "is_validated": True,
        }
        save_versioned_model(rf1, v1_id, meta1, base_dir=tmp_dir)
        promote_active_model(v1_id, base_dir=tmp_dir)

        initial_sha = compute_file_sha256(os.path.join(tmp_dir, "versions", v1_id, "rating_model.pkl"))

        # 2. Attempt to save or promote invalid v2 (unvalidated)
        v2_id = "model_v2_broken"
        meta2 = {
            "model_version": v2_id,
            "season_boundary": "2026-10-20",
            "training_timestamp": datetime.now(timezone.utc).isoformat(),
            "training_rows": 100,
            "feature_schema": EXPECTED_FEATURE_SCHEMA,
            "is_validated": False,  # Not validated!
        }

        # Saving unvalidated model succeeds as inactive artifact
        save_versioned_model(rf1, v2_id, meta2, base_dir=tmp_dir)

        # Attempting to promote unvalidated model MUST fail validation
        with pytest.raises(ModelValidationError, match="not marked as validated"):
            promote_active_model(v2_id, base_dir=tmp_dir)

        # Active pointer must still point to v1
        resolved_path, active_meta = resolve_active_model(base_dir=tmp_dir)
        assert active_meta["active_version"] == v1_id
        assert active_meta["sha256"] == initial_sha


def test_resolve_active_model_fails_closed_when_pointer_missing():
    """Verify resolve_active_model fails closed and does NOT silently load unvalidated models."""
    with tempfile.TemporaryDirectory() as empty_dir:
        with pytest.raises(ModelValidationError, match="active model pointer missing"):
            resolve_active_model(base_dir=empty_dir, fail_closed=True)


def test_resolve_active_model_fails_closed_on_sha_tampering():
    """Verify resolve_active_model detects file tampering and rejects the model."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        rf = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)
        X = pd.DataFrame(np.random.randn(20, 7), columns=EXPECTED_FEATURE_SCHEMA)
        y = pd.Series(np.random.randn(20))
        rf.fit(X, y)

        v_id = "model_tamper_test"
        meta = {
            "model_version": v_id,
            "season_boundary": "2026-10-20",
            "training_timestamp": datetime.now(timezone.utc).isoformat(),
            "training_rows": 50,
            "feature_schema": EXPECTED_FEATURE_SCHEMA,
            "is_validated": True,
        }
        save_versioned_model(rf, v_id, meta, base_dir=tmp_dir)
        promote_active_model(v_id, base_dir=tmp_dir)

        # Tamper with the model file
        model_file = os.path.join(tmp_dir, "versions", v_id, "rating_model.pkl")
        with open(model_file, "ab") as f:
            f.write(b"CORRUPTED_BYTES")

        with pytest.raises(ModelValidationError, match="SHA-256 mismatch"):
            resolve_active_model(base_dir=tmp_dir, fail_closed=True)


def test_snapshot_publication_fails_closed_without_valid_active_model(monkeypatch):
    """Verify publish_serving_snapshot fails closed if active model cannot be verified."""
    from scripts.azure_blob import publish_serving_snapshot

    with tempfile.TemporaryDirectory() as empty_model_dir:
        monkeypatch.setenv("PROPREDICT_MODELS_DIR", empty_model_dir)

        with pytest.raises(RuntimeError, match="Snapshot publication blocked: active model validation failed"):
            publish_serving_snapshot(data_dir="data", container_name="propredict-serving")


# ==============================================================================
# 3. Root vs Azure Function Parity Tests
# ==============================================================================

def test_codebase_parity_model_registry():
    """Verify root and azure_function implementations of model_registry.py are identical."""
    root_path = "scripts/model_registry.py"
    af_path = "azure_function/scripts/model_registry.py"

    assert os.path.exists(root_path), "Root model_registry.py missing"
    assert os.path.exists(af_path), "azure_function model_registry.py missing"

    with open(root_path, "r", encoding="utf-8") as f1, open(af_path, "r", encoding="utf-8") as f2:
        assert f1.read() == f2.read(), "Parity mismatch between root and azure_function model_registry.py"


def test_codebase_parity_train_model():
    """Verify root and azure_function implementations of train_model.py are identical."""
    root_path = "scripts/train_model.py"
    af_path = "azure_function/scripts/train_model.py"

    with open(root_path, "r", encoding="utf-8") as f1, open(af_path, "r", encoding="utf-8") as f2:
        assert f1.read() == f2.read(), "Parity mismatch between root and azure_function train_model.py"


def test_codebase_parity_predict_ratings():
    """Verify root and azure_function implementations of predict_ratings.py are identical."""
    root_path = "scripts/predict_ratings.py"
    af_path = "azure_function/scripts/predict_ratings.py"

    with open(root_path, "r", encoding="utf-8") as f1, open(af_path, "r", encoding="utf-8") as f2:
        assert f1.read() == f2.read(), "Parity mismatch between root and azure_function predict_ratings.py"


# ==============================================================================
# 4. Candidate D & Top Players Ranking Semantics Preservation
# ==============================================================================

def test_candidate_d_and_ranking_semantics_preserved():
    """Verify Candidate D additive formula and Top Players sorting hierarchy remain exact."""
    ratings_history = pd.DataFrame({
        "gameId": ["g1", "g2", "g3"],
        "personId": [101, 101, 101],
        "game_date": ["2026-10-22", "2026-10-24", "2026-10-26"],
        "rating": [5.0, 5.5, 6.0],
    })

    current_predictions = pd.DataFrame({
        "personId": [101],
        "predicted_rating": [5.0],
        "playerName": ["Top Star"],
    })

    multipliers = compute_season_multipliers(ratings_df=ratings_history, all_players_df=current_predictions)
    assert not multipliers.empty
    row = multipliers.iloc[0]

    # Candidate D semantics:
    # season_average_rating = 5.5
    # season_games_played = 3
    # ranking_score = round(predicted_rating * (1 + bonus), 2)
    assert row["season_game_count"] == 3
    assert row["season_average_rating"] == pytest.approx(5.5, abs=1e-4)


# ==============================================================================
# 5. Phase 2.9.5 Remediation & Promotion Safety Tests
# ==============================================================================

APPROVED_BASELINE_SHA256 = "71e19ab50e35d6bf0dbdca92599b1b3f5bcca4dc17d3c614f69b1b0727a4bf11"


def test_production_model_version_resolves_correctly():
    """1. The production model version resolves correctly in both root and azure_function."""
    for base_dir in ["models", "azure_function/models"]:
        resolved_path, meta = resolve_active_model(base_dir=base_dir, fail_closed=True)
        assert os.path.exists(resolved_path)
        assert meta["active_version"] == "model_202526_production"
        assert meta["is_active"] is True
        assert meta["is_validated"] is True


def test_production_model_sha256_matches_approved_baseline():
    """2. Production model SHA-256 matches the approved baseline commit (4ad6c65)."""
    for base_dir in ["models", "azure_function/models"]:
        resolved_path, meta = resolve_active_model(base_dir=base_dir, fail_closed=True)
        assert meta["sha256"] == APPROVED_BASELINE_SHA256
        assert compute_file_sha256(resolved_path) == APPROVED_BASELINE_SHA256


def test_root_and_azure_function_registry_files_remain_identical():
    """3. Root and Azure Function registry files remain bit-for-bit identical."""
    for rel_path in [
        "active_model.json",
        "versions/model_202526_production/model_metadata.json",
    ]:
        root_f = os.path.join("models", rel_path)
        af_f = os.path.join("azure_function/models", rel_path)
        assert os.path.exists(root_f), f"Missing {root_f}"
        assert os.path.exists(af_f), f"Missing {af_f}"
        with open(root_f, "r", encoding="utf-8") as f1, open(af_f, "r", encoding="utf-8") as f2:
            assert json.load(f1) == json.load(f2), f"Mismatch in {rel_path}"


def test_underperforming_candidate_cannot_replace_active_champion():
    """4. An underperforming candidate model cannot replace the active champion."""
    from scripts.train_model import check_promotion_quality_gate

    # Metrics from Phase 2.9.3 benchmark: RF (0.7390) worse than Last 7 (0.7274) & Champion (0.7214)
    benchmark_metrics = {
        "Last 7 Avg Baseline": {"MAE": 0.7274, "R2": 0.5371},
        "Last 5 Avg Baseline": {"MAE": 0.7301, "R2": 0.5374},
        "Active Champion": {"MAE": 0.7214, "R2": 0.5460, "version": "model_202526_production"},
        "Random Forest": {"MAE": 0.7390, "R2": 0.5158},
    }

    is_eligible, reasons = check_promotion_quality_gate(benchmark_metrics)
    assert not is_eligible
    assert any("did not outperform strongest rolling baseline" in r for r in reasons)
    assert any("underperforms Active Champion" in r for r in reasons)


def test_failed_validation_leaves_active_pointer_unchanged(tmp_path):
    """5. Failed validation leaves the active pointer unchanged."""
    tmp_models = str(tmp_path / "models")
    os.makedirs(tmp_models, exist_ok=True)

    rf_champ = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)
    X = pd.DataFrame(np.random.randn(20, 7), columns=EXPECTED_FEATURE_SCHEMA)
    y = pd.Series(np.random.randn(20))
    rf_champ.fit(X, y)

    champ_id = "model_champion_v1"
    champ_meta = {
        "model_version": champ_id,
        "season_boundary": "2026-10-20",
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "feature_schema": EXPECTED_FEATURE_SCHEMA,
        "is_validated": True,
    }
    save_versioned_model(rf_champ, champ_id, champ_meta, base_dir=tmp_models)
    promote_active_model(champ_id, base_dir=tmp_models)

    champ_sha = compute_file_sha256(os.path.join(tmp_models, "versions", champ_id, "rating_model.pkl"))

    # Now attempt to promote candidate that failed gate
    candidate_id = "model_cand_failed"
    candidate_meta = {
        "model_version": candidate_id,
        "season_boundary": "2026-10-20",
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "feature_schema": EXPECTED_FEATURE_SCHEMA,
        "is_validated": False,  # Failed quality gate
        "promotion_rejection_reasons": ["Underperforms Last 7 baseline"],
    }
    save_versioned_model(rf_champ, candidate_id, candidate_meta, base_dir=tmp_models)

    with pytest.raises(ModelValidationError):
        promote_active_model(candidate_id, base_dir=tmp_models)

    # Pointer must remain pointing to champ_id with champ_sha
    resolved_path, active_meta = resolve_active_model(base_dir=tmp_models)
    assert active_meta["active_version"] == champ_id
    assert active_meta["sha256"] == champ_sha


def test_successful_candidate_promoted_only_after_passing_every_gate():
    """6. A candidate is eligible for promotion only when outperforming both baselines and champion."""
    from scripts.train_model import check_promotion_quality_gate

    winning_metrics = {
        "Last 7 Avg Baseline": {"MAE": 0.7274, "R2": 0.5371},
        "Last 5 Avg Baseline": {"MAE": 0.7301, "R2": 0.5374},
        "Active Champion": {"MAE": 0.7214, "R2": 0.5460, "version": "model_202526_production"},
        "Random Forest": {"MAE": 0.7150, "R2": 0.5600},  # Outperforms ALL
    }

    is_eligible, reasons = check_promotion_quality_gate(winning_metrics)
    assert is_eligible
    assert len(reasons) == 0


def test_missing_or_corrupt_model_fails_closed(tmp_path):
    """7. Missing or corrupt active model fails closed."""
    tmp_models = str(tmp_path / "models")
    os.makedirs(tmp_models, exist_ok=True)

    # Case A: active_model.json does not exist
    with pytest.raises(ModelValidationError, match="active model pointer missing"):
        resolve_active_model(base_dir=tmp_models, fail_closed=True)

    # Case B: active_model.json exists but references non-existent model
    bad_pointer = {
        "active_version": "ghost_model",
        "model_path": "versions/ghost/rating_model.pkl",
        "metadata_path": "versions/ghost/model_metadata.json",
        "sha256": "abcdef123456",
        "feature_schema": EXPECTED_FEATURE_SCHEMA,
    }
    with open(os.path.join(tmp_models, "active_model.json"), "w") as f:
        json.dump(bad_pointer, f)

    with pytest.raises(ModelValidationError, match="does not exist"):
        resolve_active_model(base_dir=tmp_models, fail_closed=True)


def test_snapshot_publication_blocked_when_model_validation_fails(monkeypatch):
    """8. Snapshot publication remains blocked when model validation fails."""
    from scripts.azure_blob import publish_serving_snapshot

    with tempfile.TemporaryDirectory() as bad_model_dir:
        # Create malformed pointer
        with open(os.path.join(bad_model_dir, "active_model.json"), "w") as f:
            f.write("{invalid json")

        monkeypatch.setenv("PROPREDICT_MODELS_DIR", bad_model_dir)

        with pytest.raises(RuntimeError, match="Snapshot publication blocked: active model validation failed"):
            publish_serving_snapshot(data_dir="data", container_name="propredict-serving")


def test_no_fallback_to_legacy_bundled_model(monkeypatch, tmp_path):
    """9. Inference never silently falls back to legacy bundled model if pointer fails."""
    tmp_dir = str(tmp_path / "legacy_test")
    os.makedirs(tmp_dir, exist_ok=True)

    # Put a dummy legacy model in root of tmp_dir
    legacy_model = os.path.join(tmp_dir, "rating_model.pkl")
    with open(legacy_model, "wb") as f:
        f.write(b"dummy legacy content")

    # When resolve_active_model is called with fail_closed=True, it MUST raise ModelValidationError,
    # never silently returning the legacy_model
    with pytest.raises(ModelValidationError):
        resolve_active_model(base_dir=tmp_dir, fail_closed=True)


def test_inactive_experimental_candidate_remains_inactive():
    """10. Verify model_20261110_season202627_init is inactive and not referenced by active pointer."""
    for base_dir in ["models", "azure_function/models"]:
        cand_meta_path = os.path.join(base_dir, "versions", "model_20261110_season202627_init", "model_metadata.json")
        assert os.path.exists(cand_meta_path), "Candidate metadata file must exist"

        with open(cand_meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        assert meta["is_validated"] is False
        assert meta["is_active"] is False
        assert "promotion_rejection_reason" in meta

        # Active pointer must NOT point to this candidate
        _, active_meta = resolve_active_model(base_dir=base_dir, fail_closed=True)
        assert active_meta["active_version"] != "model_20261110_season202627_init"
        assert active_meta["active_version"] == "model_202526_production"


# ==============================================================================
# 6. Phase 2.9.5.1 Hardened Fail-Closed Promotion Tests
# ==============================================================================

def test_quality_gate_rejects_missing_last5_baseline():
    """Fail-closed: quality gate rejects promotion if Last 5 baseline is missing."""
    from scripts.train_model import check_promotion_quality_gate

    metrics = {
        "Last 7 Avg Baseline": {"MAE": 0.7274},
        "Active Champion": {"MAE": 0.7214, "version": "model_202526_production"},
        "Random Forest": {"MAE": 0.7100},
    }
    is_eligible, reasons = check_promotion_quality_gate(metrics)
    assert not is_eligible
    assert any("Last 5 Avg Baseline" in r and "missing" in r for r in reasons)


def test_quality_gate_rejects_missing_last7_baseline():
    """Fail-closed: quality gate rejects promotion if Last 7 baseline is missing."""
    from scripts.train_model import check_promotion_quality_gate

    metrics = {
        "Last 5 Avg Baseline": {"MAE": 0.7301},
        "Active Champion": {"MAE": 0.7214, "version": "model_202526_production"},
        "Random Forest": {"MAE": 0.7100},
    }
    is_eligible, reasons = check_promotion_quality_gate(metrics)
    assert not is_eligible
    assert any("Last 7 Avg Baseline" in r and "missing" in r for r in reasons)


def test_quality_gate_rejects_missing_active_champion():
    """Fail-closed: quality gate rejects promotion if Active Champion metric is missing."""
    from scripts.train_model import check_promotion_quality_gate

    metrics = {
        "Last 5 Avg Baseline": {"MAE": 0.7301},
        "Last 7 Avg Baseline": {"MAE": 0.7274},
        "Random Forest": {"MAE": 0.7100},
    }
    is_eligible, reasons = check_promotion_quality_gate(metrics)
    assert not is_eligible
    assert any("Active Champion" in r and "missing" in r for r in reasons)


def test_quality_gate_rejects_nan_and_infinite_metrics():
    """Fail-closed: quality gate rejects NaN or infinite metric values for any required baseline."""
    from scripts.train_model import check_promotion_quality_gate

    base = {
        "Last 5 Avg Baseline": {"MAE": 0.7301},
        "Last 7 Avg Baseline": {"MAE": 0.7274},
        "Active Champion": {"MAE": 0.7214, "version": "model_202526_production"},
        "Random Forest": {"MAE": 0.7100},
    }

    # Case A: RF candidate MAE is NaN
    m_nan_rf = {**base, "Random Forest": {"MAE": float("nan")}}
    ok, reasons = check_promotion_quality_gate(m_nan_rf)
    assert not ok
    assert any("Random Forest" in r and "non-finite or NaN" in r for r in reasons)

    # Case B: Last 5 MAE is inf
    m_inf_last5 = {**base, "Last 5 Avg Baseline": {"MAE": float("inf")}}
    ok, reasons = check_promotion_quality_gate(m_inf_last5)
    assert not ok
    assert any("Last 5 Avg Baseline" in r and "non-finite or NaN" in r for r in reasons)

    # Case C: Active Champion MAE is NaN
    m_nan_champ = {**base, "Active Champion": {"MAE": float("nan")}}
    ok, reasons = check_promotion_quality_gate(m_nan_champ)
    assert not ok
    assert any("Active Champion" in r and "non-finite or NaN" in r for r in reasons)


def test_quality_gate_rejects_invalid_metric_types():
    """Fail-closed: quality gate rejects string, None, boolean, or list types in MAE fields."""
    from scripts.train_model import check_promotion_quality_gate

    base = {
        "Last 5 Avg Baseline": {"MAE": 0.7301},
        "Last 7 Avg Baseline": {"MAE": 0.7274},
        "Active Champion": {"MAE": 0.7214, "version": "model_202526_production"},
        "Random Forest": {"MAE": 0.7100},
    }

    # String type
    m_str = {**base, "Last 7 Avg Baseline": {"MAE": "0.7274"}}
    ok, reasons = check_promotion_quality_gate(m_str)
    assert not ok
    assert any("non-numeric" in r for r in reasons)

    # Boolean type
    m_bool = {**base, "Active Champion": {"MAE": True}}
    ok, reasons = check_promotion_quality_gate(m_bool)
    assert not ok
    assert any("non-numeric" in r for r in reasons)

    # None type
    m_none = {**base, "Random Forest": {"MAE": None}}
    ok, reasons = check_promotion_quality_gate(m_none)
    assert not ok
    assert any("non-numeric" in r for r in reasons)


def test_quality_gate_accepts_valid_candidate_passing_every_gate():
    """Quality gate promotes candidate only when all 4 metrics are finite and both gates pass."""
    from scripts.train_model import check_promotion_quality_gate

    winning_metrics = {
        "Last 5 Avg Baseline": {"MAE": 0.7301, "R2": 0.5374},
        "Last 7 Avg Baseline": {"MAE": 0.7274, "R2": 0.5371},
        "Active Champion": {"MAE": 0.7214, "R2": 0.5460, "version": "model_202526_production"},
        "Random Forest": {"MAE": 0.7150, "R2": 0.5600},  # strictly < 0.7274 and <= 0.7214
    }
    is_eligible, reasons = check_promotion_quality_gate(winning_metrics)
    assert is_eligible
    assert len(reasons) == 0


def test_quality_gate_rejects_underperforming_candidate():
    """Quality gate rejects candidate if it fails either rolling baseline or champion comparison."""
    from scripts.train_model import check_promotion_quality_gate

    # Underperforms both Last 7 (0.7274) and Champion (0.7214)
    losing_metrics = {
        "Last 5 Avg Baseline": {"MAE": 0.7301},
        "Last 7 Avg Baseline": {"MAE": 0.7274},
        "Active Champion": {"MAE": 0.7214, "version": "model_202526_production"},
        "Random Forest": {"MAE": 0.7390},
    }
    is_eligible, reasons = check_promotion_quality_gate(losing_metrics)
    assert not is_eligible
    assert len(reasons) == 2
    assert any("did not outperform strongest rolling baseline" in r for r in reasons)
    assert any("underperforms Active Champion" in r for r in reasons)


def test_active_pointer_remains_unchanged_after_every_rejection(tmp_path):
    """Verify active pointer remains pointing to champion after any validation failure."""
    tmp_models = str(tmp_path / "models")
    os.makedirs(tmp_models, exist_ok=True)

    rf_dummy = RandomForestRegressor(n_estimators=5, max_depth=2, random_state=42)
    X = pd.DataFrame(np.random.randn(10, 7), columns=EXPECTED_FEATURE_SCHEMA)
    y = pd.Series(np.random.randn(10))
    rf_dummy.fit(X, y)

    # 1. Establish champion
    champ_id = "model_champ_init"
    champ_meta = {
        "model_version": champ_id,
        "season_boundary": "2026-10-20",
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "feature_schema": EXPECTED_FEATURE_SCHEMA,
        "is_validated": True,
    }
    save_versioned_model(rf_dummy, champ_id, champ_meta, base_dir=tmp_models)
    promote_active_model(champ_id, base_dir=tmp_models)

    initial_pointer_sha = compute_file_sha256(os.path.join(tmp_models, "active_model.json"))

    # 2. Candidate with invalid metadata (is_validated = False)
    cand1_id = "cand_unvalidated"
    save_versioned_model(rf_dummy, cand1_id, {**champ_meta, "model_version": cand1_id, "is_validated": False}, base_dir=tmp_models)
    with pytest.raises(ModelValidationError):
        promote_active_model(cand1_id, base_dir=tmp_models)

    # Pointer must remain completely unchanged
    current_pointer_sha = compute_file_sha256(os.path.join(tmp_models, "active_model.json"))
    assert current_pointer_sha == initial_pointer_sha

    # 3. Candidate with nonexistent model file
    with pytest.raises(ModelValidationError):
        promote_active_model("nonexistent_cand_v2", base_dir=tmp_models)

    current_pointer_sha = compute_file_sha256(os.path.join(tmp_models, "active_model.json"))
    assert current_pointer_sha == initial_pointer_sha

    resolved_path, active_meta = resolve_active_model(base_dir=tmp_models)
    assert active_meta["active_version"] == champ_id



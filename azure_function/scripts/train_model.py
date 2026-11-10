import pandas as pd
import numpy as np
import os
import sys
import json
import joblib
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Tuple, List
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

try:
    from scripts.model_registry import (
        generate_version_id,
        save_versioned_model,
        promote_active_model,
        ModelValidationError,
        EXPECTED_FEATURE_SCHEMA,
    )
except ImportError:
    from model_registry import (
        generate_version_id,
        save_versioned_model,
        promote_active_model,
        ModelValidationError,
        EXPECTED_FEATURE_SCHEMA,
    )


def get_calendar_date_cutoff(df: pd.DataFrame, target_train_ratio: float = 0.8) -> pd.Timestamp:
    """
    Finds a calendar-date cutoff such that:
      - Train set: game_date < cutoff_date
      - Test set: game_date >= cutoff_date
    No calendar date is split between train and test.
    The cutoff is chosen to get the train ratio as close as possible to target_train_ratio.
    """
    unique_dates = sorted(df["game_date"].unique())
    total_rows = len(df)

    best_cutoff = unique_dates[0]
    best_diff = float("inf")

    for d in unique_dates[1:]:
        train_count = (df["game_date"] < d).sum()
        ratio = train_count / total_rows
        diff = abs(ratio - target_train_ratio)
        if diff < best_diff:
            best_diff = diff
            best_cutoff = d

    return pd.to_datetime(best_cutoff)


def evaluate_baselines(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    rf_model: RandomForestRegressor,
    model_dir: str = "models",
) -> dict:
    """
    Evaluates reproducible baseline methods on the test set:
      1. Mean baseline (predict training mean)
      2. Median baseline (predict training median)
      3. Last game baseline (last_game)
      4. Last 3 average baseline (last3_avg)
      5. Last 5 average baseline (last5_avg)
      6. Last 7 average baseline (last7_avg)
      7. Linear Regression
      8. Random Forest (trained candidate)
      9. Active Champion (if registered champion exists)
    """
    results = {}

    # 1. Mean Baseline
    p_mean = np.full_like(y_test, y_train.mean())
    results["Mean Baseline"] = {
        "MAE": float(mean_absolute_error(y_test, p_mean)),
        "R2": float(r2_score(y_test, p_mean))
    }

    # 2. Median Baseline
    p_median = np.full_like(y_test, y_train.median())
    results["Median Baseline"] = {
        "MAE": float(mean_absolute_error(y_test, p_median)),
        "R2": float(r2_score(y_test, p_median))
    }

    # 3. Last Game Baseline
    results["Last Game Baseline"] = {
        "MAE": float(mean_absolute_error(y_test, X_test["last_game"])),
        "R2": float(r2_score(y_test, X_test["last_game"]))
    }

    # 4. Last 3 Avg Baseline
    results["Last 3 Baseline"] = {
        "MAE": float(mean_absolute_error(y_test, X_test["last3_avg"])),
        "R2": float(r2_score(y_test, X_test["last3_avg"]))
    }
    results["Last 3 Avg Baseline"] = results["Last 3 Baseline"]

    # 5. Last 5 Avg Baseline
    results["Last 5 Baseline"] = {
        "MAE": float(mean_absolute_error(y_test, X_test["last5_avg"])),
        "R2": float(r2_score(y_test, X_test["last5_avg"]))
    }
    results["Last 5 Avg Baseline"] = results["Last 5 Baseline"]

    # 6. Last 7 Avg Baseline
    results["Last 7 Baseline"] = {
        "MAE": float(mean_absolute_error(y_test, X_test["last7_avg"])),
        "R2": float(r2_score(y_test, X_test["last7_avg"]))
    }
    results["Last 7 Avg Baseline"] = results["Last 7 Baseline"]

    # 7. Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    p_lr = lr.predict(X_test)
    results["Linear Regression"] = {
        "MAE": float(mean_absolute_error(y_test, p_lr)),
        "R2": float(r2_score(y_test, p_lr))
    }

    # 8. Random Forest (Candidate)
    p_rf = rf_model.predict(X_test)
    results["Random Forest"] = {
        "MAE": float(mean_absolute_error(y_test, p_rf)),
        "R2": float(r2_score(y_test, p_rf))
    }

    # 9. Active Champion Model (if available)
    try:
        from scripts.model_registry import resolve_active_model
    except ImportError:
        from model_registry import resolve_active_model

    try:
        champ_path, champ_meta = resolve_active_model(base_dir=model_dir, fail_closed=False)
        if champ_path and os.path.exists(champ_path):
            champ_model = joblib.load(champ_path)
            p_champ = champ_model.predict(X_test)
            results["Active Champion"] = {
                "MAE": float(mean_absolute_error(y_test, p_champ)),
                "R2": float(r2_score(y_test, p_champ)),
                "version": champ_meta.get("active_version", "unknown"),
            }
    except Exception as e:
        print(f"Active Champion evaluation skipped or unavailable: {e}")

    return results


def _extract_finite_mae(metrics: dict, key: str) -> Tuple[Optional[float], Optional[str]]:
    """Helper to safely extract a finite numeric MAE, returning (mae_value, error_reason)."""
    if not isinstance(metrics, dict) or key not in metrics:
        return None, f"Required metric '{key}' is missing from evaluation benchmark"

    val_dict = metrics[key]
    if not isinstance(val_dict, dict) or "MAE" not in val_dict:
        return None, f"Required metric '{key}' is missing an 'MAE' entry"

    mae = val_dict["MAE"]
    if isinstance(mae, bool) or not isinstance(mae, (int, float)):
        return None, f"Metric '{key}' MAE is non-numeric (type: {type(mae).__name__})"

    mae_float = float(mae)
    if not np.isfinite(mae_float):
        return None, f"Metric '{key}' MAE is non-finite or NaN ({mae})"

    return mae_float, None


def check_promotion_quality_gate(
    baseline_metrics: dict,
) -> Tuple[bool, list]:
    """
    Evaluates whether a candidate model qualifies for promotion.

    Enforced Promotion Criteria:
      1. Candidate Random Forest finite MAE < 3.0 (sanity check).
      2. Valid, finite Last 5 Avg Baseline MAE.
      3. Valid, finite Last 7 Avg Baseline MAE.
      4. Valid, finite Active Champion MAE.
      5. Baseline superiority: Candidate MAE must strictly outperform the strongest
         applicable rolling baseline (Last 5 Avg and Last 7 Avg):
           rf_mae < min(last5_mae, last7_mae)
         If a complex ML model cannot beat a simple 5-game or 7-game rolling average
         on the holdout set, it is statistically unjustified to deploy it.
      6. Champion-Challenger superiority: Candidate must not underperform the active champion
         on the same test set:
           rf_mae <= champion_mae

    Diagnostic Benchmark Metrics (Non-Gating):
      - Mean Baseline, Median Baseline, Last Game Baseline, Last 3 Avg Baseline, Linear Regression.
      These provide analytical context in benchmark logs but are not gating constraints.

    Fail-Closed Policy:
      If ANY of the 4 required metrics (Random Forest, Last 5, Last 7, Active Champion)
      is missing, non-numeric, NaN, or non-finite, promotion is rejected immediately.
      No comparison is ever silently skipped.

    Returns:
      (is_passed, rejection_reasons)
    """
    reasons = []

    # 1. Validate Random Forest candidate
    rf_mae, rf_err = _extract_finite_mae(baseline_metrics, "Random Forest")
    if rf_err:
        reasons.append(rf_err)
    elif rf_mae > 3.0 or rf_mae < 0.0:
        reasons.append(f"Candidate Random Forest MAE ({rf_mae:.4f}) exceeds sanity threshold of 3.0")

    # 2. Validate Last 5 Avg Baseline
    last5_mae, last5_err = _extract_finite_mae(baseline_metrics, "Last 5 Avg Baseline")
    if last5_err:
        reasons.append(last5_err)

    # 3. Validate Last 7 Avg Baseline
    last7_mae, last7_err = _extract_finite_mae(baseline_metrics, "Last 7 Avg Baseline")
    if last7_err:
        reasons.append(last7_err)

    # 4. Validate Active Champion
    champ_mae, champ_err = _extract_finite_mae(baseline_metrics, "Active Champion")
    if champ_err:
        reasons.append(champ_err)

    # If any required metric failed extraction/validation, reject fail-closed
    if reasons:
        return False, reasons

    # 5. Baseline superiority: candidate must strictly outperform the strongest rolling baseline
    rolling_baselines = [
        ("Last 5 Avg Baseline", last5_mae),
        ("Last 7 Avg Baseline", last7_mae),
    ]
    best_rolling_name, best_rolling_mae = min(rolling_baselines, key=lambda x: x[1])
    if rf_mae >= best_rolling_mae:
        reasons.append(
            f"Candidate RF MAE ({rf_mae:.4f}) did not outperform strongest rolling baseline "
            f"{best_rolling_name} ({best_rolling_mae:.4f})"
        )

    # 6. Champion comparison: candidate must outperform or match the active champion
    champ_ver = baseline_metrics["Active Champion"].get("version", "champion")
    if rf_mae > champ_mae:
        reasons.append(
            f"Candidate RF MAE ({rf_mae:.4f}) underperforms Active Champion '{champ_ver}' ({champ_mae:.4f})"
        )

    is_passed = len(reasons) == 0
    return is_passed, reasons


def train_model(
    input_path: str = "data/player_features.parquet",
    model_path: str = "models/rating_model.pkl",
    target_train_ratio: float = 0.8,
    version_id: Optional[str] = None,
    promote: bool = True,
) -> str:

    # ==============================
    # INPUT CHECK
    # ==============================

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Missing input file: {input_path}")

    # ==============================
    # LOAD DATA
    # ==============================

    df = pd.read_parquet(input_path)

    print("Dataset loaded")
    print("Rows:", len(df))

    if df.empty:
        raise Exception("Dataset is empty")

    # ==============================
    # SORT BY TIME (CRITICAL)
    # ==============================

    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values("game_date")

    # ==============================
    # FEATURES / TARGET
    # ==============================

    features = list(EXPECTED_FEATURE_SCHEMA)

    # ==============================
    # CLEAN CALENDAR-DATE SPLIT
    # ==============================

    cutoff_date = get_calendar_date_cutoff(df, target_train_ratio=target_train_ratio)

    train_df = df[df["game_date"] < cutoff_date]
    test_df = df[df["game_date"] >= cutoff_date]

    X_train = train_df[features].copy()
    y_train = train_df["rating"]

    X_test = test_df[features].copy()
    y_test = test_df["rating"]

    train_start = train_df["game_date"].min().strftime("%Y-%m-%d")
    train_end = train_df["game_date"].max().strftime("%Y-%m-%d")
    test_start = test_df["game_date"].min().strftime("%Y-%m-%d")
    test_end = test_df["game_date"].max().strftime("%Y-%m-%d")

    print("\n--- Temporal Evaluation Split ---")
    print(f"Cutoff Date: {cutoff_date.strftime('%Y-%m-%d')}")
    print(f"Training Rows: {len(X_train)} ({len(X_train)/len(df):.1%}) | Dates: {train_start} to {train_end}")
    print(f"Testing Rows:  {len(X_test)} ({len(X_test)/len(df):.1%}) | Dates: {test_start} to {test_end}")

    # Verify no date overlap
    overlap = set(train_df["game_date"]).intersection(set(test_df["game_date"]))
    if overlap:
        raise ValueError(f"Date overlap detected between train and test: {overlap}")

    print("\n--- Feature Completeness Check ---")
    print(f"X_train nulls: {X_train.isna().sum().to_dict()}")
    print(f"X_test nulls:  {X_test.isna().sum().to_dict()}")

    # ==============================
    # MODEL TRAINING (PRESERVED CONFIG)
    # ==============================

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    )

    print("\nTraining Random Forest model...")
    model.fit(X_train, y_train)

    model_dir = os.path.dirname(model_path) if os.path.dirname(model_path) else "models"
    os.makedirs(model_dir, exist_ok=True)

    # ==============================
    # REPRODUCIBLE BASELINE EVALUATION
    # ==============================

    baseline_metrics = evaluate_baselines(X_train, y_train, X_test, y_test, model, model_dir=model_dir)

    print("\n==================================================")
    print("BASELINE & MODEL EVALUATION BENCHMARK")
    print("==================================================")
    print(f"{'Method':<25} | {'MAE':>8} | {'R2':>8}")
    print("-" * 47)
    for method, scores in baseline_metrics.items():
        print(f"{method:<25} | {scores['MAE']:>8.4f} | {scores['R2']:>8.4f}")
    print("==================================================\n")

    # ==============================
    # QUALITY VALIDATION GATE
    # ==============================
    is_promotion_eligible, rejection_reasons = check_promotion_quality_gate(baseline_metrics)

    if is_promotion_eligible:
        print("Promotion Quality Gate: PASSED. Candidate outperforms baselines and active champion.")
    else:
        print("Promotion Quality Gate: REJECTED.")
        for r in rejection_reasons:
            print(f"   - {r}")

    # ==============================
    # VERSIONED MODEL PERSISTENCE
    # ==============================
    v_id = version_id or generate_version_id()

    std3_val = None
    if "std_3" in train_df.columns:
        valid_std3 = train_df["std_3"].dropna()
        if not valid_std3.empty:
            std3_val = float(valid_std3.median())

    meta = {
        "model_version": v_id,
        "season_boundary": "2026-10-20",
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "training_date_range": {"start": train_start, "end": train_end},
        "test_date_range": {"start": test_start, "end": test_end},
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "total_feature_rows": int(len(df)),
        "unique_players": int(df["personId"].nunique()) if "personId" in df.columns else int(len(df)),
        "feature_schema": features,
        "hyperparameters": {
            "n_estimators": 200,
            "max_depth": 8,
            "random_state": 42
        },
        "validation_metrics": baseline_metrics,
        "is_validated": is_promotion_eligible,
        "promotion_eligible": is_promotion_eligible,
        "promotion_rejection_reasons": rejection_reasons if not is_promotion_eligible else [],
        "std3_median": std3_val,
    }

    saved_model_path, metadata_path = save_versioned_model(
        model=model,
        version_id=v_id,
        metadata=meta,
        base_dir=model_dir,
    )
    print(f"Versioned model saved to {saved_model_path} (validated: {is_promotion_eligible})")
    print(f"Model metadata saved to {metadata_path}")

    # ==============================
    # ATOMIC ACTIVE POINTER PROMOTION
    # ==============================
    if promote:
        if not is_promotion_eligible:
            err_msg = (
                f"Promotion blocked: Candidate model {v_id} failed quality gate:\n  "
                + "\n  ".join(rejection_reasons)
            )
            print(f"\nActive pointer left UNCHANGED. Previous champion preserved.\n{err_msg}")
            raise ModelValidationError(err_msg)

        active_pointer = promote_active_model(
            version_id=v_id,
            base_dir=model_dir,
            update_legacy_path=True,
        )
        print(f"Active model pointer updated: {active_pointer} -> {v_id}")

    return saved_model_path


# ==============================
# RUN
# ==============================

if __name__ == "__main__":
    train_model()

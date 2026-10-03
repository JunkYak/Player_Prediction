import pandas as pd
import numpy as np
import os
import sys
import json
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


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
    rf_model: RandomForestRegressor
) -> dict:
    """
    Evaluates 8 reproducible baseline methods on the test set:
      1. Mean baseline (predict training mean)
      2. Median baseline (predict training median)
      3. Last game baseline (last_game)
      4. Last 3 average baseline (last3_avg)
      5. Last 5 average baseline (last5_avg)
      6. Last 7 average baseline (last7_avg)
      7. Linear Regression
      8. Random Forest (trained)
    """
    results = {}

    # 1. Mean Baseline
    p_mean = np.full_like(y_test, y_train.mean())
    results["Mean Baseline"] = {
        "MAE": mean_absolute_error(y_test, p_mean),
        "R2": r2_score(y_test, p_mean)
    }

    # 2. Median Baseline
    p_median = np.full_like(y_test, y_train.median())
    results["Median Baseline"] = {
        "MAE": mean_absolute_error(y_test, p_median),
        "R2": r2_score(y_test, p_median)
    }

    # 3. Last Game Baseline
    results["Last Game Baseline"] = {
        "MAE": mean_absolute_error(y_test, X_test["last_game"]),
        "R2": r2_score(y_test, X_test["last_game"])
    }

    # 4. Last 3 Baseline
    results["Last 3 Baseline"] = {
        "MAE": mean_absolute_error(y_test, X_test["last3_avg"]),
        "R2": r2_score(y_test, X_test["last3_avg"])
    }

    # 5. Last 5 Baseline
    results["Last 5 Baseline"] = {
        "MAE": mean_absolute_error(y_test, X_test["last5_avg"]),
        "R2": r2_score(y_test, X_test["last5_avg"])
    }

    # 6. Last 7 Baseline
    results["Last 7 Baseline"] = {
        "MAE": mean_absolute_error(y_test, X_test["last7_avg"]),
        "R2": r2_score(y_test, X_test["last7_avg"])
    }

    # 7. Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    p_lr = lr.predict(X_test)
    results["Linear Regression"] = {
        "MAE": mean_absolute_error(y_test, p_lr),
        "R2": r2_score(y_test, p_lr)
    }

    # 8. Random Forest
    p_rf = rf_model.predict(X_test)
    results["Random Forest"] = {
        "MAE": mean_absolute_error(y_test, p_rf),
        "R2": r2_score(y_test, p_rf)
    }

    return results


def train_model(
    input_path: str = "data/player_features.parquet",
    model_path: str = "models/rating_model.pkl",
    target_train_ratio: float = 0.8
):

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

    features = [
        "last_game",
        "last3_avg",
        "last5_avg",
        "last7_avg",
        "games_played_before_target",
        "expand_mean",
        "days_since_prev_game"
    ]

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

    # ==============================
    # REPRODUCIBLE BASELINE EVALUATION
    # ==============================

    baseline_metrics = evaluate_baselines(X_train, y_train, X_test, y_test, model)

    print("\n==================================================")
    print("BASELINE & MODEL EVALUATION BENCHMARK")
    print("==================================================")
    print(f"{'Method':<25} | {'MAE':>8} | {'R2':>8}")
    print("-" * 47)
    for method, scores in baseline_metrics.items():
        print(f"{method:<25} | {scores['MAE']:>8.4f} | {scores['R2']:>8.4f}")
    print("==================================================\n")

    # ==============================
    # SAVE MODEL
    # ==============================

    model_dir = os.path.dirname(model_path) if os.path.dirname(model_path) else "."
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    return model_path


# ==============================
# RUN
# ==============================

if __name__ == "__main__":
    train_model()
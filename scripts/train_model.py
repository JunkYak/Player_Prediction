import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib


def train_model():

    input_path = "data/player_features.parquet"
    model_path = "models/rating_model.pkl"

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
        "last7_avg"
    ]

    X = df[features]
    y = df["rating"]

    # ==============================
    # TIME-BASED SPLIT
    # ==============================

    split_idx = int(len(df) * 0.8)

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]

    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    print("Training rows:", len(X_train))
    print("Test rows:", len(X_test))

    # ==============================
    # MODEL
    # ==============================

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    )

    print("Training model...")

    model.fit(X_train, y_train)

    # ==============================
    # EVALUATION
    # ==============================

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print("\nModel Performance")
    print("MAE:", round(mae, 3))
    print("R2:", round(r2, 3))

    # ==============================
    # SAVE MODEL
    # ==============================

    os.makedirs("models", exist_ok=True)

    joblib.dump(model, model_path)

    print(f"\nModel saved to {model_path}")

    # ==============================
    # SAMPLE PREDICTIONS
    # ==============================

    sample = X_test.copy()
    sample["actual"] = y_test
    sample["predicted"] = preds

    print("\nSample predictions:")
    print(sample.head(10))

    return model_path


# ==============================
# RUN
# ==============================

if __name__ == "__main__":
    train_model()
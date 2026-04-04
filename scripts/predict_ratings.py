import pandas as pd
import os
import joblib


def predict_ratings():

    model_path = "models/rating_model.pkl"
    input_path = "data/player_features.parquet"
    output_path = "data/predicted_ratings.parquet"

    # ==============================
    # CHECK FILES
    # ==============================

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model: {model_path}")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Missing input file: {input_path}")

    # ==============================
    # LOAD MODEL
    # ==============================

    model = joblib.load(model_path)
    print("Model loaded")

    # ==============================
    # LOAD DATA
    # ==============================

    df = pd.read_parquet(input_path)

    print("Dataset loaded:", df.shape)

    if df.empty:
        raise Exception("Feature dataset is empty")

    # ==============================
    # SAFETY CHECK
    # ==============================

    if "game_date" not in df.columns:
        raise ValueError("game_date column missing — check pipeline order")

    df["game_date"] = pd.to_datetime(df["game_date"])

    # ==============================
    # GET MOST RECENT GAME PER PLAYER
    # ==============================

    latest = (
        df.sort_values("game_date")
        .groupby("personId")
        .tail(1)
        .copy()
    )

    print("Players considered:", len(latest))

    # ==============================
    # FEATURES
    # ==============================

    features = [
        "last_game",
        "last3_avg",
        "last5_avg",
        "last7_avg"
    ]

    X = latest[features]

    # ==============================
    # PREDICT
    # ==============================

    latest["predicted_rating"] = model.predict(X)

    # ==============================
    # SORT
    # ==============================

    latest = latest.sort_values(
        "predicted_rating",
        ascending=False
    )

    # ==============================
    # OUTPUT
    # ==============================

    print("\nTop Predicted Performers\n")

    print(
        latest[
            [
                "playerName",
                "teamId",
                "gameId",
                "game_date",
                "predicted_rating",
                "last_game"
            ]
        ].head(25)
    )

    # ==============================
    # SAVE
    # ==============================

    os.makedirs("data", exist_ok=True)

    latest.to_parquet(output_path, index=False)

    print(f"\nSaved predictions → {output_path}")

    return output_path


# ==============================
# RUN
# ==============================

if __name__ == "__main__":
    predict_ratings()
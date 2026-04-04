import pandas as pd
import os


def build_features():

    input_path = "data/player_game_ratings_with_dates.parquet"
    output_path = "data/player_features.parquet"

    # ==============================
    # INPUT CHECK
    # ==============================

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Missing input file: {input_path}")

    # ==============================
    # LOAD DATA
    # ==============================

    df = pd.read_parquet(input_path)

    print("Loaded ratings with dates:", df.shape)

    if df.empty:
        raise Exception("Input dataset is empty")

    # ==============================
    # FIX DATE TYPE
    # ==============================

    df["game_date"] = pd.to_datetime(df["game_date"])

    # ==============================
    # SORT CORRECTLY (CRITICAL)
    # ==============================

    df = df.sort_values(["personId", "game_date"])

    # ==============================
    # FEATURE CREATION
    # ==============================

    # previous game rating
    df["last_game"] = df.groupby("personId")["rating"].shift(1)

    # rolling averages
    df["last3_avg"] = (
        df.groupby("personId")["rating"]
        .rolling(3)
        .mean()
        .shift(1)
        .reset_index(level=0, drop=True)
    )

    df["last5_avg"] = (
        df.groupby("personId")["rating"]
        .rolling(5)
        .mean()
        .shift(1)
        .reset_index(level=0, drop=True)
    )

    df["last7_avg"] = (
        df.groupby("personId")["rating"]
        .rolling(7)
        .mean()
        .shift(1)
        .reset_index(level=0, drop=True)
    )

    # ==============================
    # REMOVE ROWS WITHOUT HISTORY
    # ==============================

    features = df.dropna()

    if features.empty:
        raise Exception("No rows left after feature engineering")

    # ==============================
    # SAVE
    # ==============================

    os.makedirs("data", exist_ok=True)

    features.to_parquet(output_path, index=False)

    print("Saved player_features.parquet")
    print("Rows:", len(features))

    print("\nSample:")
    print(features.head())

    return output_path


# ==============================
# RUN
# ==============================

if __name__ == "__main__":
    build_features()
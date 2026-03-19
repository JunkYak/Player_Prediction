import pandas as pd
import joblib

# ==============================
# LOAD MODEL
# ==============================

model = joblib.load("models/rating_model.pkl")
print("Model loaded")

# ==============================
# LOAD PLAYER FEATURES (WITH DATES)
# ==============================

df = pd.read_parquet("data/player_features_with_dates.parquet")

print("Dataset loaded:", df.shape)
print("Columns:", df.columns)

# ==============================
# SAFETY CHECK (IMPORTANT)
# ==============================

if "game_date" not in df.columns:
    raise ValueError("game_date column missing — run add_game_dates.py first")

# ==============================
# GET MOST RECENT GAME PER PLAYER
# ==============================

latest = (
    df
    .sort_values("game_date")
    .groupby("personId")
    .tail(1)
)

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
# PREDICT RATINGS
# ==============================

latest["predicted_rating"] = model.predict(X)

# ==============================
# SORT BEST PLAYERS
# ==============================

latest = latest.sort_values(
    "predicted_rating",
    ascending=False
)

# ==============================
# OUTPUT
# ==============================

print("\nTop Predicted Performers (with last game date)\n")

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
# SAVE RESULTS
# ==============================

latest.to_parquet(
    "data/predicted_ratings.parquet",
    index=False
)

print("\nSaved predictions → data/predicted_ratings.parquet")
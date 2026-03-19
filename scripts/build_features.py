import pandas as pd

# ==============================
# LOAD RATINGS
# ==============================

df = pd.read_parquet("data/player_game_ratings.parquet")

# ensure chronological order
df = df.sort_values(["personId", "gameId"])

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

# ==============================
# SAVE DATASET
# ==============================

features.to_parquet(
    "data/player_features.parquet",
    index=False
)

print("Saved player_features.parquet")
print("Rows:", len(features))
print("\nSample:")
print(features.head())
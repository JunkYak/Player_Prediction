import pandas as pd
from nba_api.stats.endpoints import leaguegamelog

# ==============================
# LOAD PLAYER FEATURES
# ==============================

df = pd.read_parquet("data/player_features.parquet")
print("Loaded features:", df.shape)

# ==============================
# FETCH GAME LOG (WITH DATES)
# ==============================

print("Fetching game logs...")

games = leaguegamelog.LeagueGameLog().get_data_frames()[0]

# Keep only needed columns
games = games[["GAME_ID", "GAME_DATE"]].drop_duplicates()

# Rename to match your dataset
games.columns = ["gameId", "game_date"]

print("Games fetched:", games.shape)

# ==============================
# MERGE
# ==============================

df = df.merge(games, on="gameId", how="left")

# ==============================
# DEBUG CHECK
# ==============================

print("\nMissing dates:", df["game_date"].isna().sum())

print("\nSample:")
print(df[["gameId", "game_date"]].head())

# ==============================
# SAVE
# ==============================

df.to_parquet("data/player_features_with_dates.parquet", index=False)

print("\nSaved successfully.")
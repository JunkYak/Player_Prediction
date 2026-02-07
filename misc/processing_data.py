import pandas as pd
import re

# ======================================
# LOAD DATASET
# ======================================

df = pd.read_parquet("nba_last_7_days.parquet")

print("Loaded dataset")
print("Rows:", df.shape[0])
print("Games:", df["gameId"].nunique())

# ======================================
# CLOCK → SECONDS
# (same logic used in formula script)
# ======================================

def clock_to_seconds(clock):
    match = re.search(r"PT(\d+)M([\d\.]+)S", str(clock))
    if match:
        minutes = int(match.group(1))
        seconds = float(match.group(2))
        return minutes * 60 + seconds
    return None


df["seconds_remaining"] = df["clock"].apply(clock_to_seconds)

# ======================================
# SCORE PLACEHOLDER
# (dataset doesn't contain score fields)
# ======================================

df["score_diff"] = 0
df["score_diff_before"] = 0

# ======================================
# IMPORT RATING ENGINE
# ======================================

from test1 import base_delta, context_delta, clutch_multiplier

# ======================================
# APPLY RATING ENGINE
# ======================================

print("Applying rating formula...")

df["base_delta"] = df.apply(base_delta, axis=1)
df["context_delta"] = df.apply(context_delta, axis=1)
df["clutch_mult"] = df.apply(clutch_multiplier, axis=1)

df["rating"] = (
    (df["base_delta"] + df["context_delta"])
    * df["clutch_mult"]
)


df = df[
    (df["personId"] > 1000) &
    (df["personId"] < 1610612700) &
    (df["teamId"] > 0)
]
# ======================================
# PLAYER GAME RATINGS
# ======================================

player_game_ratings = (
    df
    .groupby(["gameId", "teamId", "personId"])["rating"]
    .sum()
    .reset_index()
)

player_game_ratings = player_game_ratings.sort_values(
    ["gameId", "rating"],
    ascending=[True, False]
)

# ======================================
# SAVE OUTPUT
# ======================================

player_game_ratings.to_parquet(
    "player_game_ratings.parquet",
    index=False
)

print("\nSaved player_game_ratings.parquet")
print("Shape:", player_game_ratings.shape)

print("\nSample:")
print(player_game_ratings.head(20))
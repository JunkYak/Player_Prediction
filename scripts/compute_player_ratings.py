import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rating_engine.formula import clock_to_seconds,base_delta,context_delta,clutch_multiplier


# ==============================
# LOAD DATASET
# ==============================

df = pd.read_parquet("data/play_by_play.parquet")

print("Loaded play-by-play dataset")
print("Rows:", len(df))
print("Games:", df["gameId"].nunique())

# ==============================
# SCORE PROCESSING
# ==============================

df["scoreHome"] = pd.to_numeric(df["scoreHome"], errors="coerce").ffill().fillna(0)
df["scoreAway"] = pd.to_numeric(df["scoreAway"], errors="coerce").ffill().fillna(0)

df["score_diff"] = df["scoreHome"] - df["scoreAway"]
df["score_diff_before"] = df.groupby("gameId")["score_diff"].shift(1).fillna(0)

# ==============================
# CLOCK → SECONDS
# ==============================

df["seconds_remaining"] = df["clock"].apply(clock_to_seconds)

# ==============================
# APPLY RATING ENGINE
# ==============================

print("Computing base deltas...")

df["base_delta"] = df.apply(base_delta, axis=1)

print("Computing context deltas...")

df["context_delta"] = df.apply(context_delta, axis=1)

print("Computing clutch multipliers...")

df["clutch_mult"] = df.apply(clutch_multiplier, axis=1)

# ==============================
# TOTAL RATING
# ==============================

df["rating"] = (
    (df["base_delta"] + df["context_delta"])
    * df["clutch_mult"]
)

# ==============================
# CLEAN DATA
# ==============================

df = df[
    (df["personId"] > 1000) &
    (df["teamId"] > 0) &
    (df["playerName"].notna()) &
    (df["playerName"] != "")
]

# ==============================
# PLAYER GAME RATINGS
# ==============================

player_ratings = (
    df.groupby(
        ["gameId", "teamId", "personId", "playerName"],
        as_index=False
    )["rating"]
    .sum()
)

# ==============================
# SORT OUTPUT
# ==============================

player_ratings = player_ratings.sort_values(
    ["gameId", "teamId", "rating"],
    ascending=[True, True, False]
)

# ==============================
# SAVE DATASET
# ==============================

player_ratings.to_parquet(
    "data/player_game_ratings.parquet",
    index=False
)

print("\nSaved player_game_ratings.parquet")
print("Rows:", len(player_ratings))

print("\nSample:")
print(player_ratings.head(20))
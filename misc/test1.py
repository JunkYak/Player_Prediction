import pandas as pd
import numpy as np
import re
from nba_api.stats.endpoints import playbyplayv3

# ==============================
# CONFIG
# ==============================

GAME_ID = "0022500974"

# ==============================
# LOAD DATA
# ==============================

pbp = playbyplayv3.PlayByPlayV3(game_id=GAME_ID)
pbp_df = pbp.get_data_frames()[0]

# ==============================
# SCORE PROCESSING
# ==============================

pbp_df["scoreHome"] = pd.to_numeric(pbp_df["scoreHome"], errors="coerce").ffill().fillna(0)
pbp_df["scoreAway"] = pd.to_numeric(pbp_df["scoreAway"], errors="coerce").ffill().fillna(0)

pbp_df["score_diff"] = pbp_df["scoreHome"] - pbp_df["scoreAway"]
pbp_df["score_diff_before"] = pbp_df["score_diff"].shift(1).fillna(0)

# ==============================
# CLOCK → SECONDS
# ==============================

def clock_to_seconds(clock):
    match = re.search(r"PT(\d+)M([\d\.]+)S", str(clock))
    if match:
        minutes = int(match.group(1))
        seconds = float(match.group(2))
        return minutes * 60 + seconds
    return None

pbp_df["seconds_remaining"] = pbp_df["clock"].apply(clock_to_seconds)

# ==============================
# BASE DELTA MODEL
# ==============================

def base_delta(row):

    desc = str(row.get("description", "")).lower()
    action = str(row.get("actionType", ""))
    shot_value = row.get("shotValue", 0)

    if pd.isna(shot_value):
        shot_value = 0

    distance = row.get("shotDistance", 0)
    if pd.isna(distance):
        distance = 0

    if "sub:" in desc or "jump ball" in desc:
        return 0

    if "turnover" in desc:
        return -0.17

    if "steal" in desc and "block" not in desc:
        return 0.19

    if "block" in desc:
        return 0.18

    if "rebound" in desc:

        off = re.search(r"off:(\d+)", desc)
        deff = re.search(r"def:(\d+)", desc)

        if off and int(off.group(1)) > 0:
            return 0.07

        if deff and int(deff.group(1)) > 0:
            return 0.05

        return 0.05

    if "free throw" in desc:
        if "miss" in desc:
            return 0
        return 0.09

    if action == "Missed Shot":

        if "putback" in desc:
            return 0.12

        if "layup" in desc or "drive" in desc:
            return 0.10

        if shot_value == 3 and distance >= 27:
            return 0.12

        return 0.07

    if action == "Made Shot":

        if shot_value == 2:
            attempt_value = 0.15
        elif shot_value == 3:
            attempt_value = 0.18
        else:
            attempt_value = 0

        is_three = int(shot_value == 3)
        is_layup = int("layup" in desc)
        is_dunk = int("dunk" in desc)
        is_floater = int("floater" in desc)
        is_hook = int("hook" in desc)
        is_jump = int("jump" in desc and shot_value != 3)
        is_putback = int("putback" in desc)

        is_assisted = "ast" in desc
        assist_modifier = -0.04 if is_assisted else 0.03

        difficulty = 0

        if "fadeaway" in desc:
            difficulty += 0.05

        if "turnaround" in desc:
            difficulty += 0.04

        if "step back" in desc or "stepback" in desc:
            difficulty += 0.06

        if "pullup" in desc:
            difficulty += 0.03

        short_3 = int(is_three and distance <= 24)
        deep_3 = int(is_three and distance >= 27)

        residual = (
            0.194 * is_three +
            0.130 * is_floater +
            0.130 * is_hook +
            0.120 * is_jump +
            0.100 * is_layup +
            0.087 * is_dunk -
            0.030 * is_putback +
            0.010 * short_3 -
            0.050 * deep_3
        )

        return attempt_value + residual + difficulty + assist_modifier

    return 0


pbp_df["base_delta"] = pbp_df.apply(base_delta, axis=1)

# ==============================
# CONTEXT BONUS
# ==============================

def context_delta(row):

    desc = str(row.get("description", "")).lower()
    before = row["score_diff_before"]
    after = row["score_diff"]

    if row["actionType"] != "Made Shot":
        return 0

    if "game-winner" in desc:
        return 0.8

    if "game-tying" in desc:
        return 0.07

    if "lead-taking" in desc:
        return 0.05

    if after == 0 and before != 0:
        return 0.07

    if before <= 0 and after > 0:
        return 0.05

    return 0


pbp_df["context_delta"] = pbp_df.apply(context_delta, axis=1)

# ==============================
# CLUTCH MULTIPLIER
# ==============================

def clutch_multiplier(row):

    desc = str(row.get("description","")).lower()

    if "free throw" in desc:
        return 1.0

    if row["actionType"] != "Made Shot":
        return 1.0

    if row["period"] != 4:
        return 1.0

    seconds = row["seconds_remaining"]
    if seconds is None:
        return 1.0

    if seconds <= 120:
        return 1 + ((120 - seconds) / 120) * 2

    return 1.0


pbp_df["clutch_mult"] = pbp_df.apply(clutch_multiplier, axis=1)

# ==============================
# TOTAL DELTA
# ==============================

pbp_df["rating"] = (
    (pbp_df["base_delta"] + pbp_df["context_delta"])
    * pbp_df["clutch_mult"]
)

# ==============================
# CLEAN DATA
# ==============================

pbp_df = pbp_df[
    (pbp_df["personId"] > 1000) &           # remove system IDs
    (pbp_df["teamId"] > 0) &                # remove invalid teams
    (pbp_df["playerName"].notna()) &
    (pbp_df["playerName"] != "")
]

# ==============================
# PLAYER GAME RATINGS
# ==============================

player_ratings = (
    pbp_df
    .groupby(["teamId", "personId", "playerName"], as_index=False)["rating"]
    .sum()
)

player_ratings["game_id"] = GAME_ID

# sort cleanly
player_ratings = player_ratings.sort_values(
    ["teamId", "rating"], 
    ascending=[True, False]
)

print("\n--- PLAYER RATINGS FOR GAME ---\n")
print(player_ratings.to_string(index=False))
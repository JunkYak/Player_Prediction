import pandas as pd
import numpy as np
import re
from nba_api.stats.endpoints import playbyplayv3

# ==============================
# CONFIG
# ==============================

GAME_ID = "0022500711"
PLAYER_NAME = "Jokić"

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
# CLOCK TO SECONDS
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
# CONTINUOUS BASE DELTA MODEL
# ==============================

def base_delta(row):

    desc = str(row["description"]).lower()
    action = row["actionType"]
    shot_value = row["shotValue"]
    distance = row["shotDistance"] if pd.notna(row["shotDistance"]) else 0

    # -----------------------------
    # TURNOVERS
    # -----------------------------
    if "turnover" in desc:
        return -0.17

    # -----------------------------
    # STEALS
    # -----------------------------
    if "steal" in desc:
        return 0.17

    # -----------------------------
    # FREE THROWS
    # -----------------------------
    if "free throw" in desc:
        if "miss" in desc:
            return 0
        return 0.09

    # -----------------------------
    # MISSED SHOTS
    # -----------------------------
    if action == "Missed Shot":
        return 0.07

    # -----------------------------
    # MADE SHOTS
    # -----------------------------
    if action == "Made Shot":

        # ---- Attempt value ----
        if shot_value == 2:
            attempt_value = 0.15
        elif shot_value == 3:
            attempt_value = 0.18
        else:
            attempt_value = 0

        # ---- Feature flags ----
        is_three = int(shot_value == 3)
        is_layup = int("layup" in desc)
        is_dunk = int("dunk" in desc)
        is_floater = int("floater" in desc)
        is_hook = int("hook" in desc)
        is_jump = int("jump" in desc and shot_value != 3)
        is_putback = int("putback" in desc)

        # ---- Distance buckets ----
        short_3 = int(is_three and distance <= 24)
        deep_3 = int(is_three and distance >= 27)

        # ---- Learned residual model ----
        residual = (
            0.194 * is_three +
            0.130 * is_floater +
            0.130 * is_hook +
            0.120 * is_jump +
            0.100 * is_layup +
            0.087 * is_dunk +
            0.040 * is_putback +
            0.010 * short_3 -
            0.050 * deep_3
        )

        return attempt_value + residual

    return 0



pbp_df["base_delta"] = pbp_df.apply(base_delta, axis=1)

# ==============================
# CONTEXT (LEAD / TIE)
# ==============================

def context_delta(row):
    before = row["score_diff_before"]
    after = row["score_diff"]

    # Only for made shots
    if row["actionType"] != "Made Shot":
        return 0

    # Game tying
    if after == 0 and before != 0:
        return 0.07

    # Lead taking
    if before <= 0 and after > 0:
        return 0.12

    return 0


pbp_df["context_delta"] = pbp_df.apply(context_delta, axis=1)

# ==============================
# CLUTCH SCALING
# ==============================

def clutch_delta(row):
    if row["period"] != 4:
        return 0

    seconds = row["seconds_remaining"]
    if seconds is None:
        return 0

    # last 2 minutes only
    if seconds <= 120:
        # scale 0 → 0.25
        return (120 - seconds) / 120 * 0.25

    return 0


pbp_df["clutch_delta"] = pbp_df.apply(clutch_delta, axis=1)

# ==============================
# TOTAL DELTA
# ==============================

pbp_df["total_delta"] = (
    pbp_df["base_delta"]
    + pbp_df["context_delta"]
    + pbp_df["clutch_delta"]
)

# ==============================
# PLAYER FILTER
# ==============================

player_df = pbp_df[pbp_df["playerName"] == PLAYER_NAME]

print("\n--- PLAY BY PLAY DELTA ---\n")
print(player_df[[
    "period",
    "clock",
    "playerName",
    "description",
    "base_delta",
    "context_delta",
    "clutch_delta",
    "total_delta"
]])

print("\n--- TOTAL PERFORMANCE ---\n")
print(round(player_df["total_delta"].sum(), 3))

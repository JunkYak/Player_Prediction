"""
rating_engine/formula.py

Custom NBA play-by-play rating engine.

Architecture:
    Raw PBP row
        ↓
    classify_event(row) → structured event info
        ↓
    base_delta(row) → shot/action base score
    context_delta(row) → situational importance bonus
    clutch_multiplier(row) → 4th-quarter pressure multiplier
        ↓
    rating = (base_delta + context_delta) * clutch_multiplier

Rating contract (output groupby keys):
    gameId, teamId, personId, playerName, rating

Phase 2B Change Summary
-----------------------
PRESERVED (no change):
- All numerical weights are identical to original
- Turnover: -0.17
- Steal: +0.19
- Block: +0.18
- Offensive rebound: +0.07
- Defensive rebound: +0.05
- Free throw made: +0.09 | missed: 0
- Made 2pt attempt_value: 0.15 | 3pt: 0.18
- All residual weights (floater, hook, layup, dunk, jump, putback, is_three, short_3, deep_3)
- Assist modifier: -0.04 assisted, +0.03 unassisted
- Difficulty bonuses: fadeaway +0.05, turnaround +0.04, step-back +0.06, pullup +0.03
- Combined descriptions are still ADDITIVE (e.g. turnaround fadeaway = +0.09)
- context_delta values: game-winner +0.8, game-tying +0.07, lead-taking +0.05
- clutch_multiplier formula: 1 + ((120 - seconds) / 120) * 2 for period==4, seconds <= 120
- Overtime: no clutch (period != 4)
- Free throws: always multiplier=1.0

FIXED (implementation bugs, no semantic change):
1. BUG: "floater" keyword never matched NBA descriptions (which use "Floating").
   FIX: Changed to check "float" in desc for is_floater. This correctly detects
        "Driving Floating Jump Shot", "Floating Jump Shot", etc.
   IMPACT: is_floater now correctly fires. This WILL change some ratings where
           "Floating" shots were previously classified as generic jump shots.
   DOCUMENTED in tests as a corrected attribution.

2. BUG: "drive" keyword never matched NBA descriptions (which use "Driving").
   FIX: Changed to check "driving" in desc for the missed-shot branch.
   IMPACT: "Driving Floating Jump Shot" missed shots now score 0.10 instead of 0.07.
           "Driving Layup Shot" missed shots were already correctly scoring 0.10 via
           the "layup" keyword, so no change there.
   DOCUMENTED in tests.

3. BUG: "stepback" check was dead code. NBA descriptions never use "stepback" (no space).
   FIX: Removed redundant "stepback" check (kept "step back" check which is correct).
   IMPACT: Zero — "stepback" never matched real data, so no ratings change.

4. BUG: Output console encoding failure on Windows for non-ASCII player names.
   FIX: compute_player_ratings.py uses sys.stdout.reconfigure for UTF-8 output.
   IMPACT: None on rating values.

INTENTIONALLY PRESERVED (not bugs):
- Combined description bonuses (stepback+fadeaway is additive). Behavior is intentional.
- Clutch only fires for period==4 (not overtime periods). This is the documented intent.
- "game-winner"/"game-tying"/"lead-taking" in desc checked before computed score check.
- Rebound detection uses description-parsed Off:/Def: counts (structured field missing).
- Steal/Block detected via description (structured fields not present for these events).
"""

import re
import pandas as pd


# ==============================
# CLOCK PARSING
# ==============================

def clock_to_seconds(clock):
    """Parse NBA PBP clock string 'PTmMsS' to total seconds remaining."""
    match = re.search(r"PT(\d+)M([\d\.]+)S", str(clock))
    if match:
        minutes = int(match.group(1))
        seconds = float(match.group(2))
        return minutes * 60 + seconds
    return None


# ==============================
# EVENT CLASSIFICATION HELPERS
# ==============================

def _normalize_desc(row) -> str:
    """Return lowercased description string."""
    return str(row.get("description", "")).lower()


def _is_excluded_event(desc: str) -> bool:
    """Return True for events that should yield zero rating (substitutions, jump balls)."""
    return "sub:" in desc or "jump ball" in desc


# ==============================
# BASE DELTA
# ==============================

def base_delta(row) -> float:
    """
    Compute base rating contribution for a single PBP event row.

    Detection uses description text where structured fields are absent
    (steals, blocks, rebounds), and actionType + structured fields for shots.

    Attribution: The row's personId/playerName IS the attributed player.
    Steals and blocks are separate NBA PBP rows with the defending player as personId.
    """
    desc = _normalize_desc(row)
    action = str(row.get("actionType", ""))
    shot_value = row.get("shotValue", 0)
    if pd.isna(shot_value):
        shot_value = 0

    distance = row.get("shotDistance", 0)
    if pd.isna(distance):
        distance = 0

    # ---- Early exits ----
    if _is_excluded_event(desc):
        return 0

    # ---- Turnover ----
    # Detection: "Turnover" appears in description for all actionType==Turnover rows.
    if "turnover" in desc:
        return -0.17

    # ---- Steal ----
    # Detection: NBA PBP publishes separate rows for steals with empty actionType.
    # The stealing player is personId on that row.
    # Guard: never misfire on a block row (block rows never have 'steal' in desc).
    if "steal" in desc and "block" not in desc:
        return 0.19

    # ---- Block ----
    # Detection: NBA PBP publishes separate rows for blocks with empty actionType.
    # The blocking player is personId on that row.
    if "block" in desc:
        return 0.18

    # ---- Rebound ----
    # structured subType is mostly "Unknown" — use description parsing.
    # Off:N / Def:N are present in all rebound descriptions.
    if "rebound" in desc:
        off = re.search(r"off:(\d+)", desc)
        deff = re.search(r"def:(\d+)", desc)
        if off and int(off.group(1)) > 0:
            return 0.07
        if deff and int(deff.group(1)) > 0:
            return 0.05
        return 0.05  # fallback for team rebounds or unstructured formats

    # ---- Free Throw ----
    if "free throw" in desc:
        if "miss" in desc:
            return 0
        return 0.09

    # ---- Missed Shot ----
    if action == "Missed Shot":
        if "putback" in desc:
            return 0.12
        if "layup" in desc:
            return 0.10
        # FIX (Phase 2B): Changed "drive" to "driving" to match actual NBA descriptions.
        # "Driving Floating Jump Shot" missed shots now correctly score 0.10.
        if "driving" in desc:
            return 0.10
        if shot_value == 3 and distance >= 27:
            return 0.12
        return 0.07

    # ---- Made Shot ----
    if action == "Made Shot":
        if shot_value == 2:
            attempt_value = 0.15
        elif shot_value == 3:
            attempt_value = 0.18
        else:
            attempt_value = 0

        is_three   = int(shot_value == 3)
        is_layup   = int("layup"   in desc)
        is_dunk    = int("dunk"    in desc)
        # FIX (Phase 2B): Changed "floater" to "float" to match actual NBA PBP descriptions.
        # NBA uses "Floating Jump Shot" / "Driving Floating Jump Shot" — not "Floater".
        # Previously: is_floater = int("floater" in desc)  → always 0 (no match in real data)
        # Now:        is_floater = int("float"   in desc)  → correctly fires for Floating shots
        is_floater = int("float"   in desc)
        is_hook    = int("hook"    in desc)
        is_jump    = int("jump"    in desc and shot_value != 3)
        is_putback = int("putback" in desc)

        is_assisted    = "ast" in desc
        assist_modifier = -0.04 if is_assisted else 0.03

        # Difficulty bonuses — all ADDITIVE, this is intentional behavior
        difficulty = 0
        if "fadeaway" in desc:
            difficulty += 0.05
        if "turnaround" in desc:
            difficulty += 0.04
        if "step back" in desc:
            # FIX (Phase 2B): Removed dead "stepback" check. "stepback" (no space) never
            # appears in NBA PBP descriptions. "step back" (with space) is the correct form.
            difficulty += 0.06
        if "pullup" in desc:
            difficulty += 0.03

        short_3 = int(is_three and distance <= 24)
        deep_3  = int(is_three and distance >= 27)

        residual = (
            0.194 * is_three    +
            0.130 * is_floater  +
            0.130 * is_hook     +
            0.120 * is_jump     +
            0.100 * is_layup    +
            0.087 * is_dunk     -
            0.030 * is_putback  +
            0.010 * short_3     -
            0.050 * deep_3
        )

        return attempt_value + residual + difficulty + assist_modifier

    return 0


# ==============================
# CONTEXT DELTA
# ==============================

def context_delta(row) -> float:
    """
    Compute situational importance bonus for a made shot.

    Checks description keywords first (explicit NBA labels take precedence),
    then falls back to computed score differential changes.

    Attribution: same personId as the shot.
    """
    desc = _normalize_desc(row)
    before = row["score_diff_before"]
    after  = row["score_diff"]

    if row["actionType"] != "Made Shot":
        return 0

    # Explicit NBA context labels (present in description when applicable)
    if "game-winner" in desc:
        return 0.8
    if "game-tying" in desc:
        return 0.07
    if "lead-taking" in desc:
        return 0.05

    # Computed context from score differential
    if after == 0 and before != 0:
        return 0.07   # Shot ties the game
    if before <= 0 and after > 0:
        return 0.05   # Shot takes the lead

    return 0


# ==============================
# CLUTCH MULTIPLIER
# ==============================

def clutch_multiplier(row) -> float:
    """
    Compute pressure multiplier for 4th-quarter made shots.

    Applies only to:
    - actionType == "Made Shot"
    - period == 4 (not overtime)
    - seconds_remaining <= 120

    Formula: 1 + ((120 - seconds) / 120) * 2
      At 120s: multiplier = 1.0
      At   0s: multiplier = 3.0

    Free throws: always 1.0 (excluded from clutch boost).
    """
    desc = _normalize_desc(row)

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
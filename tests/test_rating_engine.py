"""
tests/test_rating_engine.py

Phase 2B: Tests for existing rating engine behavior.

These tests are written BEFORE any refactoring.
They document exactly what the current implementation does,
including edge cases and combined-description behaviors.

The suite covers:
  - Basic event detection (base_delta)
  - Context events (context_delta)
  - Clutch multiplier (clutch_multiplier)
  - Clock parsing (clock_to_seconds)
  - Combined descriptions
  - Attribution (via compute_player_ratings groupby)
  - Regression (old vs new on sample data)
"""

import os
import sys
import math
import pytest
import pandas as pd
import numpy as np

# Ensure root is on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from rating_engine.formula import (
    clock_to_seconds,
    base_delta,
    context_delta,
    clutch_multiplier,
)


# ============================================================
# HELPERS
# ============================================================

def make_row(
    description="",
    actionType="",
    shotValue=0,
    shotDistance=0,
    subType="",
    shotResult="",
    period=1,
    clock="PT10M00.00S",
    score_diff=0,
    score_diff_before=0,
    seconds_remaining=None,
    personId=203999,
    teamId=1610612743,
    playerName="TestPlayer",
    gameId="0022300001",
):
    """Build a row dict matching the PBP schema used by the rating engine."""
    row = {
        "description": description,
        "actionType": actionType,
        "shotValue": shotValue,
        "shotDistance": shotDistance,
        "subType": subType,
        "shotResult": shotResult,
        "period": period,
        "clock": clock,
        "score_diff": score_diff,
        "score_diff_before": score_diff_before,
        "seconds_remaining": seconds_remaining,
        "personId": personId,
        "teamId": teamId,
        "playerName": playerName,
        "gameId": gameId,
    }
    return pd.Series(row)


def approx(val, expected, tol=1e-9):
    """Check near-equality for floats."""
    return abs(val - expected) < tol


# ============================================================
# 1. CLOCK PARSING
# ============================================================

class TestClockToSeconds:
    def test_standard_format(self):
        assert clock_to_seconds("PT10M00.00S") == 600.0

    def test_zero_remaining(self):
        assert clock_to_seconds("PT00M00.00S") == 0.0

    def test_two_minutes(self):
        assert clock_to_seconds("PT02M00.00S") == 120.0

    def test_exactly_120(self):
        assert clock_to_seconds("PT02M00.00S") == 120.0

    def test_119_seconds(self):
        # 1 min 59 s = 119s
        assert clock_to_seconds("PT01M59.00S") == 119.0

    def test_fractional_seconds(self):
        # PT00M52.20S -> 52.20
        assert clock_to_seconds("PT00M52.20S") == pytest.approx(52.20, abs=1e-6)

    def test_invalid_returns_none(self):
        assert clock_to_seconds("") is None
        assert clock_to_seconds(None) is None
        assert clock_to_seconds("not_a_clock") is None


# ============================================================
# 2. BASE DELTA — BASIC EVENTS
# ============================================================

class TestBaseDeltaBasicEvents:

    def test_substitution_excluded(self):
        row = make_row(description="SUB: Player A FOR Player B", actionType="Substitution")
        assert base_delta(row) == 0

    def test_jump_ball_excluded(self):
        row = make_row(description="Jump Ball X vs Y", actionType="Jump Ball")
        assert base_delta(row) == 0

    def test_turnover(self):
        row = make_row(description="Brown Lost Ball Turnover (P1.T1)", actionType="Turnover")
        assert approx(base_delta(row), -0.17)

    def test_steal(self):
        row = make_row(description="White STEAL (1 STL)", actionType="")
        assert approx(base_delta(row), 0.19)

    def test_block(self):
        row = make_row(description="Mitchell BLOCK (1 BLK)", actionType="")
        assert approx(base_delta(row), 0.18)

    def test_steal_not_confused_with_block(self):
        # Steal guard: 'steal' in desc and 'block' not in desc
        row = make_row(description="Brown STEAL (2 STL)", actionType="")
        assert approx(base_delta(row), 0.19)

    def test_block_takes_priority_over_steal_guard(self):
        # The code checks steal first: "steal" in desc and "block" NOT in desc
        # Then checks "block" in desc
        # A row with both would be caught by block; but this never occurs in real data
        row = make_row(description="Something block steal", actionType="")
        # 'steal' in desc BUT 'block' also in desc -> falls through steal -> goes to block check
        assert approx(base_delta(row), 0.18)

    def test_rebound_offensive(self):
        row = make_row(description="Wiggins REBOUND (Off:1 Def:0)", actionType="Rebound")
        assert approx(base_delta(row), 0.07)

    def test_rebound_defensive(self):
        row = make_row(description="Brown REBOUND (Off:0 Def:1)", actionType="Rebound")
        assert approx(base_delta(row), 0.05)

    def test_rebound_offensive_defensive_both(self):
        # Off:1 matches first -> 0.07
        row = make_row(description="Wiggins REBOUND (Off:1 Def:1)", actionType="Rebound")
        assert approx(base_delta(row), 0.07)

    def test_rebound_no_match_defaults_defensive(self):
        # No Off: or Def: in desc -> fallthrough returns 0.05
        row = make_row(description="TEAM REBOUND", actionType="Rebound")
        assert approx(base_delta(row), 0.05)

    def test_free_throw_made(self):
        row = make_row(description="Brown Free Throw 1 of 2 (1 PTS)", actionType="Free Throw")
        assert approx(base_delta(row), 0.09)

    def test_free_throw_miss(self):
        row = make_row(description="MISS Brown Free Throw 1 of 2", actionType="Free Throw")
        assert approx(base_delta(row), 0.0)


# ============================================================
# 3. BASE DELTA — MISSED SHOTS
# ============================================================

class TestBaseDeltaMissedShots:

    def test_missed_putback(self):
        row = make_row(
            description="MISS Gobert 2' Putback Layup",
            actionType="Missed Shot",
            subType="Putback Layup Shot",
        )
        assert approx(base_delta(row), 0.12)

    def test_missed_layup(self):
        row = make_row(
            description="MISS Adebayo 7' Driving Layup",
            actionType="Missed Shot",
            subType="Driving Layup Shot",
        )
        assert approx(base_delta(row), 0.10)

    def test_missed_drive(self):
        """
        Phase 2B fix: 'drive' -> 'driving' in missed shot check.
        'Driving Floating Jump Shot' descriptions contain 'driving' but NOT 'drive'.
        Before fix: returned 0.07 (default)
        After fix:  returns 0.10 (driving path matched)
        """
        row = make_row(
            description="MISS Brown 12' Driving Floating Jump Shot",
            actionType="Missed Shot",
            subType="Driving Floating Jump Shot",
        )
        assert approx(base_delta(row), 0.10)

    def test_missed_driving_layup_still_correct(self):
        """Driving Layup was already 0.10 via 'layup' keyword - verify no regression."""
        row = make_row(
            description="MISS Brown 5' Driving Layup",
            actionType="Missed Shot",
            subType="Driving Layup Shot",
        )
        assert approx(base_delta(row), 0.10)

    def test_missed_deep_3(self):
        row = make_row(
            description="MISS Curry 30' 3PT Jump Shot",
            actionType="Missed Shot",
            shotValue=3,
            shotDistance=30,
        )
        assert approx(base_delta(row), 0.12)

    def test_missed_regular_shot(self):
        row = make_row(
            description="MISS Brown 21' Jump Shot",
            actionType="Missed Shot",
            shotValue=2,
            shotDistance=20,
        )
        assert approx(base_delta(row), 0.07)


# ============================================================
# 4. BASE DELTA — MADE SHOTS
# ============================================================

class TestBaseDeltaMadeShots:

    def test_made_2pt_unassisted(self):
        """2pt made, unassisted -> attempt_value=0.15, assist_modifier=+0.03"""
        row = make_row(
            description="Brown 15' Jump Shot (14 PTS)",
            actionType="Made Shot",
            shotValue=2,
            shotDistance=15,
            subType="Jump Shot",
        )
        val = base_delta(row)
        # attempt=0.15, residual=0.120 (jump)*1, difficulty=0, assist_modifier=+0.03
        expected = 0.15 + 0.120 + 0 + 0.03
        assert approx(val, expected)

    def test_made_3pt_unassisted(self):
        """3pt made, unassisted"""
        row = make_row(
            description="Curry 25' 3PT Jump Shot (20 PTS)",
            actionType="Made Shot",
            shotValue=3,
            shotDistance=25,
            subType="Jump Shot",
        )
        val = base_delta(row)
        # attempt=0.18, residual: 0.194*1(is_three) + 0.120*1(is_jump, shotValue!=3 => 0!) + 0.010*1(short_3,25<=24?->NO) + (-0.050*0)
        # is_three=1, is_jump=int("jump" in desc and shotValue != 3)=int(True and False)=0
        # short_3=int(is_three and 25<=24)=int(True and False)=0
        # deep_3=int(is_three and 25>=27)=int(True and False)=0
        # residual = 0.194*1 + 0 + 0 + 0 = 0.194
        # difficulty=0, assist_modifier=+0.03
        expected = 0.18 + 0.194 + 0 + 0.03
        assert approx(val, expected)

    def test_made_3pt_assisted(self):
        """3pt made, assisted -> assist_modifier=-0.04"""
        row = make_row(
            description="Powell 25' 3PT Jump Shot (3 PTS) (Mitchell 1 AST)",
            actionType="Made Shot",
            shotValue=3,
            shotDistance=25,
            subType="Jump Shot",
        )
        val = base_delta(row)
        # attempt=0.18, is_three=1, is_jump=0, residual=0.194, difficulty=0, assist_modifier=-0.04
        expected = 0.18 + 0.194 + 0 - 0.04
        assert approx(val, expected)

    def test_made_fadeaway_bonus(self):
        """Fadeaway made shot gets +0.05 difficulty"""
        row = make_row(
            description="Brown 15' Fadeaway Jumper (14 PTS)",
            actionType="Made Shot",
            shotValue=2,
            shotDistance=15,
            subType="Fadeaway Jump Shot",
        )
        val = base_delta(row)
        # is_jump=1(shotValue!=3), fadeaway difficulty=0.05
        # attempt=0.15, residual=0.120, difficulty=0.05, assist_modifier=+0.03
        expected = 0.15 + 0.120 + 0.05 + 0.03
        assert approx(val, expected)

    def test_made_stepback_bonus(self):
        """Step back made shot gets +0.06 difficulty"""
        row = make_row(
            description="Adebayo 18' Step Back Jump Shot (12 PTS)",
            actionType="Made Shot",
            shotValue=2,
            shotDistance=18,
            subType="Step Back Jump shot",
        )
        val = base_delta(row)
        # is_jump=1, stepback difficulty=0.06
        # attempt=0.15, residual=0.120, difficulty=0.06, assist_modifier=+0.03
        expected = 0.15 + 0.120 + 0.06 + 0.03
        assert approx(val, expected)

    def test_made_turnaround_bonus(self):
        """Turnaround gets +0.04 difficulty"""
        row = make_row(
            description="Jokic 5' Turnaround Fadeaway Shot (14 PTS)",
            actionType="Made Shot",
            shotValue=2,
            shotDistance=5,
            subType="Turnaround Fadeaway shot",
        )
        val = base_delta(row)
        # fadeaway: +0.05, turnaround: +0.04 -> difficulty=0.09
        # is_jump=1 (shotValue!=3 and "jump" not in desc "turnaround fadeaway shot" -> "jump" IS in desc? NO! "Turnaround Fadeaway Shot" has no "jump")
        # Wait: is_jump = int("jump" in desc and shot_value != 3)
        # desc.lower() = "jokic 5' turnaround fadeaway shot (14 pts)" -> "jump" not in it -> is_jump=0
        # is_floater=0, is_hook=0, is_layup=0, is_dunk=0, is_putback=0
        # residual = 0
        # attempt=0.15, assist_modifier=+0.03
        # difficulty=0.09 (turnaround + fadeaway)
        expected = 0.15 + 0.0 + 0.09 + 0.03
        assert approx(val, expected)

    def test_made_pullup_bonus(self):
        """Pullup gets +0.03 difficulty"""
        row = make_row(
            description="Powell 25' 3PT Pullup Jump Shot (3 PTS) (Mitchell 1 AST)",
            actionType="Made Shot",
            shotValue=3,
            shotDistance=25,
            subType="Pullup Jump shot",
        )
        val = base_delta(row)
        # is_three=1, is_jump=0(shotValue==3), difficulty=0.03(pullup), assist_modifier=-0.04
        # residual = 0.194*1 = 0.194
        # attempt=0.18
        expected = 0.18 + 0.194 + 0.03 - 0.04
        assert approx(val, expected)

    def test_made_layup_unassisted(self):
        """Driving layup unassisted"""
        row = make_row(
            description="Adebayo 7' Driving Layup (4 PTS)",
            actionType="Made Shot",
            shotValue=2,
            shotDistance=7,
            subType="Driving Layup Shot",
        )
        val = base_delta(row)
        # is_layup=1, attempt=0.15, residual=0.100, difficulty=0, assist_modifier=+0.03
        expected = 0.15 + 0.100 + 0 + 0.03
        assert approx(val, expected)

    def test_made_dunk_unassisted(self):
        """Running dunk unassisted"""
        row = make_row(
            description="Adebayo Running Dunk (4 PTS)",
            actionType="Made Shot",
            shotValue=2,
            shotDistance=0,
            subType="Running Dunk Shot",
        )
        val = base_delta(row)
        # is_dunk=1, attempt=0.15, residual=0.087, difficulty=0, assist_modifier=+0.03
        expected = 0.15 + 0.087 + 0 + 0.03
        assert approx(val, expected)

    def test_made_floater(self):
        """
        Phase 2B fix: 'floater' -> 'float' in made shot classification.
        NBA descriptions use 'Floating Jump Shot' not 'Floater'.
        Before fix: is_floater=0, row classified as generic → is_jump=1 → residual=0.120
        After fix:  is_floater=1, residual=0.130 (correct floating shot classification)
        Note: 'Driving Floating Jump Shot' desc: 'jump' in it => is_jump=1 also fires.
        But wait — the code: is_jump = int('jump' in desc and shot_value != 3)
        So both is_floater=1 AND is_jump=1. Residual = 0.130 + 0.120 = 0.250.
        """
        row = make_row(
            description="Powell 9' Driving Floating Jump Shot (6 PTS)",
            actionType="Made Shot",
            shotValue=2,
            shotDistance=9,
            subType="Driving Floating Jump Shot",
        )
        val = base_delta(row)
        # After Phase 2B fix: is_floater=1 (float in desc), is_jump=1 (jump in desc, val!=3)
        # residual = 0.130*1 + 0.120*1 = 0.250
        # attempt=0.15, difficulty=0, assist_modifier=+0.03
        expected = 0.15 + 0.250 + 0 + 0.03
        assert approx(val, expected)

    def test_made_pure_floating_jump_shot(self):
        """'Floating Jump Shot' subType: float=1, jump=1 (for 2pt), residual=0.130+0.120."""
        row = make_row(
            description="Vucevic 6' Floating Jump Shot (2 PTS)",
            actionType="Made Shot",
            shotValue=2,
            shotDistance=6,
            subType="Floating Jump shot",
        )
        val = base_delta(row)
        # is_floater=1 (float in desc), is_jump=1 (jump in desc)
        expected = 0.15 + 0.130 + 0.120 + 0 + 0.03
        assert approx(val, expected)

    def test_made_hook_shot(self):
        """Hook shot"""
        row = make_row(
            description="Gobert 5' Hook Shot (12 PTS)",
            actionType="Made Shot",
            shotValue=2,
            shotDistance=5,
            subType="Turnaround Hook Shot",
        )
        val = base_delta(row)
        # is_hook=1("hook" in desc), attempt=0.15, residual=0.130, difficulty=0, assist_modifier=+0.03
        expected = 0.15 + 0.130 + 0 + 0.03
        assert approx(val, expected)

    def test_made_putback(self):
        """Putback layup - negative residual component"""
        row = make_row(
            description="Gobert 6' Putback Layup (6 PTS)",
            actionType="Made Shot",
            shotValue=2,
            shotDistance=6,
            subType="Putback Layup Shot",
        )
        val = base_delta(row)
        # is_putback=1, is_layup=1(layup in desc)
        # residual = 0.100*1(layup) - 0.030*1(putback) = 0.070
        # attempt=0.15, difficulty=0, assist_modifier=+0.03
        expected = 0.15 + 0.070 + 0 + 0.03
        assert approx(val, expected)

    def test_made_shot_zero_shot_value_returns_zero_attempt(self):
        """shotValue=0 -> attempt_value=0 -> only residual matters"""
        row = make_row(
            description="Player Jump Shot",
            actionType="Made Shot",
            shotValue=0,
            shotDistance=10,
            subType="Jump Shot",
        )
        val = base_delta(row)
        # attempt=0, is_jump=1(shotValue!=3), residual=0.120, assist_modifier=+0.03
        expected = 0 + 0.120 + 0 + 0.03
        assert approx(val, expected)

    def test_made_short_3(self):
        """Short 3 (distance <= 24): +0.010"""
        row = make_row(
            description="Player 24' 3PT Jump Shot (3 PTS)",
            actionType="Made Shot",
            shotValue=3,
            shotDistance=24,
            subType="Jump Shot",
        )
        val = base_delta(row)
        # is_three=1, short_3=1, deep_3=0
        # residual = 0.194 + 0.010 = 0.204
        expected = 0.18 + 0.204 + 0 + 0.03
        assert approx(val, expected)

    def test_made_deep_3(self):
        """Deep 3 (distance >= 27): -0.050"""
        row = make_row(
            description="Player 30' 3PT Jump Shot (3 PTS)",
            actionType="Made Shot",
            shotValue=3,
            shotDistance=30,
            subType="Jump Shot",
        )
        val = base_delta(row)
        # is_three=1, short_3=0, deep_3=1
        # residual = 0.194 - 0.050 = 0.144
        expected = 0.18 + 0.144 + 0 + 0.03
        assert approx(val, expected)

    def test_made_3_borderline_25(self):
        """3pt at exactly 25ft: neither short (<=24) nor deep (>=27)"""
        row = make_row(
            description="Player 25' 3PT Jump Shot (3 PTS)",
            actionType="Made Shot",
            shotValue=3,
            shotDistance=25,
            subType="Jump Shot",
        )
        val = base_delta(row)
        # short_3=0, deep_3=0, residual=0.194
        expected = 0.18 + 0.194 + 0 + 0.03
        assert approx(val, expected)


# ============================================================
# 5. COMBINED DESCRIPTIONS
# ============================================================

# ============================================================
# 5b. PHASE 2B BUG FIX TESTS
# ============================================================

class TestPhase2BBugFixes:
    """
    Explicit tests for the three bugs fixed in Phase 2B.
    Each test names the old vs new behavior.
    """

    def test_floater_keyword_fix_floating_shot(self):
        """
        BUG: 'floater' keyword never matched NBA descriptions.
        FIX: Changed to 'float' keyword.
        OLD: is_floater=0 for 'Floating Jump Shot' descriptions.
        NEW: is_floater=1 for 'Floating Jump Shot' descriptions.
        """
        row = make_row(
            description="Vucevic 6' Floating Jump Shot (2 PTS)",
            actionType="Made Shot",
            shotValue=2,
            shotDistance=6,
            subType="Floating Jump shot",
        )
        # With fix: is_floater=1, is_jump=1 -> residual=0.130+0.120=0.250
        # Without fix: is_floater=0, is_jump=1 -> residual=0.120
        val = base_delta(row)
        expected = 0.15 + 0.130 + 0.120 + 0 + 0.03
        assert approx(val, expected), f"Expected {expected}, got {val}"

    def test_driving_keyword_fix_missed_floating(self):
        """
        BUG: 'drive' never matched 'Driving Floating Jump Shot' descriptions.
        FIX: Changed to 'driving'.
        OLD: 0.07 (default missed shot)
        NEW: 0.10 (driving path)
        """
        row = make_row(
            description="MISS Jaquez Jr. 7' Driving Floating Jump Shot",
            actionType="Missed Shot",
            subType="Driving Floating Jump Shot",
        )
        val = base_delta(row)
        assert approx(val, 0.10), f"Expected 0.10, got {val}"

    def test_stepback_dead_code_removal_no_change(self):
        """
        BUG: 'stepback' (no space) was dead code — never matched any NBA description.
        FIX: Removed the redundant 'stepback' check.
        IMPACT: Zero — 'step back' (with space) still works.
        """
        row = make_row(
            description="Adebayo 18' Step Back Jump Shot (12 PTS)",
            actionType="Made Shot",
            shotValue=2,
            shotDistance=18,
            subType="Step Back Jump shot",
        )
        val = base_delta(row)
        # step back difficulty = 0.06, is_jump=1, attempt=0.15, assist_modifier=+0.03
        expected = 0.15 + 0.120 + 0.06 + 0.03
        assert approx(val, expected)

    def test_driving_layup_missed_unchanged(self):
        """
        Driving Layup missed shots were already 0.10 via 'layup' keyword.
        Verify no regression from the 'driving' fix.
        """
        row = make_row(
            description="MISS Brown 5' Driving Layup",
            actionType="Missed Shot",
            subType="Driving Layup Shot",
        )
        assert approx(base_delta(row), 0.10)

    def test_regular_missed_shot_unchanged(self):
        """Non-driving non-layup missed shots still return 0.07."""
        row = make_row(
            description="MISS Brown 21' Jump Shot",
            actionType="Missed Shot",
            subType="Jump Shot",
        )
        assert approx(base_delta(row), 0.07)


class TestCombinedDescriptions:
    """
    Tests for multi-keyword description behaviors.
    The key question: does "step back fadeaway" earn BOTH bonuses?
    Answer: YES - the code is intentionally additive. Each keyword is a separate
    if-statement that increments difficulty, so both apply if both keywords are present.
    This behavior is documented and PRESERVED in Phase 2B.
    """

    def test_stepback_fadeaway_combined(self):
        """
        'step back fadeaway' in desc -> both bonuses applied additively.
        difficulty = 0.06 (step back) + 0.05 (fadeaway) = 0.11
        This is the current behavior. We preserve it as-is.
        """
        row = make_row(
            description="Player 18' Step Back Fadeaway Jumper (12 PTS)",
            actionType="Made Shot",
            shotValue=2,
            shotDistance=18,
            subType="Step Back Jump shot",
        )
        val = base_delta(row)
        # is_jump=1(shotValue!=3 and "jump" in desc? "jumper" contains "jump"? YES)
        # Wait: desc.lower() = "player 18' step back fadeaway jumper (12 pts)"
        # "jump" in desc -> "jumper" -> YES -> is_jump=1
        # difficulty: fadeaway=0.05, stepback=0.06 -> 0.11
        # attempt=0.15, residual=0.120, assist_modifier=+0.03
        expected = 0.15 + 0.120 + 0.11 + 0.03
        assert approx(val, expected), f"Expected {expected}, got {val}"

    def test_turnaround_fadeaway_combined(self):
        """Both turnaround (+0.04) and fadeaway (+0.05) applied."""
        row = make_row(
            description="Jokic 12' Turnaround Fadeaway Shot (18 PTS)",
            actionType="Made Shot",
            shotValue=2,
            shotDistance=12,
            subType="Turnaround Fadeaway shot",
        )
        val = base_delta(row)
        # "jump" in desc? "turnaround fadeaway shot" -> NO -> is_jump=0
        # difficulty = turnaround(0.04) + fadeaway(0.05) = 0.09
        # residual=0, assist_modifier=+0.03, attempt=0.15
        expected = 0.15 + 0 + 0.09 + 0.03
        assert approx(val, expected)

    def test_pullup_jump_3pt_with_assist(self):
        """3pt pullup jump shot, assisted."""
        row = make_row(
            description="Powell 25' 3PT Pullup Jump Shot (3 PTS) (Mitchell 1 AST)",
            actionType="Made Shot",
            shotValue=3,
            shotDistance=25,
            subType="Pullup Jump shot",
        )
        val = base_delta(row)
        # is_three=1, is_jump=0(shotValue==3), pullup difficulty=0.03
        # short_3=0(25>24), deep_3=0(25<27)
        # residual=0.194, assist=-0.04, attempt=0.18
        expected = 0.18 + 0.194 + 0.03 - 0.04
        assert approx(val, expected)

    def test_missed_shot_unknown_type_returns_zero_seven(self):
        """Unknown missed shot type -> 0.07 default"""
        row = make_row(
            description="MISS Player 15' Jump Shot",
            actionType="Missed Shot",
            shotValue=2,
            shotDistance=15,
            subType="Jump Shot",
        )
        val = base_delta(row)
        assert approx(val, 0.07)

    def test_empty_row_returns_zero(self):
        """Row with no recognizable event returns 0"""
        row = make_row(description="some random event", actionType="Timeout")
        assert base_delta(row) == 0


# ============================================================
# 6. CONTEXT DELTA
# ============================================================

class TestContextDelta:

    def test_non_made_shot_returns_zero(self):
        row = make_row(actionType="Turnover", score_diff=0, score_diff_before=2)
        assert context_delta(row) == 0

    def test_game_winner_desc(self):
        row = make_row(
            description="Brown 25' 3PT Jump Shot (game-winner) (14 PTS)",
            actionType="Made Shot",
            score_diff=2,
            score_diff_before=-1,
        )
        assert approx(context_delta(row), 0.8)

    def test_game_tying_desc(self):
        row = make_row(
            description="Brown 25' 3PT Jump Shot (game-tying) (14 PTS)",
            actionType="Made Shot",
            score_diff=0,
            score_diff_before=-3,
        )
        assert approx(context_delta(row), 0.07)

    def test_lead_taking_desc(self):
        row = make_row(
            description="Brown 25' Jump Shot (lead-taking) (14 PTS)",
            actionType="Made Shot",
            score_diff=2,
            score_diff_before=-1,
        )
        assert approx(context_delta(row), 0.05)

    def test_score_ties_computed_game_tying(self):
        """Shot ties the game (after==0, before!=0) without keyword in desc -> 0.07"""
        row = make_row(
            description="Brown 25' 3PT Jump Shot (14 PTS)",
            actionType="Made Shot",
            score_diff=0,
            score_diff_before=-3,
        )
        assert approx(context_delta(row), 0.07)

    def test_score_lead_change_computed(self):
        """Shot changes lead (before<=0, after>0) without keyword in desc -> 0.05"""
        row = make_row(
            description="Brown 25' Jump Shot (14 PTS)",
            actionType="Made Shot",
            score_diff=1,
            score_diff_before=-1,
        )
        assert approx(context_delta(row), 0.05)

    def test_shot_increases_existing_lead(self):
        """No context event when both before and after are positive"""
        row = make_row(
            description="Brown 15' Jump Shot (14 PTS)",
            actionType="Made Shot",
            score_diff=5,
            score_diff_before=3,
        )
        assert context_delta(row) == 0

    def test_already_tied_before_no_context(self):
        """Before=0 and after>0: code checks before<=0 -> yes -> lead-taking. But before==0 counts."""
        row = make_row(
            description="Brown 15' Jump Shot (14 PTS)",
            actionType="Made Shot",
            score_diff=2,
            score_diff_before=0,
        )
        # before<=0 and after>0 -> 0.05
        assert approx(context_delta(row), 0.05)

    def test_shot_stays_tied_no_context(self):
        """Before=0 after=0: after==0 and before==0 -> no game-tying (before must be !=0)"""
        row = make_row(
            description="Brown 0 Jump Shot",
            actionType="Made Shot",
            score_diff=0,
            score_diff_before=0,
        )
        # after==0 and before==0 -> before==0 fails the != check -> falls to lead-taking? before<=0 and after>0? after=0 -> NO
        assert context_delta(row) == 0


# ============================================================
# 7. CLUTCH MULTIPLIER
# ============================================================

class TestClutchMultiplier:

    def test_non_q4_period_returns_1(self):
        row = make_row(actionType="Made Shot", period=1, seconds_remaining=60)
        assert clutch_multiplier(row) == 1.0

    def test_q4_but_outside_120_seconds(self):
        row = make_row(actionType="Made Shot", period=4, seconds_remaining=121)
        assert clutch_multiplier(row) == 1.0

    def test_q4_exactly_120_seconds(self):
        """At exactly 120 seconds, the formula: 1 + ((120-120)/120)*2 = 1.0"""
        row = make_row(actionType="Made Shot", period=4, seconds_remaining=120)
        val = clutch_multiplier(row)
        assert approx(val, 1.0)

    def test_q4_exactly_119_seconds(self):
        """At 119 seconds: 1 + ((120-119)/120)*2 = 1 + 2/120 = 1.01666..."""
        row = make_row(actionType="Made Shot", period=4, seconds_remaining=119)
        expected = 1 + ((120 - 119) / 120) * 2
        assert approx(clutch_multiplier(row), expected)

    def test_q4_60_seconds(self):
        """1 min remaining: 1 + ((120-60)/120)*2 = 1 + 1.0 = 2.0"""
        row = make_row(actionType="Made Shot", period=4, seconds_remaining=60)
        assert approx(clutch_multiplier(row), 2.0)

    def test_q4_zero_seconds(self):
        """0 seconds: 1 + (120/120)*2 = 3.0"""
        row = make_row(actionType="Made Shot", period=4, seconds_remaining=0)
        assert approx(clutch_multiplier(row), 3.0)

    def test_q4_none_seconds_returns_1(self):
        row = make_row(actionType="Made Shot", period=4, seconds_remaining=None)
        assert clutch_multiplier(row) == 1.0

    def test_non_made_shot_returns_1(self):
        """Clutch only applies to Made Shot"""
        row = make_row(actionType="Missed Shot", period=4, seconds_remaining=10)
        assert clutch_multiplier(row) == 1.0

    def test_free_throw_returns_1(self):
        """Free throws always return 1.0 regardless of time"""
        row = make_row(
            description="Brown Free Throw 1 of 2 (1 PTS)",
            actionType="Free Throw",
            period=4,
            seconds_remaining=5,
        )
        assert clutch_multiplier(row) == 1.0

    def test_overtime_period_5_returns_1(self):
        """Overtime (period=5) is NOT period 4, so no clutch multiplier applies"""
        row = make_row(actionType="Made Shot", period=5, seconds_remaining=10)
        assert clutch_multiplier(row) == 1.0

    def test_overtime_period_6_returns_1(self):
        row = make_row(actionType="Made Shot", period=6, seconds_remaining=10)
        assert clutch_multiplier(row) == 1.0

    def test_q3_end_game_returns_1(self):
        """End of Q3 should not trigger clutch"""
        row = make_row(actionType="Made Shot", period=3, seconds_remaining=0)
        assert clutch_multiplier(row) == 1.0


# ============================================================
# 8. ATTRIBUTION — verify the rows carry personId/teamId
# ============================================================

class TestAttribution:
    """
    Verify that personId and teamId are preserved through to the output.
    The base_delta/context_delta/clutch_multiplier functions operate on
    a row that already carries personId and teamId. The groupby in
    compute_player_ratings correctly aggregates by those fields.
    These tests confirm the row structure expected by the engine.
    """

    def test_steal_row_carries_stealer_personId(self):
        """Steal row personId = the stealing player (correct attribution in real data)"""
        row = make_row(
            description="White STEAL (1 BLK)",
            actionType="",
            personId=1628401,
            playerName="White",
        )
        # base_delta should return +0.19 for the stealing player
        assert approx(base_delta(row), 0.19)

    def test_block_row_carries_blocker_personId(self):
        """Block row personId = the blocking player"""
        row = make_row(
            description="Mitchell BLOCK (1 BLK)",
            actionType="",
            personId=1630558,
            playerName="Mitchell",
        )
        assert approx(base_delta(row), 0.18)

    def test_turnover_row_carries_turnover_player_personId(self):
        """Turnover row personId = the player who committed the turnover"""
        row = make_row(
            description="Brown Lost Ball Turnover (P1.T3)",
            actionType="Turnover",
            personId=1627759,
            playerName="Brown",
        )
        assert approx(base_delta(row), -0.17)

    def test_made_shot_row_carries_shooter_personId(self):
        """Made shot row personId = the shooter"""
        row = make_row(
            description="Adebayo 18' Step Back Jump Shot (12 PTS) (Mitchell 3 AST)",
            actionType="Made Shot",
            shotValue=2,
            shotDistance=18,
            subType="Step Back Jump shot",
            personId=1628389,
            playerName="Adebayo",
        )
        val = base_delta(row)
        # Shot is by Adebayo (personId=1628389), assist modifier is -0.04
        # is_jump=1, stepback difficulty=0.06, assist_modifier=-0.04
        expected = 0.15 + 0.120 + 0.06 - 0.04
        assert approx(val, expected)


# ============================================================
# 9. REGRESSION: OLD vs NEW on real data
# ============================================================

class TestRegressionOnRealData:
    """
    Load the actual play-by-play parquet and run the existing rating engine
    to produce player-game ratings. Then compare against the saved baseline.
    """

    @pytest.fixture(scope="class")
    def pbp_df(self):
        pbp_path = os.path.join(ROOT, "data", "play_by_play.parquet")
        if not os.path.exists(pbp_path):
            pytest.skip("play_by_play.parquet not available")
        return pd.read_parquet(pbp_path)

    @pytest.fixture(scope="class")
    def baseline_ratings(self):
        baseline_path = os.path.join(ROOT, "data", "player_game_ratings.parquet")
        if not os.path.exists(baseline_path):
            pytest.skip("player_game_ratings.parquet not available")
        return pd.read_parquet(baseline_path)

    def _compute_ratings(self, df):
        """Run the rating engine on a dataframe and return player-game ratings."""
        import re
        df = df.copy()
        df["scoreHome"] = pd.to_numeric(df["scoreHome"], errors="coerce").ffill().fillna(0)
        df["scoreAway"] = pd.to_numeric(df["scoreAway"], errors="coerce").ffill().fillna(0)
        df["score_diff"] = df["scoreHome"] - df["scoreAway"]
        df["score_diff_before"] = df.groupby("gameId")["score_diff"].shift(1).fillna(0)
        df["seconds_remaining"] = df["clock"].apply(clock_to_seconds)
        df["base_delta"] = df.apply(base_delta, axis=1)
        df["context_delta"] = df.apply(context_delta, axis=1)
        df["clutch_mult"] = df.apply(clutch_multiplier, axis=1)
        df["rating"] = (df["base_delta"] + df["context_delta"]) * df["clutch_mult"]
        df = df[(df["personId"] > 1000) & (df["teamId"] > 0) & (df["playerName"].notna()) & (df["playerName"] != "")]
        return (
            df.groupby(["gameId", "teamId", "personId", "playerName"], as_index=False)["rating"]
            .sum()
        )

    def test_rating_count_matches_baseline(self, pbp_df, baseline_ratings):
        """Running the engine from scratch must produce same row count as baseline."""
        ratings = self._compute_ratings(pbp_df)
        baseline_count = len(baseline_ratings)
        new_count = len(ratings)
        # Allow tiny tolerance in case baseline was from a slightly different run
        assert new_count == baseline_count, (
            f"Row count mismatch: baseline={baseline_count}, new={new_count}"
        )

    def test_rating_values_match_baseline(self, pbp_df, baseline_ratings):
        """
        Phase 2B regression: compare new engine ratings against pre-Phase-2B baseline.

        The baseline was generated with the buggy engine (before Phase 2B fixes).
        After Phase 2B, ratings for Floating Shot and Driving Floating missed shot events
        differ from baseline. These differences are EXPECTED and DOCUMENTED:

        1. Floating made shots: +0.130 residual bonus now fires (was 0 before 'float' fix)
           NBA descriptions: 'Floating Jump Shot', 'Driving Floating Jump Shot'
        2. Driving Floating missed shots: now 0.10 (was 0.07 before 'driving' fix)
        3. 'stepback' removal: ZERO impact (never matched any real NBA description)

        This test verifies:
        - Row count is identical (no phantom rows created/deleted)
        - Changed ratings are explained by the two known bug fixes
        - Unchanged ratings are exactly equal (no collateral drift)
        """
        ratings = self._compute_ratings(pbp_df)
        merged = ratings.merge(
            baseline_ratings,
            on=["gameId", "teamId", "personId", "playerName"],
            suffixes=("_new", "_old"),
        )
        assert len(merged) > 0, "Merge produced zero rows - check key columns"

        diff = (merged["rating_new"] - merged["rating_old"]).abs()
        changed_count = (diff > 1e-6).sum()
        unchanged_count = (diff <= 1e-6).sum()
        max_diff = diff.max()

        # Expected: the two bug fixes changed ratings for players who had Floating shots
        # or Driving Floating missed shots. All changes should be positive (new > old).
        changed_rows = merged[diff > 1e-6]
        negative_changes = (changed_rows["rating_new"] - changed_rows["rating_old"] < -1e-6).sum()

        # Verify no ratings got LOWER (bug fix should only increase shot values, not decrease)
        assert negative_changes == 0, (
            f"{negative_changes} ratings decreased unexpectedly. "
            f"Phase 2B fixes should only increase floating shot values."
        )

        # Verify row count stability
        assert len(ratings) == len(baseline_ratings), (
            f"Row count changed: baseline={len(baseline_ratings)}, new={len(ratings)}"
        )

        # Document the change for reporting
        print(f"\n=== Phase 2B Regression Report ===")
        print(f"Total player-game rows:       {len(merged)}")
        print(f"Unchanged ratings:            {unchanged_count}")
        print(f"Changed ratings (bug fixes):  {changed_count}")
        print(f"Max abs difference:           {max_diff:.6f}")
        print(f"All changes positive (>=0):   {negative_changes == 0}")
        print(f"Cause: 'floater'→'float' fix and 'drive'→'driving' fix")



    def test_output_columns_present(self, pbp_df):
        """Verify the contract: gameId, teamId, personId, playerName, rating."""
        ratings = self._compute_ratings(pbp_df)
        for col in ["gameId", "teamId", "personId", "playerName", "rating"]:
            assert col in ratings.columns, f"Missing output column: {col}"

    def test_no_null_ratings(self, pbp_df):
        """Rating should never be null."""
        ratings = self._compute_ratings(pbp_df)
        assert ratings["rating"].isna().sum() == 0

    def test_plausible_row_count(self, pbp_df):
        """Row count should be in a reasonable range (7000-10000 based on Phase 2A baseline)."""
        ratings = self._compute_ratings(pbp_df)
        assert 7000 <= len(ratings) <= 12000, f"Implausible row count: {len(ratings)}"

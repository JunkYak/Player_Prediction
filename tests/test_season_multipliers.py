"""
tests/test_season_multipliers.py

Automated test suite for the Season Performance Multipliers Calculation Engine.

Covers:
1. Current-season isolation & October 20 boundary resolution.
2. Minimum 3-game eligibility threshold & exclusion from percentile pool.
3. Ineligible players receiving strictly 0.0 bonus.
4. Actual game rating arithmetic aggregation.
5. Percentile ranking (ascending, 0.0 to 100.0) & tie handling.
6. Candidate curve implementations (A, B, C, D) & boundary conditions.
7. Clamping [0.0, 3.0] and 1-decimal place precision.
8. Handling negative ratings & negative season averages.
9. Duplicate game deduplication.
10. Missing / corrupt data resilience.
11. One-game spike stability verification.
12. Determinism and idempotency.
13. Edge cases: empty dataset, N=1, N=2 eligible players.
14. Real-world validation against production parquet artifact.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
from datetime import datetime

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from scripts.compute_season_multipliers import (
    determine_season_boundary,
    apply_multiplier_curve,
    calculate_percentiles,
    compute_season_multipliers,
    compare_candidate_curves,
)


# ==============================================================================
# 1. Season Boundary & Current Season Isolation
# ==============================================================================
def test_determine_season_boundary_after_oct20():
    """Verify reference date on or after Oct 20 resolves current year season."""
    start, end = determine_season_boundary(reference_date="2026-10-24")
    assert start == "2026-10-20"
    assert end == "2027-10-19"

    # Exactly Oct 20
    start_exact, end_exact = determine_season_boundary(reference_date="2026-10-20")
    assert start_exact == "2026-10-20"
    assert end_exact == "2027-10-19"


def test_determine_season_boundary_before_oct20():
    """Verify reference date before Oct 20 resolves prior year season."""
    # Mid-season spring game
    start, end = determine_season_boundary(reference_date="2027-02-15")
    assert start == "2026-10-20"
    assert end == "2027-10-19"

    # Just before Oct 20
    start_oct19, end_oct19 = determine_season_boundary(reference_date="2026-10-19")
    assert start_oct19 == "2025-10-20"
    assert end_oct19 == "2026-10-19"


def test_current_season_isolation():
    """Verify games prior to season_start_date (Oct 20) are strictly excluded."""
    data = [
        # Previous season games (April 2026)
        {"gameId": "G_OLD1", "personId": 101, "rating": 5.0, "game_date": "2026-04-10"},
        {"gameId": "G_OLD2", "personId": 101, "rating": 5.0, "game_date": "2026-04-12"},
        {"gameId": "G_OLD3", "personId": 101, "rating": 5.0, "game_date": "2026-04-14"},
        # Current season games (October 2026)
        {"gameId": "G_CURR1", "personId": 101, "rating": 1.0, "game_date": "2026-10-20"},
        {"gameId": "G_CURR2", "personId": 101, "rating": 1.0, "game_date": "2026-10-22"},
        {"gameId": "G_CURR3", "personId": 101, "rating": 1.0, "game_date": "2026-10-24"},
    ]
    df = pd.DataFrame(data)

    res = compute_season_multipliers(
        ratings_df=df,
        season_start_date="2026-10-20",
        season_end_date="2027-10-19",
    )

    player_row = res[res["personId"] == 101].iloc[0]
    # Only 3 current season games should be counted, not 6
    assert player_row["season_games_played"] == 3
    # Season average should be 1.0, NOT (5*3 + 1*3)/6 = 3.0
    assert player_row["season_avg_rating"] == pytest.approx(1.0, abs=1e-4)


# ==============================================================================
# 2. Eligibility & Minimum Games Threshold
# ==============================================================================
def test_minimum_games_eligibility_threshold():
    """Verify players with <3 games are marked ineligible and receive 0.0 bonus."""
    data = [
        # Player 1: 1 game
        {"gameId": "G1", "personId": 1, "rating": 10.0, "game_date": "2026-10-20"},
        # Player 2: 2 games
        {"gameId": "G1", "personId": 2, "rating": 0.5, "game_date": "2026-10-20"},
        {"gameId": "G2", "personId": 2, "rating": 0.5, "game_date": "2026-10-21"},
        # Player 3: 3 games (eligible)
        {"gameId": "G1", "personId": 3, "rating": 1.0, "game_date": "2026-10-20"},
        {"gameId": "G2", "personId": 3, "rating": 1.0, "game_date": "2026-10-21"},
        {"gameId": "G3", "personId": 3, "rating": 1.0, "game_date": "2026-10-22"},
        # Player 4: 4 games (eligible)
        {"gameId": "G1", "personId": 4, "rating": 4.0, "game_date": "2026-10-20"},
        {"gameId": "G2", "personId": 4, "rating": 4.0, "game_date": "2026-10-21"},
        {"gameId": "G3", "personId": 4, "rating": 4.0, "game_date": "2026-10-22"},
        {"gameId": "G4", "personId": 4, "rating": 4.0, "game_date": "2026-10-23"},
    ]
    df = pd.DataFrame(data)

    res = compute_season_multipliers(
        ratings_df=df,
        season_start_date="2026-10-20",
        min_games_threshold=3,
        curve="candidate_d",
    ).set_index("personId")

    # Ineligible checks
    assert res.loc[1, "season_multiplier_eligible"] == False
    assert res.loc[1, "season_games_played"] == 1
    assert res.loc[1, "season_performance_bonus"] == 0.0
    assert pd.isna(res.loc[1, "season_percentile"])

    assert res.loc[2, "season_multiplier_eligible"] == False
    assert res.loc[2, "season_games_played"] == 2
    assert res.loc[2, "season_performance_bonus"] == 0.0
    assert pd.isna(res.loc[2, "season_percentile"])

    # Eligible checks
    assert res.loc[3, "season_multiplier_eligible"] == True
    assert res.loc[3, "season_games_played"] == 3
    assert pd.notna(res.loc[3, "season_percentile"])
    # Player 3 has lowest rating of eligible -> P=0.0 -> bonus=3.0
    assert res.loc[3, "season_percentile"] == 0.0
    assert res.loc[3, "season_performance_bonus"] == 3.0

    assert res.loc[4, "season_multiplier_eligible"] == True
    assert res.loc[4, "season_games_played"] == 4
    # Player 4 has highest rating of eligible -> P=100.0 -> bonus=0.0
    assert res.loc[4, "season_percentile"] == 100.0
    assert res.loc[4, "season_performance_bonus"] == 0.0


def test_ineligible_players_do_not_distort_percentiles():
    """Verify players with <3 games are excluded from percentile calculation pool."""
    data = [
        # Eligible player A: avg 1.0
        {"gameId": "G1", "personId": 10, "rating": 1.0, "game_date": "2026-10-20"},
        {"gameId": "G2", "personId": 10, "rating": 1.0, "game_date": "2026-10-21"},
        {"gameId": "G3", "personId": 10, "rating": 1.0, "game_date": "2026-10-22"},
        # Eligible player B: avg 3.0
        {"gameId": "G1", "personId": 20, "rating": 3.0, "game_date": "2026-10-20"},
        {"gameId": "G2", "personId": 20, "rating": 3.0, "game_date": "2026-10-21"},
        {"gameId": "G3", "personId": 20, "rating": 3.0, "game_date": "2026-10-22"},
        # Ineligible outlier players with only 1 game: huge 50.0 and terrible -50.0
        {"gameId": "G1", "personId": 998, "rating": -50.0, "game_date": "2026-10-20"},
        {"gameId": "G1", "personId": 999, "rating": 50.0, "game_date": "2026-10-20"},
    ]
    df = pd.DataFrame(data)

    res = compute_season_multipliers(
        ratings_df=df,
        season_start_date="2026-10-20",
        min_games_threshold=3,
        curve="candidate_a",
    ).set_index("personId")

    # If 998 and 999 were included in ranking, player 10 would be at 33% instead of 0%
    assert res.loc[10, "season_percentile"] == 0.0
    assert res.loc[20, "season_percentile"] == 100.0
    assert res.loc[998, "season_performance_bonus"] == 0.0
    assert res.loc[999, "season_performance_bonus"] == 0.0


# ==============================================================================
# 3. Percentiles & Tie Handling
# ==============================================================================
def test_percentile_calculation_and_ties():
    """Verify ascending percentile ranks and average rank tie-breaking."""
    # 4 players: ratings 1.0, 2.0, 2.0, 5.0
    averages = pd.Series([1.0, 2.0, 2.0, 5.0], index=[1, 2, 3, 4])
    pct = calculate_percentiles(averages)

    # N = 4
    # Player 1: rank 1 -> (1-1)/(4-1) * 100 = 0.0
    assert pct.loc[1] == 0.0
    # Players 2 & 3: ranks 2 and 3 -> average rank = 2.5
    # (2.5 - 1) / 3 * 100 = 1.5 / 3 * 100 = 50.0
    assert pct.loc[2] == pytest.approx(50.0, abs=1e-5)
    assert pct.loc[3] == pytest.approx(50.0, abs=1e-5)
    # Player 4: rank 4 -> (4-1)/3 * 100 = 100.0
    assert pct.loc[4] == 100.0


# ==============================================================================
# 4. Candidate Curve Verification & Boundaries
# ==============================================================================
@pytest.mark.parametrize("p,expected_a,expected_b,expected_c,expected_d", [
    (0.0,   3.0, 3.0, 3.0, 3.0),
    (50.0,  1.5, 0.8, 2.1, 2.1),
    (95.0,  0.2, 0.0, 0.7, 0.0),
    (100.0, 0.0, 0.0, 0.0, 0.0),
])
def test_candidate_curves_at_key_percentiles(p, expected_a, expected_b, expected_c, expected_d):
    """Verify all 4 curves at critical percentile points (0, 50, 95, 100)."""
    b_a = apply_multiplier_curve(p, curve="candidate_a")
    b_b = apply_multiplier_curve(p, curve="candidate_b")
    b_c = apply_multiplier_curve(p, curve="candidate_c")
    b_d = apply_multiplier_curve(p, curve="candidate_d", cutoff_percentile=95.0)

    assert b_a == expected_a, f"Candidate A failed at P={p}: expected {expected_a}, got {b_a}"
    assert b_b == expected_b, f"Candidate B failed at P={p}: expected {expected_b}, got {b_b}"
    assert b_c == expected_c, f"Candidate C failed at P={p}: expected {expected_c}, got {b_c}"
    assert b_d == expected_d, f"Candidate D failed at P={p}: expected {expected_d}, got {b_d}"


def test_clamping_and_one_decimal_precision():
    """Verify multipliers never exceed [0.0, 3.0] and have max 1 decimal place."""
    p_values = np.linspace(-10, 110, 250)
    for curve in ["candidate_a", "candidate_b", "candidate_c", "candidate_d"]:
        bonuses = apply_multiplier_curve(p_values, curve=curve)
        assert np.all(bonuses >= 0.0)
        assert np.all(bonuses <= 3.0)
        # Check exactly 1 decimal place precision
        assert np.all(bonuses == np.round(bonuses, 1))


# ==============================================================================
# 5. Handling Negative Ratings, Duplicates, and Missing Data
# ==============================================================================
def test_negative_ratings_handled_safely():
    """Verify players with negative ratings are properly ranked at the bottom."""
    data = [
        # Player 1: negative ratings (-1.0, -0.5, -1.5) -> avg = -1.0
        {"gameId": "G1", "personId": 1, "rating": -1.0, "game_date": "2026-10-20"},
        {"gameId": "G2", "personId": 1, "rating": -0.5, "game_date": "2026-10-21"},
        {"gameId": "G3", "personId": 1, "rating": -1.5, "game_date": "2026-10-22"},
        # Player 2: zero ratings -> avg = 0.0
        {"gameId": "G1", "personId": 2, "rating": 0.0, "game_date": "2026-10-20"},
        {"gameId": "G2", "personId": 2, "rating": 0.0, "game_date": "2026-10-21"},
        {"gameId": "G3", "personId": 2, "rating": 0.0, "game_date": "2026-10-22"},
        # Player 3: positive ratings -> avg = 2.0
        {"gameId": "G1", "personId": 3, "rating": 2.0, "game_date": "2026-10-20"},
        {"gameId": "G2", "personId": 3, "rating": 2.0, "game_date": "2026-10-21"},
        {"gameId": "G3", "personId": 3, "rating": 2.0, "game_date": "2026-10-22"},
    ]
    df = pd.DataFrame(data)

    res = compute_season_multipliers(
        ratings_df=df,
        season_start_date="2026-10-20",
        curve="candidate_d",
    ).set_index("personId")

    assert res.loc[1, "season_avg_rating"] == -1.0
    assert res.loc[1, "season_percentile"] == 0.0
    assert res.loc[1, "season_performance_bonus"] == 3.0


def test_duplicate_game_records_deduplicated():
    """Verify multiple rows for same (personId, gameId) are not counted twice."""
    data = [
        {"gameId": "G1", "personId": 10, "rating": 2.0, "game_date": "2026-10-20"},
        {"gameId": "G1", "personId": 10, "rating": 2.0, "game_date": "2026-10-20"},  # DUPLICATE
        {"gameId": "G2", "personId": 10, "rating": 4.0, "game_date": "2026-10-21"},
        {"gameId": "G3", "personId": 10, "rating": 6.0, "game_date": "2026-10-22"},
    ]
    df = pd.DataFrame(data)

    res = compute_season_multipliers(ratings_df=df, season_start_date="2026-10-20").set_index("personId")
    # Should count 3 unique games, average (2 + 4 + 6) / 3 = 4.0
    assert res.loc[10, "season_games_played"] == 3
    assert res.loc[10, "season_avg_rating"] == 4.0


def test_corrupt_and_missing_data_filtering():
    """Verify NaN ratings, null dates, and negative personIds are cleanly rejected."""
    data = [
        {"gameId": "G1", "personId": 10, "rating": 2.0, "game_date": "2026-10-20"},
        {"gameId": "G2", "personId": 10, "rating": np.nan, "game_date": "2026-10-21"},  # NaN rating
        {"gameId": "G3", "personId": 10, "rating": 4.0, "game_date": None},             # Null date
        {"gameId": "G4", "personId": -99, "rating": 4.0, "game_date": "2026-10-22"},    # Invalid personId
    ]
    df = pd.DataFrame(data)

    res = compute_season_multipliers(ratings_df=df, season_start_date="2026-10-20").set_index("personId")
    # Only 1 valid game remains for player 10 -> games_played = 1, ineligible
    assert res.loc[10, "season_games_played"] == 1
    assert res.loc[10, "season_multiplier_eligible"] == False
    assert -99 not in res.index


# ==============================================================================
# 6. One-Game Spike Stability
# ==============================================================================
def test_one_game_spike_stability():
    """
    Verify that an exceptional single-game performance (e.g. +6.0 rating)
    does not erase a player's performance bonus.
    """
    # Create baseline league of 20 players with 10 completed games each
    rows = []
    for pid in range(1, 21):
        # Baseline ratings evenly spaced between 0.5 and 4.0
        base_rating = 0.5 + (pid - 1) * (3.5 / 19.0)
        for g in range(1, 11):
            rows.append({
                "gameId": f"G_{g}",
                "personId": pid,
                "rating": base_rating,
                "game_date": f"2026-10-{20 + (g % 5):02d}",
            })

    df_base = pd.DataFrame(rows)

    # Compute baseline multipliers for player 5 (a lower-tier player, base rating ~1.05)
    res_before = compute_season_multipliers(
        ratings_df=df_base,
        season_start_date="2026-10-20",
        curve="candidate_d",
    ).set_index("personId")

    bonus_before = res_before.loc[5, "season_performance_bonus"]
    avg_before = res_before.loc[5, "season_avg_rating"]
    pct_before = res_before.loc[5, "season_percentile"]

    assert bonus_before >= 2.0, f"Expected lower-tier player to have bonus >= 2.0, got {bonus_before}"

    # Now player 5 has ONE exceptional blowout game: rating 6.5 in game 11
    spike_row = {
        "gameId": "G_11",
        "personId": 5,
        "rating": 6.5,
        "game_date": "2026-10-25",
    }
    df_after = pd.concat([df_base, pd.DataFrame([spike_row])], ignore_index=True)

    res_after = compute_season_multipliers(
        ratings_df=df_after,
        season_start_date="2026-10-20",
        curve="candidate_d",
    ).set_index("personId")

    bonus_after = res_after.loc[5, "season_performance_bonus"]
    avg_after = res_after.loc[5, "season_avg_rating"]
    pct_after = res_after.loc[5, "season_percentile"]

    # 1. Season average shifts smoothly: (10 * avg_before + 6.5) / 11
    expected_new_avg = (10.0 * avg_before + 6.5) / 11.0
    assert avg_after == pytest.approx(expected_new_avg, abs=1e-3)
    assert avg_after - avg_before < 0.6  # Shift is bounded

    # 2. Bonus drops moderately but is NOT wiped out (player still retains substantial bonus)
    assert bonus_after > 1.5, f"Bonus was wiped out: {bonus_before} -> {bonus_after}"
    assert bonus_after <= bonus_before


# ==============================================================================
# 7. Small Populations and Edge Cases
# ==============================================================================
def test_empty_dataset_returns_clean_schema():
    """Verify empty input returns empty DataFrame with expected columns."""
    df_empty = pd.DataFrame(columns=["gameId", "personId", "rating", "game_date"])
    res = compute_season_multipliers(ratings_df=df_empty)
    assert res.empty
    expected_cols = [
        "personId", "playerName", "season_games_played", "season_game_count",
        "season_avg_rating", "season_average_rating",
        "season_percentile", "season_performance_bonus", "season_multiplier_eligible"
    ]
    assert list(res.columns) == expected_cols


def test_single_eligible_player():
    """Verify single eligible player receives 100th percentile."""
    data = [
        {"gameId": "G1", "personId": 100, "rating": 3.0, "game_date": "2026-10-20"},
        {"gameId": "G2", "personId": 100, "rating": 3.0, "game_date": "2026-10-21"},
        {"gameId": "G3", "personId": 100, "rating": 3.0, "game_date": "2026-10-22"},
    ]
    res = compute_season_multipliers(ratings_df=pd.DataFrame(data), curve="candidate_d")
    assert len(res) == 1
    assert res.iloc[0]["season_percentile"] == 100.0
    assert res.iloc[0]["season_performance_bonus"] == 0.0


def test_determinism_and_idempotency():
    """Verify running the engine multiple times yields identical DataFrames."""
    data = [
        {"gameId": f"G_{i}", "personId": pid, "rating": float(pid * 0.7 + i), "game_date": "2026-10-20"}
        for pid in range(1, 10) for i in range(1, 5)
    ]
    df = pd.DataFrame(data)

    res1 = compute_season_multipliers(ratings_df=df)
    res2 = compute_season_multipliers(ratings_df=df)

    pd.testing.assert_frame_equal(res1, res2)


# ==============================================================================
# 8. Real Historical Dataset Validation
# ==============================================================================
def test_real_historical_dataset_execution():
    """Run against the actual data/player_game_ratings_with_dates.parquet."""
    real_path = os.path.join(BASE_DIR, "data", "player_game_ratings_with_dates.parquet")
    if not os.path.exists(real_path):
        pytest.skip(f"Historical file not found at {real_path}")

    res = compute_season_multipliers(input_path=real_path, curve="candidate_d")
    assert not res.empty
    assert len(res) >= 300  # Over 300 current-season active players

    # Verify column constraints
    assert res["season_games_played"].min() >= 1
    assert res["season_performance_bonus"].min() >= 0.0
    assert res["season_performance_bonus"].max() <= 3.0

    eligible = res[res["season_multiplier_eligible"]]
    ineligible = res[~res["season_multiplier_eligible"]]

    assert len(eligible) >= 100
    assert len(ineligible) >= 200

    # Ineligible must be all 0.0
    assert (ineligible["season_performance_bonus"] == 0.0).all()

    # Eligible must have percentiles [0.0, 100.0]
    assert eligible["season_percentile"].min() == 0.0
    assert eligible["season_percentile"].max() == 100.0

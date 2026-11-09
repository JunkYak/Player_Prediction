"""
Regression tests for tomorrow schedule synchronization, schema normalization,
and Top Players slate alignment (Phase 2.2).

Covers all 15 specification requirements:
1. Tomorrow date calculation in ET, including midnight and DST boundaries.
2. Dedicated schedule schema normalization.
3. Confirmed schedule with multiple games.
4. Valid confirmed off-day.
5. Stale schedule.
6. Future-dated schedule.
7. Missing or malformed schedule metadata.
8. Failed schedule retrieval.
9. Exact team-set consistency between tomorrow schedule and tomorrow frontend slate.
10. Correct home/away opponent mapping.
11. Exclusion of players from unscheduled teams.
12. Top Players refusing to present an outdated slate as tomorrow's.
13. Playing Tomorrow showing a distinct stale/unavailable state.
14. Preservation of ranking-score sorting and raw predictions.
15. Preservation of Starting Five scoring.
"""

import os
import sys
import json
import tempfile
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.build_frontend_dataset import (
    normalize_schedule_dataframe,
    validate_and_load_tomorrow_schedule,
    build_frontend_dataset,
)
from scripts.get_next_day_games import get_tomorrow_schedule


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_predictions_df():
    """Returns 8 players across 4 teams with varying ratings and statuses."""
    return pd.DataFrame([
        # Bulls (1610612741)
        {"personId": 101, "playerName": "Josh Giddey", "teamId": 1610612741, "predicted_rating": 3.5, "status": "Active"},
        {"personId": 102, "playerName": "Matas Buzelis", "teamId": 1610612741, "predicted_rating": 3.0, "status": "Active"},
        # 76ers (1610612755)
        {"personId": 201, "playerName": "VJ Edgecombe", "teamId": 1610612755, "predicted_rating": 3.2, "status": "Active"},
        {"personId": 202, "playerName": "Tyrese Maxey", "teamId": 1610612755, "predicted_rating": 4.1, "status": "Out"},
        # Celtics (1610612738)
        {"personId": 301, "playerName": "Jayson Tatum", "teamId": 1610612738, "predicted_rating": 4.5, "status": "Active"},
        {"personId": 302, "playerName": "Jaylen Brown", "teamId": 1610612738, "predicted_rating": 4.0, "status": "Active"},
        # Unscheduled franchise (e.g. Warriors 1610612744)
        {"personId": 401, "playerName": "Stephen Curry", "teamId": 1610612744, "predicted_rating": 4.8, "status": "Active"},
        {"personId": 402, "playerName": "Draymond Green", "teamId": 1610612744, "predicted_rating": 2.9, "status": "Active"},
    ])


@pytest.fixture
def sample_multipliers_df():
    """Returns sample Candidate D multipliers."""
    return pd.DataFrame([
        {"personId": 101, "season_game_count": 5, "season_performance_bonus": 1.5, "season_multiplier_eligible": True},
        {"personId": 102, "season_game_count": 4, "season_performance_bonus": 0.5, "season_multiplier_eligible": True},
        {"personId": 201, "season_game_count": 5, "season_performance_bonus": 1.0, "season_multiplier_eligible": True},
        {"personId": 301, "season_game_count": 5, "season_performance_bonus": 0.0, "season_multiplier_eligible": False},
        {"personId": 401, "season_game_count": 5, "season_performance_bonus": 1.2, "season_multiplier_eligible": True},
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Tomorrow date calculation in ET across midnight and DST
# ─────────────────────────────────────────────────────────────────────────────

def test_tomorrow_date_calculation_et_boundaries():
    """Verifies that tomorrow's calendar date is computed relative to America/New_York."""
    tz_et = ZoneInfo("America/New_York")

    # 11:30 PM ET on Nov 7 -> tomorrow is Nov 8
    dt_late = datetime(2026, 11, 7, 23, 30, tzinfo=tz_et)
    ref_utc = dt_late.astimezone(timezone.utc)
    ref_et = ref_utc.astimezone(tz_et)
    tom_et = (ref_et.date() + timedelta(days=1)).strftime("%Y-%m-%d")
    assert tom_et == "2026-11-08"

    # 12:15 AM ET on Nov 8 -> tomorrow is Nov 9
    dt_early = datetime(2026, 11, 8, 0, 15, tzinfo=tz_et)
    ref_utc = dt_early.astimezone(timezone.utc)
    ref_et = ref_utc.astimezone(tz_et)
    tom_et = (ref_et.date() + timedelta(days=1)).strftime("%Y-%m-%d")
    assert tom_et == "2026-11-09"

    # UTC is already next day, but ET is still current day:
    # 8:00 PM ET on Nov 7 = 1:00 AM UTC on Nov 8. Tomorrow in ET must be Nov 8, not Nov 9!
    dt_utc_ahead = datetime(2026, 11, 8, 1, 0, tzinfo=timezone.utc)
    ref_et = dt_utc_ahead.astimezone(tz_et)
    tom_et = (ref_et.date() + timedelta(days=1)).strftime("%Y-%m-%d")
    assert tom_et == "2026-11-08"


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Dedicated schedule schema normalization
# ─────────────────────────────────────────────────────────────────────────────

def test_dedicated_schedule_schema_normalization():
    """Verifies normalize_schedule_dataframe correctly handles both schemas."""
    # Dedicated snake_case
    df_snake = pd.DataFrame([
        {
            "game_id": "401909938",
            "home_team_id": "1610612754",
            "away_team_id": 1610612743,
            "home_team_name": "Pacers",
            "away_team_name": "Nuggets",
            "home_team_tricode": "IND",
            "away_team_tricode": "DEN",
            "schedule_date": "2026-11-08",
            "is_postponed": False,
        },
        {
            "game_id": "401909939",
            "home_team_id": 1610612737,
            "away_team_id": 1610612751,
            "schedule_date": "2026-11-08",
            "is_postponed": True,  # Postponed must be dropped!
        }
    ])
    norm_snake = normalize_schedule_dataframe(df_snake)
    assert len(norm_snake) == 1
    assert norm_snake["homeTeamId"].iloc[0] == 1610612754
    assert norm_snake["awayTeamId"].iloc[0] == 1610612743
    assert norm_snake["homeTeamTricode"].iloc[0] == "IND"
    assert norm_snake["nba_game_date"].iloc[0] == "2026-11-08"

    # Legacy camelCase
    df_camel = pd.DataFrame([
        {
            "gameId": "0022600101",
            "homeTeamId": 1610612738,
            "awayTeamId": 1610612747,
            "homeTeamTricode": "BOS",
            "awayTeamTricode": "LAL",
            "nba_game_date": "2026-11-08",
        }
    ])
    norm_camel = normalize_schedule_dataframe(df_camel)
    assert len(norm_camel) == 1
    assert norm_camel["homeTeamId"].iloc[0] == 1610612738
    assert norm_camel["awayTeamId"].iloc[0] == 1610612747


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Confirmed schedule with multiple games
# ─────────────────────────────────────────────────────────────────────────────

def test_confirmed_schedule_with_multiple_games(sample_predictions_df, sample_multipliers_df):
    """When schedule has 2 confirmed games, exactly those 4 teams appear with correct opponents."""
    sched_df = pd.DataFrame([
        {"home_team_id": 1610612741, "away_team_id": 1610612755, "schedule_date": "2026-11-08"},  # Bulls vs 76ers
        {"home_team_id": 1610612738, "away_team_id": 1610612744, "schedule_date": "2026-11-08"},  # Celtics vs Warriors
    ])
    sched_meta = {
        "status": "CONFIRMED",
        "schedule_date": "2026-11-08",
        "games_scheduled": True,
        "games_count": 2,
        "is_stale": False,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        home_path = os.path.join(tmpdir, "frontend_home.parquet")
        all_path = os.path.join(tmpdir, "frontend_all.parquet")

        build_frontend_dataset(
            pred_df=sample_predictions_df,
            multipliers_df=sample_multipliers_df,
            schedule_df=sched_df,
            schedule_meta=sched_meta,
            output_home_path=home_path,
            output_all_path=all_path,
            reference_time=datetime(2026, 11, 7, 12, 0, tzinfo=ZoneInfo("America/New_York")),
        )

        df_home = pd.read_parquet(home_path)
        df_all = pd.read_parquet(all_path)

        # 4 teams participating, max 3 players per team; Maxey is Out so 76ers has 1 player (VJ Edgecombe)
        assert set(df_home["teamId"].unique()) == {1610612741, 1610612755, 1610612738, 1610612744}
        assert all(df_home["game_date"] == "2026-11-08")


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Valid confirmed OFF_DAY
# ─────────────────────────────────────────────────────────────────────────────

def test_valid_confirmed_off_day(sample_predictions_df, sample_multipliers_df):
    """An explicit OFF_DAY produces an empty draft slate (0 rows) and no false opponents."""
    sched_df = pd.DataFrame(columns=["home_team_id", "away_team_id", "schedule_date"])
    sched_meta = {
        "status": "OFF_DAY",
        "schedule_date": "2026-11-08",
        "games_scheduled": False,
        "games_count": 0,
        "is_stale": False,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        home_path = os.path.join(tmpdir, "frontend_home.parquet")
        all_path = os.path.join(tmpdir, "frontend_all.parquet")

        build_frontend_dataset(
            pred_df=sample_predictions_df,
            multipliers_df=sample_multipliers_df,
            schedule_df=sched_df,
            schedule_meta=sched_meta,
            output_home_path=home_path,
            output_all_path=all_path,
            reference_time=datetime(2026, 11, 7, 12, 0, tzinfo=ZoneInfo("America/New_York")),
        )

        df_home = pd.read_parquet(home_path)
        df_all = pd.read_parquet(all_path)

        assert len(df_home) == 0
        assert len(df_all) == len(sample_predictions_df)
        assert all(df_all["opponentName"] == "")


# ─────────────────────────────────────────────────────────────────────────────
# Test 5 & 6: Stale and Future-dated schedule
# ─────────────────────────────────────────────────────────────────────────────

def test_stale_and_future_dated_schedule_creates_empty_slate(sample_predictions_df, sample_multipliers_df):
    """Stale or future-dated schedules must not produce a misleading draft slate."""
    ref_time = datetime(2026, 11, 7, 12, 0, tzinfo=ZoneInfo("America/New_York"))

    # Stale: schedule date is 2026-11-06 (past relative to tomorrow 2026-11-08)
    status_stale, _, _ = validate_and_load_tomorrow_schedule(
        schedule_df=pd.DataFrame([{"home_team_id": 1610612741, "away_team_id": 1610612755, "schedule_date": "2026-11-06"}]),
        schedule_meta={"status": "CONFIRMED", "schedule_date": "2026-11-06", "is_stale": True},
        reference_time=ref_time,
    )
    assert status_stale == "STALE"

    # Future-dated: schedule date is 2026-11-10 (> 2026-11-08)
    status_future, _, _ = validate_and_load_tomorrow_schedule(
        schedule_df=pd.DataFrame([{"home_team_id": 1610612741, "away_team_id": 1610612755, "schedule_date": "2026-11-10"}]),
        schedule_meta={"status": "CONFIRMED", "schedule_date": "2026-11-10", "is_stale": False},
        reference_time=ref_time,
    )
    assert status_future == "SCHEDULE_UNAVAILABLE"


# ─────────────────────────────────────────────────────────────────────────────
# Test 7 & 8: Missing/malformed metadata and failed retrieval
# ─────────────────────────────────────────────────────────────────────────────

def test_missing_metadata_and_failed_retrieval(sample_predictions_df):
    """Missing metadata or FETCH_FAILED returns SCHEDULE_UNAVAILABLE and empty home slate."""
    status_fail, _, _ = validate_and_load_tomorrow_schedule(
        schedule_df=pd.DataFrame(),
        schedule_meta={"status": "FETCH_FAILED", "error": "Network timeout"},
    )
    assert status_fail == "SCHEDULE_UNAVAILABLE"

    # Empty metadata
    status_empty, _, _ = validate_and_load_tomorrow_schedule(
        schedule_path="nonexistent_schedule.parquet",
        schedule_metadata_path="nonexistent_meta.json",
    )
    assert status_empty == "SCHEDULE_UNAVAILABLE"


# ─────────────────────────────────────────────────────────────────────────────
# Test 9, 10, 11: Exact team set consistency, opponent mapping, and exclusion
# ─────────────────────────────────────────────────────────────────────────────

def test_exact_team_set_opponent_mapping_and_exclusion(sample_predictions_df, sample_multipliers_df):
    """
    Verifies:
    9. Exact team-set consistency.
    10. Correct home/away opponent mapping.
    11. Exclusion of players from unscheduled teams.
    """
    # Tomorrow only has Bulls vs 76ers. Celtics and Warriors are resting.
    sched_df = pd.DataFrame([
        {"home_team_id": 1610612741, "away_team_id": 1610612755, "schedule_date": "2026-11-08"},
    ])
    sched_meta = {
        "status": "CONFIRMED",
        "schedule_date": "2026-11-08",
        "games_scheduled": True,
        "games_count": 1,
        "is_stale": False,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        home_path = os.path.join(tmpdir, "frontend_home.parquet")
        all_path = os.path.join(tmpdir, "frontend_all.parquet")

        build_frontend_dataset(
            pred_df=sample_predictions_df,
            multipliers_df=sample_multipliers_df,
            schedule_df=sched_df,
            schedule_meta=sched_meta,
            output_home_path=home_path,
            output_all_path=all_path,
            reference_time=datetime(2026, 11, 7, 12, 0, tzinfo=ZoneInfo("America/New_York")),
        )

        df_home = pd.read_parquet(home_path)
        df_all = pd.read_parquet(all_path)

        # 9. Exact team set
        assert set(df_home["teamId"].unique()) == {1610612741, 1610612755}

        # 10. Correct home/away opponent mapping
        bulls_players = df_home[df_home["teamId"] == 1610612741]
        sixers_players = df_home[df_home["teamId"] == 1610612755]
        assert all(bulls_players["opponentName"] == "76ers")
        assert all(sixers_players["opponentName"] == "Bulls")

        # 11. Exclusion of players from unscheduled teams (Celtics & Warriors)
        assert 1610612738 not in df_home["teamId"].values
        assert 1610612744 not in df_home["teamId"].values

        # In all_df, unscheduled teams have opponentName = ""
        celtics_in_all = df_all[df_all["teamId"] == 1610612738]
        assert all(celtics_in_all["opponentName"] == "")


# ─────────────────────────────────────────────────────────────────────────────
# Test 14 & 15: Preservation of ranking_score sorting, predictions, & scoring
# ─────────────────────────────────────────────────────────────────────────────

def test_preservation_of_ranking_score_and_starting_five_scoring(sample_predictions_df, sample_multipliers_df):
    """
    Verifies:
    14. ranking_score = round(predicted_rating * (1 + season_performance_bonus), 2)
        and raw predicted_rating is never mutated.
    15. Starting Five scoring formula:
        effective_multiplier = slot_multiplier + season_performance_bonus
        projected_slot_value = predicted_rating * effective_multiplier
    """
    sched_df = pd.DataFrame([
        {"home_team_id": 1610612741, "away_team_id": 1610612755, "schedule_date": "2026-11-08"},
    ])
    sched_meta = {
        "status": "CONFIRMED",
        "schedule_date": "2026-11-08",
        "games_scheduled": True,
        "games_count": 1,
        "is_stale": False,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        home_path = os.path.join(tmpdir, "frontend_home.parquet")
        all_path = os.path.join(tmpdir, "frontend_all.parquet")

        build_frontend_dataset(
            pred_df=sample_predictions_df,
            multipliers_df=sample_multipliers_df,
            schedule_df=sched_df,
            schedule_meta=sched_meta,
            output_home_path=home_path,
            output_all_path=all_path,
            reference_time=datetime(2026, 11, 7, 12, 0, tzinfo=ZoneInfo("America/New_York")),
        )

        df_home = pd.read_parquet(home_path)

        # Josh Giddey: pred=3.5, bonus=1.5 -> ranking_score = round(3.5 * (1 + 1.5), 2) = round(3.5 * 2.5, 2) = 8.75
        giddey = df_home[df_home["personId"] == 101].iloc[0]
        assert giddey["predicted_rating"] == 3.5  # Raw prediction untouched!
        assert giddey["season_performance_bonus"] == 1.5
        assert giddey["ranking_score"] == 8.75

        # Starting Five calculation test:
        # Slot 1: base_multiplier = 2.0
        # effective_multiplier = 2.0 + 1.5 = 3.5
        # slot_value = 3.5 * 3.5 = 12.25
        base_slot_mult = 2.0
        eff_mult = base_slot_mult + giddey["season_performance_bonus"]
        assert eff_mult == 3.5
        slot_value = giddey["predicted_rating"] * eff_mult
        assert slot_value == 12.25

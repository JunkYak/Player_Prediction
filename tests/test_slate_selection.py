"""
tests/test_slate_selection.py

Deterministic tests for timezone-aware NBA slate selection.
Covers:
  - October 21, 2026 at 10 PM IST (the root-cause problem case)
  - October 22 morning IST games
  - Official NBA date differing from IST date
  - Games crossing midnight in IST
  - In-progress / live games (not replacing live games prematurely)
  - Completed games advancing to next slate
  - Postponed games handling
  - Multiple future slates
  - Off-days / zero scheduled games in target window
  - US Daylight Saving Time transition boundaries
  - Missing or malformed schedule timestamps fallback
"""

import pytest
import os
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.get_next_day_games import select_upcoming_slate, get_next_day_games, parse_game_timestamps

TZ_IST = ZoneInfo("Asia/Kolkata")
TZ_ET = ZoneInfo("America/New_York")
TZ_UTC = timezone.utc


def make_mock_game(
    game_id: str,
    home_id: int,
    away_id: int,
    game_date_et: str,
    time_utc_str: str,
    status: int = 1,
    status_text: str = "7:00 pm ET",
    game_et_str: str = ""
):
    return {
        "gameId": game_id,
        "homeTeam": {"teamId": home_id, "teamTricode": f"T{home_id}"},
        "awayTeam": {"teamId": away_id, "teamTricode": f"T{away_id}"},
        "gameStatus": status,
        "gameStatusText": status_text,
        "gameTimeUTC": time_utc_str,
        "gameEt": game_et_str or f"{game_date_et}T19:00:00Z",
    }


def test_october_21_10pm_ist_selects_october_21_us_slate():
    """
    ROOT CAUSE BUG REGRESSION TEST:
    On October 21, 2026, at 10:00 PM IST (16:30 UTC / 12:30 PM EDT):
    The immediate upcoming games are the October 21 US slate (tipping off October 22 morning IST).
    The pipeline MUST select the October 21 US slate (11 games), NOT jump to October 22 US games!
    """
    ref_time = datetime(2026, 10, 21, 22, 0, 0, tzinfo=TZ_IST)

    schedule_data = {
        "2026-10-20": [
            make_mock_game("0022600001", 1610612738, 1610612752, "2026-10-20", "2026-10-20T23:30:00Z", status=3, status_text="Final"),
        ],
        "2026-10-21": [
            # October 21 US games (tipoff 2026-10-21 19:00 ET = 2026-10-21 23:00 UTC = 2026-10-22 04:30 IST)
            make_mock_game("0022600085", 1610612753, 1610612737, "2026-10-21", "2026-10-21T23:00:00Z", status=1, status_text="7:00 pm ET"),
            make_mock_game("0022600005", 1610612747, 1610612744, "2026-10-21", "2026-10-22T02:00:00Z", status=1, status_text="10:00 pm ET"),
        ],
        "2026-10-22": [
            # October 22 US games (tipoff morning of Oct 23 IST)
            make_mock_game("0022600006", 1610612755, 1610612739, "2026-10-22", "2026-10-22T23:00:00Z", status=1, status_text="7:00 pm ET"),
        ],
    }

    def mock_fetcher(dt_str):
        return schedule_data.get(dt_str, [])

    df = select_upcoming_slate(reference_time=ref_time, schedule_fetcher=mock_fetcher)
    assert not df.empty
    assert len(df) == 2
    assert set(df["gameId"]) == {"0022600085", "0022600005"}
    assert df["nba_game_date"].iloc[0] == "2026-10-21"
    # IST date should be Oct 22 morning
    assert df["ist_game_date"].iloc[0] == "2026-10-22"
    assert df["ist_tipoff_time"].iloc[0] == "04:30"


def test_games_differing_nba_date_and_ist_date():
    """
    US evening games starting on Oct 21 at 7:00 PM ET start at 4:30 AM IST on Oct 22.
    Official NBA game date must remain 2026-10-21, while ist_game_date is 2026-10-22.
    """
    raw_game = make_mock_game(
        "0022600085", 1610612753, 1610612737,
        "2026-10-21", "2026-10-21T23:00:00Z", status=1, status_text="7:00 pm ET",
        game_et_str="2026-10-21T19:00:00Z"
    )
    meta = parse_game_timestamps(raw_game, default_nba_date="2026-10-21")
    assert meta["nba_game_date"] == "2026-10-21"
    assert meta["ist_game_date"] == "2026-10-22"
    assert meta["ist_tipoff_time"] == "04:30"


def test_games_crossing_midnight_in_ist():
    """
    Afternoon US games on a Sunday (e.g. 1:00 PM ET = 10:30 PM IST same day)
    vs evening games on the same US slate (7:00 PM ET = 4:30 AM IST next day).
    All games in that slate must be grouped together under the same official NBA date.
    """
    ref_time = datetime(2026, 11, 15, 20, 0, 0, tzinfo=TZ_IST)
    schedule_data = {
        "2026-11-14": [],
        "2026-11-15": [
            # 1:00 PM EST = 18:00 UTC = 23:30 IST on Nov 15
            make_mock_game("G_AFT", 1610612738, 1610612751, "2026-11-15", "2026-11-15T18:00:00Z", status=1, status_text="1:00 pm ET"),
            # 7:00 PM EST = 00:00 UTC (Nov 16) = 05:30 IST on Nov 16
            make_mock_game("G_EVE", 1610612747, 1610612744, "2026-11-15", "2026-11-16T00:00:00Z", status=1, status_text="7:00 pm ET"),
        ],
        "2026-11-16": []
    }
    df = select_upcoming_slate(reference_time=ref_time, schedule_fetcher=lambda dt: schedule_data.get(dt, []))
    assert len(df) == 2
    assert set(df["gameId"]) == {"G_AFT", "G_EVE"}
    assert all(df["nba_game_date"] == "2026-11-15")
    # G_AFT is Nov 15 in IST, G_EVE is Nov 16 in IST
    row_aft = df[df["gameId"] == "G_AFT"].iloc[0]
    row_eve = df[df["gameId"] == "G_EVE"].iloc[0]
    assert row_aft["ist_game_date"] == "2026-11-15"
    assert row_eve["ist_game_date"] == "2026-11-16"


def test_in_progress_games_are_preserved_not_replaced():
    """
    When games on Oct 21 US date are currently underway in morning of Oct 22 IST (e.g. 7:00 AM IST),
    the active slate must remain Oct 21 US games, not jump forward to Oct 22 US games.
    """
    ref_time = datetime(2026, 10, 22, 7, 0, 0, tzinfo=TZ_IST)  # 01:30 UTC
    schedule_data = {
        "2026-10-21": [
            # Game 1 is finished, Game 2 is in progress (Q3)
            make_mock_game("G1", 1610612753, 1610612737, "2026-10-21", "2026-10-21T23:00:00Z", status=3, status_text="Final"),
            make_mock_game("G2", 1610612747, 1610612744, "2026-10-21", "2026-10-22T02:00:00Z", status=2, status_text="Q3 08:14"),
        ],
        "2026-10-22": [
            make_mock_game("G3", 1610612755, 1610612739, "2026-10-22", "2026-10-22T23:00:00Z", status=1, status_text="7:00 pm ET"),
        ]
    }
    df = select_upcoming_slate(reference_time=ref_time, schedule_fetcher=lambda dt: schedule_data.get(dt, []))
    assert len(df) == 2
    assert set(df["gameId"]) == {"G1", "G2"}
    assert df["nba_game_date"].iloc[0] == "2026-10-21"


def test_after_all_games_finish_advances_to_next_slate():
    """
    Once all games on Oct 21 US date have finished (e.g. at 11:30 AM IST on Oct 22),
    the pipeline advances cleanly to the Oct 22 US slate.
    """
    ref_time = datetime(2026, 10, 22, 11, 30, 0, tzinfo=TZ_IST)
    schedule_data = {
        "2026-10-21": [
            make_mock_game("G1", 1610612753, 1610612737, "2026-10-21", "2026-10-21T23:00:00Z", status=3, status_text="Final"),
            make_mock_game("G2", 1610612747, 1610612744, "2026-10-21", "2026-10-22T02:00:00Z", status=3, status_text="Final"),
        ],
        "2026-10-22": [
            make_mock_game("G3", 1610612755, 1610612739, "2026-10-22", "2026-10-22T23:00:00Z", status=1, status_text="7:00 pm ET"),
        ],
        "2026-10-23": [
            make_mock_game("G4", 1610612738, 1610612752, "2026-10-23", "2026-10-23T23:00:00Z", status=1, status_text="7:00 pm ET"),
        ]
    }
    df = select_upcoming_slate(reference_time=ref_time, schedule_fetcher=lambda dt: schedule_data.get(dt, []))
    assert len(df) == 1
    assert df["gameId"].iloc[0] == "G3"
    assert df["nba_game_date"].iloc[0] == "2026-10-22"


def test_postponed_game_is_excluded_from_playable_slate():
    """
    If a game is marked PPD or Postponed, it must be excluded from the playable slate.
    """
    ref_time = datetime(2026, 10, 22, 15, 0, 0, tzinfo=TZ_IST)
    schedule_data = {
        "2026-10-22": [
            make_mock_game("G_PPD", 1610612755, 1610612739, "2026-10-22", "2026-10-22T23:00:00Z", status=1, status_text="PPD"),
            make_mock_game("G_OK", 1610612760, 1610612743, "2026-10-22", "2026-10-23T01:30:00Z", status=1, status_text="9:30 pm ET"),
        ]
    }
    df = select_upcoming_slate(reference_time=ref_time, schedule_fetcher=lambda dt: schedule_data.get(dt, []))
    assert len(df) == 1
    assert df["gameId"].iloc[0] == "G_OK"


def test_off_day_returns_clean_empty_slate_without_jumping():
    """
    If target day has 0 games scheduled (e.g. Thanksgiving or offseason),
    the system must return an empty DataFrame (0 rows), not jump ahead to subsequent days.
    """
    ref_time = datetime(2026, 11, 26, 12, 0, 0, tzinfo=TZ_IST)  # Thanksgiving off-day
    schedule_data = {
        "2026-11-25": [
            make_mock_game("G_PREV", 1610612755, 1610612739, "2026-11-25", "2026-11-25T23:00:00Z", status=3, status_text="Final"),
        ],
        "2026-11-26": [],  # Off day
        "2026-11-27": [
            make_mock_game("G_NEXT", 1610612738, 1610612752, "2026-11-27", "2026-11-27T23:00:00Z", status=1, status_text="7:00 pm ET"),
        ]
    }
    df = select_upcoming_slate(reference_time=ref_time, schedule_fetcher=lambda dt: schedule_data.get(dt, []))
    assert df.empty
    assert len(df) == 0


def test_dst_transition_us_eastern():
    """
    Verify timezone handling across US Daylight Saving Time transition.
    In 2026, DST ends on Sunday, November 1, 2026 at 2:00 AM EDT (shifts EDT UTC-4 -> EST UTC-5).
    """
    # 1. Before DST end (EDT: UTC-4)
    # Oct 31, 2026, 19:00 EDT = 23:00 UTC = Nov 1 04:30 IST
    raw_before = make_mock_game("G_EDT", 1, 2, "2026-10-31", "2026-10-31T23:00:00Z", game_et_str="2026-10-31T19:00:00Z")
    meta_before = parse_game_timestamps(raw_before, "2026-10-31")
    assert meta_before["ist_tipoff_time"] == "04:30"

    # 2. After DST end (EST: UTC-5)
    # Nov 2, 2026, 19:00 EST = 00:00 UTC (Nov 3) = Nov 3 05:30 IST
    raw_after = make_mock_game("G_EST", 1, 2, "2026-11-02", "2026-11-03T00:00:00Z", game_et_str="2026-11-02T19:00:00Z")
    meta_after = parse_game_timestamps(raw_after, "2026-11-02")
    assert meta_after["ist_tipoff_time"] == "05:30"


def test_missing_and_malformed_timestamps_handled_safely():
    """
    Missing or malformed gameTimeUTC should fall back gracefully to gameEt or default 19:00 ET.
    """
    game_malformed = {
        "gameId": "G_BAD",
        "homeTeam": {"teamId": 10, "teamTricode": "HOM"},
        "awayTeam": {"teamId": 20, "teamTricode": "AWY"},
        "gameStatus": 1,
        "gameStatusText": "7:00 pm ET",
        "gameTimeUTC": "invalid-timestamp",
        "gameEt": "2026-12-01T19:00:00Z",
    }
    meta = parse_game_timestamps(game_malformed, default_nba_date="2026-12-01")
    assert meta["nba_game_date"] == "2026-12-01"
    assert meta["ist_game_date"] == "2026-12-02"
    assert meta["ist_tipoff_time"] == "05:30"


def test_get_next_day_games_writes_valid_parquet_and_csv(tmp_path):
    """
    End-to-end execution of get_next_day_games writing to disk artifacts.
    """
    parquet_path = str(tmp_path / "next_day_games.parquet")
    csv_path = str(tmp_path / "next_day_team_ids.csv")

    ref_time = datetime(2026, 10, 21, 22, 0, 0, tzinfo=TZ_IST)
    schedule_data = {
        "2026-10-21": [
            make_mock_game("0022600085", 1610612753, 1610612737, "2026-10-21", "2026-10-21T23:00:00Z"),
        ]
    }

    out_file = get_next_day_games(
        reference_time=ref_time,
        schedule_fetcher=lambda dt: schedule_data.get(dt, []),
        output_path=parquet_path,
        team_ids_path=csv_path,
    )
    assert out_file == parquet_path
    saved_df = pd.read_parquet(parquet_path)
    assert len(saved_df) == 1
    assert "homeTeamId" in saved_df.columns
    assert "awayTeamId" in saved_df.columns
    assert "nba_game_date" in saved_df.columns
    assert saved_df["homeTeamId"].iloc[0] == 1610612753
    assert saved_df["awayTeamId"].iloc[0] == 1610612737

    teams_csv = pd.read_csv(csv_path, header=None)
    assert set(teams_csv[0]) == {1610612753, 1610612737}


def test_espn_fallback_when_nba_api_fails(monkeypatch):
    """
    When stats.nba.com times out, default_nba_scoreboard_fetcher must fall back to ESPN API.
    """
    from scripts.get_next_day_games import default_nba_scoreboard_fetcher

    def mock_failing_nba(dt):
        return []

    mock_espn_games = [
        make_mock_game("ESPN_1", 1610612749, 1610612741, "2026-10-23", "2026-10-23T20:00:00Z", status=1, status_text="Scheduled")
    ]

    monkeypatch.setattr("scripts.get_next_day_games.fetch_nba_api_scoreboard", mock_failing_nba)
    monkeypatch.setattr("scripts.get_next_day_games.fetch_espn_scoreboard", lambda dt: mock_espn_games if dt == "2026-10-23" else [])

    games = default_nba_scoreboard_fetcher("2026-10-23")
    assert len(games) == 1
    assert games[0]["gameId"] == "ESPN_1"
    assert games[0]["homeTeam"]["teamId"] == 1610612749


def test_stale_cached_games_rejected_on_live_timeout(tmp_path):
    """
    CRITICAL REGRESSION TEST:
    If live fetch returns 0 games and an existing cached file exists,
    but the cached games are from a past date, the cache MUST BE DISCARDED,
    preventing yesterday's finished games from persisting into today's prediction.
    """
    parquet_path = str(tmp_path / "next_day_games.parquet")
    csv_path = str(tmp_path / "next_day_team_ids.csv")

    # Seed the cache with October 22 games
    stale_games = pd.DataFrame([{
        "gameId": "0022600006",
        "homeTeamId": 1610612755,
        "awayTeamId": 1610612739,
        "homeTeamTricode": "PHI",
        "awayTeamTricode": "CLE",
        "nba_game_date": "2026-10-22",
        "game_time_utc": "2026-10-22T23:00:00+00:00",
        "ist_game_date": "2026-10-23",
        "ist_tipoff_time": "04:30",
        "game_status": 1,
        "game_status_text": "7:00 pm ET",
    }])
    stale_games.to_parquet(parquet_path, index=False)

    # Reference time is October 23, 2026, at 13:00 IST (07:30 UTC / 03:30 ET)
    ref_time = datetime(2026, 10, 23, 13, 0, 0, tzinfo=TZ_IST)

    # Live query returns empty (e.g. timeout)
    out_file = get_next_day_games(
        reference_time=ref_time,
        schedule_fetcher=lambda dt: [],
        output_path=parquet_path,
        team_ids_path=csv_path,
    )

    saved_df = pd.read_parquet(parquet_path)
    # Stale cache must be rejected -> empty DataFrame
    assert len(saved_df) == 0


def test_schedule_rollover_when_today_completed():
    """
    When all games on curr_et_date (Oct 22) are completed, the selector MUST advance
    to the next scheduled slate (Oct 23).
    """
    ref_time = datetime(2026, 10, 23, 13, 0, 0, tzinfo=TZ_IST)  # Oct 23 03:30 ET

    schedule = {
        "2026-10-22": [
            make_mock_game("G_OLD_1", 1610612755, 1610612739, "2026-10-22", "2026-10-22T23:00:00Z", status=3, status_text="Final"),
            make_mock_game("G_OLD_2", 1610612760, 1610612743, "2026-10-22", "2026-10-23T01:30:00Z", status=3, status_text="Final"),
        ],
        "2026-10-23": [
            make_mock_game("G_NEW_1", 1610612749, 1610612741, "2026-10-23", "2026-10-23T20:00:00Z", status=1, status_text="4:00 pm ET"),
            make_mock_game("G_NEW_2", 1610612738, 1610612752, "2026-10-23", "2026-10-23T23:00:00Z", status=1, status_text="7:00 pm ET"),
        ],
    }

    df = select_upcoming_slate(reference_time=ref_time, schedule_fetcher=lambda dt: schedule.get(dt, []))
    assert not df.empty
    assert len(df) == 2
    assert set(df["gameId"]) == {"G_NEW_1", "G_NEW_2"}
    assert df["nba_game_date"].iloc[0] == "2026-10-23"


def test_off_day_returns_empty_slate():
    """
    If today has 0 scheduled games (genuine off-day), returns clean empty DataFrame.
    """
    ref_time = datetime(2026, 11, 10, 12, 0, 0, tzinfo=TZ_ET)  # Nov 10 noon ET

    schedule = {
        "2026-11-09": [
            make_mock_game("G_PREV", 1610612755, 1610612739, "2026-11-09", "2026-11-09T23:00:00Z", status=3, status_text="Final"),
        ],
        "2026-11-10": [],  # Off day
        "2026-11-11": [
            make_mock_game("G_AHEAD", 1610612738, 1610612752, "2026-11-11", "2026-11-11T23:00:00Z", status=1, status_text="7:00 pm ET"),
        ],
    }

    df = select_upcoming_slate(reference_time=ref_time, schedule_fetcher=lambda dt: schedule.get(dt, []))
    assert df.empty


def test_publication_guard_rejects_stale_slate(tmp_path):
    """
    Publication guard in azure_blob.validate_serving_parquet must reject a stale slate.
    """
    from scripts.azure_blob import validate_serving_parquet
    from api.loader import REQUIRED_COLUMNS

    fake_home = pd.DataFrame([{col: 1 for col in REQUIRED_COLUMNS}])
    fake_home["game_date"] = "2026-10-20"  # Older than yesterday

    test_file = str(tmp_path / "frontend_home.parquet")
    fake_home.to_parquet(test_file, index=False)

    with pytest.raises(ValueError, match="failed slate freshness validation"):
        validate_serving_parquet(test_file, "frontend_home.parquet", allow_zero_rows=True)


def test_publication_guard_accepts_valid_slate(tmp_path):
    """
    Publication guard accepts current or upcoming slate dates.
    """
    from scripts.azure_blob import validate_serving_parquet
    from api.loader import REQUIRED_COLUMNS

    now_et = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    fake_home = pd.DataFrame([{col: 1 for col in REQUIRED_COLUMNS}])
    fake_home["game_date"] = now_et

    test_file = str(tmp_path / "frontend_home.parquet")
    fake_home.to_parquet(test_file, index=False)

    meta = validate_serving_parquet(test_file, "frontend_home.parquet", allow_zero_rows=True)
    assert meta["row_count"] == 1


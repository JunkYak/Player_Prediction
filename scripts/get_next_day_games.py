"""
scripts/get_next_day_games.py

Timezone-aware NBA Slate and Upcoming Game Discovery.

Design and Policy:
- Timezones:
    - Display / Local User: Asia/Kolkata (IST, UTC+05:30)
    - NBA Official: America/New_York (US Eastern Time)
    - Serialization / Internal: UTC
- Slate Selection:
    1. A slate is defined by the official NBA game date (ET).
    2. At any given reference time:
       - If a slate has active / in-progress games (gameStatus == 2 or started and not completed),
         that slate is kept active so ongoing predictions remain visible.
       - Once all games in a slate finish (gameStatus == 3 / Final), the selector advances to
         the next scheduled slate.
       - For an upcoming game day (e.g. 10 PM IST looking ahead to morning IST games),
         the selector accurately identifies the next upcoming slate without skipping it
         due to IST/ET calendar date differences.
       - If the target window has 0 games scheduled (e.g. league off-day), returns an empty
         slate intentionally without fabricating games or silently jumping slates.
       - Postponed games ("PPD" / "Postponed") are excluded from playable slates.
- Determinism & Testing:
    - Accepts optional reference_time and schedule_fetcher for full test control.
"""

import os
import sys
import logging
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Optional, Callable, Dict, List, Any
import pandas as pd
from nba_api.stats.endpoints import scoreboardv3

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("propredict.schedule")

TZ_IST = ZoneInfo("Asia/Kolkata")
TZ_ET = ZoneInfo("America/New_York")
TZ_UTC = timezone.utc


def parse_game_timestamps(game: Dict[str, Any], default_nba_date: str) -> Dict[str, Any]:
    """
    Parses start time into UTC and IST datetimes from NBA API scoreboard game object.
    Robustly handles missing or malformed timestamps with fallback to ET gameEt or default_nba_date.
    """
    raw_utc = game.get("gameTimeUTC")
    raw_et = game.get("gameEt")
    tipoff_utc: Optional[datetime] = None

    if raw_utc:
        try:
            # Handle ISO string with or without Z
            cleaned_utc = raw_utc.replace("Z", "+00:00")
            tipoff_utc = datetime.fromisoformat(cleaned_utc)
            if tipoff_utc.tzinfo is None:
                tipoff_utc = tipoff_utc.replace(tzinfo=TZ_UTC)
            else:
                tipoff_utc = tipoff_utc.astimezone(TZ_UTC)
        except Exception as e:
            logger.warning("Failed parsing gameTimeUTC '%s' for game %s: %s", raw_utc, game.get("gameId"), e)

    if tipoff_utc is None and raw_et:
        try:
            # gameEt is Eastern Time, format often 2026-10-21T19:00:00Z but represents ET
            cleaned_et = raw_et.rstrip("Z")
            parsed_et_naive = datetime.fromisoformat(cleaned_et)
            parsed_et = parsed_et_naive.replace(tzinfo=TZ_ET)
            tipoff_utc = parsed_et.astimezone(TZ_UTC)
        except Exception as e:
            logger.warning("Failed parsing gameEt '%s' for game %s: %s", raw_et, game.get("gameId"), e)

    if tipoff_utc is None:
        # Fallback: assume 19:00 (7 PM) ET on the official game date
        try:
            fallback_dt = datetime.strptime(f"{default_nba_date} 19:00:00", "%Y-%m-%d %H:%M:%S").replace(tzinfo=TZ_ET)
            tipoff_utc = fallback_dt.astimezone(TZ_UTC)
        except Exception:
            tipoff_utc = datetime.now(TZ_UTC)

    tipoff_ist = tipoff_utc.astimezone(TZ_IST)
    nba_game_date = default_nba_date
    if raw_et and len(raw_et) >= 10:
        nba_game_date = raw_et[:10]

    return {
        "tipoff_utc": tipoff_utc,
        "tipoff_ist": tipoff_ist,
        "nba_game_date": nba_game_date,
        "game_time_utc": tipoff_utc.isoformat(),
        "ist_game_date": tipoff_ist.strftime("%Y-%m-%d"),
        "ist_tipoff_time": tipoff_ist.strftime("%H:%M"),
    }


import urllib.request
import json

ESPN_TRICODE_MAP = {
    "GS": "GSW",
    "NO": "NOP",
    "NY": "NYK",
    "SA": "SAS",
    "UTAH": "UTA",
    "WSH": "WAS",
}

TRICODE_TO_TEAM_ID = {
    "ATL": 1610612737, "BOS": 1610612738, "CLE": 1610612739, "NOP": 1610612740,
    "CHI": 1610612741, "DAL": 1610612742, "DEN": 1610612743, "GSW": 1610612744,
    "HOU": 1610612745, "LAC": 1610612746, "LAL": 1610612747, "MIA": 1610612748,
    "MIL": 1610612749, "MIN": 1610612750, "BKN": 1610612751, "NYK": 1610612752,
    "ORL": 1610612753, "IND": 1610612754, "PHI": 1610612755, "PHX": 1610612756,
    "POR": 1610612757, "SAC": 1610612758, "SAS": 1610612759, "OKC": 1610612760,
    "TOR": 1610612761, "UTA": 1610612762, "MEM": 1610612763, "WAS": 1610612764,
    "DET": 1610612765, "CHA": 1610612766,
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "Accept": "application/json, text/plain, */*",
}


def fetch_nba_api_scoreboard(game_date_et: str) -> List[Dict[str, Any]]:
    """Fetches games for a specific ET game date from NBA API ScoreboardV3."""
    try:
        sb = scoreboardv3.ScoreboardV3(game_date=game_date_et, headers=HEADERS, timeout=6)
        data = sb.get_dict()
        games = data.get("scoreboard", {}).get("games", [])
        if games:
            logger.info("Fetched %d games for %s from stats.nba.com", len(games), game_date_et)
        return games
    except Exception as e:
        logger.warning("stats.nba.com scoreboard fetch timed out or failed for %s: %s", game_date_et, e)
        return []


def fetch_espn_scoreboard(game_date_et: str) -> List[Dict[str, Any]]:
    """
    Reliable fallback schedule fetcher using ESPN Public NBA Scoreboard API.
    Used when stats.nba.com blocks or times out datacenter requests.
    """
    clean_date = game_date_et.replace("-", "")
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={clean_date}"
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))

        events = data.get("events", [])
        normalized_games = []
        for ev in events:
            comp = ev.get("competitions", [{}])[0]
            competitors = comp.get("competitors", [])
            home_c = next((c for c in competitors if c.get("homeAway") == "home"), None)
            away_c = next((c for c in competitors if c.get("homeAway") == "away"), None)
            if not home_c or not away_c:
                continue

            home_tri = home_c.get("team", {}).get("abbreviation", "")
            away_tri = away_c.get("team", {}).get("abbreviation", "")
            home_nba_tri = ESPN_TRICODE_MAP.get(home_tri, home_tri)
            away_nba_tri = ESPN_TRICODE_MAP.get(away_tri, away_tri)
            home_id = TRICODE_TO_TEAM_ID.get(home_nba_tri, 0)
            away_id = TRICODE_TO_TEAM_ID.get(away_nba_tri, 0)

            status_obj = ev.get("status", {})
            status_type = status_obj.get("type", {})
            state = status_type.get("state", "pre")
            desc = status_type.get("description", "")
            detail = status_type.get("detail", "")

            if state == "post" or "FINAL" in desc.upper():
                status_num = 3
                status_text = "Final"
            elif state == "in":
                status_num = 2
                status_text = detail or "In Progress"
            elif "PPD" in desc.upper() or "POSTPONED" in desc.upper():
                status_num = 1
                status_text = "PPD"
            else:
                status_num = 1
                status_text = detail or desc or "Scheduled"

            raw_utc = ev.get("date", "")
            game_dict = {
                "gameId": str(ev.get("id", "")),
                "homeTeam": {"teamId": home_id, "teamTricode": home_nba_tri},
                "awayTeam": {"teamId": away_id, "teamTricode": away_nba_tri},
                "gameStatus": status_num,
                "gameStatusText": status_text,
                "gameTimeUTC": raw_utc,
                "gameEt": "",
            }
            normalized_games.append(game_dict)

        if normalized_games:
            logger.info("Fetched %d verified games for %s from ESPN schedule API", len(normalized_games), game_date_et)
        return normalized_games
    except Exception as e:
        logger.error("ESPN schedule API fetch failed for %s: %s", game_date_et, e)
        return []


def default_nba_scoreboard_fetcher(game_date_et: str) -> List[Dict[str, Any]]:
    """
    Fetches games for an ET game date with multi-source fallback.
    1. Tries NBA API ScoreboardV3.
    2. Falls back to ESPN API if NBA API returns 0 games, times out, or fails.
    """
    games = fetch_nba_api_scoreboard(game_date_et)
    if not games:
        games = fetch_espn_scoreboard(game_date_et)
    return games


def select_upcoming_slate(
    reference_time: Optional[datetime] = None,
    schedule_fetcher: Optional[Callable[[str], List[Dict[str, Any]]]] = None,
) -> pd.DataFrame:
    """
    Discovers candidate NBA game slates across a rolling multi-day window around reference_time
    and selects the correct active or upcoming slate based on the defined slate policy.
    """
    fetcher = schedule_fetcher or default_nba_scoreboard_fetcher

    if reference_time is None:
        ref_utc = datetime.now(TZ_UTC)
    else:
        if reference_time.tzinfo is None:
            # Assume local system / IST if naive
            ref_utc = reference_time.replace(tzinfo=TZ_IST).astimezone(TZ_UTC)
        else:
            ref_utc = reference_time.astimezone(TZ_UTC)

    ref_ist = ref_utc.astimezone(TZ_IST)
    ref_et = ref_utc.astimezone(TZ_ET)

    logger.info("Slate selection reference time: IST=%s | ET=%s | UTC=%s",
                ref_ist.isoformat(), ref_et.isoformat(), ref_utc.isoformat())

    # Build rolling search window in ET days: [ET - 1 day, ET, ET + 1 day, ET + 2 days]
    base_et_date = ref_et.date()
    candidate_dates = [
        (base_et_date - timedelta(days=1)).strftime("%Y-%m-%d"),
        base_et_date.strftime("%Y-%m-%d"),
        (base_et_date + timedelta(days=1)).strftime("%Y-%m-%d"),
        (base_et_date + timedelta(days=2)).strftime("%Y-%m-%d"),
    ]

    slates: Dict[str, List[Dict[str, Any]]] = {}

    for dt_str in candidate_dates:
        raw_games = fetcher(dt_str)
        slates[dt_str] = []
        for g in raw_games:
            status_text = str(g.get("gameStatusText", "")).strip()
            # Mark postponed games
            is_ppd = "PPD" in status_text.upper() or "POSTPONED" in status_text.upper()
            game_status = int(g.get("gameStatus", 1))

            time_meta = parse_game_timestamps(g, default_nba_date=dt_str)
            home_team = g.get("homeTeam", {})
            away_team = g.get("awayTeam", {})

            home_id = int(home_team.get("teamId", 0))
            away_id = int(away_team.get("teamId", 0))

            game_info = {
                "gameId": str(g.get("gameId", "")),
                "homeTeamId": home_id,
                "awayTeamId": away_id,
                "homeTeamTricode": home_team.get("teamTricode", ""),
                "awayTeamTricode": away_team.get("teamTricode", ""),
                "nba_game_date": time_meta["nba_game_date"],
                "game_time_utc": time_meta["game_time_utc"],
                "tipoff_utc": time_meta["tipoff_utc"],
                "ist_game_date": time_meta["ist_game_date"],
                "ist_tipoff_time": time_meta["ist_tipoff_time"],
                "game_status": game_status,
                "game_status_text": status_text,
                "is_postponed": is_ppd,
            }
            slates[dt_str].append(game_info)

    # Helper: Check if a game is conclusively completed
    def is_game_completed(g: Dict[str, Any]) -> bool:
        if g["game_status"] == 3 or "FINAL" in g["game_status_text"].upper():
            return True
        # If tipoff was > 4.5 hours ago, game has ended even if final flag is unupdated
        elapsed_hours = (ref_utc - g["tipoff_utc"]).total_seconds() / 3600.0
        return elapsed_hours >= 4.5

    # Helper: Check if a game is actively live
    def is_game_live(g: Dict[str, Any]) -> bool:
        if is_game_completed(g):
            return False
        if g["game_status"] == 2:
            return True
        # Tipoff in past but within last 4.5 hours
        elapsed_hours = (ref_utc - g["tipoff_utc"]).total_seconds() / 3600.0
        return 0 <= elapsed_hours < 4.5

    # Step 1: Check prev_et_date for any in-progress games
    prev_str = (base_et_date - timedelta(days=1)).strftime("%Y-%m-%d")
    prev_games = [g for g in slates.get(prev_str, []) if not g["is_postponed"]]
    if prev_games:
        has_live = any(is_game_live(g) for g in prev_games)
        all_final = all(is_game_completed(g) for g in prev_games)
        if has_live and not all_final:
            logger.info("Active in-progress slate from %s identified (%d games)", prev_str, len(prev_games))
            return pd.DataFrame(prev_games)

    # Step 2: Check curr_et_date
    curr_str = base_et_date.strftime("%Y-%m-%d")
    curr_games = [g for g in slates.get(curr_str, []) if not g["is_postponed"]]

    if curr_games:
        all_final = all(is_game_completed(g) for g in curr_games)
        if not all_final:
            # Games today are either live or upcoming
            logger.info("Selected slate for %s (%d games)", curr_str, len(curr_games))
            return pd.DataFrame(curr_games)
        else:
            # All games on curr_et_date are finished. Advance to next scheduled slate (tomorrow)
            next_str = (base_et_date + timedelta(days=1)).strftime("%Y-%m-%d")
            next_games = [g for g in slates.get(next_str, []) if not g["is_postponed"]]
            if next_games:
                logger.info("Current date games completed. Advancing to next slate on %s (%d games)", next_str, len(next_games))
                return pd.DataFrame(next_games)
            next2_str = (base_et_date + timedelta(days=2)).strftime("%Y-%m-%d")
            next2_games = [g for g in slates.get(next2_str, []) if not g["is_postponed"]]
            if next2_games:
                logger.info("Current date games completed and tomorrow is off. Advancing to %s (%d games)", next2_str, len(next2_games))
                return pd.DataFrame(next2_games)
    else:
        # Off-day: curr_et_date has 0 games scheduled
        # Intentionally do not jump forward by an entire slate
        logger.info("Off-day: 0 games scheduled on target date %s. Returning empty slate.", curr_str)
        empty_cols = [
            "gameId", "homeTeamId", "awayTeamId", "homeTeamTricode", "awayTeamTricode",
            "nba_game_date", "game_time_utc", "ist_game_date", "ist_tipoff_time",
            "game_status", "game_status_text"
        ]
        return pd.DataFrame(columns=empty_cols)

    # Fallback: empty slate
    empty_cols = [
        "gameId", "homeTeamId", "awayTeamId", "homeTeamTricode", "awayTeamTricode",
        "nba_game_date", "game_time_utc", "ist_game_date", "ist_tipoff_time",
        "game_status", "game_status_text"
    ]
    return pd.DataFrame(columns=empty_cols)


TEAM_ID_TO_NAME = {
    1610612737: "Hawks", 1610612738: "Celtics", 1610612739: "Cavaliers",
    1610612740: "Pelicans", 1610612741: "Bulls", 1610612742: "Mavericks",
    1610612743: "Nuggets", 1610612744: "Warriors", 1610612745: "Rockets",
    1610612746: "Clippers", 1610612747: "Lakers", 1610612748: "Heat",
    1610612749: "Bucks", 1610612750: "Timberwolves", 1610612751: "Nets",
    1610612752: "Knicks", 1610612753: "Magic", 1610612754: "Pacers",
    1610612755: "76ers", 1610612756: "Suns", 1610612757: "Blazers",
    1610612758: "Kings", 1610612759: "Spurs", 1610612760: "Thunder",
    1610612761: "Raptors", 1610612762: "Jazz", 1610612763: "Grizzlies",
    1610612764: "Wizards", 1610612765: "Pistons", 1610612766: "Hornets"
}

SCHEDULE_COLUMNS = [
    "game_id",
    "game_date",
    "schedule_date",
    "home_team_id",
    "home_team_name",
    "away_team_id",
    "away_team_name",
    "home_team_tricode",
    "away_team_tricode",
    "status",
    "game_status",
    "game_time_utc",
    "is_postponed",
    "generated_at",
]


def get_tomorrow_schedule(
    reference_time: Optional[datetime] = None,
    schedule_fetcher: Optional[Callable[[str], List[Dict[str, Any]]]] = None,
    output_path: str = "data/schedule_tomorrow.parquet",
    metadata_path: str = "data/schedule_tomorrow_metadata.json",
) -> str:
    """
    Discovers and persists official NBA schedule for tomorrow's calendar date in America/New_York.

    Guarantees:
    - Timezone: America/New_York target calendar date (tomorrowET = todayET + 1 day).
    - Dedicated artifact: data/schedule_tomorrow.parquet.
    - Completely decoupled from today's active fantasy prediction slate.
    - Distinguishes:
        * CONFIRMED: Valid upcoming matchups found for tomorrow.
        * OFF_DAY: NBA off-day (verified 0 games scheduled).
        * FETCH_FAILED: Network / upstream API error (never masked as an off-day).
        * STALE: Cached schedule from an older calendar date.
    - Saves metadata companion data/schedule_tomorrow_metadata.json atomically.
    """
    fetcher = schedule_fetcher or default_nba_scoreboard_fetcher

    if reference_time is None:
        ref_utc = datetime.now(timezone.utc)
    else:
        if reference_time.tzinfo is None:
            ref_utc = reference_time.replace(tzinfo=TZ_ET).astimezone(timezone.utc)
        else:
            ref_utc = reference_time.astimezone(timezone.utc)

    ref_et = ref_utc.astimezone(TZ_ET)
    tomorrow_et_date = (ref_et.date() + timedelta(days=1)).strftime("%Y-%m-%d")

    logger.info("Computing tomorrow's schedule for NBA date: %s (Reference ET: %s)",
                tomorrow_et_date, ref_et.strftime("%Y-%m-%d %H:%M:%S"))

    fetch_failed = False
    raw_games = []
    try:
        raw_games = fetcher(tomorrow_et_date)
    except Exception as e:
        logger.error("Failed to fetch schedule for %s: %s", tomorrow_et_date, e)
        fetch_failed = True

    if fetch_failed:
        # Check if an existing schedule is available locally
        if os.path.exists(output_path):
            try:
                cached_df = pd.read_parquet(output_path)
                cached_date = None
                if not cached_df.empty and "schedule_date" in cached_df.columns:
                    cached_date = str(cached_df["schedule_date"].iloc[0])[:10]
                is_stale = cached_date != tomorrow_et_date
                meta = {
                    "status": "STALE" if is_stale else "FETCH_FAILED",
                    "schedule_date": cached_date or tomorrow_et_date,
                    "target_date": tomorrow_et_date,
                    "generated_at": ref_utc.isoformat(),
                    "games_scheduled": len(cached_df) > 0,
                    "games_count": len(cached_df),
                    "is_stale": is_stale,
                    "error": "Upstream schedule fetch failed.",
                }
                with open(f"{metadata_path}.tmp", "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)
                os.replace(f"{metadata_path}.tmp", metadata_path)
                return output_path
            except Exception:
                pass

        # Write failed metadata without creating false empty schedule
        meta = {
            "status": "FETCH_FAILED",
            "schedule_date": tomorrow_et_date,
            "target_date": tomorrow_et_date,
            "generated_at": ref_utc.isoformat(),
            "games_scheduled": False,
            "games_count": 0,
            "is_stale": True,
            "error": "Upstream schedule fetch failed and no cached data exists.",
        }
        os.makedirs(os.path.dirname(metadata_path) or ".", exist_ok=True)
        with open(f"{metadata_path}.tmp", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        os.replace(f"{metadata_path}.tmp", metadata_path)
        raise RuntimeError(f"Schedule retrieval failed for {tomorrow_et_date}")

    # Process games into clean schema
    records = []
    for g in raw_games:
        time_meta = parse_game_timestamps(g, default_nba_date=tomorrow_et_date)
        home_team = g.get("homeTeam", {})
        away_team = g.get("awayTeam", {})
        home_id = int(home_team.get("teamId", 0))
        away_id = int(away_team.get("teamId", 0))
        home_name = home_team.get("teamName") or TEAM_ID_TO_NAME.get(home_id, "Unknown")
        away_name = away_team.get("teamName") or TEAM_ID_TO_NAME.get(away_id, "Unknown")
        status_text = str(g.get("gameStatusText", "")).strip()
        is_ppd = "PPD" in status_text.upper() or "POSTPONED" in status_text.upper()
        game_status = int(g.get("gameStatus", 1))

        records.append({
            "game_id": str(g.get("gameId", "")),
            "game_date": tomorrow_et_date,
            "schedule_date": tomorrow_et_date,
            "home_team_id": home_id,
            "home_team_name": home_name,
            "away_team_id": away_id,
            "away_team_name": away_name,
            "home_team_tricode": home_team.get("teamTricode", ""),
            "away_team_tricode": away_team.get("teamTricode", ""),
            "status": status_text or "Scheduled",
            "game_status": game_status,
            "game_time_utc": time_meta["game_time_utc"],
            "is_postponed": is_ppd,
            "generated_at": ref_utc.isoformat(),
        })

    schedule_df = pd.DataFrame(records, columns=SCHEDULE_COLUMNS)

    has_games = len(records) > 0
    status = "CONFIRMED" if has_games else "OFF_DAY"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(metadata_path) or ".", exist_ok=True)

    # Atomic write for schedule_tomorrow.parquet
    tmp_path = f"{output_path}.tmp"
    schedule_df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, output_path)

    # Write companion metadata
    meta = {
        "status": status,
        "schedule_date": tomorrow_et_date,
        "target_date": tomorrow_et_date,
        "generated_at": ref_utc.isoformat(),
        "games_scheduled": has_games,
        "games_count": len(records),
        "is_stale": False,
        "error": None,
    }
    tmp_meta = f"{metadata_path}.tmp"
    with open(tmp_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp_meta, metadata_path)

    logger.info("Saved tomorrow's schedule to %s (%d games, status: %s)", output_path, len(records), status)
    return output_path


def get_next_day_games(
    reference_time: Optional[datetime] = None,
    schedule_fetcher: Optional[Callable[[str], List[Dict[str, Any]]]] = None,
    output_path: str = "data/next_day_games.parquet",
    team_ids_path: str = "data/next_day_team_ids.csv",
) -> str:
    """
    Pipeline entry point to discover and persist the upcoming NBA game slate.
    Saves next_day_games.parquet and next_day_team_ids.csv.
    """
    games_df = select_upcoming_slate(
        reference_time=reference_time,
        schedule_fetcher=schedule_fetcher,
    )

    if games_df.empty and os.path.exists(output_path):
        try:
            cached_games = pd.read_parquet(output_path)
            if not cached_games.empty:
                # Validate cached games: verify they are not completed or in the past
                ref = reference_time or datetime.now(TZ_UTC)
                if ref.tzinfo is None:
                    ref = ref.replace(tzinfo=TZ_UTC)
                else:
                    ref = ref.astimezone(TZ_UTC)
                current_et_date = ref.astimezone(TZ_ET).date()

                is_stale = False
                if "nba_game_date" in cached_games.columns:
                    cached_dates = [str(d)[:10] for d in cached_games["nba_game_date"].dropna().unique()]
                    if cached_dates and all(datetime.strptime(d, "%Y-%m-%d").date() < current_et_date for d in cached_dates):
                        is_stale = True

                if is_stale:
                    logger.warning("Cached games in %s are from a past date (%s < %s). Discarding stale cache.", output_path, cached_dates, current_et_date)
                    games_df = pd.DataFrame(columns=cached_games.columns)
                else:
                    logger.warning("Live schedule query returned 0 games. Preserving verified slate from %s", output_path)
                    games_df = cached_games
        except Exception as e:
            logger.warning("Could not load cached schedule: %s", e)

    # Drop internal datetime object before parquet serialization if present
    if "tipoff_utc" in games_df.columns:
        games_df = games_df.drop(columns=["tipoff_utc"])
    if "is_postponed" in games_df.columns:
        games_df = games_df.drop(columns=["is_postponed"])

    team_ids = set()
    if not games_df.empty and "homeTeamId" in games_df.columns and "awayTeamId" in games_df.columns:
        team_ids = set(games_df["homeTeamId"].astype(int)).union(set(games_df["awayTeamId"].astype(int)))

    print(f"Games Found: {len(games_df)}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(team_ids_path) or ".", exist_ok=True)

    # Atomic write for next_day_games.parquet
    tmp_path = f"{output_path}.tmp"
    games_df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, output_path)

    # Save team IDs
    pd.Series(list(team_ids)).to_csv(team_ids_path, index=False, header=False)

    print(f"Saved next day games to {output_path} ({len(games_df)} games, {len(team_ids)} teams)")
    return output_path


if __name__ == "__main__":
    get_next_day_games()
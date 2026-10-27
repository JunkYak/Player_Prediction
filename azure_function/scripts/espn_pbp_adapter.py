"""
scripts/espn_pbp_adapter.py

Robust ESPN Play-by-Play and Completed Game Ingestion Adapter for UN-REAL.

Provides secondary/fallback ingestion of completed NBA game statistics when
outbound calls to stats.nba.com time out or are blocked in cloud environments.

Guarantees:
1. Strict schema compliance with canonical play_by_play.parquet REQUIRED_COLUMNS.
2. 100% semantic compatibility with rating_engine/formula.py:
   - Preserves shot types, difficulty bonuses (fadeaway, turnaround, step back, pullup),
     residual bonuses (layup, dunk, float, hook, jump, putback), assists,
     rebound tags (Off:1 Def:0 / Off:0 Def:1), turnovers, steals, and blocks.
3. Deterministic player identity resolution via PlayerIdentityResolver.
4. Preserves ISO clock format (PTmMsS) and chronological score tracking.
"""

import os
import re
import json
import logging
import urllib.request
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np

from scripts.injury_provider import PlayerIdentityResolver

logger = logging.getLogger("propredict.espn_adapter")

REQUIRED_COLUMNS = [
    "gameId",
    "teamId",
    "personId",
    "playerName",
    "period",
    "clock",
    "actionType",
    "subType",
    "shotResult",
    "shotDistance",
    "shotValue",
    "description",
    "scoreHome",
    "scoreAway"
]

DEFAULT_TIMEOUT = 10
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
}


def clean_score_str(val: Any) -> str:
    """Normalize score values to canonical string representation.

    Numeric integers/floats are converted to string integers (e.g. 102 -> '102').
    Nulls, empty strings, and NaN are converted to empty string ('').
    """
    if pd.isna(val) or val is None or val == "":
        return ""
    if isinstance(val, (int, np.integer)):
        return str(val)
    if isinstance(val, (float, np.floating)):
        return str(int(val)) if val.is_integer() else str(val)
    s = str(val).strip()
    if s == "" or s.lower() in ("nan", "none", "<na>"):
        return ""
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except (ValueError, OverflowError):
        pass
    return s


def normalize_pbp_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize DataFrame to match canonical play_by_play.parquet schema.

    Guarantees:
    - Exactly REQUIRED_COLUMNS in canonical order.
    - String columns: object dtype with non-null str values (default '').
    - Integer columns: int64 dtype with non-null int values (default 0, period=1).
    - scoreHome/scoreAway: object dtype with clean numeric string or ''.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    df = df.copy()
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = "" if col in [
                "gameId", "playerName", "clock", "actionType",
                "subType", "shotResult", "description", "scoreHome", "scoreAway"
            ] else 0

    # 1. Clean score strings
    df["scoreHome"] = df["scoreHome"].apply(clean_score_str).astype(str)
    df["scoreAway"] = df["scoreAway"].apply(clean_score_str).astype(str)

    # 2. Integer columns
    for int_col in ["teamId", "personId", "period", "shotDistance", "shotValue"]:
        default_val = 1 if int_col == "period" else 0
        df[int_col] = pd.to_numeric(df[int_col], errors="coerce").fillna(default_val).astype("int64")

    # 3. String columns
    for str_col in ["gameId", "playerName", "clock", "actionType", "subType", "shotResult", "description"]:
        df[str_col] = df[str_col].fillna("").astype(str).str.strip()

    return df[REQUIRED_COLUMNS].reset_index(drop=True)


def fetch_espn_scoreboard(date_str: str, timeout: int = DEFAULT_TIMEOUT) -> List[Dict[str, Any]]:
    """
    Fetch completed NBA games for a specific calendar date (YYYY-MM-DD) from ESPN.
    Returns list of dicts with game metadata.
    """
    date_compact = date_str.replace("-", "")
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_compact}"
    
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        logger.warning(f"Failed to fetch ESPN scoreboard for {date_str}: {e}")
        return []

    events = data.get("events", [])
    completed_games = []

    for ev in events:
        status_info = ev.get("status", {})
        type_info = status_info.get("type", {})
        is_completed = type_info.get("completed", False) is True or type_info.get("state") == "post"
        
        if not is_completed:
            continue

        game_id = str(ev.get("id"))
        ev_date = str(ev.get("date", date_str))[:10]
        
        competitors = ev.get("competitions", [{}])[0].get("competitors", [])
        home_comp = next((c for c in competitors if c.get("homeAway") == "home"), {})
        away_comp = next((c for c in competitors if c.get("homeAway") == "away"), {})

        completed_games.append({
            "game_id": game_id,
            "game_date": ev_date,
            "home_team_name": home_comp.get("team", {}).get("displayName", ""),
            "home_team_tri": home_comp.get("team", {}).get("abbreviation", ""),
            "away_team_name": away_comp.get("team", {}).get("displayName", ""),
            "away_team_tri": away_comp.get("team", {}).get("abbreviation", ""),
            "home_score": int(home_comp.get("score", 0)),
            "away_score": int(away_comp.get("score", 0)),
        })

    logger.info(f"ESPN Scoreboard for {date_str}: found {len(completed_games)} completed games.")
    return completed_games


def fetch_espn_completed_games(
    start_date: str,
    end_date: str,
    cutoff_date: str = "2026-02-06",
    timeout: int = DEFAULT_TIMEOUT
) -> List[Dict[str, Any]]:
    """
    Fetch all completed games between start_date and end_date (inclusive),
    strictly enforcing cutoff_date.
    """
    start_dt = max(datetime.strptime(start_date, "%Y-%m-%d"), datetime.strptime(cutoff_date, "%Y-%m-%d"))
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    all_completed = []
    curr = start_dt
    while curr <= end_dt:
        d_str = curr.strftime("%Y-%m-%d")
        games = fetch_espn_scoreboard(d_str, timeout=timeout)
        all_completed.extend(games)
        curr += timedelta(days=1)

    return all_completed


def parse_espn_summary_to_pbp(
    espn_game_id: str,
    game_date: Optional[str] = None,
    resolver: Optional[PlayerIdentityResolver] = None,
    timeout: int = DEFAULT_TIMEOUT
) -> pd.DataFrame:
    """
    Fetches ESPN game summary and parses play-by-play events into the canonical
    REQUIRED_COLUMNS format compatible with rating_engine/formula.py.
    """
    if resolver is None:
        resolver = PlayerIdentityResolver()

    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={espn_game_id}"
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        logger.error(f"Failed to fetch ESPN game summary for {espn_game_id}: {e}")
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    # 1. Build athlete registry from boxscore
    athlete_map = {}
    for team_data in data.get("boxscore", {}).get("players", []):
        team_name = team_data.get("team", {}).get("displayName", "")
        team_tri = team_data.get("team", {}).get("abbreviation", "")
        nba_tid = resolver.resolve_team_id(team_tri) or resolver.resolve_team_id(team_name)

        for stat_grp in team_data.get("statistics", []):
            for ath in stat_grp.get("athletes", []):
                a_info = ath.get("athlete", {})
                espn_aid = str(a_info.get("id"))
                raw_name = a_info.get("displayName", "")
                
                nba_pid, canonical_name, _ = resolver.resolve_player(raw_name, team_name or team_tri)
                
                athlete_map[espn_aid] = {
                    "personId": nba_pid,
                    "playerName": canonical_name or raw_name,
                    "teamId": nba_tid,
                    "teamName": team_name,
                    "teamTricode": team_tri
                }

    # 2. Iterate plays and construct canonical rows
    plays = data.get("plays", [])
    if not plays:
        logger.warning(f"No play-by-play events found in ESPN summary for game {espn_game_id}")
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    canonical_rows = []
    
    for p in plays:
        text = str(p.get("text", "")).strip()
        type_text = str(p.get("type", {}).get("text", "")).strip()
        period = int(p.get("period", {}).get("number", 1))
        clock_disp = str(p.get("clock", {}).get("displayValue", "12:00")).strip()
        away_score = clean_score_str(p.get("awayScore", 0))
        home_score = clean_score_str(p.get("homeScore", 0))

        # Format clock to ISO PT{minutes}M{seconds}S
        if ":" in clock_disp:
            parts_c = clock_disp.split(":")
            m_val = int(parts_c[0])
            s_val = float(parts_c[1])
            clock_iso = f"PT{m_val:02d}M{s_val:05.2f}S"
        else:
            clock_iso = "PT12M00.00S"

        # Participant athletes
        parts = [str(part.get("athlete", {}).get("id")) for part in p.get("participants", []) if "athlete" in part]
        main_ath = athlete_map.get(parts[0]) if parts else None

        # Excluded events (substitutions, jump balls)
        low_text = text.lower()
        low_type = type_text.lower()

        if "substitution" in low_type or "enters the game" in low_text:
            continue
        if "jumpball" in low_type or "jump ball" in low_text:
            continue
        if "timeout" in low_type:
            continue

        # Rebounds
        if "rebound" in low_type or "rebound" in low_text:
            if main_ath and main_ath["personId"]:
                is_off = "offensive" in low_type or "offensive" in low_text
                reb_tag = "(Off:1 Def:0)" if is_off else "(Off:0 Def:1)"
                canonical_rows.append({
                    "gameId": espn_game_id,
                    "teamId": main_ath["teamId"],
                    "personId": main_ath["personId"],
                    "playerName": main_ath["playerName"],
                    "period": period,
                    "clock": clock_iso,
                    "actionType": "Rebound",
                    "subType": "Offensive" if is_off else "Defensive",
                    "shotResult": "",
                    "shotDistance": 0,
                    "shotValue": 0,
                    "description": f"{main_ath['playerName']} REBOUND {reb_tag}",
                    "scoreHome": home_score,
                    "scoreAway": away_score
                })
            continue

        # Turnovers
        if "turnover" in low_type or "turnover" in low_text:
            if main_ath and main_ath["personId"]:
                canonical_rows.append({
                    "gameId": espn_game_id,
                    "teamId": main_ath["teamId"],
                    "personId": main_ath["personId"],
                    "playerName": main_ath["playerName"],
                    "period": period,
                    "clock": clock_iso,
                    "actionType": "Turnover",
                    "subType": type_text,
                    "shotResult": "",
                    "shotDistance": 0,
                    "shotValue": 0,
                    "description": f"{main_ath['playerName']} Turnover",
                    "scoreHome": home_score,
                    "scoreAway": away_score
                })
            # Secondary Steal Row (attributed to stealer)
            if "steal" in low_text and len(parts) > 1:
                stealer = athlete_map.get(parts[1])
                if stealer and stealer["personId"]:
                    canonical_rows.append({
                        "gameId": espn_game_id,
                        "teamId": stealer["teamId"],
                        "personId": stealer["personId"],
                        "playerName": stealer["playerName"],
                        "period": period,
                        "clock": clock_iso,
                        "actionType": "",
                        "subType": "",
                        "shotResult": "",
                        "shotDistance": 0,
                        "shotValue": 0,
                        "description": f"{stealer['playerName']} STEAL (1 STL)",
                        "scoreHome": home_score,
                        "scoreAway": away_score
                    })
            continue

        # Free Throws
        if "free throw" in low_type or "free throw" in low_text:
            if main_ath and main_ath["personId"]:
                is_miss = "misses" in low_text
                canonical_rows.append({
                    "gameId": espn_game_id,
                    "teamId": main_ath["teamId"],
                    "personId": main_ath["personId"],
                    "playerName": main_ath["playerName"],
                    "period": period,
                    "clock": clock_iso,
                    "actionType": "Free Throw",
                    "subType": type_text,
                    "shotResult": "Missed" if is_miss else "Made",
                    "shotDistance": 0,
                    "shotValue": 0,
                    "description": f"{main_ath['playerName']} {'MISS ' if is_miss else ''}Free Throw",
                    "scoreHome": home_score,
                    "scoreAway": away_score
                })
            continue

        # Field Goal Shots (Made / Missed)
        is_shot = (
            p.get("shootingPlay", False)
            or "shot" in low_type
            or "dunk" in low_type
            or "layup" in low_type
            or "makes" in low_text
            or "misses" in low_text
        )

        if is_shot and main_ath and main_ath["personId"]:
            is_made = p.get("scoringPlay", False) or "makes" in low_text
            
            # Distance
            dist_match = re.search(r"(\d+)-foot", text) or re.search(r"(\d+)'", text)
            dist = int(dist_match.group(1)) if dist_match else 0
            
            # 3PT detection
            is_three = (
                p.get("scoreValue") == 3
                or p.get("pointsAttempted") == 3
                or "three point" in low_text
                or "3pt" in low_text
                or "3-point" in low_text
                or "three-point" in low_text
            )
            shot_val = 3 if is_three else 2

            # Build canonical description preserving rating engine keywords
            full_context = f"{type_text} {text}".lower()
            desc_parts = []
            if not is_made:
                desc_parts.append("MISS")
            desc_parts.append(main_ath["playerName"])
            if dist > 0:
                desc_parts.append(f"{dist}'")
            if is_three:
                desc_parts.append("3PT")

            # Preserved formula keywords
            if "driving" in full_context:
                desc_parts.append("Driving")
            if "step back" in full_context:
                desc_parts.append("Step Back")
            if "pullup" in full_context:
                desc_parts.append("Pullup")
            if "floating" in full_context or "float" in full_context:
                desc_parts.append("Floating")
            if "fadeaway" in full_context or "fade away" in full_context:
                desc_parts.append("Fadeaway")
            if "turnaround" in full_context:
                desc_parts.append("Turnaround")

            if "layup" in full_context:
                desc_parts.append("Layup Shot")
            elif "dunk" in full_context:
                desc_parts.append("Dunk")
            elif "hook" in full_context:
                desc_parts.append("Hook Shot")
            elif "putback" in full_context:
                desc_parts.append("Putback")
            elif "jump" in full_context:
                desc_parts.append("Jump Shot")
            else:
                desc_parts.append("Shot")

            # Assist detection
            if is_made and "assists" in low_text and len(parts) > 1:
                ast_ath = athlete_map.get(parts[1])
                ast_name = ast_ath["playerName"] if ast_ath else "Player"
                desc_parts.append(f"({ast_name} 1 AST)")

            shot_desc = " ".join(desc_parts)

            canonical_rows.append({
                "gameId": espn_game_id,
                "teamId": main_ath["teamId"],
                "personId": main_ath["personId"],
                "playerName": main_ath["playerName"],
                "period": period,
                "clock": clock_iso,
                "actionType": "Made Shot" if is_made else "Missed Shot",
                "subType": type_text,
                "shotResult": "Made" if is_made else "Missed",
                "shotDistance": dist,
                "shotValue": shot_val,
                "description": shot_desc,
                "scoreHome": home_score,
                "scoreAway": away_score
            })

            # Secondary Block Row (attributed to blocker)
            if not is_made and "blocks" in low_text and len(parts) > 1:
                blocker = athlete_map.get(parts[1])
                if blocker and blocker["personId"]:
                    canonical_rows.append({
                        "gameId": espn_game_id,
                        "teamId": blocker["teamId"],
                        "personId": blocker["personId"],
                        "playerName": blocker["playerName"],
                        "period": period,
                        "clock": clock_iso,
                        "actionType": "",
                        "subType": "",
                        "shotResult": "",
                        "shotDistance": 0,
                        "shotValue": 0,
                        "description": f"{blocker['playerName']} BLOCK (1 BLK)",
                        "scoreHome": home_score,
                        "scoreAway": away_score
                    })

    df = pd.DataFrame(canonical_rows)
    if df.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    # Filter invalid records
    df = df[
        (df["personId"].notna())
        & (df["personId"] > 1000)
        & (df["teamId"].notna())
        & (df["teamId"] > 0)
        & (df["playerName"].notna())
        & (df["playerName"] != "")
    ].copy()

    return normalize_pbp_dataframe(df)

"""
scripts/roster_provider.py

Authoritative NBA Roster Provider for the 2026-27 Season.

Responsibilities:
- Ingests authoritative 30-team current roster data using NBA API CommonAllPlayers / CommonTeamRoster.
- Validates player IDs, current team assignments, and 30-team completeness.
- Persists data/current_rosters.parquet and data/roster_metadata.json.
- Provides fallback to cached roster if live NBA API call fails or yields implausible data.
- Ensures trades, signings, waivers, and rookies are reflected accurately while preserving historical game logs.
"""

import os
import sys
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple
import pandas as pd
from nba_api.stats.endpoints import commonallplayers, commonteamroster

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("propredict.rosters")

NBA_TEAMS = {
    1610612737: {"tricode": "ATL", "name": "Hawks", "city": "Atlanta"},
    1610612738: {"tricode": "BOS", "name": "Celtics", "city": "Boston"},
    1610612739: {"tricode": "CLE", "name": "Cavaliers", "city": "Cleveland"},
    1610612740: {"tricode": "NOP", "name": "Pelicans", "city": "New Orleans"},
    1610612741: {"tricode": "CHI", "name": "Bulls", "city": "Chicago"},
    1610612742: {"tricode": "DAL", "name": "Mavericks", "city": "Dallas"},
    1610612743: {"tricode": "DEN", "name": "Nuggets", "city": "Denver"},
    1610612744: {"tricode": "GSW", "name": "Warriors", "city": "Golden State"},
    1610612745: {"tricode": "HOU", "name": "Rockets", "city": "Houston"},
    1610612746: {"tricode": "LAC", "name": "Clippers", "city": "LA"},
    1610612747: {"tricode": "LAL", "name": "Lakers", "city": "Los Angeles"},
    1610612748: {"tricode": "MIA", "name": "Heat", "city": "Miami"},
    1610612749: {"tricode": "MIL", "name": "Bucks", "city": "Milwaukee"},
    1610612750: {"tricode": "MIN", "name": "Timberwolves", "city": "Minnesota"},
    1610612751: {"tricode": "BKN", "name": "Nets", "city": "Brooklyn"},
    1610612752: {"tricode": "NYK", "name": "Knicks", "city": "New York"},
    1610612753: {"tricode": "ORL", "name": "Magic", "city": "Orlando"},
    1610612754: {"tricode": "IND", "name": "Pacers", "city": "Indiana"},
    1610612755: {"tricode": "PHI", "name": "76ers", "city": "Philadelphia"},
    1610612756: {"tricode": "PHX", "name": "Suns", "city": "Phoenix"},
    1610612757: {"tricode": "POR", "name": "Blazers", "city": "Portland"},
    1610612758: {"tricode": "SAC", "name": "Kings", "city": "Sacramento"},
    1610612759: {"tricode": "SAS", "name": "Spurs", "city": "San Antonio"},
    1610612760: {"tricode": "OKC", "name": "Thunder", "city": "Oklahoma City"},
    1610612761: {"tricode": "TOR", "name": "Raptors", "city": "Toronto"},
    1610612762: {"tricode": "UTA", "name": "Jazz", "city": "Utah"},
    1610612763: {"tricode": "MEM", "name": "Grizzlies", "city": "Memphis"},
    1610612764: {"tricode": "WAS", "name": "Wizards", "city": "Washington"},
    1610612765: {"tricode": "DET", "name": "Pistons", "city": "Detroit"},
    1610612766: {"tricode": "CHA", "name": "Hornets", "city": "Charlotte"},
}


def validate_roster_dataframe(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validates roster dataset integrity:
    - Required columns present.
    - No null values in personId, teamId, playerName.
    - teamId belongs to valid 30 NBA teams.
    - At least 400 total active players.
    - At least 28 unique teams represented.
    - Unique personId (no duplicate players).
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return False, "Roster DataFrame is empty or None"

    req_cols = ["personId", "playerName", "teamId", "teamName"]
    missing = [c for c in req_cols if c not in df.columns]
    if missing:
        return False, f"Missing required roster columns: {missing}"

    if df["personId"].isna().any() or df["teamId"].isna().any() or df["playerName"].isna().any():
        return False, "Null values detected in essential roster columns"

    # Check for duplicates
    dup_pids = df[df.duplicated(subset=["personId"], keep=False)]["personId"].tolist()
    if dup_pids:
        return False, f"Duplicate personId entries found in roster: {dup_pids[:5]}"

    # Check team counts
    unique_teams = set(df["teamId"].unique())
    valid_team_ids = set(NBA_TEAMS.keys())
    unknown_teams = unique_teams - valid_team_ids
    if unknown_teams:
        return False, f"Unknown teamIds in roster: {unknown_teams}"

    if len(unique_teams) < 28:
        return False, f"Implausible team count: only {len(unique_teams)} teams present (expected 30)"

    if len(df) < 400:
        return False, f"Implausible player count: only {len(df)} players present (expected >= 400)"

    return True, "Valid"


def fetch_live_rosters_from_nba_api(season: str = "2026-27") -> pd.DataFrame:
    """
    Fetches active rosters using CommonAllPlayers endpoint for the specified season.
    Standardizes schema to personId, playerName, teamId, teamName, teamTricode.
    """
    logger.info("Fetching authoritative current rosters from CommonAllPlayers (season=%s)...", season)
    cap = commonallplayers.CommonAllPlayers(season=season, is_only_current_season=1)
    df_raw = cap.get_data_frames()[0]

    if df_raw.empty:
        raise ValueError(f"Empty response from CommonAllPlayers for season {season}")

    # Keep only active players with a valid NBA team (teamId in 30 teams)
    valid_team_ids = set(NBA_TEAMS.keys())
    active_df = df_raw[df_raw["TEAM_ID"].isin(valid_team_ids)].copy()

    records = []
    for _, row in active_df.iterrows():
        pid = int(row["PERSON_ID"])
        name = str(row["DISPLAY_FIRST_LAST"]).strip()
        tid = int(row["TEAM_ID"])
        t_info = NBA_TEAMS.get(tid, {})
        team_name = t_info.get("name", str(row.get("TEAM_NAME", "")))
        team_tricode = t_info.get("tricode", str(row.get("TEAM_ABBREVIATION", "")))

        records.append({
            "personId": pid,
            "playerName": name,
            "teamId": tid,
            "teamName": team_name,
            "teamTricode": team_tricode,
            "season": season,
            "rosterStatus": int(row.get("ROSTERSTATUS", 1)),
        })

    roster_df = pd.DataFrame(records)
    # Deduplicate if necessary (keep first)
    roster_df = roster_df.drop_duplicates(subset=["personId"]).reset_index(drop=True)
    return roster_df


def refresh_current_rosters(
    season: str = "2026-27",
    output_parquet: str = "data/current_rosters.parquet",
    output_metadata: str = "data/roster_metadata.json",
    force_fetch: bool = False
) -> pd.DataFrame:
    """
    Refreshes current roster membership for all 30 NBA teams.
    Saves to parquet and records verification metadata.
    Fails safely back to cached roster if live fetch fails.
    """
    existing_df = None
    if os.path.exists(output_parquet):
        try:
            existing_df = pd.read_parquet(output_parquet)
            is_valid, _ = validate_roster_dataframe(existing_df)
            if not is_valid:
                existing_df = None
        except Exception:
            existing_df = None

    fetched_df = None
    fetch_error = None

    try:
        candidate_df = fetch_live_rosters_from_nba_api(season=season)
        is_valid, msg = validate_roster_dataframe(candidate_df)
        if is_valid:
            fetched_df = candidate_df
        else:
            fetch_error = f"Live roster validation failed: {msg}"
            logger.warning(fetch_error)
    except Exception as e:
        fetch_error = f"Error querying NBA API for rosters: {e}"
        logger.warning(fetch_error)

    if fetched_df is not None:
        final_df = fetched_df
        source_desc = f"nba_api.stats.endpoints.commonallplayers (season={season})"
    elif existing_df is not None:
        logger.warning("Using existing cached roster due to fetch failure: %s", fetch_error)
        final_df = existing_df
        source_desc = "cached_roster_fallback"
    else:
        raise RuntimeError(f"Unable to obtain valid NBA rosters and no valid cache exists: {fetch_error}")

    # Atomic write to disk
    os.makedirs(os.path.dirname(output_parquet) or ".", exist_ok=True)
    tmp_parquet = f"{output_parquet}.tmp"
    final_df.to_parquet(tmp_parquet, index=False)
    os.replace(tmp_parquet, output_parquet)

    meta = {
        "season": season,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": source_desc,
        "total_players": len(final_df),
        "team_count": int(final_df["teamId"].nunique()),
        "status": "VALIDATED",
    }
    with open(output_metadata, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info("Current rosters successfully updated (%d players across %d teams)",
                len(final_df), final_df["teamId"].nunique())
    return final_df


def get_player_team_roster_map(
    roster_path: str = "data/current_rosters.parquet",
    season: str = "2026-27"
) -> Dict[int, Dict[str, Any]]:
    """
    Returns a dictionary mapping personId -> {teamId, teamName, playerName}.
    Loads from disk cache or refreshes if missing.
    """
    if not os.path.exists(roster_path):
        df = refresh_current_rosters(season=season, output_parquet=roster_path)
    else:
        try:
            df = pd.read_parquet(roster_path)
        except Exception:
            df = refresh_current_rosters(season=season, output_parquet=roster_path)

    roster_map = {}
    for _, row in df.iterrows():
        roster_map[int(row["personId"])] = {
            "teamId": int(row["teamId"]),
            "teamName": str(row["teamName"]),
            "playerName": str(row["playerName"]),
        }
    return roster_map


if __name__ == "__main__":
    refresh_current_rosters()

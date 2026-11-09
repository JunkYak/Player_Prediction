import os
import sys
import json
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure project root in sys.path for compute_season_multipliers
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from scripts.compute_season_multipliers import compute_season_multipliers

logger = logging.getLogger("propredict.frontend")

TEAM_MAP = {
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


def normalize_schedule_dataframe(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Normalizes schedule DataFrame columns between legacy camelCase and dedicated snake_case schemas.
    Always yields integer homeTeamId / awayTeamId and drops postponed games.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            "gameId", "homeTeamId", "awayTeamId", "homeTeamTricode", "awayTeamTricode", "nba_game_date"
        ])
    res = df.copy()
    col_map = {
        "home_team_id": "homeTeamId",
        "away_team_id": "awayTeamId",
        "home_team_name": "homeTeamName",
        "away_team_name": "awayTeamName",
        "home_team_tricode": "homeTeamTricode",
        "away_team_tricode": "awayTeamTricode",
        "schedule_date": "nba_game_date",
        "game_date": "nba_game_date",
        "game_id": "gameId",
    }
    for old_col, new_col in col_map.items():
        if old_col in res.columns and new_col not in res.columns:
            res[new_col] = res[old_col]

    # Ensure homeTeamId and awayTeamId are integer numeric
    if "homeTeamId" in res.columns:
        res["homeTeamId"] = pd.to_numeric(res["homeTeamId"], errors="coerce").fillna(0).astype(int)
    if "awayTeamId" in res.columns:
        res["awayTeamId"] = pd.to_numeric(res["awayTeamId"], errors="coerce").fillna(0).astype(int)

    # Exclude postponed games if marked
    if "is_postponed" in res.columns:
        res = res[res["is_postponed"] != True].copy()

    return res


def validate_and_load_tomorrow_schedule(
    schedule_path: Optional[str] = "data/schedule_tomorrow.parquet",
    schedule_metadata_path: Optional[str] = "data/schedule_tomorrow_metadata.json",
    schedule_df: Optional[pd.DataFrame] = None,
    schedule_meta: Optional[Dict[str, Any]] = None,
    games_df: Optional[pd.DataFrame] = None,
    games_path: Optional[str] = None,
    reference_time: Optional[datetime] = None,
) -> Tuple[str, pd.DataFrame, Optional[str]]:
    """
    Loads and validates tomorrow's schedule according to official America/New_York calendar policy.

    Returns:
        (status, normalized_schedule_df, schedule_date_str)
        where status is one of: 'CONFIRMED', 'OFF_DAY', 'STALE', 'SCHEDULE_UNAVAILABLE'
    """
    # Determine reference tomorrow in America/New_York
    if reference_time is None:
        now_et = datetime.now(ZoneInfo("America/New_York"))
    else:
        if reference_time.tzinfo is None:
            now_et = reference_time.replace(tzinfo=ZoneInfo("America/New_York"))
        else:
            now_et = reference_time.astimezone(ZoneInfo("America/New_York"))

    tomorrow_et = (now_et.date() + timedelta(days=1)).strftime("%Y-%m-%d")

    # 1. Direct DataFrame overrides (e.g. unit tests)
    if schedule_df is not None:
        norm = normalize_schedule_dataframe(schedule_df)
        if schedule_meta:
            meta_status = schedule_meta.get("status")
            raw_sched_date = schedule_meta.get("schedule_date")
            sched_date = raw_sched_date[:10] if raw_sched_date else None
            if meta_status in ("FETCH_FAILED", "UNAVAILABLE"):
                return ("SCHEDULE_UNAVAILABLE", pd.DataFrame(), None)
            if sched_date and sched_date > tomorrow_et:
                return ("SCHEDULE_UNAVAILABLE", pd.DataFrame(), None)
            if meta_status == "STALE" or schedule_meta.get("is_stale") or (sched_date and sched_date < tomorrow_et):
                return ("STALE", norm, sched_date)
            if meta_status == "OFF_DAY" or (len(norm) == 0 and schedule_meta.get("games_scheduled") is False):
                return ("OFF_DAY", norm, sched_date)
            if meta_status == "CONFIRMED" and len(norm) > 0:
                return ("CONFIRMED", norm, sched_date)
        if len(norm) == 0:
            return ("OFF_DAY", norm, None)
        date_str = str(norm["nba_game_date"].iloc[0]) if "nba_game_date" in norm.columns and pd.notna(norm["nba_game_date"].iloc[0]) else None
        return ("CONFIRMED", norm, date_str)

    if games_df is not None:
        norm = normalize_schedule_dataframe(games_df)
        if len(norm) == 0:
            return ("OFF_DAY", norm, None)
        date_str = str(norm["nba_game_date"].iloc[0]) if "nba_game_date" in norm.columns and pd.notna(norm["nba_game_date"].iloc[0]) else None
        return ("CONFIRMED", norm, date_str)

    # 2. Legacy explicit games_path override (if explicitly passed and exists)
    if games_path is not None:
        if not os.path.exists(games_path):
            logger.warning("Specified games_path does not exist: %s", games_path)
            return ("SCHEDULE_UNAVAILABLE", pd.DataFrame(), None)
        try:
            raw = pd.read_parquet(games_path)
            norm = normalize_schedule_dataframe(raw)
            if len(norm) == 0:
                return ("OFF_DAY", norm, None)
            date_str = str(norm["nba_game_date"].iloc[0]) if "nba_game_date" in norm.columns and pd.notna(norm["nba_game_date"].iloc[0]) else None
            return ("CONFIRMED", norm, date_str)
        except Exception as e:
            logger.warning("Failed to read games_path: %s", e)
            return ("SCHEDULE_UNAVAILABLE", pd.DataFrame(), None)

    # 3. Canonical dedicated tomorrow schedule artifacts
    if not schedule_path or not os.path.exists(schedule_path):
        logger.warning("Tomorrow schedule artifact not found: %s", schedule_path)
        return ("SCHEDULE_UNAVAILABLE", pd.DataFrame(), None)

    if not schedule_metadata_path or not os.path.exists(schedule_metadata_path):
        logger.warning("Tomorrow schedule metadata not found: %s", schedule_metadata_path)
        return ("SCHEDULE_UNAVAILABLE", pd.DataFrame(), None)

    try:
        with open(schedule_metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as e:
        logger.warning("Unreadable schedule metadata: %s", e)
        return ("SCHEDULE_UNAVAILABLE", pd.DataFrame(), None)

    meta_status = meta.get("status")
    if meta_status in ("FETCH_FAILED", "UNAVAILABLE"):
        logger.warning("Schedule metadata status is %s", meta_status)
        return ("SCHEDULE_UNAVAILABLE", pd.DataFrame(), None)

    raw_date = meta.get("schedule_date")
    if not raw_date or not isinstance(raw_date, str):
        logger.warning("Missing or malformed schedule_date in metadata")
        return ("SCHEDULE_UNAVAILABLE", pd.DataFrame(), None)

    sched_date = raw_date[:10]

    try:
        raw_df = pd.read_parquet(schedule_path)
        norm = normalize_schedule_dataframe(raw_df)
    except Exception as e:
        logger.warning("Unreadable schedule parquet artifact: %s", e)
        return ("SCHEDULE_UNAVAILABLE", pd.DataFrame(), None)

    # Determine reference tomorrow in America/New_York
    if reference_time is None:
        now_et = datetime.now(ZoneInfo("America/New_York"))
    else:
        if reference_time.tzinfo is None:
            now_et = reference_time.replace(tzinfo=ZoneInfo("America/New_York"))
        else:
            now_et = reference_time.astimezone(ZoneInfo("America/New_York"))

    tomorrow_et = (now_et.date() + timedelta(days=1)).strftime("%Y-%m-%d")

    # Future-dated check
    if sched_date > tomorrow_et:
        logger.warning("Future-dated schedule detected (%s > %s); unavailable", sched_date, tomorrow_et)
        return ("SCHEDULE_UNAVAILABLE", pd.DataFrame(), None)

    # Stale check
    is_stale = meta.get("is_stale", False) or (sched_date < tomorrow_et)
    if is_stale:
        logger.warning("Tomorrow schedule artifact is stale (%s < %s)", sched_date, tomorrow_et)
        return ("STALE", norm, sched_date)

    # Consistency check between row count and metadata
    meta_games_count = meta.get("games_count")
    if meta_games_count is not None and meta_games_count != len(norm):
        logger.warning("Schedule row count mismatch: parquet=%d, meta=%s", len(norm), meta_games_count)
        return ("SCHEDULE_UNAVAILABLE", pd.DataFrame(), None)

    if len(norm) == 0:
        if meta_status != "OFF_DAY" or meta.get("games_scheduled") is not False:
            logger.warning("Empty schedule rejected: metadata does not establish OFF_DAY")
            return ("SCHEDULE_UNAVAILABLE", pd.DataFrame(), None)
        return ("OFF_DAY", norm, sched_date)

    if meta_status != "CONFIRMED" or meta.get("games_scheduled") is not True:
        logger.warning("Schedule with games rejected: metadata does not establish CONFIRMED")
        return ("SCHEDULE_UNAVAILABLE", pd.DataFrame(), None)

    return ("CONFIRMED", norm, sched_date)


def build_frontend_dataset(
    pred_path: str = "data/predicted_ratings_with_status.parquet",
    games_path: Optional[str] = None,
    ratings_path: str = "data/player_game_ratings_with_dates.parquet",
    output_home_path: str = "data/frontend_home.parquet",
    output_all_path: str = "data/frontend_all.parquet",
    pred_df: Optional[pd.DataFrame] = None,
    games_df: Optional[pd.DataFrame] = None,
    multipliers_df: Optional[pd.DataFrame] = None,
    schedule_path: str = "data/schedule_tomorrow.parquet",
    schedule_metadata_path: str = "data/schedule_tomorrow_metadata.json",
    schedule_df: Optional[pd.DataFrame] = None,
    schedule_meta: Optional[Dict[str, Any]] = None,
    reference_time: Optional[datetime] = None,
) -> str:
    """
    Builds the frontend dataset artifacts:
      1. frontend_home.parquet: The active draft slate containing top available players
         belonging ONLY to teams playing in tomorrow's confirmed schedule.
         If zero games are scheduled tomorrow or the schedule is unconfirmed/stale,
         this dataset contains exactly 0 rows (empty slate).
      2. frontend_all.parquet: Complete league-wide player predictions grouped by team
         for roster browsing, with tomorrow's opponents populated for scheduled teams.
      3. Integrates Candidate D Season Performance Multipliers and computes
         ranking_score = round(predicted_rating * (1 + season_performance_bonus), 2).
         Retains adjusted_predicted_rating as a backward-compatible alias.

    Safety Contract:
      - Validated dedicated schedule (schedule_tomorrow.parquet / metadata) is the canonical source.
      - Never presents stale or unconfirmed matchups as tomorrow's games.
      - If tomorrow's schedule is OFF_DAY, STALE, or SCHEDULE_UNAVAILABLE:
        * frontend_home.parquet contains 0 players.
        * No player is assigned a false opponent. Unscheduled teams receive opponentName = "".
        * frontend_all.parquet safely preserves complete player predictions.
    """
    # 1. LOAD OR USE PROVIDED PREDICTIONS
    if pred_df is not None:
        pred = pred_df.copy()
    else:
        if not os.path.exists(pred_path):
            raise FileNotFoundError(f"Missing predictions file: {pred_path}")
        pred = pd.read_parquet(pred_path)

    if pred.empty:
        raise ValueError("Prediction dataset cannot be empty")

    print(f"Loaded predictions: {pred.shape}")

    # 2. VALIDATE AND LOAD TOMORROW'S SCHEDULE
    sched_status, games, schedule_date = validate_and_load_tomorrow_schedule(
        schedule_path=schedule_path,
        schedule_metadata_path=schedule_metadata_path,
        schedule_df=schedule_df,
        schedule_meta=schedule_meta,
        games_df=games_df,
        games_path=games_path,
        reference_time=reference_time,
    )
    print(f"Schedule resolution: status={sched_status}, games={len(games)}, date={schedule_date}")

    # 3. COMPUTE OR USE PROVIDED SEASON PERFORMANCE MULTIPLIERS
    if multipliers_df is not None:
        mult_df = multipliers_df.copy()
    else:
        if os.path.exists(ratings_path):
            print(f"Computing season performance multipliers from {ratings_path}...")
            mult_df = compute_season_multipliers(input_path=ratings_path, curve="candidate_d")
        else:
            print(f"Warning: {ratings_path} not found. Defaulting multipliers to 0.0.")
            mult_df = pd.DataFrame(columns=[
                "personId", "season_game_count", "season_games_played",
                "season_average_rating", "season_avg_rating",
                "season_percentile", "season_performance_bonus", "season_multiplier_eligible"
            ])

    # 4. MERGE SEASON MULTIPLIERS (SAFE LEFT MERGE ON personId)
    mult_cols = [
        "personId",
        "season_game_count",
        "season_games_played",
        "season_average_rating",
        "season_avg_rating",
        "season_percentile",
        "season_performance_bonus",
        "season_multiplier_eligible",
    ]
    available_mult_cols = [c for c in mult_cols if c in mult_df.columns]
    if "personId" in available_mult_cols:
        pred["personId"] = pred["personId"].astype(np.int64)
        mult_df["personId"] = mult_df["personId"].astype(np.int64)
        pred = pred.merge(mult_df[available_mult_cols].drop_duplicates(subset=["personId"]), on="personId", how="left")

    # Safe defaults for all players (including ineligible / missing)
    if "season_performance_bonus" not in pred.columns:
        pred["season_performance_bonus"] = 0.0
    else:
        pred["season_performance_bonus"] = pred["season_performance_bonus"].fillna(0.0).round(1)

    if "season_game_count" not in pred.columns:
        pred["season_game_count"] = 0
    else:
        pred["season_game_count"] = pred["season_game_count"].fillna(0).astype(int)

    if "season_games_played" not in pred.columns:
        pred["season_games_played"] = pred["season_game_count"]
    else:
        pred["season_games_played"] = pred["season_games_played"].fillna(0).astype(int)

    if "season_average_rating" not in pred.columns:
        pred["season_average_rating"] = np.nan

    if "season_avg_rating" not in pred.columns:
        pred["season_avg_rating"] = pred["season_average_rating"]

    if "season_percentile" not in pred.columns:
        pred["season_percentile"] = np.nan

    if "season_multiplier_eligible" not in pred.columns:
        pred["season_multiplier_eligible"] = False
    else:
        pred["season_multiplier_eligible"] = (pred["season_multiplier_eligible"] == True)

    # 5. COMPUTE TOP PLAYERS RANKING SCORE
    # Formula: ranking_score = round(predicted_rating * (1 + season_performance_bonus), 2)
    # The ranking score is exclusively a sorting metric for Top Players.
    # Raw predicted_rating remains untouched!
    pred["ranking_score"] = (
        pred["predicted_rating"] * (1.0 + pred["season_performance_bonus"])
    ).round(2)
    # Retain adjusted_predicted_rating as a deprecated backward-compatible alias
    pred["adjusted_predicted_rating"] = pred["ranking_score"]

    # 6. STATUS SORTING PRIORITY
    status_priority = {
        "Active": 0,
        "Available": 0,
        "Probable": 1,
        "Questionable": 2,
        "Doubtful": 3,
        "Out": 4,
        "Unknown": 5,
    }
    pred["status_rank"] = pred["status"].map(status_priority).fillna(5).astype(int)

    pred = pred.sort_values(
        ["status_rank", "ranking_score", "predicted_rating"],
        ascending=[True, False, False]
    )

    # 7. TEAMS PLAYING & OPPONENT MAPPING
    teams = set()
    opponent_map = {}

    if sched_status == "CONFIRMED" and not games.empty and "homeTeamId" in games.columns and "awayTeamId" in games.columns:
        for _, row in games.iterrows():
            h = int(row["homeTeamId"])
            a = int(row["awayTeamId"])
            if h > 0 and a > 0:
                teams.add(h)
                teams.add(a)
                opponent_map[h] = a
                opponent_map[a] = h

    # 8. ADD METADATA
    pred["teamName"] = pred["teamId"].map(TEAM_MAP).fillna("Unknown")
    pred["opponentId"] = pred["teamId"].map(opponent_map)
    # Opponent name is assigned if scheduled; unscheduled teams receive "" (not "Unknown")
    pred["opponentName"] = pred["opponentId"].map(TEAM_MAP).fillna("")

    pred["headshot"] = pred["personId"].apply(
        lambda x: f"https://cdn.nba.com/headshots/nba/latest/1040x760/{x}.png"
    )

    # 9. FULL LEAGUE-WIDE DATASET
    full_df = pred.sort_values(
        ["teamId", "status_rank", "ranking_score", "predicted_rating"],
        ascending=[True, True, False, False]
    )

    # 10. HOMEPAGE / DRAFT SLATE DATASET (ONLY CONFIRMED TOMORROW TEAMS)
    if sched_status == "CONFIRMED" and len(teams) > 0:
        home_df = pred[pred["teamId"].isin(teams)].copy()
        home_df = (
            home_df.sort_values(["status_rank", "ranking_score", "predicted_rating"], ascending=[True, False, False])
            .groupby("teamId")
            .head(3)
        )
        if schedule_date:
            home_df["game_date"] = schedule_date
        home_df = home_df.sort_values(
            ["status_rank", "ranking_score", "predicted_rating"],
            ascending=[True, False, False]
        )
    else:
        # Zero games scheduled or unconfirmed/stale schedule -> empty draft slate (0 rows)
        home_df = pred.iloc[0:0].copy()

    # 11. ATOMIC SAVE PARQUET ARTIFACTS
    os.makedirs(os.path.dirname(output_all_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(output_home_path) or ".", exist_ok=True)

    tmp_all = f"{output_all_path}.tmp"
    tmp_home = f"{output_home_path}.tmp"
    try:
        full_df.to_parquet(tmp_all, index=False)
        home_df.to_parquet(tmp_home, index=False)

        # Integrity validation before atomic swap
        verify_all = pd.read_parquet(tmp_all)
        verify_home = pd.read_parquet(tmp_home)
        if len(verify_all) != len(full_df) or len(verify_home) != len(home_df):
            raise IOError("Temporary parquet artifact failed row count integrity check.")

        # Atomic replacement
        os.replace(tmp_all, output_all_path)
        os.replace(tmp_home, output_home_path)
    except Exception as e:
        if os.path.exists(tmp_all):
            try:
                os.remove(tmp_all)
            except OSError:
                pass
        if os.path.exists(tmp_home):
            try:
                os.remove(tmp_home)
            except OSError:
                pass
        raise e

    print("\nSaved:")
    print(f"→ {output_all_path} ({len(full_df)} total players)")
    print(f"→ {output_home_path} ({len(home_df)} active slate players across {len(teams)} teams)")

    if len(home_df) > 0:
        print("\nTop Players (Tomorrow Slate):")
        print(home_df[["playerName", "teamName", "predicted_rating", "season_performance_bonus", "ranking_score", "status"]].head(15))
    else:
        if sched_status == "OFF_DAY":
            print("\nℹ️ Official NBA off-day tomorrow — draft slate is empty (0 players).")
        else:
            print(f"\n⚠️ Schedule status is {sched_status} — draft slate is empty (0 players) to avoid false matchups.")

    return output_home_path


if __name__ == "__main__":
    build_frontend_dataset()

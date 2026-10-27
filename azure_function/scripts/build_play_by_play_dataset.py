import os
import sys
import time
import json
import logging
import argparse
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any, Tuple, Union
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog, playbyplayv3

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from scripts.espn_pbp_adapter import (
        fetch_espn_completed_games,
        parse_espn_summary_to_pbp,
        normalize_pbp_dataframe,
        clean_score_str,
        REQUIRED_COLUMNS
    )
except ImportError:
    from espn_pbp_adapter import (
        fetch_espn_completed_games,
        parse_espn_summary_to_pbp,
        normalize_pbp_dataframe,
        clean_score_str,
        REQUIRED_COLUMNS
    )
from scripts.injury_provider import PlayerIdentityResolver

# ==============================
# LOGGING SETUP
# ==============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ==============================
# CONFIG DEFAULTS & CUTOFF
# ==============================
HISTORICAL_CUTOFF_DATE = "2026-02-06"
DEFAULT_DAYS_BACK = 60
DEFAULT_OUTPUT_FILE = "data/play_by_play.parquet"
DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_BACKOFF = 1.5
DEFAULT_SLEEP_TIME = 0.5

# Preserve exact legacy constants for backward compatibility
DAYS_BACK = DEFAULT_DAYS_BACK
OUTPUT_FILE = DEFAULT_OUTPUT_FILE


# ==============================
# VALIDATION
# ==============================
def validate_pbp_dataset(df: pd.DataFrame) -> bool:
    """
    Lightweight validation for play-by-play DataFrame.
    Verifies that the dataset is non-empty, contains required columns,
    and has valid gameId entries.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("Validation failed: dataset is not a pandas DataFrame.")

    if df.empty:
        raise ValueError("Validation failed: dataset is empty.")

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Validation failed: missing required columns {missing_cols}")

    if df["gameId"].isna().all():
        raise ValueError("Validation failed: gameId contains only null values.")

    return True


# ==============================
# DATA MERGING & DEDUPLICATION
# ==============================
def merge_pbp_data(existing_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """
    Safely merges existing play-by-play data with newly fetched data.
    - If existing_df is empty/None, returns new_df (or empty template).
    - If new_df is empty/None, returns existing_df.
    - Normalizes both datasets to canonical schema to guarantee type compatibility.
    - For gameIds present in new_df, replaces the existing records with the newly fetched ones.
    - Drops exact duplicate rows.
    """
    has_existing = existing_df is not None and isinstance(existing_df, pd.DataFrame) and not existing_df.empty
    has_new = new_df is not None and isinstance(new_df, pd.DataFrame) and not new_df.empty

    if not has_existing and not has_new:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    if not has_existing:
        return normalize_pbp_dataframe(new_df).drop_duplicates().reset_index(drop=True)

    if not has_new:
        return normalize_pbp_dataframe(existing_df).drop_duplicates().reset_index(drop=True)

    # Check that both DataFrames have the basic required 'gameId' column
    if has_new and "gameId" not in new_df.columns:
        logger.warning("Newly fetched dataset is missing 'gameId' column; cannot merge.")
        return normalize_pbp_dataframe(existing_df).drop_duplicates().reset_index(drop=True) if has_existing else pd.DataFrame(columns=REQUIRED_COLUMNS)

    if has_existing and "gameId" not in existing_df.columns:
        logger.warning("Existing dataset is missing 'gameId' column.")
        return normalize_pbp_dataframe(new_df).drop_duplicates().reset_index(drop=True) if has_new else pd.DataFrame(columns=REQUIRED_COLUMNS)

    # Standardize both DataFrames to canonical schema and types
    existing_df = normalize_pbp_dataframe(existing_df)
    new_df = normalize_pbp_dataframe(new_df)

    new_game_ids = set(new_df["gameId"].unique())
    # Keep existing records for games that were NOT in the newly fetched batch
    retained_existing = existing_df[~existing_df["gameId"].isin(new_game_ids)]

    # Concatenate and normalize resulting dataset
    merged = pd.concat([retained_existing, new_df[REQUIRED_COLUMNS]], ignore_index=True)
    merged = normalize_pbp_dataframe(merged)
    merged = merged.drop_duplicates().reset_index(drop=True)
    return merged


class GameInfo(str):
    """
    Backwards-compatible string subclass representing a game ID
    that can also hold metadata (game_date, source) and support dict-like access.
    """
    def __new__(cls, game_id, game_date="", source="", **kwargs):
        obj = super().__new__(cls, str(game_id))
        obj.gameId = str(game_id)
        obj.game_date = str(game_date)
        obj.source = str(source)
        for k, v in kwargs.items():
            setattr(obj, k, v)
        return obj

    def __getitem__(self, item):
        if item == "gameId":
            return self.gameId
        elif item == "game_date":
            return self.game_date
        elif item == "source":
            return self.source
        return super().__getitem__(item)

    def get(self, item, default=None):
        if item == "gameId":
            return self.gameId
        elif item == "game_date":
            return self.game_date
        elif item == "source":
            return self.source
        return getattr(self, item, default)


def _extract_game_meta(g: Any) -> Tuple[str, str, str]:
    if isinstance(g, dict):
        return str(g.get("gameId", "")), str(g.get("game_date", "")), str(g.get("source", "unknown"))
    gid = str(g)
    gdate = getattr(g, "game_date", "")
    source = getattr(g, "source", "unknown")
    return gid, gdate, source


# ==============================
# FETCH GAME LIST (DUAL ENGINE)
# ==============================
def fetch_games(
    days_back: int = DEFAULT_DAYS_BACK,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_backoff: float = DEFAULT_BASE_BACKOFF,
    start_date_str: Optional[str] = None,
    end_date_str: Optional[str] = None,
    cutoff_date: str = HISTORICAL_CUTOFF_DATE,
) -> list:
    """
    Discovers completed games within the lookback window.
    Enforces cutoff_date (February 6, 2026).
    Primary: stats.nba.com LeagueGameLog (with retries).
    Fallback: ESPN Completed Games Engine (site.api.espn.com).
    Returns list of GameInfo objects (backwards-compatible with list of strings).
    """
    end_dt = datetime.strptime(end_date_str, "%Y-%m-%d") if end_date_str else datetime.today()
    if start_date_str:
        start_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
    else:
        start_dt = end_dt - timedelta(days=days_back)

    # Strictly enforce historical cutoff
    cutoff_dt = datetime.strptime(cutoff_date, "%Y-%m-%d")
    start_dt = max(start_dt, cutoff_dt)

    start_str_nba = start_dt.strftime("%m/%d/%Y")
    end_str_nba = end_dt.strftime("%m/%d/%Y")
    start_str_iso = start_dt.strftime("%Y-%m-%d")
    end_str_iso = end_dt.strftime("%Y-%m-%d")

    logger.info(f"Discovering completed games from {start_str_iso} to {end_str_iso} (cutoff: {cutoff_date})...")

    discovered = []

    # 1. Primary: stats.nba.com LeagueGameLog with retries
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Attempting primary discovery via stats.nba.com LeagueGameLog (attempt {attempt}/{max_retries})...")
            log_endpoint = leaguegamelog.LeagueGameLog(
                date_from_nullable=start_str_nba,
                date_to_nullable=end_str_nba,
                timeout=5
            )
            frames = log_endpoint.get_data_frames()
            if frames and not frames[0].empty and "GAME_ID" in frames[0].columns:
                df_log = frames[0]
                has_date = "GAME_DATE" in df_log.columns
                for _, row in df_log.drop_duplicates("GAME_ID").iterrows():
                    gid = str(row["GAME_ID"])
                    if has_date:
                        gdate = pd.to_datetime(row["GAME_DATE"]).strftime("%Y-%m-%d")
                    else:
                        gdate = ""
                    if not gdate or gdate >= cutoff_date:
                        discovered.append(GameInfo(gid, game_date=gdate, source="nba_stats"))
                if discovered:
                    logger.info(f"Successfully discovered {len(discovered)} games via stats.nba.com.")
                    return discovered
            break  # If LeagueGameLog returns empty frame without exception, break to fallback
        except Exception as e:
            if attempt < max_retries:
                sleep_dur = base_backoff * (2 ** (attempt - 1))
                logger.warning(f"Retrying primary discovery (attempt {attempt}/{max_retries}) after error: {e}. Backoff: {sleep_dur:.2f}s")
                time.sleep(sleep_dur)
            else:
                logger.warning(f"Primary discovery via stats.nba.com failed after {max_retries} attempts: {e}. Switching to ESPN fallback...")

    # 2. Secondary Fallback: ESPN Completed Games API
    try:
        logger.info(f"Querying ESPN Completed Games fallback from {start_str_iso} to {end_str_iso}...")
        espn_games = fetch_espn_completed_games(start_str_iso, end_str_iso, cutoff_date=cutoff_date)
        for eg in espn_games:
            gid = str(eg["game_id"])
            gdate = str(eg["game_date"])
            if gdate >= cutoff_date:
                discovered.append(GameInfo(
                    gid,
                    game_date=gdate,
                    source="espn",
                    home_team=eg.get("home_team_tri"),
                    away_team=eg.get("away_team_tri")
                ))
        if discovered:
            logger.info(f"Discovered {len(discovered)} completed games via ESPN fallback.")
            return discovered
    except Exception as espn_err:
        logger.error(f"Both NBA Stats and ESPN game discovery failed: {espn_err}")
        raise RuntimeError(f"Unable to discover completed games: {espn_err}") from espn_err

    return discovered


# ==============================
# FETCH PLAY BY PLAY (DUAL ENGINE)
# ==============================
def fetch_pbp(
    game_info: Union[str, Dict[str, Any]],
    resolver: Optional[PlayerIdentityResolver] = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_backoff: float = DEFAULT_BASE_BACKOFF
) -> pd.DataFrame:
    """
    Fetches play-by-play data for a single game with dual-engine fallback.
    Returns DataFrame matching REQUIRED_COLUMNS.
    """
    if isinstance(game_info, dict):
        game_id = str(game_info.get("gameId", ""))
        source = str(game_info.get("source", "nba_stats"))
        game_date = game_info.get("game_date")
    else:
        game_id = str(game_info)
        source = getattr(game_info, "source", "nba_stats")
        game_date = getattr(game_info, "game_date", None)

    # If discovered from ESPN, directly parse ESPN summary
    if source == "espn":
        df_espn = parse_espn_summary_to_pbp(game_id, game_date=game_date, resolver=resolver)
        if not df_espn.empty:
            return df_espn
        raise ValueError(f"Empty play-by-play parsed from ESPN for game {game_id}")

    # Primary: stats.nba.com PlayByPlayV3
    for attempt in range(1, max_retries + 1):
        try:
            pbp = playbyplayv3.PlayByPlayV3(game_id=game_id, timeout=5)
            frames = pbp.get_data_frames()
            if frames and not frames[0].empty:
                df = frames[0]
                missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
                if not missing:
                    df = df[REQUIRED_COLUMNS].copy()
                    df["gameId"] = df["gameId"].astype(str)
                    return df
        except Exception as e:
            if attempt < max_retries:
                time.sleep(base_backoff * (2 ** (attempt - 1)))
            else:
                logger.warning(f"NBA API PlayByPlayV3 failed for {game_id} ({e}). Attempting ESPN summary fallback...")
                try:
                    df_fallback = parse_espn_summary_to_pbp(game_id, game_date=game_date, resolver=resolver)
                    if not df_fallback.empty:
                        return df_fallback
                except Exception:
                    pass
                raise e

    raise RuntimeError(f"Could not fetch play-by-play for game {game_id}")


# ==============================
# BUILD DATASET
# ==============================
def build_dataset(
    days_back: int = DEFAULT_DAYS_BACK,
    output_file: str = DEFAULT_OUTPUT_FILE,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_backoff: float = DEFAULT_BASE_BACKOFF,
    sleep_time: float = DEFAULT_SLEEP_TIME,
    cutoff_date: str = HISTORICAL_CUTOFF_DATE,
) -> pd.DataFrame:
    """
    Primary ingestion pipeline function with incremental ingestion and historical cutoff.
    - Loads existing dataset if present to preserve canonical historical data.
    - Discovers missing completed games since the latest ingested date.
    - Enforces February 6, 2026 cutoff strictly.
    - Uses NBA Stats as primary and ESPN as verified production fallback.
    - Deduplicates idempotently and validates before saving.
    - Saves ingestion metadata for freshness reporting.
    """
    logger.info("=" * 60)
    logger.info("STARTING INCREMENTAL PLAY-BY-PLAY DATA INGESTION")
    logger.info("=" * 60)
    logger.info(f"Config: output_file='{output_file}', cutoff_date='{cutoff_date}'")

    # Step 1: Load existing dataset if available
    existing_df = None
    existing_game_ids = set()
    latest_ingested_date = cutoff_date

    data_dir = os.path.dirname(output_file) or "."
    dates_cache_path = os.path.join(data_dir, "game_dates_cache.json")
    game_dates_map = {}

    if os.path.exists(dates_cache_path):
        try:
            with open(dates_cache_path, "r", encoding="utf-8") as f:
                game_dates_map = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load game dates cache: {e}")

    # Also backfill dates map from player_game_ratings_with_dates.parquet
    ratings_with_dates_path = os.path.join(data_dir, "player_game_ratings_with_dates.parquet")
    if os.path.exists(ratings_with_dates_path):
        try:
            rwdf = pd.read_parquet(ratings_with_dates_path)
            if "gameId" in rwdf.columns and "game_date" in rwdf.columns:
                for _, r in rwdf[["gameId", "game_date"]].dropna().drop_duplicates().iterrows():
                    game_dates_map[str(r["gameId"])] = str(r["game_date"])[:10]
        except Exception:
            pass

    if os.path.exists(output_file):
        try:
            existing_df = pd.read_parquet(output_file)
            existing_df["gameId"] = existing_df["gameId"].astype(str)
            existing_game_ids = set(existing_df["gameId"].unique())
            logger.info(f"Loaded existing canonical dataset: {len(existing_df)} rows, {len(existing_game_ids)} unique games.")
            
            # Find latest game date
            known_dates = [game_dates_map[gid] for gid in existing_game_ids if gid in game_dates_map]
            if known_dates:
                latest_ingested_date = max(known_dates)
                logger.info(f"Latest completed game date in existing dataset: {latest_ingested_date}")
        except Exception as e:
            logger.warning(f"Could not read existing file at {output_file}: {e}. Proceeding fresh.")

    # Step 2: Determine search window
    today_str = datetime.today().strftime("%Y-%m-%d")
    # Search from latest_ingested_date (or cutoff_date) through today
    start_search = max(latest_ingested_date, cutoff_date)

    discovered_games = []
    try:
        discovered_games = fetch_games(
            start_date_str=start_search,
            end_date_str=today_str,
            cutoff_date=cutoff_date,
            max_retries=max_retries,
            base_backoff=base_backoff
        )
    except Exception as e:
        logger.error(f"Failed to discover completed games: {e}")
        if existing_df is not None and not existing_df.empty:
            logger.warning("Preserving existing historical dataset despite discovery failure.")
            _save_ingestion_metadata(
                status="PRESERVED_EXISTING",
                source="none",
                new_games=0,
                total_games=len(existing_game_ids),
                total_rows=len(existing_df),
                latest_date=latest_ingested_date,
                cutoff_date=cutoff_date,
                data_dir=data_dir,
                skipped_games=0,
                fallback_occurred=False,
                is_fresh=False
            )
            return existing_df
        raise e

    # Update dates map with all newly discovered games
    for dg in discovered_games:
        gid, gdate, _ = _extract_game_meta(dg)
        if gdate:
            game_dates_map[gid] = gdate

    # Step 3: Identify missing games
    missing_games = []
    for g in discovered_games:
        gid, _, _ = _extract_game_meta(g)
        if gid not in existing_game_ids:
            missing_games.append(g)

    logger.info(f"Incremental analysis: {len(discovered_games)} completed games in window, {len(missing_games)} new games to ingest.")

    if not missing_games:
        logger.info("Canonical historical dataset is already up to date. No new completed games to ingest.")
        _save_game_dates_cache(game_dates_map, dates_cache_path)
        skipped_count = len(discovered_games)
        _save_ingestion_metadata(
            status="UP_TO_DATE",
            source="nba_stats",
            new_games=0,
            skipped_games=skipped_count,
            total_games=len(existing_game_ids),
            total_rows=len(existing_df) if existing_df is not None else 0,
            latest_date=latest_ingested_date,
            cutoff_date=cutoff_date,
            data_dir=data_dir,
            fallback_occurred=False,
            is_fresh=True
        )
        return existing_df if existing_df is not None else pd.DataFrame(columns=REQUIRED_COLUMNS)

    # Step 4: Fetch play-by-play for missing games
    resolver = PlayerIdentityResolver()
    fetched_games = []
    failed_games = []
    successful_game_ids = []

    logger.info(f"Ingesting play-by-play for {len(missing_games)} new completed games...")

    for i, g_info in enumerate(missing_games):
        gid, gdate, src = _extract_game_meta(g_info)
        logger.info(f"[{i+1}/{len(missing_games)}] Ingesting game {gid} ({gdate}, source: {src})...")
        try:
            df = fetch_pbp(g_info, resolver=resolver, max_retries=max_retries, base_backoff=base_backoff)
            fetched_games.append(df)
            successful_game_ids.append(gid)
        except Exception as e:
            failed_games.append({
                "game_id": str(gid),
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            logger.error(f"Failed to ingest game {gid}: {e}")

        if sleep_time > 0 and i < len(missing_games) - 1:
            time.sleep(sleep_time)

    # Step 5: Merge with existing dataset safely
    new_df = pd.concat(fetched_games, ignore_index=True) if fetched_games else None
    merged_dataset = merge_pbp_data(existing_df, new_df)

    # Step 6: Validate merged dataset before saving
    if not merged_dataset.empty:
        try:
            validate_pbp_dataset(merged_dataset)
            # Ensure game count and row count non-decreasing
            if existing_df is not None and not existing_df.empty:
                if len(merged_dataset) < len(existing_df):
                    raise ValueError(f"Row count decreased from {len(existing_df)} to {len(merged_dataset)}")
                if merged_dataset["gameId"].nunique() < existing_df["gameId"].nunique():
                    raise ValueError(f"Game count decreased from {existing_df['gameId'].nunique()} to {merged_dataset['gameId'].nunique()}")
        except ValueError as ve:
            logger.error(f"Validation failed on merged dataset: {ve}")
            if existing_df is not None and not existing_df.empty:
                logger.error("Preserving existing parquet file without overwriting.")
                return existing_df
            raise ve

        # Step 7: Atomic persistence
        merged_dataset = normalize_pbp_dataframe(merged_dataset)
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        temp_output_file = f"{output_file}.tmp"
        try:
            merged_dataset.to_parquet(temp_output_file, index=False)
            os.replace(temp_output_file, output_file)
            logger.info(f"Successfully saved canonical dataset to {output_file} ({len(merged_dataset)} rows, {merged_dataset['gameId'].nunique()} games)")
        except Exception as e:
            if os.path.exists(temp_output_file):
                os.remove(temp_output_file)
            logger.error(f"Failed to atomically save parquet file: {e}")
            raise e

        # Save game dates cache
        _save_game_dates_cache(game_dates_map, dates_cache_path)

        # Step 8: Save ingestion metadata
        new_max_date = max([game_dates_map[gid] for gid in merged_dataset["gameId"].unique() if gid in game_dates_map] or [latest_ingested_date])
        actual_sources = set([_extract_game_meta(g)[2] for g in missing_games])
        fallback_used = "espn" in actual_sources
        actual_provider = "espn" if actual_sources == {"espn"} else ("mixed" if len(actual_sources) > 1 else "nba_stats")
        skipped_count = max(0, len(discovered_games) - len(missing_games))

        # Check for partial failure vs full success
        if failed_games and successful_game_ids:
            run_status = "PARTIAL_SUCCESS"
            is_fresh = False  # Partial ingestion failures cannot silently report full freshness
        elif failed_games and not successful_game_ids:
            run_status = "FAILED"
            is_fresh = False
        else:
            run_status = "SUCCESS"
            is_fresh = (len(successful_game_ids) > 0)

        per_game_provenance = {
            _extract_game_meta(g)[0]: _extract_game_meta(g)[2]
            for g in missing_games if _extract_game_meta(g)[0] in successful_game_ids
        }

        _save_ingestion_metadata(
            status=run_status,
            source=actual_provider,
            new_games=len(successful_game_ids),
            skipped_games=skipped_count,
            total_games=merged_dataset["gameId"].nunique(),
            total_rows=len(merged_dataset),
            latest_date=new_max_date,
            cutoff_date=cutoff_date,
            data_dir=data_dir,
            fallback_occurred=fallback_used,
            is_fresh=is_fresh,
            per_game_provenance=per_game_provenance,
            failed_games=failed_games
        )
    else:
        logger.warning("No data available to save.")

    # Summary
    logger.info("=" * 60)
    logger.info("INCREMENTAL INGESTION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"New games successfully ingested: {len(successful_game_ids)}")
    logger.info(f"Failed game fetches:            {len(failed_games)}")
    logger.info(f"Total unique games in dataset:  {merged_dataset['gameId'].nunique() if not merged_dataset.empty else 0}")
    logger.info(f"Total rows in dataset:          {len(merged_dataset)}")
    logger.info("=" * 60)

    return merged_dataset


def _save_game_dates_cache(dates_map: Dict[str, str], cache_path: str):
    """Atomically persist gameId -> game_date dictionary to disk."""
    try:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        tmp = f"{cache_path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(dates_map, f, indent=2)
        os.replace(tmp, cache_path)
    except Exception as e:
        logger.warning(f"Could not persist game dates cache: {e}")


def _save_ingestion_metadata(
    status: str,
    source: str,
    new_games: int,
    total_games: int,
    total_rows: int,
    latest_date: str,
    cutoff_date: str,
    data_dir: str,
    skipped_games: int = 0,
    fallback_occurred: bool = False,
    is_fresh: bool = True,
    per_game_provenance: Optional[Dict[str, str]] = None,
    failed_games: Optional[List[Dict[str, Any]]] = None,
):
    """Record ingestion telemetry metadata for pipeline and manifest verification."""
    meta = {
        "status": status,
        "primary_configured_provider": "nba_stats",
        "actual_provider": source,
        "fallback_occurred": fallback_occurred,
        "source": source,
        "new_games_ingested": new_games,
        "existing_games_skipped": skipped_games,
        "failed_games_count": len(failed_games or []),
        "total_games": total_games,
        "total_rows": total_rows,
        "latest_completed_game_date": latest_date,
        "historical_cutoff_date": cutoff_date,
        "is_fresh": is_fresh,
        "per_game_provenance": per_game_provenance or {},
        "failed_games": failed_games or [],
        "ingested_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = os.path.join(data_dir, "ingestion_metadata.json")
    try:
        tmp = f"{meta_path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        os.replace(tmp, meta_path)
    except Exception as e:
        logger.warning(f"Could not persist ingestion metadata: {e}")


# ==============================
# CLI ENTRYPOINT
# ==============================
def parse_args():
    parser = argparse.ArgumentParser(description="Incremental NBA Play-by-Play Data Ingestion")
    parser.add_argument("--days-back", type=int, default=DEFAULT_DAYS_BACK)
    parser.add_argument("--output-file", type=str, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--sleep-time", type=float, default=DEFAULT_SLEEP_TIME)
    parser.add_argument("--cutoff-date", type=str, default=HISTORICAL_CUTOFF_DATE)
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    build_dataset(
        days_back=cli_args.days_back,
        output_file=cli_args.output_file,
        max_retries=cli_args.max_retries,
        sleep_time=cli_args.sleep_time,
        cutoff_date=cli_args.cutoff_date
    )
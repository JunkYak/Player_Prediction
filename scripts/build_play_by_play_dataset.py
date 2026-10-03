import os
import sys
import time
import logging
import argparse
from datetime import datetime, timedelta
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog, playbyplayv3

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
# CONFIG DEFAULTS
# ==============================
DEFAULT_DAYS_BACK = 60
DEFAULT_OUTPUT_FILE = "data/play_by_play.parquet"
DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_BACKOFF = 1.5
DEFAULT_SLEEP_TIME = 0.6

# Preserve exact legacy constants for backward compatibility
DAYS_BACK = DEFAULT_DAYS_BACK
OUTPUT_FILE = DEFAULT_OUTPUT_FILE

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
    - For gameIds present in new_df, replaces the existing records with the newly fetched ones.
    - Drops exact duplicate rows.
    """
    has_existing = existing_df is not None and isinstance(existing_df, pd.DataFrame) and not existing_df.empty
    has_new = new_df is not None and isinstance(new_df, pd.DataFrame) and not new_df.empty

    if not has_existing and not has_new:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    if not has_existing:
        return new_df.drop_duplicates().reset_index(drop=True)

    if not has_new:
        return existing_df.drop_duplicates().reset_index(drop=True)

    # Check that both DataFrames have the basic required 'gameId' column
    if has_new and "gameId" not in new_df.columns:
        logger.warning("Newly fetched dataset is missing 'gameId' column; cannot merge.")
        return existing_df.drop_duplicates().reset_index(drop=True) if has_existing else pd.DataFrame(columns=REQUIRED_COLUMNS)

    if has_existing and "gameId" not in existing_df.columns:
        logger.warning("Existing dataset is missing 'gameId' column.")
        return new_df.drop_duplicates().reset_index(drop=True) if has_new else pd.DataFrame(columns=REQUIRED_COLUMNS)

    # Standardize gameId types to string to ensure clean set matching
    existing_df = existing_df.copy()
    new_df = new_df.copy()
    existing_df["gameId"] = existing_df["gameId"].astype(str)
    new_df["gameId"] = new_df["gameId"].astype(str)

    new_game_ids = set(new_df["gameId"].unique())
    # Keep existing records for games that were NOT in the newly fetched batch
    retained_existing = existing_df[~existing_df["gameId"].isin(new_game_ids)]

    # Keep only available required columns
    valid_cols = [c for c in REQUIRED_COLUMNS if c in new_df.columns]
    merged = pd.concat([retained_existing, new_df[valid_cols]], ignore_index=True)
    merged = merged.drop_duplicates().reset_index(drop=True)
    return merged


# ==============================
# FETCH GAME LIST
# ==============================
def fetch_games(
    days_back: int = DEFAULT_DAYS_BACK,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_backoff: float = DEFAULT_BASE_BACKOFF
) -> list:
    """
    Discovers unique game IDs within the lookback window.
    Includes bounded retry behavior against transient network/API failures.
    """
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days_back)
    start_str = start_date.strftime("%m/%d/%Y")
    end_str = end_date.strftime("%m/%d/%Y")

    logger.info(f"Discovering games from {start_str} to {end_str} (lookback: {days_back} days)...")

    for attempt in range(1, max_retries + 1):
        try:
            games = leaguegamelog.LeagueGameLog(
                date_from_nullable=start_str,
                date_to_nullable=end_str
            )
            df = games.get_data_frames()[0]
            if "GAME_ID" in df.columns:
                game_ids = df["GAME_ID"].astype(str).unique().tolist()
                logger.info(f"Found {len(game_ids)} games across {days_back} days.")
                return game_ids
            else:
                logger.warning(f"No GAME_ID column in API response (attempt {attempt}/{max_retries})")
        except Exception as e:
            logger.warning(f"Error discovering games (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                sleep_dur = base_backoff * (2 ** (attempt - 1))
                time.sleep(sleep_dur)
            else:
                logger.error(f"Failed to fetch game list after {max_retries} attempts.")
                raise e

    return []


# ==============================
# FETCH PLAY BY PLAY
# ==============================
def fetch_pbp(
    game_id: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_backoff: float = DEFAULT_BASE_BACKOFF
) -> pd.DataFrame:
    """
    Fetches play-by-play data for a single game ID with bounded retries and exponential backoff.
    Returns a DataFrame containing only REQUIRED_COLUMNS.
    """
    game_id = str(game_id)

    for attempt in range(1, max_retries + 1):
        try:
            pbp = playbyplayv3.PlayByPlayV3(game_id=game_id)
            frames = pbp.get_data_frames()
            if not frames or frames[0].empty:
                raise ValueError(f"Empty play-by-play data received for game {game_id}")

            df = frames[0]
            missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
            if missing:
                raise ValueError(f"Play-by-play response missing expected columns {missing} for game {game_id}")

            df = df[REQUIRED_COLUMNS].copy()
            df["gameId"] = df["gameId"].astype(str)
            return df

        except Exception as e:
            if attempt < max_retries:
                sleep_dur = base_backoff * (2 ** (attempt - 1))
                logger.warning(f"Retrying game {game_id} (attempt {attempt}/{max_retries}) after error: {e}. Backoff: {sleep_dur:.1f}s")
                time.sleep(sleep_dur)
            else:
                logger.error(f"Final failure fetching game {game_id} after {max_retries} attempts: {e}")
                raise e


# ==============================
# BUILD DATASET
# ==============================
def build_dataset(
    days_back: int = DEFAULT_DAYS_BACK,
    output_file: str = DEFAULT_OUTPUT_FILE,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_backoff: float = DEFAULT_BASE_BACKOFF,
    sleep_time: float = DEFAULT_SLEEP_TIME
) -> pd.DataFrame:
    """
    Primary ingestion pipeline function.
    - Loads existing dataset if present to preserve historical data.
    - Discovers games within lookback window.
    - Fetches play-by-play with retry handling.
    - Tracks and reports any failed game fetches explicitly.
    - Safely merges and deduplicates against existing parquet data.
    - Performs validation before writing to disk.
    """
    logger.info("=" * 50)
    logger.info("STARTING PLAY-BY-PLAY DATA INGESTION")
    logger.info("=" * 50)
    logger.info(f"Config: days_back={days_back}, output_file='{output_file}', max_retries={max_retries}, sleep_time={sleep_time}s")

    # Step 1: Load existing dataset if available
    existing_df = None
    if os.path.exists(output_file):
        try:
            existing_df = pd.read_parquet(output_file)
            logger.info(f"Loaded existing dataset from {output_file}: {len(existing_df)} rows, {existing_df['gameId'].nunique()} unique games.")
        except Exception as e:
            logger.warning(f"Could not load existing file at {output_file}: {e}. Proceeding fresh.")

    # Step 2: Discover games
    try:
        game_ids = fetch_games(days_back=days_back, max_retries=max_retries, base_backoff=base_backoff)
    except Exception as e:
        logger.error(f"Failed to discover games: {e}")
        if existing_df is not None and not existing_df.empty:
            logger.warning("Preserving existing historical dataset despite game discovery failure.")
            return existing_df
        raise e

    if not game_ids:
        logger.warning("No games discovered for the requested time period.")
        if existing_df is not None:
            return existing_df
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    # Step 3: Fetch games with retry and track results
    fetched_games = []
    failed_games = []
    successful_game_ids = []

    logger.info(f"Fetching play-by-play for {len(game_ids)} games...")

    for i, gid in enumerate(game_ids):
        logger.info(f"[{i+1}/{len(game_ids)}] Fetching game {gid}...")
        try:
            df = fetch_pbp(gid, max_retries=max_retries, base_backoff=base_backoff)
            fetched_games.append(df)
            successful_game_ids.append(gid)
        except Exception as e:
            failed_games.append({
                "game_id": str(gid),
                "error": str(e),
                "attempts": max_retries,
                "timestamp": datetime.now().isoformat()
            })

        if sleep_time > 0 and i < len(game_ids) - 1:
            time.sleep(sleep_time)

    # Step 4: Combine newly fetched games
    new_df = pd.concat(fetched_games, ignore_index=True) if fetched_games else None

    # Step 5: Merge with existing dataset safely
    merged_dataset = merge_pbp_data(existing_df, new_df)

    # Step 6: Validate merged dataset before saving
    if not merged_dataset.empty:
        try:
            validate_pbp_dataset(merged_dataset)
        except ValueError as ve:
            logger.error(f"Data validation failed on merged dataset: {ve}")
            if existing_df is not None and not existing_df.empty:
                logger.error("Preserving existing parquet file without overwriting.")
                return existing_df
            raise ve

        # Step 7: Save to parquet safely
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        # Write to temporary file first then rename to guarantee atomic write
        temp_output_file = f"{output_file}.tmp"
        try:
            merged_dataset.to_parquet(temp_output_file, index=False)
            if os.path.exists(output_file):
                os.replace(temp_output_file, output_file)
            else:
                os.rename(temp_output_file, output_file)
            logger.info(f"Successfully saved canonical dataset to {output_file}")
        except Exception as e:
            if os.path.exists(temp_output_file):
                os.remove(temp_output_file)
            logger.error(f"Failed to save parquet file: {e}")
            raise e
    else:
        logger.warning("No data available to save.")

    # Step 8: Surface clear ingestion summary
    logger.info("=" * 50)
    logger.info("INGESTION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Total games discovered in window: {len(game_ids)}")
    logger.info(f"Successfully fetched new games:   {len(successful_game_ids)}")
    logger.info(f"Failed games:                     {len(failed_games)}")
    logger.info(f"Total games in canonical dataset: {merged_dataset['gameId'].nunique() if not merged_dataset.empty else 0}")
    logger.info(f"Total rows in canonical dataset:  {len(merged_dataset)}")

    if failed_games:
        logger.warning("----------------------------------------")
        logger.warning("FAILED GAMES DETAIL:")
        for fg in failed_games:
            logger.warning(f"  - Game ID: {fg['game_id']} | Attempts: {fg['attempts']} | Error: {fg['error']}")
        logger.warning("----------------------------------------")
    logger.info("=" * 50)

    return merged_dataset


# ==============================
# CLI ENTRYPOINT
# ==============================
def parse_args():
    parser = argparse.ArgumentParser(description="NBA Play-by-Play Data Ingestion")
    parser.add_argument(
        "--days-back",
        type=int,
        default=DEFAULT_DAYS_BACK,
        help=f"Number of days to look back for games (default: {DEFAULT_DAYS_BACK})"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=DEFAULT_OUTPUT_FILE,
        help=f"Output parquet file path (default: {DEFAULT_OUTPUT_FILE})"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=f"Maximum retries for failed API calls (default: {DEFAULT_MAX_RETRIES})"
    )
    parser.add_argument(
        "--sleep-time",
        type=float,
        default=DEFAULT_SLEEP_TIME,
        help=f"Sleep time in seconds between API requests (default: {DEFAULT_SLEEP_TIME})"
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    build_dataset(
        days_back=cli_args.days_back,
        output_file=cli_args.output_file,
        max_retries=cli_args.max_retries,
        sleep_time=cli_args.sleep_time
    )
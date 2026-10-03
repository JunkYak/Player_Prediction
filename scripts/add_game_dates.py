import os
import sys
import time
import logging
import pandas as pd
from typing import Optional
from nba_api.stats.endpoints import leaguegamelog

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

DEFAULT_INPUT_PATH = "data/player_game_ratings.parquet"
DEFAULT_OUTPUT_PATH = "data/player_game_ratings_with_dates.parquet"
DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_BACKOFF = 1.5


def fetch_game_dates(
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_backoff: float = DEFAULT_BASE_BACKOFF
) -> pd.DataFrame:
    """
    Fetches game ID to date mapping from NBA API with bounded retries.
    Returns DataFrame with columns ['gameId', 'game_date'].
    """
    logger.info("Fetching game logs from NBA API...")
    for attempt in range(1, max_retries + 1):
        try:
            log_endpoint = leaguegamelog.LeagueGameLog()
            frames = log_endpoint.get_data_frames()
            if not frames or frames[0].empty:
                raise ValueError("Empty response from LeagueGameLog endpoint")

            df = frames[0]
            if "GAME_ID" not in df.columns or "GAME_DATE" not in df.columns:
                raise ValueError(f"Missing required columns in LeagueGameLog response: {df.columns.tolist()}")

            games = df[["GAME_ID", "GAME_DATE"]].drop_duplicates().copy()
            games.columns = ["gameId", "game_date"]
            games["gameId"] = games["gameId"].astype(str)
            games["game_date"] = pd.to_datetime(games["game_date"]).dt.strftime("%Y-%m-%d")
            logger.info(f"Successfully retrieved {len(games)} game dates from NBA API.")
            return games

        except Exception as e:
            logger.warning(f"Error fetching game logs (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                sleep_dur = base_backoff * (2 ** (attempt - 1))
                time.sleep(sleep_dur)
            else:
                logger.error(f"Failed to fetch game logs after {max_retries} attempts.")
                raise e

    return pd.DataFrame(columns=["gameId", "game_date"])


def add_game_dates(
    input_path: str = DEFAULT_INPUT_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
    game_dates_df: Optional[pd.DataFrame] = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_backoff: float = DEFAULT_BASE_BACKOFF
) -> str:
    """
    Enriches player game ratings with calendar dates safely.

    Safety Guarantees:
      1. Validates input dataset exists and is non-empty.
      2. Validates fetched game-date mapping is non-empty.
      3. Leverages existing output artifact to backfill dates for historical games
         if newly fetched mapping covers a smaller lookback window.
      4. Validates that 100% of game IDs receive valid non-null dates.
      5. If validation fails for ANY reason, existing output artifact is NEVER overwritten.
      6. Writes atomically via temporary file replacement.
    """
    # 1. INPUT VALIDATION
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Missing input ratings file: {input_path}")

    df = pd.read_parquet(input_path)
    logger.info(f"Loaded player ratings: {df.shape}")

    if df.empty:
        raise ValueError(f"Input ratings dataset at {input_path} is empty")

    if "gameId" not in df.columns:
        raise ValueError(f"Input dataset missing required 'gameId' column: {df.columns.tolist()}")

    df["gameId"] = df["gameId"].astype(str)

    # 2. FETCH OR USE PROVIDED GAME DATES
    if game_dates_df is not None:
        games = game_dates_df.copy()
        if "GAME_ID" in games.columns and "GAME_DATE" in games.columns:
            games = games.rename(columns={"GAME_ID": "gameId", "GAME_DATE": "game_date"})
        if "gameId" not in games.columns or "game_date" not in games.columns:
            raise ValueError(f"Supplied game_dates_df missing required columns: {games.columns.tolist()}")
        games["gameId"] = games["gameId"].astype(str)
        games["game_date"] = pd.to_datetime(games["game_date"]).dt.strftime("%Y-%m-%d")
    else:
        try:
            games = fetch_game_dates(max_retries=max_retries, base_backoff=base_backoff)
        except Exception as api_err:
            logger.error(f"Cannot enrich dates due to API failure: {api_err}")
            if os.path.exists(output_path):
                logger.warning(f"Preserving existing valid artifact at {output_path} without changes.")
            raise RuntimeError(f"Date enrichment failed: unable to fetch game dates ({api_err})") from api_err

    if games.empty:
        if os.path.exists(output_path):
            logger.warning(f"Preserving existing valid artifact at {output_path} without changes.")
        raise ValueError("Date enrichment failed: retrieved game dates dataset is unexpectedly empty")

    # 3. COMBINE WITH KNOWN HISTORICAL DATES FROM EXISTING ARTIFACT IF PRESENT
    # This prevents lookback-window truncation from stripping dates of older games
    if os.path.exists(output_path):
        try:
            existing_out = pd.read_parquet(output_path)
            if "gameId" in existing_out.columns and "game_date" in existing_out.columns:
                existing_dates = existing_out[["gameId", "game_date"]].dropna().drop_duplicates()
                existing_dates["gameId"] = existing_dates["gameId"].astype(str)
                existing_dates["game_date"] = pd.to_datetime(existing_dates["game_date"]).dt.strftime("%Y-%m-%d")
                # Merge: newly fetched games take precedence, existing dates backfill older games
                games = pd.concat([games, existing_dates], ignore_index=True).drop_duplicates(subset=["gameId"], keep="first")
        except Exception as e:
            logger.warning(f"Could not read existing artifact for date backfilling: {e}")

    # 4. PERFORM MERGE
    merged_df = df.merge(games[["gameId", "game_date"]].drop_duplicates(subset=["gameId"]), on="gameId", how="left")

    # 5. STRICT VALIDATION CHECKS
    missing_mask = merged_df["game_date"].isna()
    missing_count = int(missing_mask.sum())
    total_rows = len(merged_df)
    unique_games_missing = merged_df.loc[missing_mask, "gameId"].nunique()

    if missing_count > 0:
        err_msg = (
            f"Date enrichment validation FAILED: {missing_count}/{total_rows} rows "
            f"({unique_games_missing} unique games) failed to match a calendar date. "
            f"Preserving existing artifact at {output_path}."
        )
        logger.error(err_msg)
        raise ValueError(err_msg)

    # 6. ENSURE CLEAN SORTING AND CONSISTENT TYPES
    # Retain exact chronological and team ordering expected by downstream scripts
    if "teamId" in merged_df.columns and "rating" in merged_df.columns:
        merged_df = merged_df.sort_values(["game_date", "gameId", "teamId", "rating"], ascending=[True, True, True, False]).reset_index(drop=True)

    # 7. ATOMIC PERSISTENCE
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    temp_output_path = f"{output_path}.tmp"
    try:
        merged_df.to_parquet(temp_output_path, index=False)
        if os.path.exists(output_path):
            os.replace(temp_output_path, output_path)
        else:
            os.rename(temp_output_path, output_path)
        logger.info(f"Successfully saved validated date-enriched ratings to {output_path} ({len(merged_df)} rows)")
    except Exception as save_err:
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
        logger.error(f"Failed to atomically write {output_path}: {save_err}")
        raise save_err

    return output_path


if __name__ == "__main__":
    add_game_dates()
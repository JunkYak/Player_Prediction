import pandas as pd
import numpy as np
import os
import sys
from typing import Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure project root in sys.path for compute_season_multipliers
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from scripts.compute_season_multipliers import compute_season_multipliers


def build_frontend_dataset(
    pred_path: str = "data/predicted_ratings_with_status.parquet",
    games_path: str = "data/next_day_games.parquet",
    ratings_path: str = "data/player_game_ratings_with_dates.parquet",
    output_home_path: str = "data/frontend_home.parquet",
    output_all_path: str = "data/frontend_all.parquet",
    pred_df: Optional[pd.DataFrame] = None,
    games_df: Optional[pd.DataFrame] = None,
    multipliers_df: Optional[pd.DataFrame] = None,
) -> str:
    """
    Builds the frontend dataset artifacts:
      1. frontend_home.parquet: The active draft slate containing top available players
         belonging ONLY to teams playing in the target day's schedule.
         If zero games are scheduled tomorrow, this dataset is EMPTY (0 rows).
      2. frontend_all.parquet: Complete league-wide player predictions grouped by team
         for roster browsing.
      3. Integrates Candidate D Season Performance Multipliers and computes
         adjusted_predicted_rating = round(predicted_rating * (1 + season_performance_bonus), 2).

    Offseason / No-Games Semantics:
      - If games is empty (0 games scheduled):
        * home_df contains 0 players.
        * No player is assigned opponentName = "Unknown" as an artificial fallback.
        * Unscheduled teams receive opponentName = "" (empty string).
    """
    # 1. LOAD OR USE PROVIDED DATASETS
    if pred_df is not None:
        pred = pred_df.copy()
    else:
        if not os.path.exists(pred_path):
            raise FileNotFoundError(f"Missing predictions file: {pred_path}")
        pred = pd.read_parquet(pred_path)

    if games_df is not None:
        games = games_df.copy()
    else:
        if not os.path.exists(games_path):
            raise FileNotFoundError(f"Missing games schedule file: {games_path}")
        games = pd.read_parquet(games_path)

    print(f"Loaded predictions: {pred.shape}")
    print(f"Loaded next-day games: {games.shape}")

    if pred.empty:
        raise ValueError("Prediction dataset cannot be empty")

    # 2. COMPUTE OR USE PROVIDED SEASON PERFORMANCE MULTIPLIERS
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

    # 3. MERGE SEASON MULTIPLIERS (SAFE LEFT MERGE ON personId)
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
    # Filter to columns that exist in mult_df
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

    # 4. COMPUTE ADJUSTED PREDICTED RATING (FOR TOP PLAYERS RANKING)
    # Formula: adjusted_predicted_rating = round(predicted_rating * (1 + season_performance_bonus), 2)
    # Raw predicted_rating remains untouched!
    pred["adjusted_predicted_rating"] = (
        pred["predicted_rating"] * (1.0 + pred["season_performance_bonus"])
    ).round(2)

    # 5. STATUS SORTING PRIORITY
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
        ["status_rank", "adjusted_predicted_rating", "predicted_rating"],
        ascending=[True, False, False]
    )

    # 6. TEAMS PLAYING & OPPONENT MAPPING
    # If games is empty, NO teams are playing tomorrow.
    if not games.empty and "homeTeamId" in games.columns and "awayTeamId" in games.columns:
        teams = set(games["homeTeamId"]).union(set(games["awayTeamId"]))
    else:
        teams = set()

    opponent_map = {}
    if not games.empty and "homeTeamId" in games.columns and "awayTeamId" in games.columns:
        for _, row in games.iterrows():
            home = row["homeTeamId"]
            away = row["awayTeamId"]
            opponent_map[home] = away
            opponent_map[away] = home

    # 7. TEAM NAME MAP
    team_map = {
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

    # 8. ADD METADATA
    pred["teamName"] = pred["teamId"].map(team_map).fillna("Unknown")
    pred["opponentId"] = pred["teamId"].map(opponent_map)
    # Opponent name is assigned if scheduled; unscheduled teams receive "" (not "Unknown")
    pred["opponentName"] = pred["opponentId"].map(team_map).fillna("")

    pred["headshot"] = pred["personId"].apply(
        lambda x: f"https://cdn.nba.com/headshots/nba/latest/1040x760/{x}.png"
    )

    # 9. FULL LEAGUE-WIDE DATASET
    full_df = pred.sort_values(
        ["teamId", "status_rank", "adjusted_predicted_rating", "predicted_rating"],
        ascending=[True, True, False, False]
    )

    # 10. HOMEPAGE / DRAFT SLATE DATASET (ONLY TEAMS PLAYING TOMORROW)
    if len(teams) > 0:
        home_df = pred[pred["teamId"].isin(teams)].copy()
        home_df = (
            home_df.sort_values(["status_rank", "adjusted_predicted_rating", "predicted_rating"], ascending=[True, False, False])
            .groupby("teamId")
            .head(3)
        )
        if not games.empty and "nba_game_date" in games.columns and pd.notna(games["nba_game_date"].iloc[0]):
            home_df["game_date"] = str(games["nba_game_date"].iloc[0])
        home_df = home_df.sort_values(
            ["status_rank", "adjusted_predicted_rating", "predicted_rating"],
            ascending=[True, False, False]
        )
    else:
        # Zero games scheduled -> empty draft slate
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
        print("\nTop Players (Next Day):")
        print(home_df[["playerName", "teamName", "predicted_rating", "season_performance_bonus", "adjusted_predicted_rating", "status"]].head(15))
    else:
        print("\nℹ️ No games scheduled for tomorrow — draft slate is empty (0 players).")

    return output_home_path


if __name__ == "__main__":
    build_frontend_dataset()
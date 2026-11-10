import pandas as pd
import os
import sys
import json
import joblib
from datetime import datetime
from typing import Optional, Dict, Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def construct_prediction_features(
    df_history: pd.DataFrame,
    target_date: Optional[str] = None,
    std3_impute_value: Optional[float] = None,
    upcoming_games: Optional[pd.DataFrame] = None,
    roster_map: Optional[Dict[int, Dict[str, Any]]] = None,
) -> pd.DataFrame:
    """
    Given completed historical player games through game T:
    Constructs feature vectors to predict upcoming game T+1.

    Variable-window semantics (matches build_features.py exactly):
      - last_game:  rating from game T
      - last3_avg:  mean of up to 3 most recent games (T-2, T-1, T);
                    if fewer than 3 exist, uses all available
      - last5_avg:  mean of up to 5 most recent games; falls back to
                    all available if fewer than 5 exist
      - last7_avg:  mean of up to 7 most recent games; falls back to
                    all available if fewer than 7 exist
      - games_played_before_target: total completed games through T
      - expand_mean: mean of all completed games through T
      - days_since_prev_game: calendar days between upcoming target_date and
                              player's most recent completed game
      - is_b2b: binary indicator (1 if days_since_prev_game == 1, else 0)
      - last3_avg_fga: mean field goal attempts across up to 3 most recent
                       completed games before target
      - is_home: binary indicator (1 if player's team is home team, else 0)
      - std_3: sample standard deviation (ddof=1) of up to 3 most recent
               completed games before target. NaN if only 1 game exists.
               Optionally imputed with std3_impute_value (e.g. training median).

    For players with >=7 completed games the values are identical to
    the previous strict-window (>=7 required) implementation.

    Requires at least 1 completed game per player.
    Associates the prediction with the upcoming target_date.
    """
    df_sorted = df_history.sort_values(["personId", "game_date"]).copy()
    df_sorted["game_date"] = pd.to_datetime(df_sorted["game_date"])

    # Enforce system-wide historical cutoff (data archive floor: 2026-02-06)
    historical_cutoff = pd.to_datetime("2026-02-06")
    df_sorted = df_sorted[df_sorted["game_date"] >= historical_cutoff].copy()

    # Enforce current NBA season boundary: strictly on or after season start date.
    # Season boundary resolves dynamically (October 20 rule) rather than hardcoding 2026.
    try:
        from scripts.compute_season_multipliers import determine_season_boundary
    except ImportError:
        from compute_season_multipliers import determine_season_boundary

    season_start_str, _ = determine_season_boundary(df_sorted["game_date"], reference_date=target_date)
    season_start_dt = pd.to_datetime(season_start_str)
    df_sorted = df_sorted[df_sorted["game_date"] >= season_start_dt].copy()

    # If fga is not in history, merge from play_by_play.parquet if available
    if "fga" not in df_sorted.columns:
        pbp_path = "data/play_by_play.parquet"
        if os.path.exists(pbp_path):
            pbp = pd.read_parquet(pbp_path)
            shots = pbp[(pbp["personId"] > 1000) & (pbp["actionType"].isin(["Made Shot", "Missed Shot"]))]
            fga_df = shots.groupby(["gameId", "personId"]).size().reset_index(name="fga")
            df_sorted = df_sorted.merge(fga_df, on=["gameId", "personId"], how="left")
            df_sorted["fga"] = df_sorted["fga"].fillna(0)
        else:
            df_sorted["fga"] = 0

    # Determine home team IDs from upcoming_games if provided or from next_day_games.parquet
    home_team_ids = set()
    if upcoming_games is not None and not upcoming_games.empty and "homeTeamId" in upcoming_games.columns:
        home_team_ids = set(upcoming_games["homeTeamId"].unique())
    elif os.path.exists("data/next_day_games.parquet"):
        try:
            ndg = pd.read_parquet("data/next_day_games.parquet")
            if not ndg.empty and "homeTeamId" in ndg.columns:
                home_team_ids = set(ndg["homeTeamId"].unique())
        except Exception:
            pass

    # Determine default target date: align with upcoming slate if available
    if target_date is None:
        if upcoming_games is not None and not upcoming_games.empty and "nba_game_date" in upcoming_games.columns and pd.notna(upcoming_games["nba_game_date"].iloc[0]):
            target_date = str(upcoming_games["nba_game_date"].iloc[0])
        elif os.path.exists("data/next_day_games.parquet"):
            try:
                ndg = pd.read_parquet("data/next_day_games.parquet")
                if not ndg.empty and "nba_game_date" in ndg.columns and pd.notna(ndg["nba_game_date"].iloc[0]):
                    target_date = str(ndg["nba_game_date"].iloc[0])
            except Exception:
                pass

    if target_date is None:
        max_dt = df_sorted["game_date"].max()
        if pd.notna(max_dt):
            target_date = (max_dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        else:
            target_date = datetime.today().strftime("%Y-%m-%d")

    target_dt = pd.to_datetime(target_date)

    records = []

    for person_id, group in df_sorted.groupby("personId"):
        n_games = len(group)
        if n_games < 1:
            # No history at all — cannot form any feature
            continue

        latest_row = group.iloc[-1]
        ratings = group["rating"].tolist()

        # last_game: rating of most recent completed game
        last_game = float(ratings[-1])

        # Variable-window averages: mean of up to N most recent games.
        # If fewer than N games exist, use all available (min_periods=1 semantics).
        last3_avg = float(pd.Series(ratings[-3:]).mean())
        last5_avg = float(pd.Series(ratings[-5:]).mean())
        last7_avg = float(pd.Series(ratings[-7:]).mean())

        # Expanding historical mean: mean of ALL completed games through T.
        # Matches build_features.py's expanding().mean().shift(1) semantics:
        # the value for the target game T+1 is mean(g1 ... gT).
        expand_mean = float(pd.Series(ratings).mean())

        # Days since previous completed game (rest context):
        # difference in calendar days between target_date and last completed game date.
        last_game_dt = pd.to_datetime(latest_row["game_date"])
        days_since_prev_game = int((target_dt - last_game_dt).days)
        is_b2b = 1 if days_since_prev_game == 1 else 0

        # Rolling 3-game average FGA: mean of up to 3 most recent completed games
        fga_list = [float(x) for x in group["fga"].tolist()]
        last3_avg_fga = float(pd.Series(fga_list[-3:]).mean())

        # Resolve current team and player name from authoritative roster if available
        if roster_map and person_id in roster_map:
            team_id = roster_map[person_id]["teamId"]
            player_name = roster_map[person_id]["playerName"]
        else:
            team_id = latest_row["teamId"]
            player_name = latest_row["playerName"]

        # Home / Away game context
        is_home = 1 if team_id in home_team_ids else 0

        # Rolling 3-game sample standard deviation of completed ratings before target.
        # Requires >= 2 completed games. If only 1 game exists, std_3 is NaN before imputation.
        if len(ratings) < 2:
            std_3 = float("nan")
        else:
            std_3 = float(pd.Series(ratings[-3:]).std())

        records.append({
            "gameId": latest_row.get("gameId", ""),
            "teamId": team_id,
            "personId": person_id,
            "playerName": player_name,
            "rating": latest_row["rating"],
            "last_game_id": latest_row.get("gameId", ""),
            "last_game_date": latest_row["game_date"].strftime("%Y-%m-%d"),
            "game_date": target_date,
            "last_game": last_game,
            "last3_avg": last3_avg,
            "last5_avg": last5_avg,
            "last7_avg": last7_avg,
            "games_played_before_target": n_games,
            "expand_mean": expand_mean,
            "days_since_prev_game": days_since_prev_game,
            "is_b2b": is_b2b,
            "last3_avg_fga": last3_avg_fga,
            "is_home": is_home,
            "std_3": std_3,
        })

    # Add active roster players / rookies who have no completed historical games
    if roster_map:
        seen_pids = set(df_sorted["personId"].unique())
        for r_pid, r_info in roster_map.items():
            if r_pid not in seen_pids:
                r_team_id = r_info["teamId"]
                r_is_home = 1 if r_team_id in home_team_ids else 0
                records.append({
                    "gameId": "",
                    "teamId": r_team_id,
                    "personId": r_pid,
                    "playerName": r_info["playerName"],
                    "rating": 0.0,
                    "last_game_id": "",
                    "last_game_date": "",
                    "game_date": target_date,
                    "last_game": 0.0,
                    "last3_avg": 0.0,
                    "last5_avg": 0.0,
                    "last7_avg": 0.0,
                    "games_played_before_target": 0,
                    "expand_mean": 0.0,
                    "days_since_prev_game": 30,
                    "is_b2b": 0,
                    "last3_avg_fga": 0.0,
                    "is_home": r_is_home,
                    "std_3": std3_impute_value if std3_impute_value is not None else 0.0,
                })

    df_res = pd.DataFrame(records)
    if std3_impute_value is not None and not df_res.empty and "std_3" in df_res.columns:
        df_res["std_3"] = df_res["std_3"].fillna(std3_impute_value)

    return df_res


def predict_ratings(
    input_path: str = "data/player_game_ratings_with_dates.parquet",
    model_path: str = "models/rating_model.pkl",
    output_path: str = "data/predicted_ratings.parquet",
    target_date: Optional[str] = None,
) -> str:

    # ==============================
    # LOAD ACTIVE VALIDATED MODEL & METADATA
    # ==============================

    model_dir = os.path.dirname(model_path) if os.path.dirname(model_path) else "models"
    resolved_model_path = model_path
    active_meta = {}

    try:
        from scripts.model_registry import resolve_active_model, ModelValidationError
    except ImportError:
        from model_registry import resolve_active_model, ModelValidationError

    pointer_path = os.path.join(model_dir, "active_model.json")
    if os.path.exists(pointer_path) or model_path in ("models/rating_model.pkl", "azure_function/models/rating_model.pkl"):
        resolved_model_path, active_meta = resolve_active_model(base_dir=model_dir, fail_closed=True)
        print(f"Active validated model resolved: {active_meta.get('active_version', 'unknown')} ({resolved_model_path})")
        print(f"Active model SHA-256: {active_meta.get('sha256', '')[:12]}...")
    elif not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model: {model_path}")

    model = joblib.load(resolved_model_path)
    print("Model loaded")

    metadata_path = active_meta.get("metadata_path")
    if metadata_path and not os.path.isabs(metadata_path):
        metadata_path = os.path.join(model_dir, metadata_path)
    if not metadata_path or not os.path.exists(metadata_path):
        metadata_path = os.path.join(model_dir, "feature_metadata.json")

    std3_median = None
    if metadata_path and os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        std3_median = metadata.get("std3_median", None)
        print(f"Loaded std_3 training imputation median: {std3_median:.4f}" if std3_median is not None else "No std3_median in metadata")

    # ==============================
    # LOAD DATA & CONSTRUCT T+1 FEATURES
    # ==============================

    df = pd.read_parquet(input_path)
    print("Dataset loaded:", df.shape)

    if df.empty:
        raise Exception("Input dataset is empty")

    if "game_date" not in df.columns:
        raise ValueError("game_date column missing — check pipeline order")

    # Load authoritative current 2026-27 rosters
    roster_map = None
    try:
        from scripts.roster_provider import get_player_team_roster_map
        roster_map = get_player_team_roster_map()
        print(f"Loaded current roster map with {len(roster_map)} active players")
    except Exception as e:
        print(f"Warning: could not load current rosters ({e}). Falling back to historical assignments.")

    # Construct features for upcoming game T+1 using completed games through T
    latest = construct_prediction_features(
        df,
        target_date=target_date,
        std3_impute_value=std3_median,
        roster_map=roster_map
    )

    if latest.empty:
        raise Exception("No eligible players with at least 1 completed game for prediction")

    print(f"Players considered for next-game prediction: {len(latest)}")

    # ==============================
    # FEATURES
    # ==============================

    features = [
        "last_game",
        "last3_avg",
        "last5_avg",
        "last7_avg",
        "games_played_before_target",
        "expand_mean",
        "days_since_prev_game"
    ]

    X = latest[features]

    # ==============================
    # PREDICT
    # ==============================

    latest["predicted_rating"] = model.predict(X)
    latest["model_version"] = active_meta.get("active_version", "untracked")
    latest["model_sha256"] = active_meta.get("sha256", "")

    # ==============================
    # SORT
    # ==============================

    latest = latest.sort_values(
        "predicted_rating",
        ascending=False
    )

    # ==============================
    # OUTPUT
    # ==============================

    print("\nTop Predicted Performers (Upcoming Game)\n")

    display_cols = [
        "playerName",
        "teamId",
        "last_game_date",
        "game_date",
        "last_game",
        "last3_avg",
        "last5_avg",
        "last7_avg",
        "expand_mean",
        "days_since_prev_game",
        "predicted_rating"
    ]
    present_cols = [c for c in display_cols if c in latest.columns]
    print(latest[present_cols].head(25))

    # ==============================
    # SAVE
    # ==============================

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    latest.to_parquet(output_path, index=False)

    print(f"\nSaved predictions → {output_path}")

    return output_path


# ==============================
# RUN
# ==============================

if __name__ == "__main__":
    predict_ratings()

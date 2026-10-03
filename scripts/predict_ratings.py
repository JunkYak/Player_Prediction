import pandas as pd
import os
import sys
import json
import joblib
from datetime import datetime
from typing import Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def construct_prediction_features(
    df_history: pd.DataFrame,
    target_date: Optional[str] = None,
    std3_impute_value: Optional[float] = None,
    upcoming_games: Optional[pd.DataFrame] = None
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

    # Determine default target date (day after max historical game date) if not provided
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

        # Home / Away game context
        team_id = latest_row["teamId"]
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
            "playerName": latest_row["playerName"],
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
    # CHECK FILES
    # ==============================

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model: {model_path}")

    # Fallback to player_features.parquet if player_game_ratings_with_dates.parquet is missing
    if not os.path.exists(input_path):
        fallback_path = "data/player_features.parquet"
        if os.path.exists(fallback_path):
            input_path = fallback_path
        else:
            raise FileNotFoundError(f"Missing input file: {input_path}")

    # ==============================
    # LOAD MODEL & FEATURE METADATA
    # ==============================

    model = joblib.load(model_path)
    print("Model loaded")

    model_dir = os.path.dirname(model_path) if os.path.dirname(model_path) else "."
    metadata_path = os.path.join(model_dir, "feature_metadata.json")
    std3_median = None
    if os.path.exists(metadata_path):
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

    # Construct features for upcoming game T+1 using completed games through T
    latest = construct_prediction_features(df, target_date=target_date, std3_impute_value=std3_median)

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
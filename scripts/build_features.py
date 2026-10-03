import pandas as pd
import os


def build_features():

    input_path = "data/player_game_ratings_with_dates.parquet"
    output_path = "data/player_features.parquet"

    # ==============================
    # INPUT CHECK
    # ==============================

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Missing input file: {input_path}")

    # ==============================
    # LOAD DATA
    # ==============================

    df = pd.read_parquet(input_path)

    print("Loaded ratings with dates:", df.shape)

    if df.empty:
        raise Exception("Input dataset is empty")

    # ==============================
    # FIX DATE TYPE & MERGE FGA
    # ==============================

    df["game_date"] = pd.to_datetime(df["game_date"])

    pbp_path = "data/play_by_play.parquet"
    if os.path.exists(pbp_path):
        pbp = pd.read_parquet(pbp_path)
        shots = pbp[(pbp["personId"] > 1000) & (pbp["actionType"].isin(["Made Shot", "Missed Shot"]))]
        fga_df = shots.groupby(["gameId", "personId"]).size().reset_index(name="fga")
        df = df.merge(fga_df, on=["gameId", "personId"], how="left")
        df["fga"] = df["fga"].fillna(0)
    elif "fga" not in df.columns:
        df["fga"] = 0

    # ==============================
    # SORT CORRECTLY (CRITICAL)
    # ==============================

    df = df.sort_values(["personId", "game_date"])

    # ==============================
    # FEATURE CREATION
    # ==============================

    # previous game rating — requires >= 1 previous game
    df["last_game"] = df.groupby("personId")["rating"].shift(1)

    # Variable-window rolling averages (min_periods=1):
    #   - For players with N previous games where N < window size, computes the
    #     mean of all N available prior games rather than producing NaN.
    #   - For players with >= window games the result is IDENTICAL to the
    #     previous strict rolling(N).mean().shift(1) implementation.
    #   - All features are constructed strictly from games BEFORE the target
    #     row (shift(1) applied after the rolling, same as before).
    df["last3_avg"] = (
        df.groupby("personId")["rating"]
        .rolling(3, min_periods=1)
        .mean()
        .shift(1)
        .reset_index(level=0, drop=True)
    )

    df["last5_avg"] = (
        df.groupby("personId")["rating"]
        .rolling(5, min_periods=1)
        .mean()
        .shift(1)
        .reset_index(level=0, drop=True)
    )

    df["last7_avg"] = (
        df.groupby("personId")["rating"]
        .rolling(7, min_periods=1)
        .mean()
        .shift(1)
        .reset_index(level=0, drop=True)
    )

    # Explicit count of previous games available for each training row.
    # groupby cumcount() gives 0 for a player's first game, 1 for second, etc.
    # So games_played_before_target == number of prior-game ratings that
    # contributed to the rolling features above.
    df["games_played_before_target"] = df.groupby("personId").cumcount()

    # Expanding historical mean: mean of ALL completed ratings before the target
    # game.  expanding().mean() at row i is mean(g0..gi); shift(1) moves it
    # forward so row i receives mean(g0..gi-1) — strictly before-target.
    # With min_periods=1 (expanding default), this is defined for every row
    # after the first, i.e. all rows that survive the last_game.notna() filter.
    # No imputation is needed.
    df["expand_mean"] = (
        df.groupby("personId")["rating"]
        .expanding()
        .mean()
        .shift(1)
        .reset_index(level=0, drop=True)
    )

    # Days since previous completed game (rest context).
    # Computed as the difference in calendar days between the target game date
    # and the player's immediately preceding completed game date (shift(1)).
    # For a player's first game, shift(1) is NaT, producing NaN (dropped below).
    # For all subsequent games, this represents exact pre-game rest in days.
    df["days_since_prev_game"] = (
        df["game_date"] - df.groupby("personId")["game_date"].shift(1)
    ).dt.days

    # Back-to-back indicator: 1 if days_since_prev_game == 1, else 0.
    df["is_b2b"] = (df["days_since_prev_game"] == 1).astype(int)

    # Rolling 3-game average field goal attempts (workload context).
    # Uses sample FGA from up to 3 most recent completed games before target.
    # For players with <3 prior games, averages available prior games (min_periods=1).
    # Target game FGA is strictly excluded via shift(1).
    df["last3_avg_fga"] = (
        df.groupby("personId")["fga"]
        .rolling(3, min_periods=1)
        .mean()
        .shift(1)
        .reset_index(level=0, drop=True)
    )

    # Rolling 3-game sample standard deviation of completed ratings before target.
    # rolling(3, min_periods=2).std() computes sample standard deviation (ddof=1)
    # over up to 3 most recent games. shift(1) excludes the target game.
    # For players with exactly 1 previous game, std_3 is NaN (requires min_periods=2).
    df["std_3"] = (
        df.groupby("personId")["rating"]
        .rolling(3, min_periods=2)
        .std()
        .shift(1)
        .reset_index(level=0, drop=True)
    )

    # ==============================
    # REMOVE ROWS WITHOUT ANY HISTORY
    # ==============================
    # Keep only rows where at least 1 previous game exists (last_game is not
    # NaN).  This is the absolute minimum: without a prior game there is no
    # historical signal whatsoever and we cannot define any rolling feature.

    features = df[df["last_game"].notna()].copy()

    if features.empty:
        raise Exception("No rows left after feature engineering")

    # ==============================
    # SAVE
    # ==============================

    os.makedirs("data", exist_ok=True)

    features.to_parquet(output_path, index=False)

    print("Saved player_features.parquet")
    print("Rows:", len(features))
    print("Players:", features["personId"].nunique())
    print("Min games_played_before_target:", features["games_played_before_target"].min())
    print("expand_mean range: [{:.3f}, {:.3f}]".format(
        features["expand_mean"].min(), features["expand_mean"].max()
    ))
    print("days_since_prev_game non-null count: {} / {} (mean: {:.2f}, median: {:.1f})".format(
        features["days_since_prev_game"].notna().sum(), len(features),
        features["days_since_prev_game"].mean(), features["days_since_prev_game"].median()
    ))
    print("last3_avg_fga non-null count: {} / {} (mean: {:.2f}, median: {:.2f})".format(
        features["last3_avg_fga"].notna().sum(), len(features),
        features["last3_avg_fga"].mean(), features["last3_avg_fga"].median()
    ))
    print("is_b2b positive count: {} / {} ({:.1%})".format(
        features["is_b2b"].sum(), len(features), features["is_b2b"].mean()
    ))
    print("std_3 non-null count: {} / {} (NaN for 1-prior-game rows: {})".format(
        features["std_3"].notna().sum(), len(features), features["std_3"].isna().sum()
    ))

    print("\nSample:")
    print(features.head())

    return output_path


# ==============================
# RUN
# ==============================

if __name__ == "__main__":
    build_features()
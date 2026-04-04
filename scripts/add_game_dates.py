import pandas as pd
import os
from nba_api.stats.endpoints import leaguegamelog


def add_game_dates():

    input_path = "data/player_game_ratings.parquet"
    output_path = "data/player_game_ratings_with_dates.parquet"

    # ==============================
    # INPUT CHECK
    # ==============================

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Missing input file: {input_path}")

    # ==============================
    # LOAD DATA
    # ==============================

    df = pd.read_parquet(input_path)
    print("Loaded player ratings:", df.shape)

    if df.empty:
        raise Exception("Input dataset is empty")

    # ==============================
    # FETCH GAME LOG (DATES)
    # ==============================

    print("Fetching game logs...")

    games = leaguegamelog.LeagueGameLog().get_data_frames()[0]

    games = games[["GAME_ID", "GAME_DATE"]].drop_duplicates()
    games.columns = ["gameId", "game_date"]

    print("Games fetched:", games.shape)

    # ==============================
    # MERGE
    # ==============================

    df = df.merge(games, on="gameId", how="left")

    missing = df["game_date"].isna().sum()

    print("\nMissing dates:", missing)

    if missing > 0:
        print("⚠️ Warning: Some games missing dates")

    # ==============================
    # SAVE
    # ==============================

    os.makedirs("data", exist_ok=True)

    df.to_parquet(output_path, index=False)

    print("\nSaved:", output_path)

    return output_path


# ==============================
# RUN
# ==============================

if __name__ == "__main__":
    add_game_dates()
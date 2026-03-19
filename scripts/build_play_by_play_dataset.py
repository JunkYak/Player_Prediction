import time
import pandas as pd
from datetime import datetime, timedelta
from nba_api.stats.endpoints import leaguegamelog, playbyplayv3


# ==============================
# CONFIG
# ==============================

DAYS_BACK = 60
OUTPUT_FILE = "data/play_by_play.parquet"


# ==============================
# FETCH GAME LIST
# ==============================

def fetch_games():

    end_date = datetime.today()
    start_date = end_date - timedelta(days=DAYS_BACK)

    games = leaguegamelog.LeagueGameLog(
    date_from_nullable=start_date.strftime("%m/%d/%Y"),
    date_to_nullable=end_date.strftime("%m/%d/%Y")
)

    df = games.get_data_frames()[0]

    return df["GAME_ID"].unique()


# ==============================
# FETCH PLAY BY PLAY
# ==============================

def fetch_pbp(game_id):

    pbp = playbyplayv3.PlayByPlayV3(game_id=game_id)
    df = pbp.get_data_frames()[0]

    # keep only required columns
    cols = [
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

    df = df[cols]

    return df


# ==============================
# BUILD DATASET
# ==============================

def build_dataset():

    game_ids = fetch_games()

    print(f"Found {len(game_ids)} games")

    all_games = []

    for i, gid in enumerate(game_ids):

        print(f"Fetching game {i+1}/{len(game_ids)}  {gid}")

        try:
            df = fetch_pbp(gid)
            all_games.append(df)

        except Exception as e:
            print("Failed:", gid, e)

        time.sleep(0.6)  # avoid API rate limit

    dataset = pd.concat(all_games, ignore_index=True)

    dataset.to_parquet(OUTPUT_FILE, index=False)

    print("\nSaved dataset")
    print("Rows:", len(dataset))
    print("Games:", dataset["gameId"].nunique())


# ==============================
# RUN
# ==============================

if __name__ == "__main__":

    build_dataset()
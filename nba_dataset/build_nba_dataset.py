import time
import pandas as pd
from datetime import datetime, timedelta
from nba_api.stats.endpoints import leaguegamelog, playbyplayv3

def build_last_7_days_dataset():
    end_date = datetime.today()
    start_date = end_date - timedelta(days=7)

    games_df = leaguegamelog.LeagueGameLog(
        date_from_nullable=start_date.strftime("%m/%d/%Y"),
        date_to_nullable=end_date.strftime("%m/%d/%Y"),
    ).get_data_frames()[0]

    game_ids = games_df["GAME_ID"].unique()

    rows = []

    for idx, game_id in enumerate(game_ids, start=1):
        print(f"Processing game {idx}/{len(game_ids)}: {game_id}")

        pbp = playbyplayv3.PlayByPlayV3(game_id=game_id)
        actions = pbp.get_dict()["game"]["actions"]

        for a in actions:
            if a.get("personId") is None:
                continue

            rows.append({
                "personId": a.get("personId"),
                "period": a.get("period"),
                "clock": a.get("clock"),
                "actionType": a.get("actionType"),
                "subType": a.get("subType"),
                "shotResult": a.get("shotResult"),
                "shotDistance": a.get("shotDistance"),
                "shotValue": a.get("shotValue"),
                "description": a.get("description"),
                "gameId": game_id
            })

        time.sleep(0.7)  # SAFE throttle

    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = build_last_7_days_dataset()
    df.to_parquet("nba_last_7_days.parquet", index=False)
    print("\nDataset saved as nba_last_7_days.parquet")

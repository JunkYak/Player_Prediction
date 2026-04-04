import pandas as pd
import os
from datetime import datetime
from nba_api.stats.endpoints import scoreboardv3


def get_next_day_games():

    game_date = datetime.today().strftime("%Y-%m-%d")
    print(f"Fetching games for: {game_date}")

    try:
        scoreboard = scoreboardv3.ScoreboardV3(game_date=game_date)
        data = scoreboard.get_dict()
        games = data["scoreboard"]["games"]
    except Exception as e:
        print("Error fetching schedule:", e)
        games = []

    game_list = []
    team_ids = set()

    for game in games:
        game_id = game["gameId"]

        home_id = int(game["homeTeam"]["teamId"])
        away_id = int(game["awayTeam"]["teamId"])

        game_list.append({
            "gameId": game_id,
            "homeTeamId": home_id,
            "awayTeamId": away_id
        })

        team_ids.add(home_id)
        team_ids.add(away_id)

    games_df = pd.DataFrame(game_list)

    print("Games Found:", len(games_df))

    os.makedirs("data", exist_ok=True)

    games_df.to_parquet("data/next_day_games.parquet", index=False)

    pd.Series(list(team_ids)).to_csv(
        "data/next_day_team_ids.csv",
        index=False
    )

    print("Saved next day games")

    return "data/next_day_games.parquet"


if __name__ == "__main__":
    get_next_day_games()
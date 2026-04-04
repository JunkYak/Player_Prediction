import pandas as pd
import os


def build_frontend_dataset():

    pred_path = "data/predicted_ratings.parquet"
    games_path = "data/next_day_games.parquet"

    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Missing: {pred_path}")

    if not os.path.exists(games_path):
        raise FileNotFoundError(f"Missing: {games_path}")

    pred = pd.read_parquet(pred_path)
    games = pd.read_parquet(games_path)

    print("Loaded predictions:", pred.shape)
    print("Loaded games:", games.shape)

    if pred.empty or games.empty:
        raise Exception("Input datasets cannot be empty")

    # ==============================
    # TEAMS PLAYING
    # ==============================

    teams = set(games["homeTeamId"]).union(set(games["awayTeamId"]))

    # ==============================
    # OPPONENT MAP
    # ==============================

    opponent_map = {}

    for _, row in games.iterrows():
        home = row["homeTeamId"]
        away = row["awayTeamId"]

        opponent_map[home] = away
        opponent_map[away] = home

    # ==============================
    # TEAM NAME MAP
    # ==============================

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

    # ==============================
    # ADD METADATA
    # ==============================

    pred["teamName"] = pred["teamId"].map(team_map)
    pred["opponentId"] = pred["teamId"].map(opponent_map)
    pred["opponentName"] = pred["opponentId"].map(team_map)

    # Fix missing mappings
    pred["teamName"] = pred["teamName"].fillna("Unknown")
    pred["opponentName"] = pred["opponentName"].fillna("Unknown")

    # Player headshots
    pred["headshot"] = pred["personId"].apply(
        lambda x: f"https://cdn.nba.com/headshots/nba/latest/1040x760/{x}.png"
    )

    # ==============================
    # FULL DATASET
    # ==============================

    full_df = pred.sort_values(
        ["teamId", "predicted_rating"],
        ascending=[True, False]
    )

    # ==============================
    # HOMEPAGE DATASET (SMARTER)
    # ==============================

    home_df = pred[pred["teamId"].isin(teams)].copy()

    # Take top 3 per team (balanced)
    home_df = (
        home_df.sort_values("predicted_rating", ascending=False)
        .groupby("teamId")
        .head(3)
    )

    home_df = home_df.sort_values("predicted_rating", ascending=False)

    # ==============================
    # SAVE
    # ==============================

    os.makedirs("data", exist_ok=True)

    full_df.to_parquet("data/frontend_all.parquet", index=False)
    home_df.to_parquet("data/frontend_home.parquet", index=False)

    print("\nSaved:")
    print("→ data/frontend_all.parquet")
    print("→ data/frontend_home.parquet")

    print("\nTop Players (Next Day):")
    print(home_df[["playerName", "teamName", "opponentName", "predicted_rating"]].head(15))

    return "data/frontend_home.parquet"


if __name__ == "__main__":
    build_frontend_dataset()
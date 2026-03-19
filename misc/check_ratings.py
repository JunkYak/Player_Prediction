import pandas as pd
from nba_api.stats.static import players, teams

# -----------------------------
# LOAD DATA
# -----------------------------
ratings = pd.read_parquet("player_game_ratings.parquet")

# -----------------------------
# FETCH PLAYER LIST FROM NBA API
# -----------------------------
nba_players = players.get_players()
player_df = pd.DataFrame(nba_players)

player_df = player_df.rename(columns={
    "id": "personId",
    "full_name": "playerName"
})

player_df = player_df[["personId", "playerName"]]

# -----------------------------
# FETCH TEAM LIST FROM NBA API
# -----------------------------
nba_teams = teams.get_teams()
team_df = pd.DataFrame(nba_teams)

team_df = team_df.rename(columns={
    "id": "teamId",
    "full_name": "teamName"
})

team_df = team_df[["teamId", "teamName"]]

# -----------------------------
# MERGE DATA
# -----------------------------
df = ratings.merge(player_df, on="personId", how="left")
df = df.merge(team_df, on="teamId", how="left")

# reorder columns
df = df[["gameId", "teamName", "playerName", "personId", "rating"]]

print("\nFirst rows:\n")
print(df.head())

print("\nLast rows:\n")
print(df.tail())
import pandas as pd

df = pd.read_parquet("data/player_game_ratings.parquet")

print("\nBasic stats:")
print(df["rating"].describe())

print("\nTop ratings:")
print(df.sort_values("rating", ascending=False).head(10))

print("\nLowest ratings:")
print(df.sort_values("rating").head(10))
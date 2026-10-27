import pandas as pd
import os
import json


def build_team_rankings():

    input_path = "data/predicted_ratings.parquet"
    output_path = "data/team_rankings.json"

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Missing input: {input_path}")

    df = pd.read_parquet(input_path)

    print("Loaded predictions:", df.shape)

    if df.empty:
        raise Exception("Predictions dataset is empty")

    # CLEAN
    df = df[
        (df["teamId"] > 0) &
        (df["playerName"].notna()) &
        (df["playerName"] != "")
    ]

    # 🔥 FIX: convert timestamps → string
    if "game_date" in df.columns:
        df["game_date"] = df["game_date"].astype(str)

    teams = df.groupby("teamId")

    output = {}

    for team_id, group in teams:
        group = group.sort_values("predicted_rating", ascending=False)
        output[str(team_id)] = group.to_dict(orient="records")

    os.makedirs("data", exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved → {output_path}")

    return output_path


if __name__ == "__main__":
    build_team_rankings()
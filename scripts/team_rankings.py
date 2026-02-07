import pandas as pd

# ==============================
# LOAD PREDICTIONS
# ==============================

df = pd.read_parquet("data/predicted_ratings.parquet")

print("Loaded predictions:", df.shape)

# ==============================
# CLEAN (just in case)
# ==============================

df = df[
    (df["teamId"] > 0) &
    (df["playerName"].notna()) &
    (df["playerName"] != "")
]

# ==============================
# GROUP BY TEAM
# ==============================

teams = df.groupby("teamId")

# ==============================
# DISPLAY TEAM-WISE RANKINGS
# ==============================

for team_id, group in teams:

    print(f"\n==============================")
    print(f"TEAM: {team_id}")
    print(f"==============================")

    group = group.sort_values(
        "predicted_rating",
        ascending=False
    )

    for _, row in group.iterrows():
        print(f"{row['playerName']:<20} {row['predicted_rating']:.2f}")

# ==============================
# SAVE TEAM SPLIT (important for UI later)
# ==============================

output = {}

for team_id, group in teams:
    group = group.sort_values("predicted_rating", ascending=False)
    output[str(team_id)] = group.to_dict(orient="records")

import json

with open("data/team_rankings.json", "w") as f:
    json.dump(output, f, indent=2)

print("\nSaved team_rankings.json")
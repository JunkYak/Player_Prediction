import pandas as pd
import joblib
import json

# Load model

model = joblib.load("models/rating_model.pkl")

# Load data

df = pd.read_parquet("data/player_features_with_dates.parquet")

# Get latest game per player

latest = (
df.sort_values("game_date")
.groupby("personId")
.tail(1)
)

# Features

features = ["last_game", "last3_avg", "last5_avg", "last7_avg"]
X = latest[features]

# Predict

latest["predicted_rating"] = model.predict(X)

# Sort

latest = latest.sort_values("predicted_rating", ascending=False)

# ✅ TAKE TOP 10

top_players = latest.head(10)

# Convert to JSON format

players = []

for _, row in top_players.iterrows():
    players.append({
        "name": row["playerName"],
        "team": int(row["teamId"]),
        "rating": float(row["predicted_rating"])
    })

print(json.dumps({"players": players}))
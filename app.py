from flask import Flask, jsonify
from flask_cors import CORS
import subprocess
import json
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)  # ✅ ADD THIS

@app.route("/predict", methods=["GET"])
def predict():
    result = subprocess.run(
        ["py", "predict_api.py"],
        capture_output=True,
        text=True
    )

    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    return result.stdout

@app.route("/teams", methods=["GET"])
def get_teams():
    model = joblib.load("models/rating_model.pkl")
    df = pd.read_parquet("data/player_features_with_dates.parquet")

    latest = (
    df.sort_values("game_date")
    .groupby("personId")
    .tail(1)
    )

    features = ["last_game", "last3_avg", "last5_avg", "last7_avg"]
    latest["predicted_rating"] = model.predict(latest[features])

    teams = {}

    for team_id, group in latest.groupby("teamId"):
        top_players = group.sort_values(
            "predicted_rating",
            ascending=False
        ).head(5)

        teams[int(team_id)] = [
            {
                "name": row["playerName"],
                "rating": float(row["predicted_rating"])
            }
            for _, row in top_players.iterrows()
        ]

    return jsonify(teams)






if __name__ == "__main__":
    app.run(debug=True)
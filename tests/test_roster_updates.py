"""
tests/test_roster_updates.py

Phase B Regression Tests:
Verifies that:
1. Current rosters reflect trades, signings, and roster changes.
2. Players traded from Team A to Team B have Team B in next-game prediction
   and on the Teams page (frontend_all.parquet).
3. Historical game logs for the traded player remain associated with Team A
   (historical statistics, ratings, and features remain uncorrupted).
4. Rookies / players with 0 historical completed games are assigned to their
   current team with baseline features and finite model ratings.
5. Home / away (is_home) context uses current team assignment relative to upcoming games.
6. Roster validation fails safely on corrupted / incomplete roster data.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.predict_ratings import construct_prediction_features
from scripts.build_frontend_dataset import build_frontend_dataset
from scripts.roster_provider import validate_roster_dataframe, NBA_TEAMS


def test_traded_player_reflects_new_team_while_preserving_history():
    """
    Simulate Paul George (202331) having played 5 games for Celtics (1610612738)
    in historical data, but traded/signed to 76ers (1610612755) in current roster.

    Expected:
    - In historical game logs: all 5 rows remain teamId=1610612738.
    - In construct_prediction_features: row has teamId=1610612755 (76ers).
    - Historical statistics (last_game, last3_avg, expand_mean) reflect historical ratings.
    """
    history_df = pd.DataFrame({
        "personId": [202331] * 5,
        "playerName": ["Paul George"] * 5,
        "teamId": [1610612738] * 5,  # Celtics in past games
        "gameId": [f"002250000{i}" for i in range(1, 6)],
        "rating": [12.0, 14.0, 10.0, 16.0, 18.0],
        "game_date": pd.to_datetime(["2026-10-20", "2026-10-20", "2026-10-21", "2026-10-21", "2026-10-21"]),
    })

    roster_map = {
        202331: {
            "teamId": 1610612755,  # 76ers in current 2026-27 roster
            "teamName": "76ers",
            "playerName": "Paul George",
        }
    }

    # Upcoming game: 76ers (home) vs Cavaliers
    upcoming_games = pd.DataFrame([{
        "gameId": "0022600006",
        "homeTeamId": 1610612755,  # 76ers
        "awayTeamId": 1610612739,
        "nba_game_date": "2026-10-22",
    }])

    feat = construct_prediction_features(
        history_df,
        target_date="2026-10-22",
        upcoming_games=upcoming_games,
        roster_map=roster_map
    )

    assert len(feat) == 1
    row = feat.iloc[0]

    # Current team must be 76ers
    assert row["teamId"] == 1610612755
    assert row["is_home"] == 1  # 76ers are home team

    # Historical stats are calculated from past games
    assert row["last_game"] == 18.0
    assert row["last3_avg"] == (10.0 + 16.0 + 18.0) / 3.0
    assert row["expand_mean"] == (12.0 + 14.0 + 10.0 + 16.0 + 18.0) / 5.0
    assert row["games_played_before_target"] == 5

    # Original historical dataframe was NOT mutated
    assert all(history_df["teamId"] == 1610612738)


def test_rookie_without_historical_games_gets_baseline_features():
    """
    Rookies on 2026-27 rosters with 0 completed historical games should be added
    with 0-game baseline features and assigned to their current team.
    """
    empty_history = pd.DataFrame(columns=["personId", "playerName", "teamId", "gameId", "rating", "game_date"])

    roster_map = {
        999001: {
            "teamId": 1610612758,  # Kings
            "teamName": "Kings",
            "playerName": "Rookie Guard",
        }
    }

    feat = construct_prediction_features(
        empty_history,
        target_date="2026-10-22",
        roster_map=roster_map
    )

    assert len(feat) == 1
    row = feat.iloc[0]
    assert row["personId"] == 999001
    assert row["teamId"] == 1610612758
    assert row["playerName"] == "Rookie Guard"
    assert row["games_played_before_target"] == 0
    assert row["last_game"] == 0.0
    assert row["last3_avg"] == 0.0
    assert row["expand_mean"] == 0.0


def test_teams_page_displays_current_team_membership(tmp_path):
    """
    Verify build_frontend_dataset places traded players on their new team in frontend_all.parquet.
    """
    pred_df = pd.DataFrame([{
        "gameId": "",
        "teamId": 1610612755,  # 76ers
        "personId": 202331,
        "playerName": "Paul George",
        "rating": 15.0,
        "last_game_id": "0022500001",
        "last_game_date": "2026-03-09",
        "game_date": "2026-10-22",
        "last_game": 18.0,
        "last3_avg": 14.67,
        "last5_avg": 14.0,
        "last7_avg": 14.0,
        "games_played_before_target": 5,
        "expand_mean": 14.0,
        "days_since_prev_game": 227,
        "is_b2b": 0,
        "last3_avg_fga": 15.0,
        "is_home": 1,
        "std_3": 3.0,
        "predicted_rating": 15.2,
        "status": "Active",
        "injury_reason": "Not Listed",
        "injury_date": "2026-10-22",
        "injury_source": "NBA",
        "status_rank": 0,
    }])

    games_df = pd.DataFrame([{
        "gameId": "0022600006",
        "homeTeamId": 1610612755,
        "awayTeamId": 1610612739,
        "nba_game_date": "2026-10-22",
    }])

    home_out = str(tmp_path / "frontend_home.parquet")
    all_out = str(tmp_path / "frontend_all.parquet")

    build_frontend_dataset(
        pred_df=pred_df,
        games_df=games_df,
        output_home_path=home_out,
        output_all_path=all_out,
    )

    df_all = pd.read_parquet(all_out)
    pg = df_all[df_all["personId"] == 202331].iloc[0]
    assert pg["teamId"] == 1610612755
    assert pg["teamName"] == "76ers"

    df_home = pd.read_parquet(home_out)
    assert len(df_home) == 1
    assert df_home.iloc[0]["opponentName"] == "Cavaliers"
    assert df_home.iloc[0]["game_date"] == "2026-10-22"


def test_roster_validation_rejects_corrupted_data():
    """
    Roster validation fails safely on empty, null, or incomplete rosters.
    """
    # Empty
    valid, msg = validate_roster_dataframe(pd.DataFrame())
    assert not valid

    # Implausible count (< 400 players)
    small_df = pd.DataFrame([
        {"personId": i, "playerName": f"P{i}", "teamId": 1610612737, "teamName": "Hawks"}
        for i in range(10)
    ])
    valid, msg = validate_roster_dataframe(small_df)
    assert not valid
    assert "Implausible" in msg

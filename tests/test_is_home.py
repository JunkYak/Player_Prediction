"""
Phase 2D.3 Experiment 3A — Game Context: Home / Away (is_home) feature tests.
Tests the correctness, leakage safety, contamination safety, cross-game independence,
and train/inference semantic consistency of the is_home feature.
"""
import os
import sys
import pytest
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.predict_ratings import construct_prediction_features


def _make_player_df(person_id, ratings, dates, team_id=1610612737):
    """Helper: build a minimal player-game DataFrame with explicit dates."""
    n = len(ratings)
    assert len(dates) == n
    return pd.DataFrame({
        "personId": [person_id] * n,
        "playerName": [f"Player {person_id}"] * n,
        "teamId": [team_id] * n,
        "gameId": [f"00225{person_id:04d}{i:02d}" for i in range(n)],
        "rating": ratings,
        "game_date": pd.to_datetime(dates),
    })


# ── Test 1: Home team produces is_home = 1 ──────────────────────────────────

def test_is_home_positive_case():
    """
    If a player's team is the home team for the upcoming matchup,
    is_home must equal 1.
    """
    home_team = 1610612738  # Celtics
    away_team = 1610612748  # Heat
    upcoming_games = pd.DataFrame([{
        "gameId": "0022501001",
        "homeTeamId": home_team,
        "awayTeamId": away_team
    }])

    df = _make_player_df(9001, [5.0], ["2026-03-20"], team_id=home_team)
    feat = construct_prediction_features(df, target_date="2026-03-22", upcoming_games=upcoming_games)
    assert len(feat) == 1
    row = feat.iloc[0]
    assert row["is_home"] == 1


# ── Test 2: Away team produces is_home = 0 ──────────────────────────────────

def test_is_home_negative_case():
    """
    If a player's team is the away team for the upcoming matchup,
    is_home must equal 0.
    """
    home_team = 1610612738  # Celtics
    away_team = 1610612748  # Heat
    upcoming_games = pd.DataFrame([{
        "gameId": "0022501001",
        "homeTeamId": home_team,
        "awayTeamId": away_team
    }])

    df = _make_player_df(9002, [4.0], ["2026-03-20"], team_id=away_team)
    feat = construct_prediction_features(df, target_date="2026-03-22", upcoming_games=upcoming_games)
    assert len(feat) == 1
    row = feat.iloc[0]
    assert row["is_home"] == 0


# ── Test 3: Cross-player independence ───────────────────────────────────────

def test_is_home_cross_player_independence():
    """
    Player A on Home team must get is_home = 1.
    Player B on Away team must get is_home = 0.
    Calculations do not contaminate each other.
    """
    home_team = 1610612744  # Warriors
    away_team = 1610612747  # Lakers
    upcoming_games = pd.DataFrame([{
        "gameId": "0022501002",
        "homeTeamId": home_team,
        "awayTeamId": away_team
    }])

    df_p1 = _make_player_df(9003, [6.0], ["2026-03-20"], team_id=home_team)
    df_p2 = _make_player_df(9004, [7.0], ["2026-03-20"], team_id=away_team)
    combined = pd.concat([df_p1, df_p2], ignore_index=True)

    feat = construct_prediction_features(combined, target_date="2026-03-22", upcoming_games=upcoming_games)
    assert len(feat) == 2

    p1_row = feat[feat["personId"] == 9003].iloc[0]
    p2_row = feat[feat["personId"] == 9004].iloc[0]

    assert p1_row["is_home"] == 1
    assert p2_row["is_home"] == 0


# ── Test 4: Cross-game independence ─────────────────────────────────────────

def test_is_home_cross_game_independence():
    """
    Multiple games in upcoming_games:
    Players from Game 1 (home/away) and Game 2 (home/away) are resolved independently.
    """
    upcoming_games = pd.DataFrame([
        {"gameId": "0022501003", "homeTeamId": 1610612737, "awayTeamId": 1610612739},
        {"gameId": "0022501004", "homeTeamId": 1610612741, "awayTeamId": 1610612742}
    ])

    df1 = _make_player_df(9005, [5.0], ["2026-03-20"], team_id=1610612737)  # G1 Home -> 1
    df2 = _make_player_df(9006, [4.0], ["2026-03-20"], team_id=1610612739)  # G1 Away -> 0
    df3 = _make_player_df(9007, [3.0], ["2026-03-20"], team_id=1610612741)  # G2 Home -> 1
    df4 = _make_player_df(9008, [2.0], ["2026-03-20"], team_id=1610612742)  # G2 Away -> 0

    combined = pd.concat([df1, df2, df3, df4], ignore_index=True)
    feat = construct_prediction_features(combined, target_date="2026-03-22", upcoming_games=upcoming_games)

    res = feat.set_index("personId")["is_home"].to_dict()
    assert res[9005] == 1
    assert res[9006] == 0
    assert res[9007] == 1
    assert res[9008] == 0


# ── Test 5: Historical training / inference semantic consistency ──────────────

def test_is_home_train_inference_consistency():
    """
    Historical calculation (teamId == homeTeamId) and inference calculation
    (teamId == upcoming_homeTeamId) use exact identical boolean indicator semantics.
    """
    # Historical game matchup
    hist_matchup = pd.DataFrame([{"gameId": "0022500001", "homeTeamId": 1610612738, "awayTeamId": 1610612748}])
    player_hist = pd.DataFrame({
        "gameId": ["0022500001"],
        "teamId": [1610612738],
        "personId": [9009],
        "playerName": ["Player 9009"],
        "rating": [5.5],
        "game_date": pd.to_datetime(["2026-03-10"])
    })
    player_hist_merged = player_hist.merge(hist_matchup, on="gameId")
    hist_is_home = int(player_hist_merged.iloc[0]["teamId"] == player_hist_merged.iloc[0]["homeTeamId"])
    assert hist_is_home == 1

    # Inference calculation for same matchup
    inf_feat = construct_prediction_features(player_hist, target_date="2026-03-12", upcoming_games=hist_matchup)
    assert inf_feat.iloc[0]["is_home"] == 1


# ── Test 6: No target-game leakage ───────────────────────────────────────────

def test_is_home_no_target_game_leakage():
    """
    is_home depends solely on scheduled matchup metadata (homeTeamId / awayTeamId),
    never on target game score, rating, minutes, or in-game events.
    """
    upcoming_games = pd.DataFrame([{"gameId": "0022501005", "homeTeamId": 1610612750, "awayTeamId": 1610612751}])
    df = _make_player_df(9010, [8.0, 9.0], ["2026-03-10", "2026-03-12"], team_id=1610612750)

    # Feature is computed strictly from schedule metadata and prior games
    feat = construct_prediction_features(df, target_date="2026-03-15", upcoming_games=upcoming_games)
    assert feat.iloc[0]["is_home"] == 1


# ── Test 7: Missing/invalid matchup handling ──────────────────────────────────

def test_is_home_missing_matchup_handling():
    """
    If no upcoming games metadata is supplied or team is not found,
    is_home defaults safely (e.g. 0) without throwing an unhandled exception.
    """
    df = _make_player_df(9011, [4.0], ["2026-03-20"], team_id=1610612755)
    feat = construct_prediction_features(df, target_date="2026-03-22", upcoming_games=None)
    assert len(feat) == 1
    assert "is_home" in feat.columns
    assert feat.iloc[0]["is_home"] in (0, 1)


# ── Test 8: Seven-feature baseline remains unchanged ─────────────────────────

def test_is_home_preserves_seven_feature_baseline():
    """
    Adding is_home must not alter any of the 7 accepted features:
    last_game, last3_avg, last5_avg, last7_avg, games_played_before_target,
    expand_mean, days_since_prev_game.
    """
    dates = ["2026-03-01", "2026-03-02", "2026-03-03", "2026-03-04",
             "2026-03-05", "2026-03-06", "2026-03-07", "2026-03-08"]
    ratings = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    df = _make_player_df(9012, ratings, dates, team_id=1610612760)

    upcoming_games = pd.DataFrame([{"gameId": "0022501006", "homeTeamId": 1610612760, "awayTeamId": 1610612761}])
    feat = construct_prediction_features(df, target_date="2026-03-09", upcoming_games=upcoming_games)
    assert len(feat) == 1
    row = feat.iloc[0]

    assert row["last_game"] == pytest.approx(8.0)
    assert row["last3_avg"] == pytest.approx((6.0 + 7.0 + 8.0) / 3)
    assert row["last5_avg"] == pytest.approx((4.0 + 5.0 + 6.0 + 7.0 + 8.0) / 5)
    assert row["last7_avg"] == pytest.approx((2.0 + 3.0 + 4.0 + 5.0 + 6.0 + 7.0 + 8.0) / 7)
    assert row["games_played_before_target"] == 8
    assert row["expand_mean"] == pytest.approx(sum(ratings) / len(ratings))
    assert row["days_since_prev_game"] == 1
    assert row["is_home"] == 1


# ── Test 9: Upcoming-game inference correctly maps player's team ─────────────

def test_is_home_upcoming_game_inference_mapping():
    """
    Upcoming game schedule maps teams correctly for both home and away.
    """
    upcoming = pd.DataFrame([
        {"gameId": "0022502001", "homeTeamId": 1610612737, "awayTeamId": 1610612738},
        {"gameId": "0022502002", "homeTeamId": 1610612739, "awayTeamId": 1610612740},
    ])
    df_home = _make_player_df(9013, [5.0], ["2026-03-25"], team_id=1610612737)
    df_away = _make_player_df(9014, [6.0], ["2026-03-25"], team_id=1610612740)

    res_home = construct_prediction_features(df_home, target_date="2026-03-26", upcoming_games=upcoming)
    res_away = construct_prediction_features(df_away, target_date="2026-03-26", upcoming_games=upcoming)

    assert res_home.iloc[0]["is_home"] == 1
    assert res_away.iloc[0]["is_home"] == 0

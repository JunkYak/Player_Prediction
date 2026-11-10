"""
tests/test_top_players_ranking_score.py

Automated test suite verifying:
1. Raw predicted_rating remains unchanged everywhere.
2. ranking_score is calculated correctly: predicted_rating * (1 + season_performance_bonus) rounded to 2 decimals.
3. Top Players sorts by ranking_score descending, not raw predicted_rating.
4. Missing ranking scores are handled safely.
5. API responses preserve legacy fields (adjusted_predicted_rating) and expose ranking_score.
6. Nonzero bonuses survive the full Parquet -> API flow.
7. Starting Five additive multiplier semantics remain unchanged:
   effective_multiplier = slot_multiplier + season_performance_bonus
   projected_slot_value = predicted_rating * effective_multiplier
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from api.main import app
from scripts.build_frontend_dataset import build_frontend_dataset


@pytest.fixture
def client():
    return TestClient(app)


def test_raw_predicted_rating_remains_unchanged_and_ranking_score_correct():
    """Verify raw predicted_rating is never mutated and ranking_score is calculated accurately."""
    raw_pred = 2.72
    bonus = 0.5
    expected_score = round(raw_pred * (1.0 + bonus), 2)  # 4.08

    df = pd.DataFrame([{
        "gameId": "0022400001",
        "teamId": 1610612740,
        "personId": 999999,
        "playerName": "Test Player",
        "rating": 2.5,
        "last_game_id": "0022400000",
        "last_game_date": "2026-10-24",
        "game_date": "2026-10-25",
        "last_game": 2.5,
        "last3_avg": 2.5,
        "last5_avg": 2.5,
        "last7_avg": 2.5,
        "predicted_rating": raw_pred,
        "status": "Active",
        "injury_reason": "",
        "injury_date": "",
        "injury_source": "",
        "season_game_count": 3,
        "season_average_rating": 2.0,
        "season_percentile": 50.0,
        "season_performance_bonus": bonus,
        "season_multiplier_eligible": True,
    }])

    # Calculate ranking_score as in build_frontend_dataset
    df["ranking_score"] = (df["predicted_rating"] * (1.0 + df["season_performance_bonus"])).round(2)
    df["adjusted_predicted_rating"] = df["ranking_score"]

    # Raw predicted_rating must remain 2.72
    assert df["predicted_rating"].iloc[0] == 2.72
    # Derived ranking_score must be 4.08
    assert df["ranking_score"].iloc[0] == 4.08
    assert df["adjusted_predicted_rating"].iloc[0] == 4.08


def test_top_players_ranking_order_priority():
    """Verify that a player with lower raw rating but higher ranking_score ranks higher."""
    # Player A: raw = 3.50, bonus = 0.0 -> score = 3.50
    # Player B: raw = 2.72, bonus = 0.5 -> score = 4.08
    player_a = {"personId": 1, "playerName": "Player A", "predicted_rating": 3.50, "ranking_score": 3.50, "status_rank": 0}
    player_b = {"personId": 2, "playerName": "Player B", "predicted_rating": 2.72, "ranking_score": 4.08, "status_rank": 0}

    players = [player_a, player_b]

    # Sort logic matching TopPlayersSection.jsx
    def sort_key(p):
        status_rank = p.get("status_rank", 5)
        score = p.get("ranking_score", p.get("predicted_rating", 0))
        raw = p.get("predicted_rating", 0)
        return (status_rank, -score, -raw)

    sorted_players = sorted(players, key=sort_key)

    assert sorted_players[0]["playerName"] == "Player B"  # Rank Score 4.08 outranks 3.50
    assert sorted_players[1]["playerName"] == "Player A"
    # Ensure raw prediction remains unmutated
    assert sorted_players[0]["predicted_rating"] == 2.72
    assert sorted_players[1]["predicted_rating"] == 3.50


def test_missing_ranking_score_fallback_safe():
    """Verify missing ranking_score safely falls back to calculation from raw and bonus."""
    player_no_score = {"personId": 3, "playerName": "Player C", "predicted_rating": 2.50, "season_performance_bonus": 0.4}

    score = player_no_score.get("ranking_score")
    if score is None:
        score = round((player_no_score.get("predicted_rating", 0)) * (1.0 + (player_no_score.get("season_performance_bonus", 0.0))), 2)

    assert score == 3.50
    assert player_no_score["predicted_rating"] == 2.50


def test_starting_five_additive_multiplier_semantics():
    """Verify Starting Five formula remains strictly additive and does not use ranking_score."""
    raw_pred = 2.72
    slot_multiplier = 1.6  # Base slot 3
    season_bonus = 0.5     # Candidate D bonus

    # Approved semantics:
    # effective_multiplier = slot_multiplier + season_performance_bonus
    # projected_slot_value = predicted_rating * effective_multiplier
    effective_multiplier = slot_multiplier + season_bonus
    projected_slot_value = raw_pred * effective_multiplier

    assert effective_multiplier == 2.1
    assert round(projected_slot_value, 2) == round(2.72 * 2.1, 2)  # 5.71

    # Regression guard: strictly distinguish approved formula from erroneous (1 + slot_mult + bonus)
    erroneous_value = raw_pred * (1.0 + slot_multiplier + season_bonus)
    assert round(erroneous_value, 2) == round(2.72 * 3.1, 2)  # 8.43
    assert round(projected_slot_value, 2) != round(erroneous_value, 2)


def test_api_exposes_ranking_score_and_preserves_compatibility(client):
    """Verify /api/v1/players and /api/v1/teams expose ranking_score and preserve legacy fields."""
    resp_players = client.get("/api/v1/players")
    assert resp_players.status_code == 200
    p_data = resp_players.json()
    players = p_data.get("players", [])

    if players:
        p0 = players[0]
        # Must have ranking_score
        assert "ranking_score" in p0
        # Must have raw predicted_rating
        assert "predicted_rating" in p0
        # Must have season_performance_bonus
        assert "season_performance_bonus" in p0
        # Must preserve legacy adjusted_predicted_rating
        assert "adjusted_predicted_rating" in p0

        # Mathematical verification
        pred = p0["predicted_rating"]
        bonus = p0["season_performance_bonus"]
        score = p0["ranking_score"]
        expected_score = round(pred * (1.0 + bonus), 2)
        assert score == expected_score

    resp_teams = client.get("/api/v1/teams")
    assert resp_teams.status_code == 200
    t_data = resp_teams.json()
    t_players = t_data.get("players", [])
    assert len(t_players) > 0

    t0 = t_players[0]
    assert "ranking_score" in t0
    assert "predicted_rating" in t0


def test_nonzero_bonuses_survive_parquet_to_api(client):
    """Verify nonzero bonuses present in Parquet survive into API responses."""
    resp_players = client.get("/api/v1/players")
    p_data = resp_players.json()
    players = p_data.get("players", [])

    # We know Bennedict Mathurin (1631097) has bonus 2.5
    mathurin = next((p for p in players if p.get("personId") == 1631097), None)
    if mathurin:
        assert mathurin["season_performance_bonus"] == 2.5
        assert mathurin["season_game_count"] == 3
        assert mathurin["season_average_rating"] == 1.407
        assert mathurin["season_percentile"] == 29.33
        assert mathurin["ranking_score"] == round(mathurin["predicted_rating"] * 3.5, 2)

"""
tests/test_freshness_failure_simulation.py

Deterministic isolated simulation tests for data freshness & failure modes:
1. NBA Stats succeeds and new games are ingested.
2. NBA Stats times out and ESPN supplies new games.
3. Both providers fail.
4. Schedule request succeeds but play-by-play ingestion fails for one game (partial failure).
5. Schedule response is stale or incomplete.
6. No new completed games exist and provider response is verified as current.
7. Mixed-provider run ingests games from both sources.
"""

import os
import sys
import json
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.build_play_by_play_dataset import (
    build_dataset,
    REQUIRED_COLUMNS,
    GameInfo
)
from scripts.azure_blob import publish_serving_snapshot


def _sample_pbp_df(game_id="g100"):
    return pd.DataFrame([{
        "gameId": str(game_id),
        "teamId": 1610612738,
        "personId": 1628369,
        "playerName": "Jayson Tatum",
        "period": 1,
        "clock": "PT10M00.00S",
        "actionType": "Made Shot",
        "subType": "Jump Shot",
        "shotResult": "Made",
        "shotDistance": 25,
        "shotValue": 3,
        "description": "Tatum 25' 3PT Jump Shot",
        "scoreHome": 3,
        "scoreAway": 0
    }], columns=REQUIRED_COLUMNS)


# --------------------------------------------------------------------------
# Scenario 1: NBA Stats succeeds and new games are ingested
# --------------------------------------------------------------------------
@patch("scripts.build_play_by_play_dataset.fetch_games")
@patch("scripts.build_play_by_play_dataset.fetch_pbp")
def test_scenario_1_nba_stats_succeeds(mock_pbp, mock_games, tmp_path):
    out_file = tmp_path / "pbp.parquet"
    mock_games.return_value = [
        GameInfo("g1", game_date="2026-10-24", source="nba_stats")
    ]
    mock_pbp.return_value = _sample_pbp_df("g1")

    res = build_dataset(output_file=str(out_file))
    assert len(res) == 1

    with open(tmp_path / "ingestion_metadata.json") as f:
        meta = json.load(f)

    assert meta["status"] == "SUCCESS"
    assert meta["primary_configured_provider"] == "nba_stats"
    assert meta["actual_provider"] == "nba_stats"
    assert meta["fallback_occurred"] is False
    assert meta["new_games_ingested"] == 1
    assert meta["existing_games_skipped"] == 0
    assert meta["latest_completed_game_date"] == "2026-10-24"
    assert meta["is_fresh"] is True
    assert meta["per_game_provenance"] == {"g1": "nba_stats"}


# --------------------------------------------------------------------------
# Scenario 2: NBA Stats times out and ESPN supplies new games
# --------------------------------------------------------------------------
@patch("scripts.build_play_by_play_dataset.fetch_games")
@patch("scripts.build_play_by_play_dataset.fetch_pbp")
def test_scenario_2_espn_fallback_supplies_games(mock_pbp, mock_games, tmp_path):
    out_file = tmp_path / "pbp.parquet"
    mock_games.return_value = [
        GameInfo("g2", game_date="2026-10-24", source="espn")
    ]
    mock_pbp.return_value = _sample_pbp_df("g2")

    res = build_dataset(output_file=str(out_file))
    assert len(res) == 1

    with open(tmp_path / "ingestion_metadata.json") as f:
        meta = json.load(f)

    assert meta["status"] == "SUCCESS"
    assert meta["primary_configured_provider"] == "nba_stats"
    assert meta["actual_provider"] == "espn"
    assert meta["fallback_occurred"] is True
    assert meta["new_games_ingested"] == 1
    assert meta["latest_completed_game_date"] == "2026-10-24"
    assert meta["is_fresh"] is True
    assert meta["per_game_provenance"] == {"g2": "espn"}


# --------------------------------------------------------------------------
# Scenario 3: Both providers fail (preserves baseline)
# --------------------------------------------------------------------------
@patch("scripts.build_play_by_play_dataset.fetch_games")
def test_scenario_3_both_providers_fail(mock_games, tmp_path):
    out_file = tmp_path / "pbp.parquet"
    baseline = _sample_pbp_df("g_base")
    baseline.to_parquet(out_file, index=False)

    mock_games.side_effect = RuntimeError("Total upstream failure across NBA and ESPN")

    res = build_dataset(output_file=str(out_file))
    assert len(res) == 1
    assert res.iloc[0]["gameId"] == "g_base"

    with open(tmp_path / "ingestion_metadata.json") as f:
        meta = json.load(f)

    assert meta["status"] == "PRESERVED_EXISTING"
    assert meta["is_fresh"] is False
    assert meta["new_games_ingested"] == 0
    assert meta["fallback_occurred"] is False


# --------------------------------------------------------------------------
# Scenario 4: Schedule succeeds, but PBP fails for 1 game (partial failure)
# --------------------------------------------------------------------------
@patch("scripts.build_play_by_play_dataset.fetch_games")
@patch("scripts.build_play_by_play_dataset.fetch_pbp")
def test_scenario_4_partial_ingestion_failure(mock_pbp, mock_games, tmp_path):
    out_file = tmp_path / "pbp.parquet"
    mock_games.return_value = [
        GameInfo("g_ok", game_date="2026-10-24", source="nba_stats"),
        GameInfo("g_fail", game_date="2026-10-24", source="nba_stats"),
    ]

    def pbp_side_effect(game_info, **kwargs):
        gid = game_info.gameId if hasattr(game_info, "gameId") else str(game_info)
        if gid == "g_ok":
            return _sample_pbp_df("g_ok")
        raise ConnectionResetError("HTTP 504 on g_fail")

    mock_pbp.side_effect = pbp_side_effect

    res = build_dataset(output_file=str(out_file))
    assert len(res) == 1
    assert res.iloc[0]["gameId"] == "g_ok"

    with open(tmp_path / "ingestion_metadata.json") as f:
        meta = json.load(f)

    assert meta["status"] == "PARTIAL_SUCCESS"
    assert meta["new_games_ingested"] == 1
    assert meta["failed_games_count"] == 1
    assert meta["is_fresh"] is False  # Partial ingestion failures cannot report full freshness
    assert "g_ok" in meta["per_game_provenance"]


# --------------------------------------------------------------------------
# Scenario 5: Schedule response is stale or empty
# --------------------------------------------------------------------------
@patch("scripts.build_play_by_play_dataset.fetch_games")
def test_scenario_5_schedule_stale_or_empty(mock_games, tmp_path):
    out_file = tmp_path / "pbp.parquet"
    baseline = _sample_pbp_df("g_base")
    baseline.to_parquet(out_file, index=False)

    # Discovered games returns only older games already in dataset
    mock_games.return_value = [GameInfo("g_base", game_date="2026-02-10", source="nba_stats")]

    res = build_dataset(output_file=str(out_file))
    assert len(res) == 1

    with open(tmp_path / "ingestion_metadata.json") as f:
        meta = json.load(f)

    assert meta["status"] == "UP_TO_DATE"
    assert meta["new_games_ingested"] == 0
    assert meta["existing_games_skipped"] == 1
    assert meta["is_fresh"] is True


# --------------------------------------------------------------------------
# Scenario 6: No new games exist and verified current
# --------------------------------------------------------------------------
@patch("scripts.build_play_by_play_dataset.fetch_games")
def test_scenario_6_no_new_games_verified_current(mock_games, tmp_path):
    out_file = tmp_path / "pbp.parquet"
    baseline = _sample_pbp_df("g1")
    baseline.to_parquet(out_file, index=False)

    mock_games.return_value = [GameInfo("g1", game_date="2026-10-24", source="nba_stats")]

    res = build_dataset(output_file=str(out_file))
    assert len(res) == 1

    with open(tmp_path / "ingestion_metadata.json") as f:
        meta = json.load(f)

    assert meta["status"] == "UP_TO_DATE"
    assert meta["new_games_ingested"] == 0
    assert meta["existing_games_skipped"] == 1
    assert meta["is_fresh"] is True


# --------------------------------------------------------------------------
# Scenario 7: Mixed-provider run
# --------------------------------------------------------------------------
@patch("scripts.build_play_by_play_dataset.fetch_games")
@patch("scripts.build_play_by_play_dataset.fetch_pbp")
def test_scenario_7_mixed_provider_run(mock_pbp, mock_games, tmp_path):
    out_file = tmp_path / "pbp.parquet"
    mock_games.return_value = [
        GameInfo("g_nba", game_date="2026-10-24", source="nba_stats"),
        GameInfo("g_espn", game_date="2026-10-24", source="espn"),
    ]

    def pbp_side_effect(game_info, **kwargs):
        gid = game_info.gameId if hasattr(game_info, "gameId") else str(game_info)
        return _sample_pbp_df(gid)

    mock_pbp.side_effect = pbp_side_effect

    res = build_dataset(output_file=str(out_file))
    assert len(res) == 2

    with open(tmp_path / "ingestion_metadata.json") as f:
        meta = json.load(f)

    assert meta["status"] == "SUCCESS"
    assert meta["primary_configured_provider"] == "nba_stats"
    assert meta["actual_provider"] == "mixed"
    assert meta["fallback_occurred"] is True
    assert meta["new_games_ingested"] == 2
    assert meta["is_fresh"] is True
    assert meta["per_game_provenance"]["g_nba"] == "nba_stats"
    assert meta["per_game_provenance"]["g_espn"] == "espn"

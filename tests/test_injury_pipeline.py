"""
tests/test_injury_pipeline.py

Unit and integration tests for NBA injury/availability pipeline.
All tests are hermetic and mock external sources.
"""

import os
import pytest
import pandas as pd
from datetime import datetime
from unittest.mock import patch, MagicMock

from scripts.injury_provider import (
    CanonicalStatus,
    CanonicalInjuryRecord,
    InjuryReportResult,
    PlayerIdentityResolver,
    InjuryProvider,
    normalize_status_string,
)
from scripts.apply_injury_status import apply_injury_status


# ==============================
# 1. STATUS NORMALIZATION TESTS
# ==============================

def test_status_normalization_out_variants():
    assert normalize_status_string("Out") == CanonicalStatus.OUT
    assert normalize_status_string("Out - Injury/Illness") == CanonicalStatus.OUT
    assert normalize_status_string("G League - Two-Way") == CanonicalStatus.OUT
    assert normalize_status_string("G League - On Assignment") == CanonicalStatus.OUT
    assert normalize_status_string("Suspended") == CanonicalStatus.OUT
    assert normalize_status_string("Not With Team") == CanonicalStatus.OUT


def test_status_normalization_uncertain_variants():
    assert normalize_status_string("Questionable") == CanonicalStatus.QUESTIONABLE
    assert normalize_status_string("Questionable - Left Ankle Sprain") == CanonicalStatus.QUESTIONABLE
    assert normalize_status_string("Doubtful") == CanonicalStatus.DOUBTFUL
    assert normalize_status_string("Probable") == CanonicalStatus.PROBABLE


def test_status_normalization_available_and_unknown():
    assert normalize_status_string("Available") == CanonicalStatus.ACTIVE
    assert normalize_status_string("Active") == CanonicalStatus.ACTIVE
    assert normalize_status_string(None) == CanonicalStatus.UNKNOWN
    assert normalize_status_string("") == CanonicalStatus.UNKNOWN
    assert normalize_status_string("SomethingRandom") == CanonicalStatus.UNKNOWN


# ==============================
# 2. IDENTITY RESOLUTION TESTS
# ==============================

def test_identity_resolver_full_name_parsing():
    resolver = PlayerIdentityResolver()
    assert resolver.parse_raw_name("Jackson, Quenton") == "Quenton Jackson"
    assert resolver.parse_raw_name("Luka Doncic") == "Luka Doncic"
    assert resolver.parse_raw_name("") == ""
    assert resolver.parse_raw_name(None) == ""


def test_identity_resolver_surname_collision_prevention():
    """
    CRITICAL TEST: Ensure players sharing a surname (Williams)
    have distinct personIds and DO NOT collide.
    """
    resolver = PlayerIdentityResolver()

    # Known distinct NBA players with surname Williams
    id_jalen, name_jalen, _ = resolver.resolve_player("Williams, Jalen")
    id_jaylin, name_jaylin, _ = resolver.resolve_player("Williams, Jaylin")
    id_patrick, name_patrick, _ = resolver.resolve_player("Williams, Patrick")
    id_grant, name_grant, _ = resolver.resolve_player("Williams, Grant")

    assert id_jalen is not None
    assert id_jaylin is not None
    assert id_patrick is not None
    assert id_grant is not None

    # All must be distinct IDs
    williams_ids = {id_jalen, id_jaylin, id_patrick, id_grant}
    assert len(williams_ids) == 4, "Surname collision detected: Williams players must have unique IDs!"


def test_identity_resolver_suffix_handling():
    resolver = PlayerIdentityResolver()
    # Jimmy Butler III in static db vs "Butler, Jimmy"
    pid, name, amb = resolver.resolve_player("Butler, Jimmy")
    assert pid is not None
    assert "Jimmy Butler" in name


def test_identity_resolver_ambiguity_no_guessing():
    """
    When a mock resolver encounters multiple active candidates,
    it must return None and flag is_ambiguous=True rather than guessing.
    """
    resolver = PlayerIdentityResolver()
    # Inject an ambiguous player name with 2 active candidates
    resolver._name_to_players["mock ambiguous"] = [
        {"id": 999001, "full_name": "Mock Ambiguous", "is_active": True},
        {"id": 999002, "full_name": "Mock Ambiguous", "is_active": True},
    ]

    pid, name, is_amb = resolver.resolve_player("Ambiguous, Mock")
    assert pid is None, "Must not arbitrarily guess between multiple active candidates!"
    assert is_amb is True


# ==============================
# 3. INJURY PROVIDER TESTS
# ==============================

def test_provider_success_normalization():
    """A valid external response is correctly converted to CanonicalInjuryRecords."""
    mock_df = pd.DataFrame([
        {
            "Game Date": "03/20/2024",
            "Game Time": "07:00 (ET)",
            "Matchup": "IND@DET",
            "Team": "Indiana Pacers",
            "Player Name": "Jackson, Quenton",
            "Current Status": "Questionable",
            "Reason": "G League - Two-Way",
        },
        {
            "Game Date": "03/20/2024",
            "Game Time": "07:00 (ET)",
            "Matchup": "IND@DET",
            "Team": "Indiana Pacers",
            "Player Name": "Mathurin, Bennedict",
            "Current Status": "Out",
            "Reason": "Injury/Illness - Right Shoulder; Labral tear",
        }
    ])

    provider = InjuryProvider()

    with patch.object(provider, "_fetch_raw_report_df", return_value=mock_df):
        result = provider.get_injury_report(target_date=datetime(2024, 3, 20))

    assert result.success is True
    assert len(result.records) == 2
    assert result.error_message is None

    # Check first record
    rec0 = result.records[0]
    assert rec0.status == CanonicalStatus.QUESTIONABLE
    assert rec0.personId is not None
    assert rec0.teamName == "Indiana Pacers"

    # Check second record
    rec1 = result.records[1]
    assert rec1.status == CanonicalStatus.OUT
    assert rec1.personId is not None


def test_provider_failure_returns_unknown_without_raising():
    """Provider exception must return success=False without raising an exception."""
    provider = InjuryProvider()

    with patch.object(provider, "_fetch_raw_report_df", side_effect=Exception("NBA CDN Connection Timeout")):
        result = provider.get_injury_report(target_date=datetime(2026, 9, 30))

    assert result.success is False
    assert len(result.records) == 0
    assert "NBA CDN Connection Timeout" in result.error_message
    assert result.source == "none"


# ==============================
# 4. PREDICTION MERGE INTEGRATION TESTS
# ==============================

def test_apply_injury_status_successful_merge(tmp_path):
    """
    Test safe merge with predictions by personId:
    - Listed players get their reported status.
    - Unlisted players get 'Active' (available).
    - Predictions are fully preserved.
    """
    input_file = tmp_path / "predicted_ratings.parquet"
    output_file = tmp_path / "predicted_ratings_with_status.parquet"

    # Mock predictions with 3 players
    pred_data = pd.DataFrame([
        {"personId": 1629029, "playerName": "Luka Doncic", "teamId": 1610612742, "predicted_rating": 6.38, "game_date": "2024-03-20"},
        {"personId": 1630552, "playerName": "Jalen Johnson", "teamId": 1610612737, "predicted_rating": 4.50, "game_date": "2024-03-20"},
        {"personId": 1631114, "playerName": "Jalen Williams", "teamId": 1610612760, "predicted_rating": 5.10, "game_date": "2024-03-20"},
    ])
    pred_data.to_parquet(input_file, index=False)

    # Mock provider reporting Jalen Johnson is OUT
    mock_provider = MagicMock()
    mock_result = InjuryReportResult(
        records=[
            CanonicalInjuryRecord(
                personId=1630552,
                playerName="Jalen Johnson",
                teamId=1610612737,
                teamName="Atlanta Hawks",
                status=CanonicalStatus.OUT,
                rawStatus="Out",
                reason="Right Ankle Sprain",
                reportDate="2024-03-20",
                source="nba_official_pdf"
            )
        ],
        success=True,
        reportDate="2024-03-20",
        source="nba_official_pdf"
    )
    mock_provider.get_injury_report.return_value = mock_result

    merged = apply_injury_status(
        input_path=str(input_file),
        output_path=str(output_file),
        provider=mock_provider,
        target_date=datetime(2024, 3, 20)
    )

    assert len(merged) == 3, "No players should be dropped or duplicated!"
    assert set(merged["personId"]) == {1629029, 1630552, 1631114}

    # Verify Jalen Johnson is OUT
    jj = merged[merged["personId"] == 1630552].iloc[0]
    assert jj["status"] == CanonicalStatus.OUT
    assert jj["injury_reason"] == "Right Ankle Sprain"
    assert jj["status_rank"] == 4

    # Verify Luka Doncic is Active (unlisted)
    ld = merged[merged["personId"] == 1629029].iloc[0]
    assert ld["status"] == CanonicalStatus.ACTIVE
    assert "Not Listed" in ld["injury_reason"]
    assert ld["status_rank"] == 0

    # Verify predictions are intact
    assert ld["predicted_rating"] == 6.38


def test_apply_injury_status_provider_failure_policy(tmp_path):
    """
    Phase 2E.5C.1 safety regression: When external provider fails:
    - InjuryProviderFailure is raised (pipeline halts).
    - Predictions are preserved (input file unchanged).
    - Output file is NOT written (no destructive overwrite).
    - The pipeline does NOT silently continue with Unknown statuses.
    """
    from scripts.apply_injury_status import InjuryProviderFailure

    input_file = tmp_path / "predicted_ratings.parquet"
    output_file = tmp_path / "predicted_ratings_with_status.parquet"

    pred_data = pd.DataFrame([
        {"personId": 1629029, "playerName": "Luka Doncic", "teamId": 1610612742, "predicted_rating": 6.38, "game_date": "2026-09-30"},
        {"personId": 1631114, "playerName": "Jalen Williams", "teamId": 1610612760, "predicted_rating": 5.10, "game_date": "2026-09-30"},
    ])
    pred_data.to_parquet(input_file, index=False)

    mock_provider = MagicMock()
    mock_result = InjuryReportResult(
        records=[],
        success=False,
        error_message="External Injury API 404",
        reportDate="2026-09-30",
        source="none"
    )
    mock_provider.get_injury_report.return_value = mock_result

    # Provider failure must raise InjuryProviderFailure
    with pytest.raises(InjuryProviderFailure, match="pipeline halted"):
        apply_injury_status(
            input_path=str(input_file),
            output_path=str(output_file),
            provider=mock_provider,
            target_date=datetime(2026, 9, 30)
        )

    # Output artifact must NOT have been written
    assert not output_file.exists(), "Provider failure must not create/overwrite output artifact"

    # Input predictions are untouched
    orig = pd.read_parquet(input_file)
    assert len(orig) == 2
    assert list(orig["personId"]) == [1629029, 1631114]


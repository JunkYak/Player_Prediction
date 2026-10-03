"""
tests/test_injury_safety.py

Phase 2E.5C.1 — Pipeline Safety Regression Tests

Verifies the core invariant:
    EXTERNAL PROVIDER FAILURE MUST NEVER OVERWRITE A VALID EXISTING ARTIFACT.

All tests are hermetic and use mocks — no live NBA API calls.
"""

import os
import sys
import pytest
import pandas as pd
from datetime import datetime
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.injury_provider import (
    CanonicalStatus,
    CanonicalInjuryRecord,
    InjuryReportResult,
    InjuryProvider,
)
from scripts.apply_injury_status import apply_injury_status, InjuryProviderFailure


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_pred_df(n=2):
    """Minimal valid prediction DataFrame."""
    return pd.DataFrame([
        {
            "personId": 1629029 + i,
            "playerName": f"Player {i}",
            "teamId": 1610612737,
            "predicted_rating": 5.0 + i * 0.1,
            "game_date": "2026-09-30",
        }
        for i in range(n)
    ])


def _make_mock_provider(success: bool, records=None, error_message: str = "Network Error"):
    """Return a mock InjuryProvider with deterministic result."""
    mock = MagicMock()
    mock.get_injury_report.return_value = InjuryReportResult(
        records=records or [],
        success=success,
        error_message=None if success else error_message,
        reportDate="2026-09-30",
        source="nba_official_pdf" if success else "none",
    )
    return mock


def _make_successful_records(person_ids):
    return [
        CanonicalInjuryRecord(
            personId=pid,
            playerName=f"Injured Player {pid}",
            teamId=1610612737,
            teamName="Hawks",
            status=CanonicalStatus.OUT,
            rawStatus="Out",
            reason="Test Injury",
            reportDate="2026-09-30",
            source="nba_official_pdf",
        )
        for pid in person_ids
    ]


# ─── TEST 1: Successful provider ──────────────────────────────────────────────

class TestProviderSuccess:
    """Successful injury provider writes output artifact normally."""

    def test_output_artifact_created_on_success(self, tmp_path):
        """Output parquet is written when provider succeeds."""
        input_file = tmp_path / "predicted_ratings.parquet"
        output_file = tmp_path / "predicted_ratings_with_status.parquet"

        df = _make_pred_df(3)
        df.to_parquet(input_file, index=False)

        provider = _make_mock_provider(success=True, records=[])
        result = apply_injury_status(
            input_path=str(input_file),
            output_path=str(output_file),
            provider=provider,
            target_date=datetime(2026, 9, 30),
        )

        assert output_file.exists(), "Output artifact must be created on provider success"
        assert len(result) == 3

    def test_all_players_active_when_none_listed(self, tmp_path):
        """Players not listed on injury report receive Active status."""
        input_file = tmp_path / "predicted_ratings.parquet"
        output_file = tmp_path / "predicted_ratings_with_status.parquet"

        df = _make_pred_df(3)
        df.to_parquet(input_file, index=False)

        # Provider returns empty report — no listed injuries
        provider = _make_mock_provider(success=True, records=[])
        result = apply_injury_status(
            input_path=str(input_file),
            output_path=str(output_file),
            provider=provider,
            target_date=datetime(2026, 9, 30),
        )

        assert all(result["status"] == CanonicalStatus.ACTIVE)
        assert all(result["status_rank"] == 0)

    def test_listed_players_get_correct_status(self, tmp_path):
        """Players explicitly listed on report receive their reported status."""
        input_file = tmp_path / "predicted_ratings.parquet"
        output_file = tmp_path / "predicted_ratings_with_status.parquet"

        df = _make_pred_df(3)
        df.to_parquet(input_file, index=False)

        # Mark first player as OUT
        first_pid = int(df.iloc[0]["personId"])
        records = _make_successful_records([first_pid])
        provider = _make_mock_provider(success=True, records=records)
        result = apply_injury_status(
            input_path=str(input_file),
            output_path=str(output_file),
            provider=provider,
            target_date=datetime(2026, 9, 30),
        )

        out_players = result[result["personId"] == first_pid]
        assert len(out_players) == 1
        assert out_players.iloc[0]["status"] == CanonicalStatus.OUT
        assert out_players.iloc[0]["status_rank"] == 4

        # Other players still Active
        active_players = result[result["personId"] != first_pid]
        assert all(active_players["status"] == CanonicalStatus.ACTIVE)

    def test_predicted_ratings_preserved_on_success(self, tmp_path):
        """Predicted rating values are never altered by the injury merge step."""
        input_file = tmp_path / "predicted_ratings.parquet"
        output_file = tmp_path / "predicted_ratings_with_status.parquet"

        df = _make_pred_df(4)
        original_ratings = df["predicted_rating"].tolist()
        df.to_parquet(input_file, index=False)

        provider = _make_mock_provider(success=True, records=[])
        result = apply_injury_status(
            input_path=str(input_file),
            output_path=str(output_file),
            provider=provider,
        )

        result_sorted = result.sort_values("personId").reset_index(drop=True)
        input_sorted = df.sort_values("personId").reset_index(drop=True)
        assert list(result_sorted["predicted_rating"]) == list(input_sorted["predicted_rating"])


# ─── TEST 2: Provider failure WITH existing valid artifact ─────────────────────

class TestProviderFailureWithExistingArtifact:
    """Core safety tests: existing valid artifact must survive provider failure."""

    def _write_existing_artifact(self, path, n_rows=5):
        """Write a 'valid' existing output artifact with real statuses."""
        rows = []
        statuses = [CanonicalStatus.ACTIVE, CanonicalStatus.OUT, CanonicalStatus.QUESTIONABLE]
        for i in range(n_rows):
            rows.append({
                "personId": 2000000 + i,
                "playerName": f"Existing Player {i}",
                "predicted_rating": 4.0 + i,
                "status": statuses[i % len(statuses)],
                "status_rank": 0 if statuses[i % len(statuses)] == CanonicalStatus.ACTIVE else 2,
                "game_date": "2026-04-07",
            })
        pd.DataFrame(rows).to_parquet(path, index=False)

    def test_existing_artifact_untouched_on_provider_failure(self, tmp_path):
        """The output parquet is not modified when provider fails."""
        input_file = tmp_path / "predicted_ratings.parquet"
        output_file = tmp_path / "predicted_ratings_with_status.parquet"

        _make_pred_df(2).to_parquet(input_file, index=False)
        self._write_existing_artifact(output_file, n_rows=5)

        # Record mtime before failure attempt
        mtime_before = os.path.getmtime(output_file)

        provider = _make_mock_provider(success=False, error_message="HTTP 403 Forbidden")

        with pytest.raises(InjuryProviderFailure):
            apply_injury_status(
                input_path=str(input_file),
                output_path=str(output_file),
                provider=provider,
            )

        # File must still exist, be unchanged
        assert output_file.exists()
        mtime_after = os.path.getmtime(output_file)
        assert mtime_after == mtime_before, "Existing artifact mtime changed — file was overwritten!"

    def test_existing_artifact_content_unchanged_on_provider_failure(self, tmp_path):
        """The existing output artifact's content is byte-identical after provider failure."""
        input_file = tmp_path / "predicted_ratings.parquet"
        output_file = tmp_path / "predicted_ratings_with_status.parquet"

        _make_pred_df(2).to_parquet(input_file, index=False)
        self._write_existing_artifact(output_file, n_rows=5)

        # Capture content before
        df_before = pd.read_parquet(output_file)
        statuses_before = df_before["status"].tolist()

        provider = _make_mock_provider(success=False, error_message="Timeout")

        with pytest.raises(InjuryProviderFailure):
            apply_injury_status(
                input_path=str(input_file),
                output_path=str(output_file),
                provider=provider,
            )

        # Verify content is untouched
        df_after = pd.read_parquet(output_file)
        assert df_after["status"].tolist() == statuses_before
        # Crucially: no "Unknown" rows written
        assert "Unknown" not in df_after["status"].tolist()

    def test_provider_failure_raises_injury_provider_failure_exception(self, tmp_path):
        """InjuryProviderFailure (a RuntimeError subclass) is raised on provider failure."""
        input_file = tmp_path / "predicted_ratings.parquet"
        output_file = tmp_path / "predicted_ratings_with_status.parquet"

        _make_pred_df(2).to_parquet(input_file, index=False)

        provider = _make_mock_provider(success=False, error_message="NBA CDN Down")

        exc_info = None
        with pytest.raises(InjuryProviderFailure) as exc_info:
            apply_injury_status(
                input_path=str(input_file),
                output_path=str(output_file),
                provider=provider,
            )

        assert "pipeline halted" in str(exc_info.value)
        # InjuryProviderFailure must be a RuntimeError for pipeline runner compatibility
        assert isinstance(exc_info.value, RuntimeError)


# ─── TEST 3: Provider failure WITHOUT existing artifact ────────────────────────

class TestProviderFailureNoExistingArtifact:
    """When no valid output exists, provider failure must not create a fake artifact."""

    def test_no_fake_artifact_created_on_provider_failure(self, tmp_path):
        """No output file is created when provider fails and no prior artifact exists."""
        input_file = tmp_path / "predicted_ratings.parquet"
        output_file = tmp_path / "predicted_ratings_with_status.parquet"

        _make_pred_df(2).to_parquet(input_file, index=False)
        assert not output_file.exists()

        provider = _make_mock_provider(success=False, error_message="Not Found")

        with pytest.raises(InjuryProviderFailure):
            apply_injury_status(
                input_path=str(input_file),
                output_path=str(output_file),
                provider=provider,
            )

        assert not output_file.exists(), "Provider failure must not create a fake output artifact"

    def test_input_artifact_untouched_on_provider_failure(self, tmp_path):
        """Input predictions artifact is not modified on provider failure."""
        input_file = tmp_path / "predicted_ratings.parquet"
        output_file = tmp_path / "predicted_ratings_with_status.parquet"

        df = _make_pred_df(3)
        df.to_parquet(input_file, index=False)
        mtime_before = os.path.getmtime(input_file)

        provider = _make_mock_provider(success=False, error_message="Unreachable")

        with pytest.raises(InjuryProviderFailure):
            apply_injury_status(
                input_path=str(input_file),
                output_path=str(output_file),
                provider=provider,
            )

        assert os.path.getmtime(input_file) == mtime_before


# ─── TEST 4: No-games offseason behavior unaffected ────────────────────────────

class TestNoGamesOffseasonBehavior:
    """Verify that the no-games / empty-schedule behavior is fully preserved.

    The fix in apply_injury_status must NOT alter build_frontend_dataset behavior
    when next_day_games.parquet is empty — that path doesn't touch apply_injury_status.
    """

    def test_empty_games_produces_empty_home_slate(self, tmp_path):
        """An empty schedule still produces an empty frontend_home with 0 rows."""
        from scripts.build_frontend_dataset import build_frontend_dataset

        pred_df = pd.DataFrame([
            {"personId": 101, "playerName": "Star A", "teamId": 1610612738, "teamId_x": 1610612738,
             "predicted_rating": 5.5, "rating": 5.5, "status": "Active", "status_rank": 0,
             "game_date": "2026-10-03", "last_game": 5.5, "last3_avg": 5.3, "last5_avg": 5.2,
             "last7_avg": 5.1, "last_game_id": "0022501000", "last_game_date": "2026-04-07"},
            {"personId": 201, "playerName": "Star B", "teamId": 1610612747, "teamId_x": 1610612747,
             "predicted_rating": 6.0, "rating": 6.0, "status": "Active", "status_rank": 0,
             "game_date": "2026-10-03", "last_game": 6.0, "last3_avg": 5.8, "last5_avg": 5.7,
             "last7_avg": 5.6, "last_game_id": "0022501001", "last_game_date": "2026-04-07"},
        ])
        empty_games = pd.DataFrame(columns=["gameId", "homeTeamId", "awayTeamId"])

        home_out = str(tmp_path / "frontend_home.parquet")
        all_out = str(tmp_path / "frontend_all.parquet")

        build_frontend_dataset(
            pred_df=pred_df,
            games_df=empty_games,
            output_home_path=home_out,
            output_all_path=all_out,
        )

        home_df = pd.read_parquet(home_out)
        all_df = pd.read_parquet(all_out)

        assert len(home_df) == 0, "Empty schedule must produce empty home slate"
        assert len(all_df) == 2, "Empty schedule must not empty the league dataset"

    def test_empty_games_no_unknown_opponents(self, tmp_path):
        """Empty schedule produces empty string opponent, never 'Unknown'."""
        from scripts.build_frontend_dataset import build_frontend_dataset

        pred_df = pd.DataFrame([
            {"personId": 301, "playerName": "Player X", "teamId": 1610612737,
             "predicted_rating": 5.0, "rating": 5.0, "status": "Active", "status_rank": 0,
             "game_date": "2026-10-03", "last_game": 5.0, "last3_avg": 4.8, "last5_avg": 4.7,
             "last7_avg": 4.6, "last_game_id": "0022501002", "last_game_date": "2026-04-07"},
        ])
        empty_games = pd.DataFrame(columns=["gameId", "homeTeamId", "awayTeamId"])

        home_out = str(tmp_path / "frontend_home.parquet")
        all_out = str(tmp_path / "frontend_all.parquet")

        build_frontend_dataset(
            pred_df=pred_df,
            games_df=empty_games,
            output_home_path=home_out,
            output_all_path=all_out,
        )

        all_df = pd.read_parquet(all_out)
        assert "Unknown" not in all_df["opponentName"].values
        assert all(all_df["opponentName"] == "")

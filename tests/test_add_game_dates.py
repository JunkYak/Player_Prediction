"""
Phase 2E.2A — Regression tests for scripts/add_game_dates.py reliability and safety.
Covers:
1. Normal successful date enrichment.
2. Empty API result rejection & artifact preservation.
3. API failure handling & artifact preservation.
4. Partial game ID matching failure handling.
5. Preservation of existing valid artifact when refresh fails.
6. Successful atomic replacement upon valid refresh.
"""
import os
import sys
import pytest
import tempfile
import pandas as pd
from unittest.mock import patch, MagicMock

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.add_game_dates import add_game_dates, fetch_game_dates


@pytest.fixture
def sample_ratings_df():
    """Returns a minimal valid player game ratings DataFrame."""
    return pd.DataFrame({
        "gameId": ["0022500001", "0022500001", "0022500002", "0022500002"],
        "teamId": [1610612737, 1610612738, 1610612739, 1610612740],
        "personId": [101, 102, 103, 104],
        "playerName": ["Player A", "Player B", "Player C", "Player D"],
        "rating": [5.2, 4.8, 6.1, 3.9]
    })


@pytest.fixture
def sample_game_dates_df():
    """Returns matching game date mappings for sample_ratings_df."""
    return pd.DataFrame({
        "gameId": ["0022500001", "0022500002"],
        "game_date": ["2026-03-01", "2026-03-02"]
    })


# ── Test 1: Normal successful date enrichment ────────────────────────────────

def test_add_game_dates_normal_success(sample_ratings_df, sample_game_dates_df):
    """Verifies complete, accurate date enrichment and atomic parquet creation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "ratings.parquet")
        output_path = os.path.join(tmpdir, "ratings_with_dates.parquet")

        sample_ratings_df.to_parquet(input_path, index=False)

        res_path = add_game_dates(
            input_path=input_path,
            output_path=output_path,
            game_dates_df=sample_game_dates_df
        )

        assert os.path.exists(res_path)
        result_df = pd.read_parquet(res_path)

        assert len(result_df) == 4
        assert "game_date" in result_df.columns
        assert result_df["game_date"].isna().sum() == 0
        assert set(result_df["game_date"].unique()) == {"2026-03-01", "2026-03-02"}


# ── Test 2: Empty API result ─────────────────────────────────────────────────

def test_add_game_dates_empty_api_result(sample_ratings_df):
    """Verifies that an empty game dates mapping raises an error and does not produce corrupt file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "ratings.parquet")
        output_path = os.path.join(tmpdir, "ratings_with_dates.parquet")

        sample_ratings_df.to_parquet(input_path, index=False)
        empty_dates_df = pd.DataFrame(columns=["gameId", "game_date"])

        with pytest.raises(ValueError, match="unexpectedly empty"):
            add_game_dates(
                input_path=input_path,
                output_path=output_path,
                game_dates_df=empty_dates_df
            )

        assert not os.path.exists(output_path)


# ── Test 3: API failure / exception handling ─────────────────────────────────

def test_add_game_dates_api_failure_handling(sample_ratings_df):
    """Verifies that an unhandled API network failure raises RuntimeError and leaves output intact."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "ratings.parquet")
        output_path = os.path.join(tmpdir, "ratings_with_dates.parquet")

        sample_ratings_df.to_parquet(input_path, index=False)

        # Mock LeagueGameLog to throw an API timeout/connection error
        with patch("scripts.add_game_dates.leaguegamelog.LeagueGameLog", side_effect=Exception("API Connection Timeout")):
            with pytest.raises(RuntimeError, match="unable to fetch game dates"):
                add_game_dates(
                    input_path=input_path,
                    output_path=output_path,
                    game_dates_df=None,
                    max_retries=1
                )

        assert not os.path.exists(output_path)


# ── Test 4: Partial game ID matching failure ─────────────────────────────────

def test_add_game_dates_partial_match_rejection(sample_ratings_df):
    """Verifies that if only some gameIds receive dates, the merge is rejected rather than producing NaNs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "ratings.parquet")
        output_path = os.path.join(tmpdir, "ratings_with_dates.parquet")

        sample_ratings_df.to_parquet(input_path, index=False)

        # Only provide date for game 1, leaving game 2 unmatched
        partial_dates_df = pd.DataFrame({
            "gameId": ["0022500001"],
            "game_date": ["2026-03-01"]
        })

        with pytest.raises(ValueError, match="failed to match a calendar date"):
            add_game_dates(
                input_path=input_path,
                output_path=output_path,
                game_dates_df=partial_dates_df
            )

        assert not os.path.exists(output_path)


# ── Test 5: Valid existing artifact preserved after failed refresh ───────────

def test_add_game_dates_preserves_existing_artifact_on_failure(sample_ratings_df, sample_game_dates_df):
    """Verifies that if a valid output artifact already exists, a subsequent failed refresh does NOT overwrite it."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "ratings.parquet")
        output_path = os.path.join(tmpdir, "ratings_with_dates.parquet")

        sample_ratings_df.to_parquet(input_path, index=False)

        # Step 1: Initial successful creation
        add_game_dates(
            input_path=input_path,
            output_path=output_path,
            game_dates_df=sample_game_dates_df
        )
        assert os.path.exists(output_path)
        original_mtime = os.path.getmtime(output_path)
        original_data = pd.read_parquet(output_path)

        # Step 2: Failed refresh attempt (e.g. empty dates)
        empty_dates_df = pd.DataFrame(columns=["gameId", "game_date"])
        with pytest.raises(ValueError):
            add_game_dates(
                input_path=input_path,
                output_path=output_path,
                game_dates_df=empty_dates_df
            )

        # Output artifact must still exist and be completely unchanged
        assert os.path.exists(output_path)
        retained_data = pd.read_parquet(output_path)
        pd.testing.assert_frame_equal(original_data, retained_data)


# ── Test 6: Successful replacement after valid refresh ───────────────────────

def test_add_game_dates_atomic_replacement(sample_ratings_df, sample_game_dates_df):
    """Verifies that a new valid refresh safely updates the output artifact."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "ratings.parquet")
        output_path = os.path.join(tmpdir, "ratings_with_dates.parquet")

        sample_ratings_df.to_parquet(input_path, index=False)

        # Initial write
        add_game_dates(input_path=input_path, output_path=output_path, game_dates_df=sample_game_dates_df)

        # Updated dates
        updated_dates_df = pd.DataFrame({
            "gameId": ["0022500001", "0022500002"],
            "game_date": ["2026-03-05", "2026-03-06"]
        })

        add_game_dates(input_path=input_path, output_path=output_path, game_dates_df=updated_dates_df)

        new_data = pd.read_parquet(output_path)
        assert set(new_data["game_date"].unique()) == {"2026-03-05", "2026-03-06"}

import os
import sys
import tempfile
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.build_play_by_play_dataset import (
    REQUIRED_COLUMNS,
    validate_pbp_dataset,
    merge_pbp_data,
    fetch_pbp,
    fetch_games,
    build_dataset
)


def sample_pbp_row(game_id="0022300001", person_id=203999, action_type="shot", description="Made Shot"):
    return {
        "gameId": str(game_id),
        "teamId": 1610612743,
        "personId": person_id,
        "playerName": "Nikola Jokic",
        "period": 1,
        "clock": "PT11M30.00S",
        "actionType": action_type,
        "subType": "Jump Shot",
        "shotResult": "Made",
        "shotDistance": 15,
        "shotValue": 2,
        "description": description,
        "scoreHome": "2",
        "scoreAway": "0"
    }


def create_sample_df(rows):
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


# ==========================================
# 1. DUPLICATE HANDLING TESTS
# ==========================================

def test_merge_pbp_data_deduplication():
    # Existing game 1 with 2 plays
    row1 = sample_pbp_row(game_id="0022300001", action_type="shot1", description="Shot 1")
    row2 = sample_pbp_row(game_id="0022300001", action_type="shot2", description="Shot 2")
    existing_df = create_sample_df([row1, row2])

    # Newly fetched data for game 1 (e.g. updated/re-fetched) and game 2
    row1_new = sample_pbp_row(game_id="0022300001", action_type="shot1", description="Shot 1 Updated")
    row3_new = sample_pbp_row(game_id="0022300002", action_type="shot1", description="Game 2 Shot")
    new_df = create_sample_df([row1_new, row3_new])

    merged = merge_pbp_data(existing_df, new_df)

    # Merged dataset should contain game 1 (new version) and game 2, no duplicates of game 1 old rows
    assert merged["gameId"].nunique() == 2
    game1_rows = merged[merged["gameId"] == "0022300001"]
    assert len(game1_rows) == 1
    assert game1_rows.iloc[0]["description"] == "Shot 1 Updated"


def test_merge_pbp_data_with_exact_duplicate_rows():
    row1 = sample_pbp_row(game_id="0022300001")
    # Duplicate row
    new_df = create_sample_df([row1, row1])
    merged = merge_pbp_data(None, new_df)
    assert len(merged) == 1


# ==========================================
# 2. VALIDATION TESTS
# ==========================================

def test_validation_success():
    df = create_sample_df([sample_pbp_row()])
    assert validate_pbp_dataset(df) is True


def test_validation_empty_df():
    empty_df = pd.DataFrame(columns=REQUIRED_COLUMNS)
    with pytest.raises(ValueError, match="dataset is empty"):
        validate_pbp_dataset(empty_df)


def test_validation_missing_columns():
    invalid_df = pd.DataFrame([{"gameId": "0022300001", "teamId": 123}])
    with pytest.raises(ValueError, match="missing required columns"):
        validate_pbp_dataset(invalid_df)


def test_validation_null_game_id():
    row = sample_pbp_row()
    row["gameId"] = None
    df = pd.DataFrame([row])
    with pytest.raises(ValueError, match="gameId contains only null values"):
        validate_pbp_dataset(df)


# ==========================================
# 3. RETRY BEHAVIOR TESTS
# ==========================================

@patch("scripts.build_play_by_play_dataset.playbyplayv3.PlayByPlayV3")
def test_fetch_pbp_retry_and_succeed(mock_pbp_class):
    mock_df = create_sample_df([sample_pbp_row("0022300005")])
    mock_instance_fail = MagicMock()
    mock_instance_fail.get_data_frames.side_effect = ConnectionError("API Timeout")

    mock_instance_success = MagicMock()
    mock_instance_success.get_data_frames.return_value = [mock_df]

    # Fails on attempt 1, succeeds on attempt 2
    mock_pbp_class.side_effect = [mock_instance_fail, mock_instance_success]

    result_df = fetch_pbp("0022300005", max_retries=2, base_backoff=0.01)
    assert len(result_df) == 1
    assert result_df.iloc[0]["gameId"] == "0022300005"
    assert mock_pbp_class.call_count == 2


@patch("scripts.build_play_by_play_dataset.playbyplayv3.PlayByPlayV3")
def test_fetch_pbp_exhausts_retries(mock_pbp_class):
    mock_pbp_class.side_effect = ConnectionError("NBA API Unavailable")

    with pytest.raises(ConnectionError):
        fetch_pbp("0022300006", max_retries=3, base_backoff=0.01)

    assert mock_pbp_class.call_count == 3


# ==========================================
# 4. EXISTING DATA PRESERVATION & FAILURE REPORTING
# ==========================================

@patch("scripts.build_play_by_play_dataset.fetch_games")
@patch("scripts.build_play_by_play_dataset.fetch_pbp")
def test_existing_data_preserved_when_api_fails(mock_fetch_pbp, mock_fetch_games):
    with tempfile.TemporaryDirectory() as tmpdir:
        parquet_path = os.path.join(tmpdir, "test_pbp.parquet")

        # Create existing valid parquet
        existing_df = create_sample_df([
            sample_pbp_row("0022300010", description="Old Valid Game")
        ])
        existing_df.to_parquet(parquet_path, index=False)

        # Mock API discovering new games but fetch_pbp failing for all of them
        mock_fetch_games.return_value = ["0022300099"]
        mock_fetch_pbp.side_effect = ConnectionError("Game fetch failed")

        result = build_dataset(
            days_back=1,
            output_file=parquet_path,
            max_retries=1,
            base_backoff=0.01,
            sleep_time=0.0
        )

        # File on disk should still have the original valid game intact
        loaded_df = pd.read_parquet(parquet_path)
        assert len(loaded_df) == 1
        assert loaded_df.iloc[0]["gameId"] == "0022300010"
        assert loaded_df.iloc[0]["description"] == "Old Valid Game"


@patch("scripts.build_play_by_play_dataset.fetch_games")
@patch("scripts.build_play_by_play_dataset.fetch_pbp")
def test_partial_success_and_failure_reporting(mock_fetch_pbp, mock_fetch_games):
    with tempfile.TemporaryDirectory() as tmpdir:
        parquet_path = os.path.join(tmpdir, "test_pbp.parquet")

        mock_fetch_games.return_value = ["0022300001", "0022300002"]

        game1_df = create_sample_df([sample_pbp_row("0022300001")])

        # Game 1 succeeds, Game 2 fails
        def side_effect(gid, **kwargs):
            if str(gid) == "0022300001":
                return game1_df
            raise ConnectionError(f"HTTP 500 on game {gid}")

        mock_fetch_pbp.side_effect = side_effect

        result_df = build_dataset(
            days_back=5,
            output_file=parquet_path,
            max_retries=2,
            base_backoff=0.01,
            sleep_time=0.0
        )

        # Result contains game 1
        assert len(result_df) == 1
        assert result_df.iloc[0]["gameId"] == "0022300001"

        # Verify parquet was saved properly
        saved_df = pd.read_parquet(parquet_path)
        assert len(saved_df) == 1
        assert saved_df.iloc[0]["gameId"] == "0022300001"


# ==========================================
# 5. DATA VALIDATION PREVENTS BAD OVERWRITES
# ==========================================

@patch("scripts.build_play_by_play_dataset.fetch_games")
@patch("scripts.build_play_by_play_dataset.fetch_pbp")
def test_bad_data_does_not_overwrite_valid_parquet(mock_fetch_pbp, mock_fetch_games):
    with tempfile.TemporaryDirectory() as tmpdir:
        parquet_path = os.path.join(tmpdir, "test_pbp.parquet")

        # Create valid existing parquet
        existing_df = create_sample_df([sample_pbp_row("0022300001")])
        existing_df.to_parquet(parquet_path, index=False)

        # Mock invalid dataframe missing columns
        bad_df = pd.DataFrame([{"bad_col": 123}])
        mock_fetch_games.return_value = ["0022300002"]
        mock_fetch_pbp.return_value = bad_df

        # Building dataset should preserve existing valid data
        result_df = build_dataset(
            days_back=5,
            output_file=parquet_path,
            max_retries=1,
            base_backoff=0.01,
            sleep_time=0.0
        )

        # Verify that the file on disk remains the original valid parquet
        saved_df = pd.read_parquet(parquet_path)
        assert len(saved_df) == 1
        assert "bad_col" not in saved_df.columns
        assert saved_df.iloc[0]["gameId"] == "0022300001"


# ==========================================
# 6. FETCH GAMES RETRY & COMPATIBILITY
# ==========================================

@patch("scripts.build_play_by_play_dataset.leaguegamelog.LeagueGameLog")
def test_fetch_games_retry_and_succeed(mock_gamelog):
    mock_df = pd.DataFrame({"GAME_ID": ["0022300001", "0022300002"]})
    mock_instance_fail = MagicMock()
    mock_instance_fail.get_data_frames.side_effect = ConnectionError("Timeout")

    mock_instance_success = MagicMock()
    mock_instance_success.get_data_frames.return_value = [mock_df]

    mock_gamelog.side_effect = [mock_instance_fail, mock_instance_success]

    game_ids = fetch_games(days_back=5, max_retries=2, base_backoff=0.01)
    assert game_ids == ["0022300001", "0022300002"]
    assert mock_gamelog.call_count == 2


def test_constants_and_columns_backward_compatibility():
    import scripts.build_play_by_play_dataset as pbp_mod
    assert hasattr(pbp_mod, "DAYS_BACK")
    assert hasattr(pbp_mod, "OUTPUT_FILE")
    assert hasattr(pbp_mod, "build_dataset")
    assert hasattr(pbp_mod, "fetch_games")
    assert hasattr(pbp_mod, "fetch_pbp")
    assert "gameId" in pbp_mod.REQUIRED_COLUMNS
    assert "scoreHome" in pbp_mod.REQUIRED_COLUMNS
    assert "scoreAway" in pbp_mod.REQUIRED_COLUMNS


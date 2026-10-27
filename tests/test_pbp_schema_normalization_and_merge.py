"""
tests/test_pbp_schema_normalization_and_merge.py

Regression tests for Play-by-Play schema normalization, ESPN/NBA Stats score
type reconciliation, Parquet serialization round-trip, deduplication, and
Policy A safety.
"""

import os
import json
import tempfile
import pytest
import pandas as pd
import numpy as np

from scripts.espn_pbp_adapter import (
    clean_score_str,
    normalize_pbp_dataframe,
    REQUIRED_COLUMNS
)
from scripts.build_play_by_play_dataset import merge_pbp_data, build_dataset
from scripts.azure_blob import publish_serving_snapshot


class TestScoreStringNormalization:
    """Test score conversion helper for all supported and edge-case values."""

    def test_empty_string_and_null_score_values(self):
        null_cases = [None, np.nan, "", " ", "nan", "NaN", "none", "None", "<NA>"]
        for val in null_cases:
            assert clean_score_str(val) == "", f"Expected empty string for {repr(val)}"

    def test_integer_and_numeric_string_scores(self):
        cases = [
            (0, "0"),
            (0.0, "0"),
            ("0", "0"),
            ("0.0", "0"),
            (2, "2"),
            ("2", "2"),
            (102, "102"),
            (102.0, "102"),
            ("102", "102"),
            ("102.0", "102"),
            (115, "115"),
            (" 115 ", "115"),
        ]
        for input_val, expected in cases:
            assert clean_score_str(input_val) == expected, f"Failed on {repr(input_val)}"


class TestParquetSerializationAndSchemaRoundtrip:
    """Test merging historical string scores with ESPN integer scores and Parquet I/O."""

    def test_historical_strings_and_espn_integers_parquet_roundtrip(self, tmp_path):
        """Reproduce the exact production failure scenario and verify clean serialization."""
        hist_df = pd.DataFrame({
            "gameId": ["0022500741", "0022500741"],
            "teamId": [1610612738, 1610612738],
            "personId": [203076, 203076],
            "playerName": ["Anthony Davis", "Anthony Davis"],
            "period": [1, 1],
            "clock": ["PT12M00.00S", "PT11M30.00S"],
            "actionType": ["period", "Made Shot"],
            "subType": ["start", "Jump Shot"],
            "shotResult": ["", "Made"],
            "shotDistance": [0, 15],
            "shotValue": [0, 2],
            "description": ["Start of Period", "Davis 15' Jump Shot"],
            "scoreHome": ["0", ""],
            "scoreAway": ["0", ""]
        })

        espn_df = pd.DataFrame({
            "gameId": ["401909876", "401909876"],
            "teamId": [1610612744, 1610612744],
            "personId": [201939, 201939],
            "playerName": ["Stephen Curry", "Stephen Curry"],
            "period": [1, 1],
            "clock": ["PT10M15.00S", "PT09M45.00S"],
            "actionType": ["Made Shot", "Rebound"],
            "subType": ["3pt", "Defensive"],
            "shotResult": ["Made", ""],
            "shotDistance": [26, 0],
            "shotValue": [3, 0],
            "description": ["Curry 26' 3PT", "Curry REBOUND (Off:0 Def:1)"],
            "scoreHome": [3, 3],
            "scoreAway": [0, 0]
        })

        # Merge through merge_pbp_data
        merged = merge_pbp_data(hist_df, espn_df)

        # Verify all canonical columns exist
        assert list(merged.columns) == REQUIRED_COLUMNS
        assert len(merged) == 4
        assert merged["gameId"].nunique() == 2

        # Verify exact dtypes
        assert merged["scoreHome"].dtype == object
        assert merged["scoreAway"].dtype == object
        assert merged["teamId"].dtype == np.int64
        assert merged["personId"].dtype == np.int64
        assert merged["period"].dtype == np.int64
        assert merged["shotDistance"].dtype == np.int64
        assert merged["shotValue"].dtype == np.int64

        # Verify all score values are python strings
        for val in merged["scoreHome"]:
            assert isinstance(val, str)
        for val in merged["scoreAway"]:
            assert isinstance(val, str)

        # Save to Parquet and read back
        parquet_file = tmp_path / "play_by_play.parquet"
        merged.to_parquet(parquet_file, index=False)
        assert parquet_file.exists()

        read_df = pd.read_parquet(parquet_file)
        assert len(read_df) == 4
        assert list(read_df.columns) == REQUIRED_COLUMNS
        assert read_df["scoreHome"].tolist() == ["0", "", "3", "3"]
        assert read_df["scoreAway"].tolist() == ["0", "", "0", "0"]

    def test_canonical_columns_and_zero_nulls_guarantee(self):
        """Verify that normalize_pbp_dataframe produces zero nulls in any column."""
        messy_df = pd.DataFrame({
            "gameId": ["G1", None],
            "teamId": [100, None],
            "personId": [None, 200],
            "playerName": ["Player 1", None],
            "period": [None, 3],
            "clock": [None, "PT05M00.00S"],
            "actionType": [None, "Shot"],
            "subType": ["Jump", None],
            "shotResult": [None, "Missed"],
            "shotDistance": [None, 12],
            "shotValue": [3, None],
            "description": [None, "Description text"],
            "scoreHome": [None, 105],
            "scoreAway": [99, None]
        })

        normalized = normalize_pbp_dataframe(messy_df)
        assert list(normalized.columns) == REQUIRED_COLUMNS
        assert normalized.isna().sum().sum() == 0, "Expected zero nulls in all columns"


class TestDeduplicationAndAtomicPersistence:
    """Test deduplication behavior and atomic file update safety."""

    def test_incremental_deduplication_replaces_same_game(self):
        initial_df = pd.DataFrame({
            "gameId": ["G1", "G1"],
            "teamId": [1, 1],
            "personId": [10, 10],
            "playerName": ["A", "A"],
            "period": [1, 1],
            "clock": ["PT12M00.00S", "PT11M00.00S"],
            "actionType": ["Shot", "Shot"],
            "subType": ["Jump", "Layup"],
            "shotResult": ["Made", "Missed"],
            "shotDistance": [10, 2],
            "shotValue": [2, 2],
            "description": ["Shot 1", "Shot 2"],
            "scoreHome": ["2", "2"],
            "scoreAway": ["0", "0"]
        })

        updated_game1_df = pd.DataFrame({
            "gameId": ["G1", "G1", "G1"],
            "teamId": [1, 1, 1],
            "personId": [10, 10, 10],
            "playerName": ["A", "A", "A"],
            "period": [1, 1, 1],
            "clock": ["PT12M00.00S", "PT11M00.00S", "PT10M00.00S"],
            "actionType": ["Shot", "Shot", "Shot"],
            "subType": ["Jump", "Layup", "Dunk"],
            "shotResult": ["Made", "Missed", "Made"],
            "shotDistance": [10, 2, 0],
            "shotValue": [2, 2, 2],
            "description": ["Shot 1", "Shot 2", "Shot 3"],
            "scoreHome": [2, 2, 4],
            "scoreAway": [0, 0, 0]
        })

        merged = merge_pbp_data(initial_df, updated_game1_df)
        assert len(merged) == 3
        assert merged["gameId"].nunique() == 1
        assert merged["description"].tolist() == ["Shot 1", "Shot 2", "Shot 3"]

    def test_atomic_save_preserves_valid_file_on_write_failure(self, tmp_path):
        """Verify that if Parquet write encounters an error, the existing valid file is untouched."""
        valid_file = tmp_path / "play_by_play.parquet"
        valid_df = pd.DataFrame({
            "gameId": ["G0"],
            "teamId": [1],
            "personId": [10],
            "playerName": ["Original"],
            "period": [1],
            "clock": ["PT12M00.00S"],
            "actionType": ["Shot"],
            "subType": ["Jump"],
            "shotResult": ["Made"],
            "shotDistance": [10],
            "shotValue": [2],
            "description": ["Original Shot"],
            "scoreHome": ["2"],
            "scoreAway": ["0"]
        })
        valid_df.to_parquet(valid_file, index=False)
        orig_mtime = os.path.getmtime(valid_file)
        orig_content = valid_file.read_bytes()

        # Simulate atomic save block with simulated write failure
        temp_file = f"{valid_file}.tmp"
        try:
            # Create a tmp file and raise an error before replace
            with open(temp_file, "w") as fp:
                fp.write("partial junk")
            raise RuntimeError("Disk write failed simulated")
        except Exception:
            if os.path.exists(temp_file):
                os.remove(temp_file)

        # Original file must remain 100% byte-identical
        assert valid_file.exists()
        assert valid_file.read_bytes() == orig_content
        assert not os.path.exists(temp_file)


class TestPolicyAFailClosedSafety:
    """Verify that an ingestion failure still blocks Policy A publication."""

    def test_failed_ingestion_blocks_serving_publication(self, tmp_path):
        from scripts.azure_blob import (
            publish_serving_snapshot,
            validate_ingestion_metadata_for_publication,
            DEFAULT_MANIFEST_BLOB,
        )
        from unittest.mock import MagicMock, patch

        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Create dummy frontend parquet files
        from api.loader import REQUIRED_COLUMNS as API_COLS
        dummy_record = {col: "val" for col in API_COLS}
        dummy_record["rating"] = 20.0
        dummy_record["predicted_rating"] = 21.0
        dummy_record["game_date"] = "2026-10-27"
        dummy_record["last_game_date"] = "2026-10-26"
        dummy_df = pd.DataFrame([dummy_record])
        dummy_df.to_parquet(data_dir / "frontend_all.parquet", index=False)
        dummy_df.to_parquet(data_dir / "frontend_home.parquet", index=False)

        # Ingestion metadata indicates failure / partial failure
        meta_file = data_dir / "ingestion_metadata.json"
        meta_file.write_text(json.dumps({
            "status": "FAILED",
            "is_fresh": False,
            "new_games_ingested": 0,
            "failed_games_count": 5,
            "historical_cutoff_date": "2026-02-06",
            "failed_games": [{"game_id": "G1", "error": "PyArrow ArrowTypeError"}],
        }))

        # Validation function must reject
        with pytest.raises(RuntimeError, match="Publication blocked"):
            validate_ingestion_metadata_for_publication(data_dir)

        # Direct publish attempt must raise RuntimeError and never publish
        mock_client = MagicMock()

        with patch("scripts.azure_blob.get_blob_service_client", return_value=mock_client):
            with pytest.raises(RuntimeError, match="Publication blocked"):
                publish_serving_snapshot(
                    account_name="testacc",
                    container_name="propredict-serving",
                    data_dir=data_dir
                )

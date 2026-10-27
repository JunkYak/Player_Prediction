"""
scripts/verify_baseline.py

Read-only baseline verification tool for play_by_play.parquet.
Verifies file integrity, Parquet schema, required columns, row count,
game count, gameId validity, and checks for duplicates or anomalies.

Usage:
    python scripts/verify_baseline.py [--path data/play_by_play.parquet]
"""

import os
import sys
import hashlib
import argparse
from typing import Dict, Any

import pandas as pd
import pyarrow.parquet as pq

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from scripts.build_play_by_play_dataset import REQUIRED_COLUMNS


def calculate_sha256(filepath: str) -> str:
    """Calculate SHA-256 checksum of a file in streaming chunks."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def verify_baseline(filepath: str) -> Dict[str, Any]:
    """Inspect and validate historical play-by-play Parquet file without modifying it."""
    results = {
        "file_path": filepath,
        "exists": False,
        "size_bytes": 0,
        "size_mb": 0.0,
        "sha256": None,
        "parquet_readable": False,
        "num_rows": 0,
        "num_columns": 0,
        "columns": [],
        "missing_required_columns": [],
        "unique_games": 0,
        "null_game_ids": 0,
        "invalid_game_ids": 0,
        "duplicate_rows": 0,
        "is_valid": False,
        "errors": [],
    }

    if not os.path.exists(filepath):
        results["errors"].append(f"File not found: {filepath}")
        return results

    results["exists"] = True
    size_bytes = os.path.getsize(filepath)
    results["size_bytes"] = size_bytes
    results["size_mb"] = round(size_bytes / (1024 * 1024), 2)

    if size_bytes == 0:
        results["errors"].append("File is empty (0 bytes)")
        return results

    # Compute SHA-256
    try:
        results["sha256"] = calculate_sha256(filepath)
    except Exception as e:
        results["errors"].append(f"Failed to calculate SHA-256 checksum: {e}")
        return results

    # Verify Parquet readability via PyArrow
    try:
        parquet_file = pq.ParquetFile(filepath)
        results["parquet_readable"] = True
        schema = parquet_file.schema.names
        results["columns"] = list(schema)
        results["num_columns"] = len(schema)
    except Exception as e:
        results["errors"].append(f"PyArrow failed to read Parquet metadata: {e}")
        return results

    # Check required columns
    missing = [col for col in REQUIRED_COLUMNS if col not in schema]
    results["missing_required_columns"] = missing
    if missing:
        results["errors"].append(f"Missing required columns: {missing}")

    # Read data into DataFrame for record-level verification
    try:
        df = pd.read_parquet(filepath)
    except Exception as e:
        results["errors"].append(f"Failed to load DataFrame from Parquet: {e}")
        return results

    results["num_rows"] = len(df)
    if len(df) == 0:
        results["errors"].append("Parquet dataset contains 0 rows")
        return results

    # Game ID validation
    if "gameId" in df.columns:
        null_games = int(df["gameId"].isna().sum())
        results["null_game_ids"] = null_games
        if null_games > 0:
            results["errors"].append(f"Found {null_games} null gameId records")

        valid_games_series = df["gameId"].dropna().astype(str)
        # NBA game IDs are 10 digits (e.g. 0022300001 or 0022400001)
        invalid_format = valid_games_series[~valid_games_series.str.match(r"^\d{10}$")]
        results["invalid_game_ids"] = len(invalid_format)
        if len(invalid_format) > 0:
            results["errors"].append(f"Found {len(invalid_format)} gameId values with non-standard format")

        results["unique_games"] = int(valid_games_series.nunique())
    else:
        results["errors"].append("Column 'gameId' not present in dataset")

    # Check duplicate rows
    results["duplicate_rows"] = int(df.duplicated().sum())

    # Final validity determination
    results["is_valid"] = len(results["errors"]) == 0
    return results


def print_report(results: Dict[str, Any]) -> None:
    print("=" * 70)
    print("      NBA PLAY-BY-PLAY BASELINE VERIFICATION REPORT")
    print("=" * 70)
    print(f"File Path:                {results['file_path']}")
    print(f"File Exists:              {results['exists']}")
    print(f"File Size:                {results['size_bytes']:,} bytes ({results['size_mb']} MB)")
    print(f"SHA-256 Checksum:         {results['sha256']}")
    print(f"Parquet Readability:      {'PASS' if results['parquet_readable'] else 'FAIL'}")
    print(f"Total Rows:               {results['num_rows']:,}")
    print(f"Total Columns:            {results['num_columns']}")
    print(f"Unique Game IDs:          {results['unique_games']:,}")
    print(f"Null Game IDs:            {results['null_game_ids']}")
    print(f"Invalid Game ID Format:   {results['invalid_game_ids']}")
    print(f"Duplicate Rows:           {results['duplicate_rows']:,}")
    print(f"Missing Required Cols:    {results['missing_required_columns'] or 'None (All Present)'}")
    print("-" * 70)
    if results["is_valid"]:
        print("OVERALL STATUS:           VALID BASELINE (Ready for seeding)")
    else:
        print("OVERALL STATUS:           INVALID / ANOMALIES DETECTED")
        print("Errors / Warnings:")
        for err in results["errors"]:
            print(f"  - {err}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Verify local play_by_play.parquet baseline.")
    parser.add_argument(
        "--path",
        default=os.path.join(BASE_DIR, "data", "play_by_play.parquet"),
        help="Path to play_by_play.parquet file",
    )
    args = parser.parse_args()

    results = verify_baseline(args.path)
    print_report(results)
    sys.exit(0 if results["is_valid"] else 1)


if __name__ == "__main__":
    main()

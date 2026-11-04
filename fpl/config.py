"""Paths and constants shared across the FPL package."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Root for every FPL artefact on disk (raw downloads, API cache, snapshots,
#: backtest output). Override with the ``FPL_DATA_DIR`` environment variable.
DATA_DIR = Path(os.environ.get("FPL_DATA_DIR", REPO_ROOT / "data" / "fpl"))

ARCHIVE_DIR = DATA_DIR / "raw" / "archive"
API_CACHE_DIR = DATA_DIR / "raw" / "api_cache"
SNAPSHOT_DIR = DATA_DIR / "snapshots"
BACKTEST_DIR = DATA_DIR / "backtest"

#: Season currently being played. The vaastav archive lags the live game, so
#: this season is rebuilt from the live FPL API instead.
CURRENT_SEASON = "2026-27"

#: Seasons available in github.com/vaastav/Fantasy-Premier-League.
ARCHIVE_SEASONS = (
    "2016-17", "2017-18", "2018-19", "2019-20", "2020-21", "2021-22",
    "2022-23", "2023-24", "2024-25", "2025-26", "2026-27",
)

#: First season with expected_goals / expected_assists in the archive.
FIRST_XG_SEASON = "2022-23"
#: First season with defensive_contribution / CBIT columns.
FIRST_DEFCON_SEASON = "2025-26"

POSITIONS = ("GKP", "DEF", "MID", "FWD")
ELEMENT_TYPE_TO_POSITION = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

#: Valid starting-XI shape (min, max) per position, plus squad club cap.
XI_POSITION_LIMITS = {"GKP": (1, 1), "DEF": (3, 5), "MID": (2, 5), "FWD": (1, 3)}
XI_SIZE = 11
MAX_PER_CLUB = 3


def season_start_year(season: str) -> int:
    """'2025-26' -> 2025."""
    return int(season.split("-")[0])

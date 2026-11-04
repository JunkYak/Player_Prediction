"""Historical per-fixture player data from github.com/vaastav/Fantasy-Premier-League.

Each season's ``gws/merged_gw.csv`` has one row per player per fixture. Column
sets drift between seasons (xG from 2022-23, ``mng_*`` assistant-manager
columns in 2024-25, CBIT / defensive_contribution from 2025-26), so everything
is normalised to :data:`CANONICAL_COLUMNS` here.

The archive's ``xP`` column is dropped on load: it is written after the
gameweek is played, so using it as a feature or baseline would leak.
"""

from __future__ import annotations

import io
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from fpl import config

logger = logging.getLogger(__name__)

RAW_BASE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
SEASON_FILES = ("gws/merged_gw.csv", "players_raw.csv", "teams.csv", "fixtures.csv")

#: Columns removed on load because they are not known before the deadline
#: and are not outcomes we want to model either.
LEAKY_COLUMNS = ("xP",)

ID_COLUMNS = ["season", "gw", "element", "code", "name", "position", "team", "team_id",
              "opponent_team", "opponent", "fixture", "kickoff_time", "was_home"]

STAT_COLUMNS = [
    "minutes", "starts", "goals_scored", "assists", "clean_sheets", "goals_conceded",
    "own_goals", "penalties_saved", "penalties_missed", "yellow_cards", "red_cards",
    "saves", "bonus", "bps", "influence", "creativity", "threat", "ict_index",
    "expected_goals", "expected_assists", "expected_goal_involvements",
    "expected_goals_conceded", "clearances_blocks_interceptions", "recoveries",
    "tackles", "defensive_contribution", "team_h_score", "team_a_score",
    "total_points",
]

MARKET_COLUMNS = ["value", "selected", "transfers_in", "transfers_out", "transfers_balance"]

CANONICAL_COLUMNS = ID_COLUMNS + STAT_COLUMNS + MARKET_COLUMNS

_POSITION_ALIASES = {"GK": "GKP", "GKP": "GKP", "DEF": "DEF", "MID": "MID", "FWD": "FWD"}

#: Completed seasons never change; the in-progress one is refreshed after this.
CURRENT_SEASON_TTL_SECONDS = 12 * 3600


def season_dir(season: str, root: Path | None = None) -> Path:
    return (root or config.ARCHIVE_DIR) / season


def download_season(season: str, root: Path | None = None, force: bool = False,
                    session: requests.Session | None = None) -> Path:
    """Fetch the season's CSVs into ``<root>/<season>/`` (cached on disk)."""
    if season not in config.ARCHIVE_SEASONS:
        raise ValueError(f"Unknown archive season {season!r}")
    out = season_dir(season, root)
    sess = session or requests.Session()
    for rel in SEASON_FILES:
        dest = out / rel
        if dest.exists() and not force:
            fresh = season != config.CURRENT_SEASON or (
                time.time() - dest.stat().st_mtime < CURRENT_SEASON_TTL_SECONDS)
            if fresh:
                continue
        url = f"{RAW_BASE}/{season}/{rel}"
        logger.info("Downloading %s", url)
        resp = sess.get(url, timeout=60)
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)
    return out


def _read_csv(path: Path) -> pd.DataFrame:
    raw = path.read_bytes()
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode {path}")


XG_COLUMNS = ["expected_goals", "expected_assists", "expected_goal_involvements",
              "expected_goals_conceded"]


def _blank_unrecorded_xg(df: pd.DataFrame) -> pd.DataFrame:
    """Set xG columns to NaN for gameweeks where FPL didn't publish them.

    FPL started publishing xG partway through 2022-23 (GW16); the archive
    stores the earlier gameweeks as 0.0 rather than missing. A gameweek in
    which goals were scored but every player's xG is exactly zero is treated
    as unrecorded.
    """
    by_gw = df.groupby("gw")
    xg_total = by_gw["expected_goals"].transform(lambda s: s.abs().sum())
    goals_total = by_gw["goals_scored"].transform("sum")
    unrecorded = (xg_total == 0) & (goals_total > 0)
    if unrecorded.any():
        df.loc[unrecorded, XG_COLUMNS] = np.nan
    return df


def normalize_merged_gw(df: pd.DataFrame, season: str,
                        players_raw: pd.DataFrame | None = None,
                        teams: pd.DataFrame | None = None) -> pd.DataFrame:
    """Map one season's raw ``merged_gw`` frame onto :data:`CANONICAL_COLUMNS`.

    * drops :data:`LEAKY_COLUMNS` and assistant-manager rows (2024-25 ``AM``);
    * adds ``season``, ``gw``, a cross-season player ``code`` (from
      ``players_raw``), ``team_id`` and the ``opponent`` name (from ``teams``);
    * creates missing stat columns as NaN so every season has the same schema.
    """
    df = df.drop(columns=[c for c in LEAKY_COLUMNS if c in df.columns]).copy()

    df["position"] = df["position"].map(_POSITION_ALIASES)
    df = df[df["position"].notna()].copy()  # drops AM (assistant manager) rows

    df["season"] = season
    gw_col = "GW" if "GW" in df.columns else "round"
    df["gw"] = pd.to_numeric(df[gw_col], errors="coerce").astype("Int64")
    df["kickoff_time"] = pd.to_datetime(df["kickoff_time"], utc=True, errors="coerce")
    df["was_home"] = df["was_home"].astype(str).str.lower().isin(["true", "1"])

    if players_raw is not None and {"id", "code"} <= set(players_raw.columns):
        df["code"] = df["element"].map(players_raw.set_index("id")["code"])
    else:
        df["code"] = np.nan

    if teams is not None and {"id", "name"} <= set(teams.columns):
        id_to_name = teams.set_index("id")["name"]
        name_to_id = {v: k for k, v in id_to_name.items()}
        df["opponent"] = df["opponent_team"].map(id_to_name)
        if "team" not in df.columns:
            df["team"] = np.nan
        df["team_id"] = df["team"].map(name_to_id)
    else:
        df["opponent"] = np.nan
        df["team_id"] = np.nan
        if "team" not in df.columns:
            df["team"] = np.nan

    for col in STAT_COLUMNS + MARKET_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = _blank_unrecorded_xg(df)
    df = df[CANONICAL_COLUMNS].sort_values(["kickoff_time", "fixture", "element"])
    return df.reset_index(drop=True)


def load_season(season: str, root: Path | None = None, download: bool = True) -> pd.DataFrame:
    """Load and normalise one archived season (downloading it if needed)."""
    folder = season_dir(season, root)
    if download:
        download_season(season, root)
    merged = folder / "gws" / "merged_gw.csv"
    if not merged.exists():
        raise FileNotFoundError(f"{merged} missing; run download_season({season!r})")
    players = folder / "players_raw.csv"
    teams = folder / "teams.csv"
    return normalize_merged_gw(
        _read_csv(merged), season,
        players_raw=_read_csv(players) if players.exists() else None,
        teams=_read_csv(teams) if teams.exists() else None,
    )


def load_seasons(seasons, root: Path | None = None, download: bool = True) -> pd.DataFrame:
    frames = [load_season(s, root, download) for s in seasons]
    return pd.concat(frames, ignore_index=True)


def max_archived_gw(season: str, root: Path | None = None) -> int:
    df = load_season(season, root)
    return int(df["gw"].max())


if __name__ == "__main__":  # pragma: no cover - manual helper
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Download vaastav FPL archive seasons.")
    p.add_argument("seasons", nargs="*", default=["2022-23", "2023-24", "2024-25", "2025-26"])
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    for s in args.seasons:
        download_season(s, force=args.force)
        d = load_season(s, download=False)
        print(f"{s}: {len(d):>6} rows, GW {d['gw'].min()}-{d['gw'].max()}, "
              f"{d['element'].nunique()} players")

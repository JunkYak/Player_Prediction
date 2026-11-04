"""Load FPL scoring rules from ``scoring.yaml`` and score stat lines with them."""

from __future__ import annotations

import copy
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

SCORING_PATH = Path(__file__).with_name("scoring.yaml")


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


@lru_cache(maxsize=None)
def _load_raw(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_rules(season: str | None = None, path: Path | None = None) -> dict[str, Any]:
    """Rules for ``season`` (the ``default`` block plus that season's overrides)."""
    raw = _load_raw(str(path or SCORING_PATH))
    rules = raw["default"]
    if season and season in (raw.get("seasons") or {}):
        rules = _deep_merge(rules, raw["seasons"][season])
    return copy.deepcopy(rules)


def points_from_stats(df: pd.DataFrame, rules: dict[str, Any]) -> pd.Series:
    """FPL points for each row of per-fixture stats (vectorised).

    Needs ``position`` plus the archive stat columns. Missing stat columns, or
    NaN values (e.g. defensive_contribution before 2025-26), count as zero.
    """
    def col(name: str) -> pd.Series:
        if name not in df.columns:
            return pd.Series(0.0, index=df.index)
        return pd.to_numeric(df[name], errors="coerce").fillna(0.0)

    pos = df["position"]

    def by_pos(mapping: dict) -> pd.Series:
        return pos.map({k: (np.nan if v is None else v) for k, v in mapping.items()}).astype(float)

    app = rules["appearance"]
    minutes = col("minutes")
    pts = np.where(minutes >= app["long_play_minutes"], app["long_play"],
                   np.where(minutes > 0, app["short_play"], 0)).astype(float)
    pts = pd.Series(pts, index=df.index)

    pts += col("goals_scored") * by_pos(rules["goals_scored"])
    pts += col("assists") * rules["assists"]
    pts += col("clean_sheets") * by_pos(rules["clean_sheets"])
    gc = rules["goals_conceded"]
    pts += np.floor(col("goals_conceded") / gc["per"]) * by_pos(gc["points"])
    sv = rules["saves"]
    pts += np.floor(col("saves") / sv["per"]) * sv["points"]
    for name in ("penalties_saved", "penalties_missed", "yellow_cards", "red_cards",
                 "own_goals", "bonus"):
        pts += col(name) * rules[name]

    dc = rules["defensive_contribution"]
    thresh = by_pos(dc["threshold"])
    hit = (col("defensive_contribution") >= thresh) & thresh.notna()
    pts += np.where(hit, by_pos(dc["points"]).fillna(0.0), 0.0)
    return pts

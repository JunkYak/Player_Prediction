"""Synthetic archive-shaped data for the FPL tests (no network, no downloads)."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from fpl.ingest.archive import CANONICAL_COLUMNS

TEAMS = ["Alpha", "Bravo", "Charlie", "Delta"]
POSITIONS = ["GKP", "DEF", "DEF", "DEF", "MID", "MID", "MID", "FWD", "FWD"]


def make_fixtures(seasons=("2023-24", "2024-25"), n_gws=8, double_gw=4, seed=0) -> pd.DataFrame:
    """Per-fixture rows: 4 teams x 9 players. ``double_gw`` has two rounds
    of fixtures, so every player plays twice that gameweek."""
    rng = np.random.default_rng(seed)
    rows = []
    fixture_id = 0
    for s_i, season in enumerate(seasons):
        year = int(season[:4])
        for gw in range(1, n_gws + 1):
            rounds = 2 if gw == double_gw else 1
            for r in range(rounds):
                pairs = [(0, 1), (2, 3)] if (gw + r) % 2 else [(0, 2), (1, 3)]
                for h, a in pairs:
                    fixture_id += 1
                    kickoff = pd.Timestamp(f"{year}-08-01", tz="UTC") + timedelta(
                        days=7 * (gw - 1) + 3 * r)
                    hs, as_ = int(rng.poisson(1.4)), int(rng.poisson(1.1))
                    for side, team_i, opp_i, gf, ga in ((True, h, a, hs, as_), (False, a, h, as_, hs)):
                        for p_i, pos in enumerate(POSITIONS):
                            element = team_i * 100 + p_i + 1
                            minutes = int(rng.choice([0, 30, 90], p=[0.2, 0.2, 0.6]))
                            goals = int(rng.poisson(0.2)) if minutes and pos in ("MID", "FWD") else 0
                            pts = (0 if minutes == 0 else (2 if minutes >= 60 else 1)) + 5 * goals
                            rows.append({
                                "season": season, "gw": gw, "element": element,
                                "code": 10_000 + element, "name": f"P{element}",
                                "position": pos, "team": TEAMS[team_i], "team_id": team_i + 1,
                                "opponent_team": opp_i + 1, "opponent": TEAMS[opp_i],
                                "fixture": fixture_id, "kickoff_time": kickoff,
                                "was_home": side, "minutes": minutes,
                                "starts": float(minutes >= 60), "goals_scored": goals,
                                "assists": 0, "clean_sheets": float(ga == 0 and minutes >= 60),
                                "goals_conceded": ga if minutes else 0, "saves": 0,
                                "bonus": 0, "bps": 10 * goals, "yellow_cards": 0,
                                "red_cards": 0, "expected_goals": round(0.15 * goals + 0.05, 2),
                                "expected_assists": 0.05, "expected_goal_involvements": 0.1,
                                "expected_goals_conceded": 1.0,
                                "defensive_contribution": (np.nan if s_i == 0 else 5.0),
                                "total_points": pts, "value": 50 + p_i,
                                "selected": 1000 * (p_i + 1), "transfers_balance": 10 * p_i,
                                "transfers_in": 0, "transfers_out": 0,
                            })
    df = pd.DataFrame(rows)
    for c in CANONICAL_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    df["gw"] = df["gw"].astype("Int64")
    return df[CANONICAL_COLUMNS]


@pytest.fixture
def fixtures_df() -> pd.DataFrame:
    return make_fixtures()

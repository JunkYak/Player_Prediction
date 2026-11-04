import pandas as pd
import pytest

from fpl import config
from fpl.scoring import load_rules, points_from_stats


def test_season_overrides_turn_off_defensive_contribution_before_2025_26():
    assert load_rules()["defensive_contribution"]["points"]["DEF"] == 2
    assert load_rules("2025-26")["defensive_contribution"]["points"]["DEF"] == 2
    assert load_rules("2024-25")["defensive_contribution"]["points"]["DEF"] == 0
    # override must not mutate the shared default
    assert load_rules()["defensive_contribution"]["points"]["MID"] == 2


@pytest.mark.parametrize("row, expected", [
    # DEF, 90 min, clean sheet, goal, 11 CBIT (>=10) -> 2 + 4 + 6 + 2
    ({"position": "DEF", "minutes": 90, "clean_sheets": 1, "goals_scored": 1,
      "defensive_contribution": 11}, 14),
    # MID, 45 min, assist, yellow; 11 CBIRT is below the MID threshold of 12
    ({"position": "MID", "minutes": 45, "assists": 1, "yellow_cards": 1,
      "defensive_contribution": 11}, 3),
    # GKP, 90 min, 7 saves (2 pts), 3 conceded (-1), 1 bonus
    ({"position": "GKP", "minutes": 90, "saves": 7, "goals_conceded": 3, "bonus": 1}, 4),
    # FWD, 0 minutes -> nothing
    ({"position": "FWD", "minutes": 0}, 0),
    # MID, 60 min (long play boundary), clean sheet, 12 CBIRT, own goal
    ({"position": "MID", "minutes": 60, "clean_sheets": 1, "defensive_contribution": 12,
      "own_goals": 1}, 3),
    # GKP with a CBIT count never earns defensive contribution points
    ({"position": "GKP", "minutes": 90, "defensive_contribution": 30}, 2),
])
def test_points_from_stats_examples(row, expected):
    pts = points_from_stats(pd.DataFrame([row]), load_rules("2025-26"))
    assert pts.iloc[0] == expected


def test_nan_defensive_contribution_counts_as_zero():
    df = pd.DataFrame([{"position": "DEF", "minutes": 90, "defensive_contribution": float("nan")}])
    assert points_from_stats(df, load_rules()).iloc[0] == 2


@pytest.mark.parametrize("season", ["2023-24", "2024-25", "2025-26"])
def test_archive_reconstruction(season):
    """scoring.yaml reproduces the archive's total_points exactly (needs the
    season downloaded via `python -m fpl.ingest.archive`)."""
    from fpl.ingest.archive import load_season, season_dir

    if not (season_dir(season) / "gws" / "merged_gw.csv").exists():
        pytest.skip(f"{season} archive not downloaded under {config.ARCHIVE_DIR}")
    df = load_season(season, download=False)
    pts = points_from_stats(df, load_rules(season))
    assert int((pts != df["total_points"]).sum()) == 0

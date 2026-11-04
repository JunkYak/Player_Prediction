import numpy as np
import pandas as pd

from fpl.features import FEATURE_COLUMNS, aggregate_player_gw, build_feature_table

KEYS = ["season", "gw", "element"]


def test_double_gameweek_rows_are_summed(fixtures_df):
    pgw = aggregate_player_gw(fixtures_df.assign(gw=fixtures_df["gw"].astype(int)))
    assert pgw.duplicated(KEYS).sum() == 0
    dgw = pgw[pgw["gw"] == 4]
    assert (dgw["n_fixtures"] == 2).all()
    assert (pgw[pgw["gw"] != 4]["n_fixtures"] == 1).all()
    raw = fixtures_df[(fixtures_df["gw"] == 4) & (fixtures_df["element"] == 101)
                      & (fixtures_df["season"] == "2024-25")]
    row = dgw[(dgw["element"] == 101) & (dgw["season"] == "2024-25")].iloc[0]
    assert row["total_points"] == raw["total_points"].sum()
    assert row["minutes"] == raw["minutes"].sum()


def test_all_feature_columns_present(fixtures_df):
    feats = build_feature_table(fixtures_df)
    missing = [c for c in FEATURE_COLUMNS if c not in feats.columns]
    assert not missing
    assert len(feats) == len(aggregate_player_gw(fixtures_df.assign(gw=fixtures_df["gw"].astype(int))))


def test_features_only_use_earlier_gameweeks(fixtures_df):
    """Scrambling every outcome from GW g onwards must not change GW g features."""
    season, g = "2024-25", 4  # the double gameweek, the trickiest case
    base = build_feature_table(fixtures_df)

    tampered = fixtures_df.copy()
    future = (tampered["season"] == season) & (tampered["gw"] >= g)
    rng = np.random.default_rng(1)
    for col in ["total_points", "minutes", "goals_scored", "expected_goals", "bps",
                "starts", "selected", "transfers_balance", "defensive_contribution"]:
        tampered.loc[future, col] = rng.integers(0, 500, future.sum())
    after = build_feature_table(tampered)

    pick = lambda d: (d[(d["season"] == season) & (d["gw"] == g)]
                      .set_index("element")[FEATURE_COLUMNS].sort_index())
    pd.testing.assert_frame_equal(pick(base), pick(after))


def test_season_to_date_rate_matches_hand_calculation(fixtures_df):
    feats = build_feature_table(fixtures_df)
    pgw = aggregate_player_gw(fixtures_df.assign(gw=fixtures_df["gw"].astype(int)))
    p = pgw[(pgw["season"] == "2024-25") & (pgw["element"] == 305)].set_index("gw")
    expected = p.loc[1:5, "total_points"].sum() / p.loc[1:5, "n_fixtures"].sum()
    got = feats[(feats["season"] == "2024-25") & (feats["element"] == 305) & (feats["gw"] == 6)]
    assert got["pts_pf_to_date"].iloc[0] == expected
    first = feats[(feats["season"] == "2024-25") & (feats["gw"] == 1)]
    assert first["pts_pf_to_date"].isna().all()       # no history yet this season
    assert first["prev_season_pts_pf"].notna().all()  # filled from 2023-24 via code


def test_defensive_contribution_rate_is_nan_before_it_existed(fixtures_df):
    feats = build_feature_table(fixtures_df)
    assert feats.loc[feats["season"] == "2023-24", "defcon90_to_date"].isna().all()
    later = feats[(feats["season"] == "2024-25") & (feats["gw"] > 2)]
    assert later["defcon90_to_date"].notna().any()


def test_team_form_carries_over_seasons_but_starts_empty(fixtures_df):
    feats = build_feature_table(fixtures_df)
    assert feats.loc[(feats["season"] == "2023-24") & (feats["gw"] == 1),
                     "team_xg_for_l6"].isna().all()
    assert feats.loc[(feats["season"] == "2024-25") & (feats["gw"] == 1),
                     "opp_xg_against_l6"].notna().all()

import pandas as pd

from fpl.ingest.archive import CANONICAL_COLUMNS, normalize_merged_gw


def _raw():
    return pd.DataFrame([
        {"name": "Keeper", "position": "GK", "team": "Alpha", "xP": 5.5, "element": 1,
         "fixture": 10, "opponent_team": 2, "kickoff_time": "2024-08-17T14:00:00Z",
         "was_home": "True", "minutes": 90, "total_points": 6, "round": 1, "GW": 1,
         "value": 45},
        {"name": "Boss", "position": "AM", "team": "Alpha", "xP": 1.0, "element": 99,
         "fixture": 10, "opponent_team": 2, "kickoff_time": "2024-08-17T14:00:00Z",
         "was_home": "True", "minutes": 0, "total_points": 3, "round": 1, "GW": 1,
         "value": 10},
        {"name": "Striker", "position": "FWD", "team": "Bravo", "xP": 2.0, "element": 2,
         "fixture": 10, "opponent_team": 1, "kickoff_time": "2024-08-17T14:00:00Z",
         "was_home": "False", "minutes": 20, "total_points": 1, "round": 1, "GW": 1,
         "value": 60},
    ])


def test_normalize_drops_leaky_xp_and_assistant_managers():
    out = normalize_merged_gw(
        _raw(), "2024-25",
        players_raw=pd.DataFrame({"id": [1, 2, 99], "code": [111, 222, 999]}),
        teams=pd.DataFrame({"id": [1, 2], "name": ["Alpha", "Bravo"]}))
    assert "xP" not in out.columns
    assert list(out.columns) == CANONICAL_COLUMNS
    assert set(out["position"]) == {"GKP", "FWD"}            # AM dropped, GK renamed
    assert out.set_index("element")["code"].to_dict() == {1: 111, 2: 222}
    assert out.set_index("element")["opponent"].to_dict() == {1: "Bravo", 2: "Alpha"}
    assert out.set_index("element")["team_id"].to_dict() == {1: 1, 2: 2}
    assert out["was_home"].tolist() == [True, False]
    assert out["season"].unique().tolist() == ["2024-25"]
    assert out["gw"].tolist() == [1, 1]
    # columns from later seasons exist as NaN rather than being absent
    assert out["defensive_contribution"].isna().all()
    assert str(out["kickoff_time"].dt.tz) == "UTC"


def test_unrecorded_xg_gameweeks_become_nan():
    """2022-23 GW1-15 store xG as 0.0 though FPL hadn't published it yet."""
    rows = []
    for gw, xg in ((1, 0.0), (1, 0.0), (2, 0.4), (2, 0.0)):
        rows.append({"name": "x", "position": "MID", "team": "Alpha", "element": len(rows) + 1,
                     "fixture": gw, "opponent_team": 2, "kickoff_time": f"2022-08-0{gw}T14:00:00Z",
                     "was_home": True, "minutes": 90, "goals_scored": 1, "expected_goals": xg,
                     "expected_assists": xg, "total_points": 7, "GW": gw})
    out = normalize_merged_gw(pd.DataFrame(rows), "2022-23")
    assert out.loc[out["gw"] == 1, "expected_goals"].isna().all()
    assert out.loc[out["gw"] == 1, "expected_assists"].isna().all()
    # a recorded gameweek keeps genuine zeros
    assert sorted(out.loc[out["gw"] == 2, "expected_goals"]) == [0.0, 0.4]

import numpy as np
import pandas as pd
import pytest

from fpl import config, metrics
from fpl.backtest import format_table, metrics_table, run_backtest
from fpl.features import build_feature_table


# -- metrics ------------------------------------------------------------------
def _gw_pool(n_per_team=8, n_teams=6, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    positions = ["GKP", "DEF", "DEF", "DEF", "MID", "MID", "FWD", "FWD"]
    for t in range(n_teams):
        for i in range(n_per_team):
            rows.append({"season": "2025-26", "gw": 1, "team": f"T{t}",
                         "position": positions[i % len(positions)],
                         "pred": rng.normal(3, 2), "total_points": rng.integers(0, 15)})
    return pd.DataFrame(rows)


def test_pick_xi_is_a_valid_lineup():
    g = _gw_pool()
    g.loc[g["team"] == "T0", "pred"] += 50  # tempt it to stack one club
    xi = metrics.pick_xi(g, "pred")
    assert len(xi) == config.XI_SIZE
    counts = xi["position"].value_counts()
    for pos, (lo, hi) in config.XI_POSITION_LIMITS.items():
        assert lo <= counts.get(pos, 0) <= hi
    assert xi["team"].value_counts().max() <= config.MAX_PER_CLUB


def test_ranking_metrics_on_perfect_and_reversed_predictions():
    g = _gw_pool()
    g["total_points"] = np.arange(len(g))
    g["perfect"] = g["total_points"]
    g["reversed"] = -g["total_points"]
    assert metrics.spearman_by_gw(g, "perfect") == pytest.approx(1.0)
    assert metrics.spearman_by_gw(g, "reversed") == pytest.approx(-1.0)
    assert metrics.topk_precision_by_gw(g, "perfect") == 1.0
    assert metrics.topk_precision_by_gw(g, "reversed") == 0.0


def test_calibration_and_brier():
    p = np.array([1.0, 2.0, 3.0, 4.0])
    assert metrics.calibration_slope(p, 2 * p) == pytest.approx(2.0)
    assert metrics.brier(np.array([1.0, 0.0]), np.array([1, 0])) == 0.0
    tbl = metrics.calibration_table(np.arange(100.0), np.arange(100.0), bins=5)
    assert len(tbl) == 5 and (tbl["n"] == 20).all()


# -- walk-forward -------------------------------------------------------------
def test_walk_forward_never_trains_on_the_test_gameweek(fixtures_df, monkeypatch):
    from fpl import backtest

    seen = []

    class Spy(backtest.SeasonPPG):
        def fit(self, train):
            seen.append((train["season"].max(), train[train["season"] == "2024-25"]["gw"].max()))
            return super().fit(train)

    monkeypatch.setattr(backtest, "SeasonPPG", Spy)
    feats = build_feature_table(fixtures_df)
    preds = run_backtest(feats, ["2024-25"], ["season_ppg"], train_from="2023-24",
                         min_train_rows=10)
    tested = sorted(preds["gw"].unique())
    assert tested == list(range(1, 9))
    # GW1 trains on 2023-24 only; GW g trains on 2024-25 rows up to g-1
    latest_train_gw = [gw for _, gw in seen]
    assert np.isnan(latest_train_gw[0])
    assert latest_train_gw[1:] == list(range(1, 8))


def test_baseline_table_runs_end_to_end(fixtures_df):
    feats = build_feature_table(fixtures_df)
    ep = feats.loc[(feats["season"] == "2024-25") & (feats["gw"] == 3), ["season", "gw", "element"]]
    ep = ep.assign(ep_next=2.0)
    names = ["season_ppg", "form_last5", "ep_next"]
    preds = run_backtest(feats, ["2024-25"], names, train_from="2023-24", min_train_rows=10,
                         ep_hist=ep)
    assert preds[["season_ppg", "form_last5"]].notna().all().all()
    assert preds["ep_next"].notna().sum() == len(ep)        # only the snapshotted GW
    table = metrics_table(preds, names)
    row = table[(table["season"] == "2024-25") & (table["model"] == "ep_next")].iloc[0]
    assert row["gws"] == 1
    text = format_table(table)
    assert "season_ppg" in text and "2024-25" in text


def test_learned_baselines_fit_and_predict(fixtures_df):
    pytest.importorskip("lightgbm")
    feats = build_feature_table(fixtures_df)
    preds = run_backtest(feats, ["2024-25"], ["lgbm_direct", "rf_total"], train_from="2023-24",
                         min_train_rows=10, retrain_every=4)
    assert preds[["lgbm_direct", "rf_total"]].notna().all().all()
    assert np.isfinite(preds["lgbm_direct"]).all()

"""Walk-forward backtest of gameweek point predictions.

For every gameweek ``g`` of each test season, models are fit only on
player-GW rows from earlier seasons and gameweeks ``< g`` of the same season,
then predict every player listed for ``g``. Features are already leak-free
(see :mod:`fpl.features.build`).

    python -m fpl.backtest --model baselines
    python -m fpl.backtest --model season_ppg,lgbm_direct --seasons 2025-26
    python -m fpl.backtest --model baselines --seasons 2026-27   # live API data

Baselines:
    season_ppg    season-to-date points per fixture x fixtures this GW
    form_last5    last-5-GW points per fixture x fixtures this GW
    ep_next       FPL's own projection (only GWs captured by fpl.ingest.snapshot)
    lgbm_direct   LightGBM regressor on the feature table -> total points
    rf_total      RandomForest on the same features (mirrors the NBA model)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fpl import config, metrics
from fpl.features import FEATURE_COLUMNS, build_feature_table
from fpl.ingest import archive, fpl_api, snapshot

logger = logging.getLogger(__name__)

TARGET = "total_points"
#: Loaded only so the first training season gets previous-season and team-form features.
WARMUP_SEASON = "2021-22"
DEFAULT_TEST_SEASONS = ("2024-25", "2025-26")
BASELINES = ("season_ppg", "form_last5", "ep_next", "lgbm_direct", "rf_total")


# -- models -------------------------------------------------------------------
class _RateBaseline:
    """Per-fixture rate x fixtures this GW, with fallbacks for players without
    a history this season: previous-season rate, then the position mean."""

    rate_col = ""

    def fit(self, train: pd.DataFrame) -> "_RateBaseline":
        per_fix = train[TARGET] / train["n_fixtures"]
        self.pos_mean_ = per_fix.groupby(train["position"]).mean()
        self.global_mean_ = float(per_fix.mean())
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        rate = test[self.rate_col].copy()
        rate = rate.fillna(test["prev_season_pts_pf"])
        rate = rate.fillna(test["position"].map(self.pos_mean_)).fillna(self.global_mean_)
        return (rate * test["n_fixtures"]).to_numpy(float)


class SeasonPPG(_RateBaseline):
    rate_col = "pts_pf_to_date"


class FormLast5(_RateBaseline):
    rate_col = "pts_pf_last5"


class EpNext:
    """FPL's ep_next from our own pre-deadline snapshots; NaN where none exist."""

    def __init__(self, ep_hist: pd.DataFrame | None):
        self.ep = ep_hist if ep_hist is not None else pd.DataFrame(
            columns=["season", "gw", "element", "ep_next"])

    def fit(self, train: pd.DataFrame) -> "EpNext":
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        merged = test[["season", "gw", "element"]].merge(
            self.ep[["season", "gw", "element", "ep_next"]], how="left",
            on=["season", "gw", "element"])
        return merged["ep_next"].to_numpy(float)


class LGBMDirect:
    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, train: pd.DataFrame) -> "LGBMDirect":
        import lightgbm as lgb

        self.model_ = lgb.LGBMRegressor(
            n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=50,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=self.seed, verbose=-1)
        self.model_.fit(train[FEATURE_COLUMNS], train[TARGET])
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(test[FEATURE_COLUMNS])


class RFTotal:
    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, train: pd.DataFrame) -> "RFTotal":
        from sklearn.ensemble import RandomForestRegressor

        self.model_ = RandomForestRegressor(
            n_estimators=150, max_depth=12, min_samples_leaf=20, max_samples=0.3,
            max_features=0.5, n_jobs=-1, random_state=self.seed)
        self.model_.fit(train[FEATURE_COLUMNS].fillna(-1.0), train[TARGET])
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(test[FEATURE_COLUMNS].fillna(-1.0))


#: Models that are cheap to refit and have no learned parameters worth caching.
_STATELESS = {"season_ppg", "form_last5", "ep_next"}


def make_model(name: str, ep_hist: pd.DataFrame | None = None):
    return {
        "season_ppg": SeasonPPG,
        "form_last5": FormLast5,
        "ep_next": lambda: EpNext(ep_hist),
        "lgbm_direct": LGBMDirect,
        "rf_total": RFTotal,
    }[name]()


# -- data ---------------------------------------------------------------------
def _seasons_between(first: str, last: str) -> list[str]:
    return [s for s in config.ARCHIVE_SEASONS if first <= s <= last]


def load_history(last_season: str, current_source: str = "api",
                 first_season: str = WARMUP_SEASON) -> pd.DataFrame:
    """Per-fixture rows for ``first_season`` .. ``last_season``."""
    frames = []
    for s in _seasons_between(first_season, last_season):
        if s == config.CURRENT_SEASON and current_source == "api":
            frames.append(fpl_api.load_current_season())
        else:
            frames.append(archive.load_season(s))
    return pd.concat(frames, ignore_index=True)


# -- walk-forward -------------------------------------------------------------
def run_backtest(features: pd.DataFrame, test_seasons, model_names,
                 train_from: str = config.FIRST_XG_SEASON, retrain_every: int = 1,
                 ep_hist: pd.DataFrame | None = None, min_train_rows: int = 1000,
                 progress=None) -> pd.DataFrame:
    """Predictions for every player-GW in ``test_seasons``, one column per model.

    Learned models are refit every ``retrain_every`` gameweeks (always on
    only-earlier rows); between refits the last fit is reused. ``progress``
    is an optional callable receiving one status string per gameweek.
    """
    pool = features[features["season"] >= train_from]
    out = []
    for season in test_seasons:
        season_rows = pool[pool["season"] == season]
        if season_rows.empty:
            raise ValueError(f"No feature rows for season {season}")
        fitted: dict = {}
        gws = sorted(season_rows["gw"].unique())
        for i, gw in enumerate(gws):
            t_gw = time.time()
            train = pool[(pool["season"] < season) | ((pool["season"] == season) & (pool["gw"] < gw))]
            test = season_rows[season_rows["gw"] == gw]
            if len(train) < min_train_rows:
                logger.warning("%s GW%s: only %d training rows, skipping", season, gw, len(train))
                continue
            preds = test[["season", "gw", "element", "code", "name", "position", "team",
                          "n_fixtures", "minutes", TARGET]].copy()
            for name in model_names:
                if name in _STATELESS or name not in fitted or i % retrain_every == 0:
                    fitted[name] = make_model(name, ep_hist).fit(train)
                preds[name] = fitted[name].predict(test)
            out.append(preds)
            msg = (f"{season} GW{gw:02d} ({i + 1}/{len(gws)}): train={len(train)} "
                   f"test={len(test)} {time.time() - t_gw:.1f}s")
            logger.info(msg)
            if progress:
                progress(msg)
    return pd.concat(out, ignore_index=True)


def metrics_table(preds: pd.DataFrame, model_names) -> pd.DataFrame:
    rows = []
    for season, g in list(preds.groupby("season")) + [("all", preds)]:
        for name in model_names:
            if g[name].notna().any():
                rows.append({"season": season, "model": name, **metrics.summarize(g, name)})
            else:
                rows.append({"season": season, "model": name, "n": 0, "gws": 0})
    return pd.DataFrame(rows)


def format_table(table: pd.DataFrame) -> str:
    cols = ["model", "gws", "mae", "rmse", "spearman_gw", "p_at_20", "bias", "calib_slope",
            "xi_points"]
    lines = []
    for season, g in table.groupby("season", sort=False):
        lines.append(f"\n== {season} ==")
        shown = g[cols].copy()
        for c in ["mae", "rmse", "spearman_gw", "p_at_20", "bias", "calib_slope"]:
            shown[c] = shown[c].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
        shown["xi_points"] = shown["xi_points"].map(lambda v: "" if pd.isna(v) else f"{v:.0f}")
        shown["gws"] = shown["gws"].fillna(0).astype(int)
        missing = g["n"] == 0
        shown.loc[missing.to_numpy(), "mae"] = "n/a (no pre-deadline snapshots)"
        lines.append(shown.to_string(index=False))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Walk-forward FPL backtest.")
    p.add_argument("--model", default="baselines",
                   help="'baselines' or a comma list of: " + ", ".join(BASELINES))
    p.add_argument("--seasons", nargs="+", default=list(DEFAULT_TEST_SEASONS))
    p.add_argument("--retrain-every", type=int, default=4,
                   help="refit learned models every N gameweeks (default 4; 1 = weekly, "
                        "~20 min for two seasons)")
    p.add_argument("-q", "--quiet", action="store_true", help="no per-gameweek progress")
    p.add_argument("--current-source", choices=["api", "archive"], default="api",
                   help=f"where {config.CURRENT_SEASON} rows come from (archive stops at GW6)")
    p.add_argument("--out", type=Path, default=config.BACKTEST_DIR)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")

    names = list(BASELINES) if args.model == "baselines" else args.model.split(",")
    unknown = set(names) - set(BASELINES)
    if unknown:
        p.error(f"unknown model(s): {', '.join(sorted(unknown))}")

    t0 = time.time()
    fx = load_history(max(args.seasons), args.current_source)
    feats = build_feature_table(fx)
    ep_hist = snapshot.load_ep_next_history()
    progress = None if args.quiet else (lambda m: print(m, file=sys.stderr, flush=True))
    preds = run_backtest(feats, args.seasons, names, retrain_every=args.retrain_every,
                         ep_hist=ep_hist, progress=progress)
    table = metrics_table(preds, names)

    print(f"Walk-forward backtest | target: GW total points per player | "
          f"train from {config.FIRST_XG_SEASON} | retrain every {args.retrain_every} GW")
    print(format_table(table))
    print("\nmae/rmse: points per player-GW (all listed players, incl. 0 minutes) | "
          "spearman_gw / p_at_20: mean per GW | bias: mean(pred-actual) | "
          "calib_slope: OLS actual~pred | xi_points: greedy best XI + captain, no budget")

    args.out.mkdir(parents=True, exist_ok=True)
    tag = "_".join(args.seasons)
    preds.to_parquet(args.out / f"predictions_{tag}.parquet", index=False)
    table.to_csv(args.out / f"metrics_{tag}.csv", index=False)
    print(f"\nSaved predictions and metrics to {args.out} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

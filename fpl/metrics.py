"""Evaluation metrics for gameweek point predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from fpl import config


def mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def brier(prob: np.ndarray, outcome: np.ndarray) -> float:
    """Brier score for a binary event (used for P(60+ min) etc. from PR 2)."""
    return float(np.mean((np.asarray(prob, float) - np.asarray(outcome, float)) ** 2))


def calibration_table(pred: np.ndarray, actual: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Mean prediction vs mean outcome per prediction-quantile bin."""
    d = pd.DataFrame({"pred": pred, "actual": actual})
    d["bin"] = pd.qcut(d["pred"].rank(method="first"), bins, labels=False)
    return d.groupby("bin").agg(pred=("pred", "mean"), actual=("actual", "mean"),
                                n=("pred", "size")).reset_index()


def calibration_slope(pred: np.ndarray, actual: np.ndarray) -> float:
    """OLS slope of actual on predicted (1.0 = well spread, <1 = overconfident)."""
    pred, actual = np.asarray(pred, float), np.asarray(actual, float)
    var = np.var(pred)
    if var == 0:
        return float("nan")
    return float(np.cov(pred, actual, bias=True)[0, 1] / var)


def spearman_by_gw(df: pd.DataFrame, pred: str, actual: str = "total_points") -> float:
    """Mean over gameweeks of the within-gameweek Spearman correlation."""
    vals = []
    for _, g in df.groupby(["season", "gw"]):
        if g[pred].nunique() > 1 and g[actual].nunique() > 1:
            vals.append(spearmanr(g[pred], g[actual]).statistic)
    return float(np.nanmean(vals)) if vals else float("nan")


def topk_precision_by_gw(df: pd.DataFrame, pred: str, actual: str = "total_points",
                         k: int = 20) -> float:
    """Share of the predicted top-k who finish in the actual top-k, averaged over GWs.

    Ties at the cut-off count as hits: a pick scores if their points are at
    least the k-th highest actual score that gameweek.
    """
    vals = []
    for _, g in df.groupby(["season", "gw"]):
        if len(g) < k:
            continue
        cutoff = g[actual].nlargest(k).iloc[-1]
        picks = g.nlargest(k, pred)
        vals.append(float((picks[actual] >= cutoff).mean()))
    return float(np.mean(vals)) if vals else float("nan")


def pick_xi(g: pd.DataFrame, pred: str) -> pd.DataFrame:
    """Greedy highest-predicted valid XI: 1 GKP, 3-5 DEF, 2-5 MID, 1-3 FWD,
    at most 3 per club. No budget; the MILP optimizer replaces this in PR 3."""
    limits = config.XI_POSITION_LIMITS
    counts = {p: 0 for p in limits}
    clubs: dict = {}
    chosen = []
    for idx, row in g.sort_values(pred, ascending=False).iterrows():
        pos, club = row["position"], row["team"]
        if counts[pos] >= limits[pos][1] or clubs.get(club, 0) >= config.MAX_PER_CLUB:
            continue
        counts[pos] += 1
        unmet = sum(max(0, lo - counts[p]) for p, (lo, _) in limits.items())
        if unmet > config.XI_SIZE - (len(chosen) + 1):
            counts[pos] -= 1
            continue
        chosen.append(idx)
        clubs[club] = clubs.get(club, 0) + 1
        if len(chosen) == config.XI_SIZE:
            break
    return g.loc[chosen]


def simulated_xi_points(df: pd.DataFrame, pred: str, actual: str = "total_points") -> float:
    """Season total of actual points for the predicted-best XI each GW, captain
    (highest predicted) doubled. A ranking-quality summary, not a full
    FPL simulation (no budget, transfers, bench or chips)."""
    total = 0.0
    for _, g in df.groupby(["season", "gw"]):
        xi = pick_xi(g, pred)
        if xi.empty:
            continue
        total += xi[actual].sum() + xi.loc[xi[pred].idxmax(), actual]
    return float(total)


def summarize(df: pd.DataFrame, pred: str, actual: str = "total_points") -> dict:
    d = df[df[pred].notna()]
    y, p = d[actual].to_numpy(float), d[pred].to_numpy(float)
    return {
        "n": len(d),
        "gws": d.groupby(["season", "gw"]).ngroups,
        "mae": mae(y, p),
        "rmse": rmse(y, p),
        "spearman_gw": spearman_by_gw(d, pred, actual),
        "p_at_20": topk_precision_by_gw(d, pred, actual, 20),
        "bias": float(np.mean(p - y)),
        "calib_slope": calibration_slope(p, y),
        "xi_points": simulated_xi_points(d, pred, actual),
    }

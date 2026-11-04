"""Player-gameweek feature table.

Unit of prediction: one player in one gameweek (a double gameweek's fixtures
are summed, a blank gameweek has no row). Every feature for gameweek ``g`` is
computed only from gameweeks strictly before ``g`` of the same season (or from
earlier seasons), so it is known at the ``g`` deadline. This also holds for
team strength: in a double gameweek the second fixture does not see the
first one's result.

Only schedule facts (number of fixtures, home/away, opponent) and the price at
the deadline (``value``) come from gameweek ``g`` itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fpl import config

GROUP = ["season", "element"]

SUM_STATS = [
    "total_points", "minutes", "starts", "goals_scored", "assists", "clean_sheets",
    "goals_conceded", "saves", "bonus", "bps", "yellow_cards", "red_cards",
    "expected_goals", "expected_assists", "expected_goal_involvements",
    "expected_goals_conceded", "defensive_contribution",
]

FEATURE_COLUMNS = [
    "pos_GKP", "pos_DEF", "pos_MID", "pos_FWD",
    "value", "n_fixtures", "home_frac", "gw", "n_prev_gws",
    "pts_pf_to_date", "pts_pf_last5", "pts_pf_last3",
    "mins_pf_to_date", "mins_pf_last3", "start_rate_last5", "played60_rate_last5",
    "xg90_to_date", "xa90_to_date", "xgi_pf_last5", "xgc90_to_date",
    "bps_pf_last5", "bonus_pf_to_date", "saves90_to_date", "defcon90_to_date",
    "prev_season_pts_pf", "prev_season_mins_pf",
    "selected_lag_log", "transfers_balance_lag",
    "team_xg_for_l6", "team_xg_against_l6", "opp_xg_for_l6", "opp_xg_against_l6",
]


# -- aggregation --------------------------------------------------------------
def aggregate_player_gw(fx: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-fixture rows to one row per (season, gw, element)."""
    fx = fx.copy()
    fx["played60"] = (fx["minutes"] >= 60).astype(float)
    keys = ["season", "gw", "element"]
    agg = {c: "sum" for c in SUM_STATS + ["played60"]}
    agg.update({
        "fixture": "count", "kickoff_time": "min", "was_home": "mean",
        "code": "first", "name": "first", "position": "last", "team": "last",
        "value": "first", "selected": "first", "transfers_balance": "first",
    })
    # NaN-preserving sums: a stat absent for the whole season stays NaN.
    out = fx.groupby(keys, sort=False).agg(agg)
    present = fx[keys + SUM_STATS].groupby(keys, sort=False).count()
    for c in SUM_STATS:
        out[c] = out[c].where(present[c] > 0)
    out = out.rename(columns={"fixture": "n_fixtures", "was_home": "home_frac"}).reset_index()
    out["gw"] = out["gw"].astype(int)
    return out.sort_values(["season", "element", "gw"]).reset_index(drop=True)


# -- exclusive (strictly-prior) rolling helpers ----------------------------------
def _prior_sum(df: pd.DataFrame, col: str, window: int | None = None) -> pd.Series:
    """Sum of ``col`` over the player's previous rows (excluding current)."""
    x = df[col].fillna(0.0)
    excl = x.groupby([df[k] for k in GROUP]).cumsum() - x
    if window is None:
        return excl
    lagged = excl.groupby([df[k] for k in GROUP]).shift(window).fillna(0.0)
    return excl - lagged


def _prior_count(df: pd.DataFrame, col: str, window: int | None = None) -> pd.Series:
    """How many previous rows had a non-null ``col``."""
    tmp = df[GROUP].copy()
    tmp["_n"] = df[col].notna().astype(float)
    return _prior_sum(tmp, "_n", window)


def _ratio(num: pd.Series, den: pd.Series, scale: float = 1.0, min_den: float = 1e-9) -> pd.Series:
    return (num * scale / den).where(den >= min_den)


# -- team strength ------------------------------------------------------------
def team_gw_strength(fx: pd.DataFrame, window: int = 6) -> pd.DataFrame:
    """Per (season, team, gw): team xG for/against over the previous ``window``
    matches, using only gameweeks before ``gw``. Carries across seasons by
    team name, so GW1 uses the end of last season (promoted sides: NaN).
    """
    tf = (fx.groupby(["season", "gw", "fixture", "team"], sort=False)
            .agg(xg=("expected_goals", lambda s: s.sum(min_count=1)),
                 opponent=("opponent", "first"))
            .reset_index())
    opp = tf[["season", "fixture", "team", "xg"]].rename(
        columns={"team": "opponent", "xg": "xga"})
    tf = tf.merge(opp, on=["season", "fixture", "opponent"], how="left")
    tf["xg_matches"] = tf["xg"].notna().astype(float)  # matches with xG recorded

    per_gw = (tf.groupby(["team", "season", "gw"])
                .agg(xg=("xg", "sum"), xga=("xga", "sum"), matches=("xg_matches", "sum"))
                .reset_index()
                .sort_values(["team", "season", "gw"]))
    g = per_gw.groupby("team")
    out = per_gw[["team", "season", "gw"]].copy()
    den = g["matches"].transform(lambda s: s.shift(1).rolling(window, min_periods=1).sum())
    for col, name in (("xg", "xg_for"), ("xga", "xg_against")):
        num = g[col].transform(lambda s: s.shift(1).rolling(window, min_periods=1).sum())
        out[f"{name}_l{window}"] = (num / den).where(den > 0)
    # One row per gameweek a team played; shift(1) makes the row for gameweek
    # g describe form entering g (a double gameweek's two matches are summed
    # into one row, so neither sees the other).
    return out.reset_index(drop=True)


def _attach_team_features(pgw: pd.DataFrame, fx: pd.DataFrame, strength: pd.DataFrame) -> pd.DataFrame:
    cols = ["xg_for_l6", "xg_against_l6"]
    fx_keys = fx[["season", "gw", "element", "team", "opponent"]].copy()
    fx_keys["gw"] = fx_keys["gw"].astype(int)
    own = strength.rename(columns={c: f"team_{c}" for c in cols})
    opp = strength.rename(columns={"team": "opponent", **{c: f"opp_{c}" for c in cols}})
    fx_keys = fx_keys.merge(own, on=["team", "season", "gw"], how="left")
    fx_keys = fx_keys.merge(opp, on=["opponent", "season", "gw"], how="left")
    feat_cols = [f"team_{c}" for c in cols] + [f"opp_{c}" for c in cols]
    per_player = fx_keys.groupby(["season", "gw", "element"])[feat_cols].mean().reset_index()
    return pgw.merge(per_player, on=["season", "gw", "element"], how="left")


# -- main entry point ---------------------------------------------------------
def _next_season(season: str) -> str:
    y = config.season_start_year(season)
    return f"{y + 1}-{(y + 2) % 100:02d}"


def build_feature_table(fx: pd.DataFrame) -> pd.DataFrame:
    """Per-fixture archive rows -> player-GW rows with leak-free features.

    The returned frame keeps identifiers and the target ``total_points``
    alongside :data:`FEATURE_COLUMNS`.
    """
    fx = fx[fx["gw"].notna()].copy()
    fx["gw"] = fx["gw"].astype(int)
    df = aggregate_player_gw(fx)

    for p in config.POSITIONS:
        df[f"pos_{p}"] = (df["position"] == p).astype(float)
    df["n_prev_gws"] = df.groupby(GROUP).cumcount().astype(float)

    pf = _prior_sum(df, "n_fixtures")
    pf5 = _prior_sum(df, "n_fixtures", 5)
    pf3 = _prior_sum(df, "n_fixtures", 3)
    df["pts_pf_to_date"] = _ratio(_prior_sum(df, "total_points"), pf)
    df["pts_pf_last5"] = _ratio(_prior_sum(df, "total_points", 5), pf5)
    df["pts_pf_last3"] = _ratio(_prior_sum(df, "total_points", 3), pf3)
    df["mins_pf_to_date"] = _ratio(_prior_sum(df, "minutes"), pf)
    df["mins_pf_last3"] = _ratio(_prior_sum(df, "minutes", 3), pf3)
    df["start_rate_last5"] = _ratio(_prior_sum(df, "starts", 5), pf5).where(
        _prior_count(df, "starts", 5) > 0)
    df["played60_rate_last5"] = _ratio(_prior_sum(df, "played60", 5), pf5)
    df["bps_pf_last5"] = _ratio(_prior_sum(df, "bps", 5), pf5)
    df["bonus_pf_to_date"] = _ratio(_prior_sum(df, "bonus"), pf)

    def per90(col: str) -> pd.Series:
        # Denominator only counts minutes from gameweeks where `col` was
        # recorded, so a partially recorded season isn't diluted.
        tmp = df[GROUP].copy()
        tmp["_m"] = df["minutes"].where(df[col].notna())
        mins = _prior_sum(tmp, "_m")
        return _ratio(_prior_sum(df, col), mins, scale=90.0, min_den=90.0).where(
            _prior_count(df, col) > 0)
    df["xg90_to_date"] = per90("expected_goals")
    df["xa90_to_date"] = per90("expected_assists")
    df["xgc90_to_date"] = per90("expected_goals_conceded")
    df["saves90_to_date"] = per90("saves")
    df["defcon90_to_date"] = per90("defensive_contribution")
    df["xgi_pf_last5"] = _ratio(_prior_sum(df, "expected_goal_involvements", 5), pf5).where(
        _prior_count(df, "expected_goal_involvements", 5) > 0)

    # Market signals from the previous GW row only (timing of the archive's
    # per-round transfer counts relative to the deadline is ambiguous).
    lag = df.groupby(GROUP)[["selected", "transfers_balance"]].shift(1)
    df["selected_lag_log"] = np.log1p(lag["selected"].clip(lower=0))
    df["transfers_balance_lag"] = lag["transfers_balance"]

    # Previous-season summary joined on the cross-season player code.
    season_sum = (df.groupby(["season", "code"])
                    .agg(pts=("total_points", "sum"), mins=("minutes", "sum"),
                         fixtures=("n_fixtures", "sum"))
                    .reset_index())
    season_sum["prev_season_pts_pf"] = season_sum["pts"] / season_sum["fixtures"]
    season_sum["prev_season_mins_pf"] = season_sum["mins"] / season_sum["fixtures"]
    season_sum["next_season"] = season_sum["season"].map(_next_season)
    df = df.merge(
        season_sum[["next_season", "code", "prev_season_pts_pf", "prev_season_mins_pf"]]
        .rename(columns={"next_season": "season"}),
        on=["season", "code"], how="left")

    df = _attach_team_features(df, fx, team_gw_strength(fx))
    return df.sort_values(["season", "gw", "element"]).reset_index(drop=True)

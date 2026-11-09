"""
scripts/compute_season_multipliers.py

Season Performance Multiplier Calculation Engine for UN-REAL NBA Prediction System.

Responsibilities:
1. Filters actual completed-game ratings strictly to the current NBA season using
   the October 20 season boundary.
2. Aggregates each player's completed games to compute:
   - season_games_played
   - season_avg_rating (actual ground-truth game rating, NOT predicted_rating)
3. Enforces the minimum games eligibility threshold (>= 3 completed games):
   - Eligible players (>= 3 games) form the league-wide percentile distribution.
   - Ineligible players (< 3 games) receive a performance bonus of 0.0 and are
     excluded from the percentile population so they do not distort rankings.
4. Calculates league-wide percentile ranks (0.0 to 100.0) with average tie-breaking.
5. Applies a continuous, configurable percentile-to-multiplier curve:
   - Candidate A: Linear: 3 * (1 - P / 100)
   - Candidate B: Quadratic: 3 * (1 - P / 100)^2
   - Candidate C: Square-root: 3 * sqrt(1 - P / 100)
   - Candidate D: Upper-percentile cutoff (default 95%) + Square-root:
                  0.0 for P >= 95.0, else 3 * sqrt(1 - P / 95.0)
6. Clamps all multiplier bonuses between 0.0 and 3.0, rounded to exactly 1 decimal place.
7. Produces deterministic, idempotent output.
"""

import os
import sys
import logging
from datetime import datetime
from typing import Optional, Union, Dict, Any, Tuple, List
import pandas as pd
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("propredict.season_multipliers")

DEFAULT_RATINGS_PATH = "data/player_game_ratings_with_dates.parquet"
SEASON_START_MONTH = 10
SEASON_START_DAY = 20
DEFAULT_MIN_GAMES = 3
DEFAULT_CUTOFF_PERCENTILE = 95.0
DEFAULT_CURVE = "candidate_d"


# ==============================================================================
# 1. Season Boundary Resolution
# ==============================================================================
def determine_season_boundary(
    game_dates: Optional[pd.Series] = None,
    reference_date: Optional[Union[str, datetime]] = None,
) -> Tuple[str, str]:
    """
    Determines the current NBA season boundary using the October 20 rule.

    Rules:
    - Season boundary is fixed at October 20 (MM-DD = "10-20").
    - If reference_date is on or after October 20 of year Y:
        Season start is Y-10-20, season end is (Y+1)-10-19.
    - If reference_date is before October 20 of year Y:
        Season start is (Y-1)-10-20, season end is Y-10-19.
    - If reference_date is None, infers the latest date from game_dates.
      If game_dates is empty or None, falls back to the current system date.

    Returns:
        Tuple of (season_start_date_str, season_end_date_str) in 'YYYY-MM-DD' format.
    """
    ref_dt = None
    if reference_date is not None:
        if isinstance(reference_date, str):
            ref_dt = pd.to_datetime(reference_date).to_pydatetime()
        elif isinstance(reference_date, datetime):
            ref_dt = reference_date
        elif hasattr(reference_date, "to_pydatetime"):
            ref_dt = reference_date.to_pydatetime()

    if ref_dt is None:
        if game_dates is not None and not game_dates.empty:
            valid_dates = pd.to_datetime(game_dates, errors="coerce").dropna()
            if not valid_dates.empty:
                ref_dt = valid_dates.max().to_pydatetime()

    if ref_dt is None:
        ref_dt = datetime.now()

    year = ref_dt.year
    oct_20_this_year = datetime(year, SEASON_START_MONTH, SEASON_START_DAY)

    if ref_dt >= oct_20_this_year:
        season_start = oct_20_this_year
        season_end = datetime(year + 1, SEASON_START_MONTH, SEASON_START_DAY - 1)
    else:
        season_start = datetime(year - 1, SEASON_START_MONTH, SEASON_START_DAY)
        season_end = datetime(year, SEASON_START_MONTH, SEASON_START_DAY - 1)

    start_str = season_start.strftime("%Y-%m-%d")
    end_str = season_end.strftime("%Y-%m-%d")
    logger.debug("Resolved season boundary: %s to %s (reference: %s)", start_str, end_str, ref_dt.strftime("%Y-%m-%d"))
    return start_str, end_str


# ==============================================================================
# 2. Percentile-to-Multiplier Formulas
# ==============================================================================
def apply_multiplier_curve(
    percentiles: Union[pd.Series, np.ndarray, float],
    curve: str = DEFAULT_CURVE,
    cutoff_percentile: float = DEFAULT_CUTOFF_PERCENTILE,
) -> Union[pd.Series, np.ndarray, float]:
    """
    Applies a continuous percentile-to-multiplier formula.

    Input:
        percentiles: Percentile rank(s) between 0.0 (lowest rating) and 100.0 (highest rating).
        curve: One of 'candidate_a' (linear), 'candidate_b' (quadratic),
               'candidate_c' (sqrt), 'candidate_d' (cutoff_sqrt).
        cutoff_percentile: Upper cutoff percentile for candidate_d (default 95.0).

    Formula Definitions:
        - Candidate A (Linear):
            B(P) = 3.0 * (1.0 - P / 100.0)
        - Candidate B (Quadratic):
            B(P) = 3.0 * ((1.0 - P / 100.0) ** 2)
        - Candidate C (Square-root):
            B(P) = 3.0 * sqrt(max(0.0, 1.0 - P / 100.0))
        - Candidate D (Upper Cutoff 95% + Square-root):
            If P >= cutoff: B(P) = 0.0
            Else:           B(P) = 3.0 * sqrt(max(0.0, 1.0 - P / cutoff))

    Guarantees:
        - Strictly clamped between 0.0 and 3.0.
        - Rounded to exactly ONE decimal place.
    """
    is_scalar = np.isscalar(percentiles)
    p_arr = np.array(percentiles, dtype=float)

    curve_key = str(curve).lower().strip()

    if curve_key in ("candidate_a", "linear", "a"):
        raw = 3.0 * (1.0 - p_arr / 100.0)

    elif curve_key in ("candidate_b", "quadratic", "b"):
        norm = np.maximum(0.0, 1.0 - p_arr / 100.0)
        raw = 3.0 * (norm ** 2)

    elif curve_key in ("candidate_c", "sqrt", "square_root", "c"):
        norm = np.maximum(0.0, 1.0 - p_arr / 100.0)
        raw = 3.0 * np.sqrt(norm)

    elif curve_key in ("candidate_d", "cutoff_sqrt", "cutoff", "d"):
        cutoff = float(cutoff_percentile)
        if cutoff <= 0.0:
            raise ValueError(f"cutoff_percentile must be > 0, got {cutoff}")
        # For P >= cutoff, bonus = 0.0
        # For P < cutoff, B(P) = 3.0 * sqrt(1.0 - P / cutoff)
        norm = np.where(p_arr >= cutoff, 0.0, np.maximum(0.0, 1.0 - p_arr / cutoff))
        raw = np.where(p_arr >= cutoff, 0.0, 3.0 * np.sqrt(norm))

    else:
        raise ValueError(
            f"Unsupported curve '{curve}'. Supported options: 'candidate_a', 'candidate_b', 'candidate_c', 'candidate_d'"
        )

    clamped = np.clip(raw, 0.0, 3.0)
    rounded = np.round(clamped, 1)

    if is_scalar:
        return float(rounded)

    if isinstance(percentiles, pd.Series):
        return pd.Series(rounded, index=percentiles.index, dtype=float)

    return rounded


# ==============================================================================
# 3. Percentile Ranking Engine
# ==============================================================================
def calculate_percentiles(season_averages: pd.Series) -> pd.Series:
    """
    Calculates ascending percentile ranks [0.0, 100.0] for eligible players.

    Ascending Semantics:
        - Lowest season average rating receives rank 1 (percentile ~0.0).
        - Highest season average rating receives rank N (percentile 100.0).
        - Ties are broken consistently using 'average' rank.

    Formulation:
        For N > 1 eligible players:
            Percentile = ((Rank - 1) / (N - 1)) * 100.0
        For N = 1 player:
            Percentile = 100.0
        For N = 0 players:
            Empty series.
    """
    if season_averages.empty:
        return pd.Series(dtype=float)

    n_players = len(season_averages)
    if n_players == 1:
        return pd.Series([100.0], index=season_averages.index, dtype=float)

    # Average rank for ties (e.g. if two players tie for rank 1 and 2, both get 1.5)
    ranks = season_averages.rank(ascending=True, method="average")
    percentiles = ((ranks - 1.0) / (n_players - 1.0)) * 100.0
    return percentiles


# ==============================================================================
# 4. Core Multiplier Calculation
# ==============================================================================
def compute_season_multipliers(
    ratings_df: Optional[pd.DataFrame] = None,
    input_path: str = DEFAULT_RATINGS_PATH,
    season_start_date: Optional[str] = None,
    season_end_date: Optional[str] = None,
    reference_date: Optional[Union[str, datetime]] = None,
    min_games_threshold: int = DEFAULT_MIN_GAMES,
    curve: str = DEFAULT_CURVE,
    cutoff_percentile: float = DEFAULT_CUTOFF_PERCENTILE,
    all_players_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Computes season performance multipliers for NBA players.

    Parameters:
        ratings_df: Optional DataFrame with columns ['gameId', 'personId', 'rating', 'game_date'].
                    If None, loaded from input_path.
        input_path: Parquet file path (default 'data/player_game_ratings_with_dates.parquet').
        season_start_date: Optional explicit season start date (e.g. '2026-10-20').
        season_end_date: Optional explicit season end date (e.g. '2027-10-19').
        reference_date: Optional date used to infer the active season boundary.
        min_games_threshold: Minimum completed games required for eligibility (default 3).
        curve: Percentile-to-multiplier formula ('candidate_a', 'candidate_b', 'candidate_c', 'candidate_d').
        cutoff_percentile: Upper cutoff percentile for candidate_d (default 95.0).
        all_players_df: Optional DataFrame of known league players (to ensure 0-game players are returned).

    Returns:
        DataFrame keyed by personId with columns:
        - personId: int64
        - playerName: str (if available)
        - season_games_played: int64
        - season_avg_rating: float (nullable, rounded to 4 decimals)
        - season_percentile: float (nullable, rounded to 2 decimals)
        - season_performance_bonus: float (strictly 1 decimal place, 0.0 to 3.0)
        - season_multiplier_eligible: bool
    """
    # 1. Load or accept input DataFrame
    if ratings_df is not None:
        df = ratings_df.copy()
    else:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Missing ratings input file: {input_path}")
        df = pd.read_parquet(input_path)

    # Empty dataset guard
    if df.empty:
        logger.warning("Input ratings dataset is empty. Returning empty multiplier DataFrame.")
        cols = [
            "personId", "playerName", "season_games_played", "season_game_count",
            "season_avg_rating", "season_average_rating",
            "season_percentile", "season_performance_bonus", "season_multiplier_eligible"
        ]
        return pd.DataFrame(columns=cols)

    # 2. Validate required columns
    required_cols = ["gameId", "personId", "rating", "game_date"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input dataset is missing required columns: {missing}")

    # 3. Clean and sanitize records
    df = df.dropna(subset=["personId", "rating", "game_date"]).copy()
    # Ensure personId is numeric integer and positive
    df["personId"] = pd.to_numeric(df["personId"], errors="coerce")
    df = df[df["personId"] > 0].copy()
    df["personId"] = df["personId"].astype(np.int64)

    # Ensure rating is numeric float
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df[df["rating"].notna()].copy()

    # Format game_date cleanly
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df[df["game_date"].notna()].copy()

    # Deduplicate: A player can only have one completed rating per gameId
    df = df.drop_duplicates(subset=["personId", "gameId"]).copy()

    # 4. Resolve season boundary
    if season_start_date is None or season_end_date is None:
        auto_start, auto_end = determine_season_boundary(
            game_dates=df["game_date"],
            reference_date=reference_date,
        )
        start_date = season_start_date or auto_start
        end_date = season_end_date or auto_end
    else:
        start_date = season_start_date
        end_date = season_end_date

    logger.info("Applying current-season filter: %s <= game_date <= %s", start_date, end_date)

    # 5. Filter strictly to current season games
    season_mask = (df["game_date"] >= start_date) & (df["game_date"] <= end_date)
    season_df = df[season_mask].copy()

    logger.info(
        "Current season dataset: %d player-game rows across %d unique games and %d unique players",
        len(season_df),
        season_df["gameId"].nunique() if not season_df.empty else 0,
        season_df["personId"].nunique() if not season_df.empty else 0,
    )

    # 6. Aggregate by player to compute games played and season average actual rating
    name_map = {}
    if "playerName" in df.columns:
        # Map personId to the most recent non-empty playerName
        for pid, group in df.groupby("personId"):
            valid_names = group["playerName"].dropna().astype(str).str.strip()
            valid_names = valid_names[valid_names != ""]
            name_map[pid] = valid_names.iloc[-1] if not valid_names.empty else f"Player {pid}"

    if not season_df.empty:
        player_stats = season_df.groupby("personId").agg(
            season_games_played=("rating", "count"),
            season_avg_rating=("rating", "mean"),
        ).reset_index()
    else:
        player_stats = pd.DataFrame(columns=["personId", "season_games_played", "season_avg_rating"])

    # 7. Merge external roster/players if provided (to represent 0-game players)
    if all_players_df is not None and not all_players_df.empty and "personId" in all_players_df.columns:
        external_pids = all_players_df["personId"].dropna().astype(np.int64).unique()
        known_pids = set(player_stats["personId"].unique()) if not player_stats.empty else set()
        missing_pids = [p for p in external_pids if p not in known_pids]
        if missing_pids:
            zero_game_rows = pd.DataFrame({
                "personId": missing_pids,
                "season_games_played": 0,
                "season_avg_rating": np.nan,
            })
            player_stats = pd.concat([player_stats, zero_game_rows], ignore_index=True)
            if "playerName" in all_players_df.columns:
                for _, r in all_players_df[all_players_df["personId"].isin(missing_pids)].iterrows():
                    pid = int(r["personId"])
                    if pid not in name_map and pd.notna(r.get("playerName")):
                        name_map[pid] = str(r["playerName"])

    if player_stats.empty:
        logger.warning("No player statistics available. Returning empty DataFrame.")
        cols = [
            "personId", "playerName", "season_games_played", "season_game_count",
            "season_avg_rating", "season_average_rating",
            "season_percentile", "season_performance_bonus", "season_multiplier_eligible"
        ]
        return pd.DataFrame(columns=cols)

    player_stats["personId"] = player_stats["personId"].astype(np.int64)
    player_stats["season_games_played"] = player_stats["season_games_played"].astype(np.int64)
    player_stats["playerName"] = player_stats["personId"].map(name_map).fillna("Unknown Player")

    # 8. Separate eligible vs. ineligible players
    # Eligibility Rule: >= min_games_threshold completed games
    eligible_mask = player_stats["season_games_played"] >= min_games_threshold
    player_stats["season_multiplier_eligible"] = eligible_mask

    # Initialize output columns
    player_stats["season_percentile"] = np.nan
    player_stats["season_performance_bonus"] = 0.0

    eligible_df = player_stats[eligible_mask].copy()
    n_eligible = len(eligible_df)
    logger.info("Eligible players for percentile distribution (>= %d games): %d", min_games_threshold, n_eligible)

    # 9. Compute percentiles and multiplier bonuses for eligible players ONLY
    if n_eligible > 0:
        eligible_percentiles = calculate_percentiles(eligible_df["season_avg_rating"])
        eligible_df["season_percentile"] = np.round(eligible_percentiles, 2)

        eligible_bonuses = apply_multiplier_curve(
            eligible_percentiles,
            curve=curve,
            cutoff_percentile=cutoff_percentile,
        )
        eligible_df["season_performance_bonus"] = eligible_bonuses

        # Update back to main DataFrame
        player_stats.loc[eligible_mask, "season_percentile"] = eligible_df["season_percentile"]
        player_stats.loc[eligible_mask, "season_performance_bonus"] = eligible_df["season_performance_bonus"]

    # Ineligible players strictly keep season_performance_bonus = 0.0, season_percentile = NaN
    player_stats.loc[~eligible_mask, "season_performance_bonus"] = 0.0
    player_stats.loc[~eligible_mask, "season_percentile"] = np.nan

    # Clean display rounding
    player_stats["season_avg_rating"] = player_stats["season_avg_rating"].round(4)
    player_stats["season_performance_bonus"] = player_stats["season_performance_bonus"].round(1)

    # Aliases for specification consistency
    player_stats["season_game_count"] = player_stats["season_games_played"]
    player_stats["season_average_rating"] = player_stats["season_avg_rating"]

    # 10. Order output columns and sort deterministically
    ordered_cols = [
        "personId",
        "playerName",
        "season_games_played",
        "season_game_count",
        "season_avg_rating",
        "season_average_rating",
        "season_percentile",
        "season_performance_bonus",
        "season_multiplier_eligible",
    ]
    result_df = player_stats[ordered_cols].sort_values(
        ["season_performance_bonus", "season_avg_rating", "personId"],
        ascending=[False, True, True],
    ).reset_index(drop=True)

    return result_df


# ==============================================================================
# 5. Multiplier Curve Comparison Tooling
# ==============================================================================
def compare_candidate_curves(
    ratings_df: Optional[pd.DataFrame] = None,
    input_path: str = DEFAULT_RATINGS_PATH,
    cutoff_percentile: float = DEFAULT_CUTOFF_PERCENTILE,
) -> Dict[str, Any]:
    """
    Computes and compares all four candidate curves on the current dataset.

    Returns a structured dictionary containing metrics, bin distributions,
    exact value counts, and player samples.
    """
    curves = [
        ("Candidate A (Linear)", "candidate_a"),
        ("Candidate B (Quadratic)", "candidate_b"),
        ("Candidate C (Square-Root)", "candidate_c"),
        ("Candidate D (95% Cutoff + Sqrt)", "candidate_d"),
    ]

    results = {}
    base_res = None

    for label, curve_id in curves:
        res = compute_season_multipliers(
            ratings_df=ratings_df,
            input_path=input_path,
            curve=curve_id,
            cutoff_percentile=cutoff_percentile,
        )
        if base_res is None:
            base_res = res.copy()

        elig = res[res["season_multiplier_eligible"]].copy()
        s = elig["season_performance_bonus"]
        n_elig = len(elig)

        bins = [(-0.01, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.01)]
        bin_labels = ["0.0-0.5", "0.6-1.0", "1.1-1.5", "1.6-2.0", "2.1-2.5", "2.6-3.0"]
        bin_counts = {}
        for (low, high), bl in zip(bins, bin_labels):
            cnt = int(((s > low) & (s <= high)).sum())
            pct = round(cnt / n_elig * 100.0, 1) if n_elig > 0 else 0.0
            bin_counts[bl] = {"count": cnt, "percentage": pct}

        exact_counts = {round(v, 1): int((s == round(v, 1)).sum()) for v in np.arange(0.0, 3.05, 0.1)}

        results[curve_id] = {
            "label": label,
            "mean": round(float(s.mean()), 2) if n_elig > 0 else 0.0,
            "median": round(float(s.median()), 2) if n_elig > 0 else 0.0,
            "min": round(float(s.min()), 1) if n_elig > 0 else 0.0,
            "max": round(float(s.max()), 1) if n_elig > 0 else 0.0,
            "count_exact_zero": int((s == 0.0).sum()),
            "count_above_two": int((s > 2.0).sum()),
            "pct_above_two": round(float((s > 2.0).mean() * 100.0), 1) if n_elig > 0 else 0.0,
            "bin_distribution": bin_counts,
            "exact_counts": exact_counts,
            "dataframe": res,
        }

    return {
        "total_players": len(base_res),
        "eligible_players": len(base_res[base_res["season_multiplier_eligible"]]),
        "ineligible_players": len(base_res[~base_res["season_multiplier_eligible"]]),
        "curves": results,
    }


# ==============================================================================
# 6. CLI Entrypoint
# ==============================================================================
def main():
    print("=" * 70)
    print("🏀 UN-REAL: Season Performance Multipliers Calculation Engine")
    print("=" * 70)

    comparison = compare_candidate_curves()

    print(f"\nTotal Players Processed:      {comparison['total_players']}")
    print(f"Eligible Players (>=3 games): {comparison['eligible_players']}")
    print(f"Ineligible Players (<3 games): {comparison['ineligible_players']}")

    print("\n" + "=" * 70)
    print("CANDIDATE CURVE COMPARISON (CURRENT SEASON DATA)")
    print("=" * 70)

    for curve_id, data in comparison["curves"].items():
        print(f"\n--- {data['label']} ---")
        print(f"  Mean Bonus:       {data['mean']}")
        print(f"  Median Bonus:     {data['median']}")
        print(f"  Count Exactly 0.0: {data['count_exact_zero']} ({data['count_exact_zero']/comparison['eligible_players']*100:.1f}%)")
        print(f"  Count > 2.0:       {data['count_above_two']} ({data['pct_above_two']}%)")
        print("  Distribution by Range:")
        for r_label, r_data in data["bin_distribution"].items():
            print(f"    {r_label:8s}: {r_data['count']:3d} ({r_data['percentage']:5.1f}%)")

    # Sample player comparison across all 4 curves
    df_a = comparison["curves"]["candidate_a"]["dataframe"].set_index("personId")
    df_b = comparison["curves"]["candidate_b"]["dataframe"].set_index("personId")
    df_c = comparison["curves"]["candidate_c"]["dataframe"].set_index("personId")
    df_d = comparison["curves"]["candidate_d"]["dataframe"].set_index("personId")

    sample_pids = [
        1630162,  # Tyrese Maxey (top)
        1629639,  # Tyler Herro (top)
        203507,   # Giannis Antetokounmpo (top 95%)
        1628401,  # Derrick White (~80th)
        203084,   # Harrison Barnes (~50th)
        1629638,  # Nickeil Alexander-Walker (lower)
        203937,   # Kyle Anderson (lowest)
    ]

    print("\n" + "=" * 70)
    print("SAMPLE PLAYERS COMPARISON ACROSS CANDIDATE CURVES")
    print("=" * 70)
    header = f"{'Player Name':<22} {'Games':<6} {'Avg Rating':<11} {'Percentile':<11} {'Bonus A':<8} {'Bonus B':<8} {'Bonus C':<8} {'Bonus D':<8}"
    print(header)
    print("-" * len(header))

    for pid in sample_pids:
        if pid in df_d.index:
            row_d = df_d.loc[pid]
            pname = str(row_d["playerName"])[:20]
            gp = row_d["season_games_played"]
            avg_r = f"{row_d['season_avg_rating']:.2f}" if pd.notna(row_d['season_avg_rating']) else "N/A"
            pct = f"{row_d['season_percentile']:.1f}%" if pd.notna(row_d['season_percentile']) else "N/A"
            b_a = f"{df_a.loc[pid, 'season_performance_bonus']:.1f}"
            b_b = f"{df_b.loc[pid, 'season_performance_bonus']:.1f}"
            b_c = f"{df_c.loc[pid, 'season_performance_bonus']:.1f}"
            b_d = f"{df_d.loc[pid, 'season_performance_bonus']:.1f}"
            print(f"{pname:<22} {gp:<6} {avg_r:<11} {pct:<11} {b_a:<8} {b_b:<8} {b_c:<8} {b_d:<8}")

    print("\n✅ Multiplier calculation complete.")


if __name__ == "__main__":
    main()

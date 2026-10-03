"""
scripts/apply_injury_status.py

Applies official NBA injury/availability statuses to player rating predictions.

Key Improvements (Phase 2C):
1. Reliable Provider Abstraction (uses InjuryProvider with browser headers).
2. Identity-Safe Matching: Merges by canonical personId (no surname-only guessing).
3. Resilient Failure Handling: Never crashes if external provider is unreachable;
   safely marks players as 'Unknown' while preserving all predictions.
4. Downstream Compatibility: Generates data/predicted_ratings_with_status.parquet
   with clean status classifications ('Active', 'Probable', 'Questionable', 'Doubtful', 'Out', 'Unknown').

Artifact Safety (Phase 2E.5C.1):
5. Provider failure NEVER overwrites a valid existing output artifact.
   When the external injury source is unavailable, InjuryProviderFailure is raised
   so the pipeline halts before any write occurs. The existing parquet is preserved.
"""

import os
import sys
import logging
import pandas as pd
from datetime import datetime
from typing import Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from scripts.injury_provider import (
    InjuryProvider,
    CanonicalStatus,
    InjuryReportResult,
)

logger = logging.getLogger(__name__)


class InjuryProviderFailure(RuntimeError):
    """Raised when the external injury provider is unavailable.

    The pipeline must halt on this exception and must NOT overwrite any
    existing valid artifact with Unknown-status records.
    """
    pass


def apply_injury_status(
    input_path: str = "data/predicted_ratings.parquet",
    output_path: str = "data/predicted_ratings_with_status.parquet",
    provider: Optional[InjuryProvider] = None,
    target_date: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Load predicted ratings, fetch availability from the provider,
    safely merge by personId, and output enriched predictions.

    Safety guarantee (Phase 2E.5C.1):
    If the external injury provider fails, InjuryProviderFailure is raised
    BEFORE any write to output_path. The existing output artifact (if any)
    is always preserved intact on provider failure.
    """
    # 1. LOAD PREDICTIONS
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Missing input predictions file: {input_path}")

    pred = pd.read_parquet(input_path)
    print(f"Loaded predictions: {pred.shape}")

    if pred.empty:
        raise ValueError("Prediction dataset cannot be empty")

    if "personId" not in pred.columns:
        raise ValueError("Missing 'personId' column in prediction dataset")

    # 2. INITIALIZE INJURY PROVIDER
    injury_provider = provider or InjuryProvider()

    # Determine query date
    if target_date is None and "game_date" in pred.columns:
        try:
            latest_date_str = str(pred["game_date"].max())
            parsed_dt = pd.to_datetime(latest_date_str)
            if not pd.isna(parsed_dt):
                target_date = parsed_dt.to_pydatetime()
        except Exception:
            target_date = None

    print(f"Fetching injury data for target date: {target_date.strftime('%Y-%m-%d') if target_date else 'current'}...")
    result: InjuryReportResult = injury_provider.get_injury_report(target_date=target_date)

    # 3. MERGE BY CANONICAL PERSON ID
    if result.success:
        print(f"Injury provider SUCCESS: {len(result.records)} records retrieved for {result.reportDate}")
        status_map = result.get_player_status_map()

        statuses = []
        reasons = []
        report_dates = []
        sources = []

        for _, row in pred.iterrows():
            pid = row["personId"]
            if pid in status_map:
                rec = status_map[pid]
                statuses.append(rec.status)
                reasons.append(rec.reason or "Listed on report")
                report_dates.append(rec.reportDate)
                sources.append(rec.source)
            else:
                # Player is playing/active (not listed on valid report)
                statuses.append(CanonicalStatus.ACTIVE)
                reasons.append("Not Listed (Available)")
                report_dates.append(result.reportDate)
                sources.append(result.source)

        pred["status"] = statuses
        pred["injury_reason"] = reasons
        pred["injury_date"] = report_dates
        pred["injury_source"] = sources

    else:
        # Phase 2E.5C.1 safety invariant:
        # EXTERNAL PROVIDER FAILURE MUST NEVER OVERWRITE A VALID EXISTING ARTIFACT.
        #
        # Do NOT write Unknown statuses to output_path.
        # Do NOT silently continue the pipeline with stale/fabricated status data.
        # Instead, raise InjuryProviderFailure so the pipeline runner halts here
        # and all downstream steps (build_frontend_dataset, generate_json) are skipped.
        #
        # If a valid output artifact already exists, it remains untouched because
        # this raise occurs before any write.
        print(f"\n❌ Injury provider unavailable: {result.error_message}")
        if os.path.exists(output_path):
            print(f"   Existing artifact preserved: {output_path}")
        else:
            print("   No existing artifact — pipeline cannot continue without fresh injury data.")
        raise InjuryProviderFailure(
            f"Injury provider failed — pipeline halted to protect artifact integrity. "
            f"Reason: {result.error_message}"
        )

    # 4. ASSIGN STATUS RANK FOR DOWNSTREAM SORTING
    pred["status_rank"] = pred["status"].map(CanonicalStatus.STATUS_RANK).fillna(5).astype(int)

    # 5. SAVE ENRICHED PREDICTIONS
    # This write is only reached when result.success is True (provider succeeded).
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    pred.to_parquet(output_path, index=False)

    print("\nStatus distribution:")
    print(pred["status"].value_counts())

    print(f"\nSaved enriched predictions → {output_path}")

    return pred


if __name__ == "__main__":
    apply_injury_status()
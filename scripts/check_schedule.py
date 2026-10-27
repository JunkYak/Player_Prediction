"""
scripts/check_schedule.py

DST-Aware schedule guard for GitHub Actions daily pipeline.
Target local schedule: 05:00 America/New_York year-round.

GitHub Actions cron fires at: 0 9,10 * * *
- During Eastern Daylight Time (EDT, UTC-4):
    09:00 UTC is 05:00 EDT (INTENDED RUN SLOT).
    10:00 UTC is 06:00 EDT (NON-MATCHING TRIGGER -> SKIPPED).
- During Eastern Standard Time (EST, UTC-5):
    09:00 UTC is 04:00 EST (NON-MATCHING TRIGGER -> SKIPPED).
    10:00 UTC is 05:00 EST (INTENDED RUN SLOT).

This guard verifies whether the current run corresponds to the intended 05:00 Eastern slot.
Handles GitHub Actions runner queue delays up to 59 minutes (execution within the 05:00 - 05:59 window).
Permits manual executions (workflow_dispatch) via --force.
"""

import os
import sys
import argparse
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Optional


def is_scheduled_slot_active(
    current_time: Optional[datetime] = None,
    target_hour_eastern: int = 5,
) -> bool:
    """
    Check if the current time in America/New_York matches the intended daily run window.

    Returns True if and only if the current local Eastern hour is target_hour_eastern (05:00 - 05:59).
    This strictly isolates the 09:00 UTC trigger during EDT and the 10:00 UTC trigger during EST,
    preventing dual execution on the same date while accommodating up to 59 minutes of runner latency.
    """
    ny_tz = ZoneInfo("America/New_York")
    now_utc = current_time or datetime.now(timezone.utc)
    now_ny = now_utc.astimezone(ny_tz)

    is_dst = bool(now_ny.dst() and now_ny.dst().total_seconds() != 0)
    target_utc_hour = 9 if is_dst else 10
    tz_abbrev = now_ny.strftime("%Z")

    print(f"Current UTC Time:         {now_utc.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"Current New York Time:    {now_ny.strftime('%Y-%m-%d %H:%M:%S')} {tz_abbrev}")
    print(f"Observing DST:            {is_dst} ({tz_abbrev})")
    print(f"Expected Trigger UTC:     {target_utc_hour:02d}:00 UTC (for 05:00 {tz_abbrev})")

    # Strict check: Local Eastern hour must be 05:xx
    if now_ny.hour == target_hour_eastern:
        print(f"Decision: MATCH - Local New York time is within the {target_hour_eastern:02d}:00 hour ({tz_abbrev}).")
        return True

    print(
        f"Decision: SKIP - Local New York time {now_ny.strftime('%H:%M')} {tz_abbrev} "
        f"is outside the {target_hour_eastern:02d}:00 window."
    )
    return False


def set_github_output(name: str, value: str) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path and os.path.exists(os.path.dirname(output_path)):
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(f"{name}={value}\n")
    print(f"GitHub Output: {name}={value}")


def main():
    parser = argparse.ArgumentParser(description="Check if current run matches 05:00 America/New_York slot.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bypass schedule check (e.g. for workflow_dispatch)",
    )
    args = parser.parse_args()

    # Always execute if explicitly forced or triggered manually
    event_name = os.environ.get("GITHUB_EVENT_NAME", "")
    if args.force or event_name == "workflow_dispatch":
        print(f"Manual or forced execution detected (event: {event_name or 'cli'}). Bypassing schedule check.")
        set_github_output("should_run", "true")
        sys.exit(0)

    should_run = is_scheduled_slot_active(target_hour_eastern=5)
    set_github_output("should_run", "true" if should_run else "false")
    sys.exit(0)


if __name__ == "__main__":
    main()

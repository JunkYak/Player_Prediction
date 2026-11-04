"""Save pre-deadline ``bootstrap-static`` snapshots so FPL's ``ep_next`` history builds up.

The live API only shows ``ep_next`` for the *upcoming* gameweek, and the
archive has no pre-deadline copy of it, so FPL's own projection can only be
used as a baseline for gameweeks we snapshot ourselves. Run this at least
once per gameweek before the deadline (e.g. daily from a scheduler):

    python -m fpl.ingest.snapshot            # save one snapshot now
    python -m fpl.ingest.snapshot --list     # show what has been captured

Layout: ``<SNAPSHOT_DIR>/<season>/gw<NN>/bootstrap_<UTC timestamp>.json.gz``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from fpl import config
from fpl.ingest.fpl_api import FPLClient, next_event, season_from_bootstrap

logger = logging.getLogger(__name__)

EP_COLUMNS = ["season", "gw", "element", "code", "ep_next", "chance_of_playing_next_round",
              "status", "now_cost", "snapshot_time", "deadline_time"]


def save_bootstrap_snapshot(client: FPLClient | None = None, out_dir: Path | None = None,
                            now: datetime | None = None,
                            bootstrap: dict | None = None) -> Path | None:
    """Fetch a fresh bootstrap-static and store it under the upcoming GW.

    Returns the written path, or None when there is no upcoming gameweek
    (season over) or the deadline has already passed.
    """
    now = now or datetime.now(timezone.utc)
    if bootstrap is None:
        bootstrap = (client or FPLClient()).bootstrap_static(use_cache=False)
    event = next_event(bootstrap)
    if event is None:
        logger.info("No upcoming gameweek; nothing to snapshot.")
        return None
    deadline = pd.Timestamp(event["deadline_time"]).to_pydatetime()
    if now >= deadline:
        logger.warning("GW%s deadline %s has passed; skipping.", event["id"], deadline)
        return None

    season = season_from_bootstrap(bootstrap)
    folder = Path(out_dir or config.SNAPSHOT_DIR) / season / f"gw{int(event['id']):02d}"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"bootstrap_{now.strftime('%Y%m%dT%H%M%SZ')}.json.gz"
    payload = {"snapshot_time": now.isoformat(), "season": season, "gw": int(event["id"]),
               "deadline_time": event["deadline_time"], "bootstrap": bootstrap}
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh)
    logger.info("Saved %s", path)
    return path


def _read(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def load_ep_next_history(season: str | None = None, snapshot_dir: Path | None = None,
                         latest_only: bool = True) -> pd.DataFrame:
    """``ep_next`` per (season, gw, element) from stored snapshots.

    With ``latest_only`` (default) only the last snapshot taken before each
    deadline is kept, i.e. FPL's final pre-deadline projection.
    """
    root = Path(snapshot_dir or config.SNAPSHOT_DIR)
    pattern = f"{season}/gw*/bootstrap_*.json.gz" if season else "*/gw*/bootstrap_*.json.gz"
    frames = []
    for path in sorted(root.glob(pattern)):
        blob = _read(path)
        snap_time = pd.Timestamp(blob["snapshot_time"])
        deadline = pd.Timestamp(blob["deadline_time"])
        if snap_time >= deadline:  # defensive: never use a post-deadline file
            continue
        els = pd.DataFrame(blob["bootstrap"]["elements"])
        els = els.rename(columns={"id": "element"})
        for c in EP_COLUMNS:
            if c not in els.columns:
                els[c] = None
        els["season"], els["gw"] = blob["season"], int(blob["gw"])
        els["snapshot_time"], els["deadline_time"] = snap_time, deadline
        frames.append(els[EP_COLUMNS])
    if not frames:
        return pd.DataFrame(columns=EP_COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    out["ep_next"] = pd.to_numeric(out["ep_next"], errors="coerce")
    if latest_only:
        out = (out.sort_values("snapshot_time")
                  .drop_duplicates(["season", "gw", "element"], keep="last"))
    return out.sort_values(["season", "gw", "element"]).reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--list", action="store_true", help="summarise stored snapshots")
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args(argv)
    if args.list:
        hist = load_ep_next_history(snapshot_dir=args.out_dir, latest_only=False)
        if hist.empty:
            print("No snapshots yet.")
        else:
            print(hist.groupby(["season", "gw"])["snapshot_time"]
                      .agg(["nunique", "max"]).rename(columns={"nunique": "snapshots",
                                                               "max": "latest"}))
        return 0
    return 0 if save_bootstrap_snapshot(out_dir=args.out_dir) else 1


if __name__ == "__main__":
    raise SystemExit(main())

import gzip
import json
from datetime import datetime, timezone

from fpl.ingest.snapshot import load_ep_next_history, save_bootstrap_snapshot


def _bootstrap(ep_value):
    return {
        "events": [
            {"id": 9, "deadline_time": "2026-10-31T11:00:00Z", "finished": True, "is_next": False},
            {"id": 10, "deadline_time": "2026-11-06T18:30:00Z", "finished": False, "is_next": True},
            {"id": 1, "deadline_time": "2026-08-21T17:30:00Z", "finished": True, "is_next": False},
        ],
        "elements": [
            {"id": 1, "code": 11, "ep_next": str(ep_value), "chance_of_playing_next_round": None,
             "status": "a", "now_cost": 62},
            {"id": 2, "code": 22, "ep_next": "2.5", "chance_of_playing_next_round": 75,
             "status": "d", "now_cost": 45},
        ],
    }


def test_snapshot_written_under_upcoming_gw(tmp_path):
    now = datetime(2026, 11, 4, 12, tzinfo=timezone.utc)
    path = save_bootstrap_snapshot(out_dir=tmp_path, now=now, bootstrap=_bootstrap(6.0))
    assert path.parent == tmp_path / "2026-27" / "gw10"
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        blob = json.load(fh)
    assert blob["gw"] == 10 and blob["season"] == "2026-27"


def test_snapshot_skipped_after_deadline(tmp_path):
    now = datetime(2026, 11, 6, 19, tzinfo=timezone.utc)
    assert save_bootstrap_snapshot(out_dir=tmp_path, now=now, bootstrap=_bootstrap(6.0)) is None
    assert not any(tmp_path.rglob("*.json.gz"))


def test_ep_next_history_keeps_latest_pre_deadline_value(tmp_path):
    save_bootstrap_snapshot(out_dir=tmp_path, bootstrap=_bootstrap(5.0),
                            now=datetime(2026, 11, 4, 9, tzinfo=timezone.utc))
    save_bootstrap_snapshot(out_dir=tmp_path, bootstrap=_bootstrap(6.5),
                            now=datetime(2026, 11, 6, 9, tzinfo=timezone.utc))
    # a file stamped after the deadline must be ignored even if present on disk
    late = tmp_path / "2026-27" / "gw10" / "bootstrap_20261107T000000Z.json.gz"
    with gzip.open(late, "wt", encoding="utf-8") as fh:
        json.dump({"snapshot_time": "2026-11-07T00:00:00+00:00", "season": "2026-27", "gw": 10,
                   "deadline_time": "2026-11-06T18:30:00Z", "bootstrap": _bootstrap(99.0)}, fh)

    hist = load_ep_next_history(snapshot_dir=tmp_path)
    assert len(hist) == 2
    row = hist.set_index("element").loc[1]
    assert row["ep_next"] == 6.5 and row["gw"] == 10 and row["season"] == "2026-27"
    assert len(load_ep_next_history(snapshot_dir=tmp_path, latest_only=False)) == 4


def test_ep_next_history_empty_when_no_snapshots(tmp_path):
    assert load_ep_next_history(snapshot_dir=tmp_path).empty

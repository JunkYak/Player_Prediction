"""Thin, polite client for the public FPL API (no auth needed).

* Every response is cached on disk as JSON with a fetch timestamp; a cached
  copy younger than its TTL is returned without touching the network.
* Network requests are spaced at least ``min_interval`` seconds apart
  (default 1 s) and retried with backoff on 429 / 5xx.

:func:`build_season_frame` rebuilds the current season's per-fixture table
from ``element-summary/{id}/`` histories in the same schema as the archive.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd
import requests

from fpl import config
from fpl.ingest import archive

logger = logging.getLogger(__name__)

BASE_URL = "https://fantasy.premierleague.com/api"
USER_AGENT = "fpl-predict/0.1 (+https://github.com/JunkYak/Player_Prediction)"

#: Default cache lifetimes (seconds) per endpoint family.
TTL = {
    "bootstrap-static": 15 * 60,
    "fixtures": 60 * 60,
    "element-summary": 12 * 3600,
    "entry": 10 * 60,
    "event": 10 * 60,
}


class FPLAPIError(RuntimeError):
    pass


def _cache_name(path: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", path.strip("/")) + ".json"


class FPLClient:
    def __init__(self, cache_dir: Path | None = None, min_interval: float = 1.0,
                 session: requests.Session | None = None, max_retries: int = 3,
                 clock: Callable[[], float] = time.time,
                 sleep: Callable[[float], None] = time.sleep):
        self.cache_dir = Path(cache_dir or config.API_CACHE_DIR)
        self.min_interval = min_interval
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", USER_AGENT)
        self.max_retries = max_retries
        self._clock = clock
        self._sleep = sleep
        self._last_request: float | None = None
        self.network_calls = 0

    # -- core -----------------------------------------------------------------
    def _throttle(self) -> None:
        if self._last_request is not None:
            wait = self.min_interval - (self._clock() - self._last_request)
            if wait > 0:
                self._sleep(wait)
        self._last_request = self._clock()

    def _cache_path(self, path: str) -> Path:
        return self.cache_dir / _cache_name(path)

    def read_cache(self, path: str, ttl: float | None) -> Any | None:
        fp = self._cache_path(path)
        if not fp.exists():
            return None
        blob = json.loads(fp.read_text(encoding="utf-8"))
        if ttl is not None and self._clock() - blob["fetched_at"] > ttl:
            return None
        return blob["data"]

    def get(self, path: str, ttl: float | None = None, use_cache: bool = True) -> Any:
        """GET ``{BASE_URL}/{path}``; ``ttl=None`` means a cached copy never expires."""
        path = path.strip("/") + "/"
        if use_cache:
            cached = self.read_cache(path, ttl)
            if cached is not None:
                return cached

        url = f"{BASE_URL}/{path}"
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._throttle()
            self.network_calls += 1
            try:
                resp = self.session.get(url, timeout=30)
            except requests.RequestException as exc:  # network blip
                last_err = exc
            else:
                if resp.status_code == 200:
                    data = resp.json()
                    fp = self._cache_path(path)
                    fp.parent.mkdir(parents=True, exist_ok=True)
                    fp.write_text(json.dumps({"fetched_at": self._clock(), "url": url,
                                              "data": data}), encoding="utf-8")
                    return data
                if resp.status_code not in (429, 500, 502, 503, 504):
                    raise FPLAPIError(f"GET {url} -> HTTP {resp.status_code}")
                last_err = FPLAPIError(f"GET {url} -> HTTP {resp.status_code}")
            if attempt < self.max_retries:
                self._sleep(self.min_interval * 2 ** (attempt + 1))
        raise FPLAPIError(f"GET {url} failed after {self.max_retries + 1} attempts") from last_err

    # -- endpoints ------------------------------------------------------------
    def bootstrap_static(self, use_cache: bool = True) -> dict:
        return self.get("bootstrap-static", TTL["bootstrap-static"], use_cache)

    def fixtures(self, use_cache: bool = True) -> list[dict]:
        return self.get("fixtures", TTL["fixtures"], use_cache)

    def element_summary(self, element_id: int, use_cache: bool = True) -> dict:
        return self.get(f"element-summary/{int(element_id)}", TTL["element-summary"], use_cache)

    def entry_picks(self, entry_id: int, gw: int, use_cache: bool = True) -> dict:
        return self.get(f"entry/{int(entry_id)}/event/{int(gw)}/picks", TTL["entry"], use_cache)

    def entry_transfers(self, entry_id: int, use_cache: bool = True) -> list[dict]:
        return self.get(f"entry/{int(entry_id)}/transfers", TTL["entry"], use_cache)


# -- helpers ------------------------------------------------------------------
def season_from_bootstrap(bootstrap: dict) -> str:
    """Infer '2026-27' from the first event deadline."""
    first = min(bootstrap["events"], key=lambda e: e["id"])
    year = pd.Timestamp(first["deadline_time"]).year
    return f"{year}-{(year + 1) % 100:02d}"


def next_event(bootstrap: dict) -> dict | None:
    return next((e for e in bootstrap["events"] if e.get("is_next")), None)


def last_finished_gw(bootstrap: dict) -> int:
    done = [e["id"] for e in bootstrap["events"] if e.get("finished")]
    return max(done) if done else 0


def elements_frame(bootstrap: dict) -> pd.DataFrame:
    els = pd.DataFrame(bootstrap["elements"])
    teams = pd.DataFrame(bootstrap["teams"])[["id", "name", "short_name"]]
    els["position"] = els["element_type"].map(config.ELEMENT_TYPE_TO_POSITION)
    els["team_name"] = els["team"].map(teams.set_index("id")["name"])
    els["name"] = (els["first_name"].fillna("") + " " + els["second_name"].fillna("")).str.strip()
    return els


def fixtures_frame(fixtures: list[dict]) -> pd.DataFrame:
    fx = pd.DataFrame(fixtures)
    keep = ["id", "event", "kickoff_time", "team_h", "team_a", "team_h_score",
            "team_a_score", "finished", "team_h_difficulty", "team_a_difficulty"]
    fx = fx[[c for c in keep if c in fx.columns]].copy()
    fx["kickoff_time"] = pd.to_datetime(fx["kickoff_time"], utc=True, errors="coerce")
    return fx


def history_to_merged_gw(histories: Iterable[dict], bootstrap: dict) -> pd.DataFrame:
    """Turn element-summary ``history`` rows into an archive-shaped raw frame."""
    rows = [h for hist in histories for h in hist]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    els = elements_frame(bootstrap).set_index("id")
    df["name"] = df["element"].map(els["name"])
    df["position"] = df["element"].map(els["position"])
    df["team"] = df["element"].map(els["team_name"])
    df["GW"] = df["round"]
    return df


def build_season_frame(client: FPLClient, through_gw: int | None = None,
                       element_ids: Iterable[int] | None = None,
                       progress_every: int = 50) -> pd.DataFrame:
    """Current-season per-fixture rows in :data:`archive.CANONICAL_COLUMNS` form.

    One ``element-summary`` request per player (cached), so a cold run takes
    roughly ``n_players * min_interval`` seconds.
    """
    bootstrap = client.bootstrap_static()
    season = season_from_bootstrap(bootstrap)
    ids = list(element_ids) if element_ids is not None else [e["id"] for e in bootstrap["elements"]]
    histories = []
    for i, eid in enumerate(ids, 1):
        histories.append(client.element_summary(eid).get("history", []))
        if progress_every and i % progress_every == 0:
            logger.info("element-summary %d/%d", i, len(ids))
    raw = history_to_merged_gw(histories, bootstrap)
    if raw.empty:
        return pd.DataFrame(columns=archive.CANONICAL_COLUMNS)
    players_raw = pd.DataFrame(bootstrap["elements"])[["id", "code"]]
    teams = pd.DataFrame(bootstrap["teams"])[["id", "name"]]
    df = archive.normalize_merged_gw(raw, season, players_raw=players_raw, teams=teams)
    if through_gw is not None:
        df = df[df["gw"] <= through_gw].reset_index(drop=True)
    return df


def load_current_season(client: FPLClient | None = None, cache_parquet: Path | None = None,
                        max_age_seconds: float = 12 * 3600) -> pd.DataFrame:
    """Finished-GW rows of the live season, cached as a parquet for reuse."""
    client = client or FPLClient()
    bootstrap = client.bootstrap_static()
    season = season_from_bootstrap(bootstrap)
    finished = last_finished_gw(bootstrap)
    out = Path(cache_parquet or config.DATA_DIR / "live" / f"{season}_merged_gw.parquet")
    if out.exists() and time.time() - out.stat().st_mtime < max_age_seconds:
        df = pd.read_parquet(out)
        if int(df["gw"].max()) >= finished:
            return df
    df = build_season_frame(client, through_gw=finished)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return df


if __name__ == "__main__":  # pragma: no cover - manual helper
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    frame = load_current_season()
    print(f"{frame['season'].iloc[0]}: {len(frame)} rows through GW {frame['gw'].max()}")

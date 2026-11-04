import pytest

from fpl.ingest import fpl_api
from fpl.ingest.fpl_api import FPLAPIError, FPLClient


class FakeClock:
    def __init__(self, t=1_000_000.0):
        self.t = t
        self.sleeps = []

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s


class FakeResponse:
    def __init__(self, status, payload=None):
        self.status_code = status
        self._payload = payload

    def json(self):
        return self._payload


class FakeSession:
    """Routes URLs (minus the base) to a queue of responses."""

    def __init__(self, routes):
        self.routes = {k: list(v) for k, v in routes.items()}
        self.headers = {}
        self.calls = []

    def get(self, url, timeout=None):
        path = url.replace(fpl_api.BASE_URL + "/", "")
        self.calls.append(path)
        queue = self.routes[path]
        return queue.pop(0) if len(queue) > 1 else queue[0]


def _client(tmp_path, routes, clock=None):
    clock = clock or FakeClock()
    return FPLClient(cache_dir=tmp_path, session=FakeSession(routes), clock=clock,
                     sleep=clock.sleep), clock


def test_cache_hit_avoids_network_and_ttl_expiry_refetches(tmp_path):
    client, clock = _client(tmp_path, {"fixtures/": [FakeResponse(200, [{"id": 1}])]})
    assert client.fixtures() == [{"id": 1}]
    assert client.fixtures() == [{"id": 1}]
    assert client.network_calls == 1
    clock.t += fpl_api.TTL["fixtures"] + 1
    client.fixtures()
    assert client.network_calls == 2


def test_requests_are_spaced_by_min_interval(tmp_path):
    client, clock = _client(tmp_path, {
        "element-summary/1/": [FakeResponse(200, {"history": []})],
        "element-summary/2/": [FakeResponse(200, {"history": []})],
    })
    client.element_summary(1)
    client.element_summary(2)
    assert clock.sleeps == [pytest.approx(1.0)]


def test_retries_on_429_then_succeeds(tmp_path):
    client, clock = _client(tmp_path, {
        "bootstrap-static/": [FakeResponse(429), FakeResponse(200, {"ok": True})]})
    assert client.bootstrap_static() == {"ok": True}
    assert client.network_calls == 2


def test_non_retryable_status_raises(tmp_path):
    client, _ = _client(tmp_path, {"entry/5/transfers/": [FakeResponse(404)]})
    with pytest.raises(FPLAPIError):
        client.entry_transfers(5)


BOOTSTRAP = {
    "events": [
        {"id": 1, "deadline_time": "2026-08-21T17:30:00Z", "finished": True, "is_next": False},
        {"id": 2, "deadline_time": "2026-08-28T17:30:00Z", "finished": False, "is_next": True},
    ],
    "teams": [{"id": 1, "name": "Alpha", "short_name": "ALP"},
              {"id": 2, "name": "Bravo", "short_name": "BRA"}],
    "elements": [
        {"id": 7, "code": 70, "first_name": "Ann", "second_name": "Ay", "element_type": 3, "team": 1},
        {"id": 8, "code": 80, "first_name": "Bo", "second_name": "Bee", "element_type": 1, "team": 2},
    ],
}


def _hist(element, opp, home, pts, minutes):
    return {"element": element, "fixture": 1, "opponent_team": opp, "total_points": pts,
            "was_home": home, "kickoff_time": "2026-08-22T14:00:00Z", "round": 1,
            "minutes": minutes, "goals_scored": 0, "value": 55, "expected_goals": "0.10"}


def test_build_season_frame_matches_archive_schema(tmp_path):
    from fpl.ingest.archive import CANONICAL_COLUMNS

    client, _ = _client(tmp_path, {
        "bootstrap-static/": [FakeResponse(200, BOOTSTRAP)],
        "element-summary/7/": [FakeResponse(200, {"history": [_hist(7, 2, True, 5, 90)]})],
        "element-summary/8/": [FakeResponse(200, {"history": [_hist(8, 1, False, 1, 90)]})],
    })
    df = fpl_api.build_season_frame(client, through_gw=1)
    assert list(df.columns) == CANONICAL_COLUMNS
    row = df.set_index("element").loc[7]
    assert (row["season"], row["position"], row["team"], row["opponent"], row["code"]) == \
        ("2026-27", "MID", "Alpha", "Bravo", 70)
    assert df.set_index("element").loc[8, "position"] == "GKP"
    assert df["expected_goals"].dtype.kind == "f"


def test_bootstrap_helpers():
    assert fpl_api.season_from_bootstrap(BOOTSTRAP) == "2026-27"
    assert fpl_api.next_event(BOOTSTRAP)["id"] == 2
    assert fpl_api.last_finished_gw(BOOTSTRAP) == 1

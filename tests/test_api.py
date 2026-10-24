import logging

import httpx
from fastapi.testclient import TestClient

from app.espn.client import EspnClient
from app.main import create_app, current_season
from tests.conftest import LEAGUE_ID, MY_TEAM, FakeEspn, signed_in

S2 = "AEB-secret-s2-value-1234567890"
SWID = "{SECRET-SWID-0000-1111}"
COOKIES = {"X-ESPN-S2": S2, "X-ESPN-SWID": SWID}


def test_league_summary(client):
    r = client.post("/api/league", json={"league_id": LEAGUE_ID, "season": 2027})
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "Chill Primera Division"
    assert body["supported"] is True
    assert len(body["teams"]) == 10


def test_analyze_and_cache(client, fake_espn):
    r = client.post("/api/analyze", json={"league_id": LEAGUE_ID, "team_id": MY_TEAM, "season": 2027})
    assert r.status_code == 200
    body = r.json()
    assert body["team_name"] == "Kucukesat Aspavas"
    calls = fake_espn.calls
    again = client.post("/api/analyze", json={"league_id": LEAGUE_ID, "team_id": MY_TEAM, "season": 2027})
    assert again.status_code == 200
    assert fake_espn.calls == calls  # served from cache


def test_analysis_is_only_for_the_verified_manager(client, fake_espn):
    app = client.app
    other = client.post("/api/analyze", json={"league_id": LEAGUE_ID, "team_id": 8, "season": 2027})
    assert other.status_code == 403 and other.json()["detail"]["code"] == "not_your_team"
    unverified = signed_in(app, team_id=None)
    r = unverified.post("/api/analyze", json={"league_id": LEAGUE_ID, "team_id": MY_TEAM, "season": 2027})
    assert r.status_code == 403 and r.json()["detail"]["code"] == "not_linked"
    anonymous = TestClient(app)
    r = anonymous.post("/api/analyze", json={"league_id": LEAGUE_ID, "team_id": MY_TEAM, "season": 2027})
    assert r.status_code == 401 and r.json()["detail"]["code"] == "not_signed_in"
    assert anonymous.post("/api/league", json={"league_id": LEAGUE_ID}).status_code == 401


def test_unknown_league(client):
    r = client.post("/api/league", json={"league_id": 1, "season": 2027})
    assert r.status_code == 404
    assert r.json()["detail"]["code"] == "league_not_found"


def test_only_one_cookie_is_rejected(client):
    r = client.post("/api/league", json={"league_id": LEAGUE_ID}, headers={"X-ESPN-S2": S2})
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "incomplete_cookies"


def test_private_league_needs_cookies_and_never_leaks_them(recorded, caplog):
    fake = FakeEspn(recorded, private=True)
    client = signed_in(create_app(EspnClient(transport=httpx.MockTransport(fake.handler))))
    caplog.set_level(logging.DEBUG)

    r = client.post("/api/league", json={"league_id": LEAGUE_ID, "season": 2027})
    assert r.status_code == 401
    assert r.json()["detail"]["code"] == "private_league"

    r = client.post("/api/analyze", json={"league_id": LEAGUE_ID, "team_id": MY_TEAM, "season": 2027}, headers=COOKIES)
    assert r.status_code == 200
    # The cookies reached ESPN...
    assert any(S2 in c and SWID in c for c in fake.seen_cookies)
    # ...but never the logs or the response.
    assert S2 not in caplog.text and SWID not in caplog.text
    assert S2 not in r.text and SWID not in r.text


def test_private_cache_is_per_user(recorded):
    fake = FakeEspn(recorded, private=True)
    client = signed_in(create_app(EspnClient(transport=httpx.MockTransport(fake.handler))))
    body = {"league_id": LEAGUE_ID, "season": 2027}
    assert client.post("/api/league", json=body, headers=COOKIES).status_code == 200
    # Same league without cookies must not be served from the cookie holder's cache.
    assert client.post("/api/league", json=body).status_code == 401


def test_espn_failure_does_not_echo_cookies(recorded, caplog):
    def broken(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    client = signed_in(create_app(EspnClient(transport=httpx.MockTransport(broken))))
    caplog.set_level(logging.DEBUG)
    r = client.post("/api/league", json={"league_id": LEAGUE_ID}, headers=COOKIES)
    assert r.status_code == 502
    assert r.json()["detail"]["code"] == "espn_unavailable"
    assert S2 not in caplog.text and SWID not in caplog.text and S2 not in r.text


def test_current_season():
    from datetime import datetime

    assert current_season(datetime(2026, 10, 22)) == 2027
    assert current_season(datetime(2027, 3, 1)) == 2027
    assert current_season(datetime(2027, 8, 1)) == 2028

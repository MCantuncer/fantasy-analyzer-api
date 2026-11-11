import logging
import time

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


def test_analysis_job_runs_in_the_background(client, fake_espn):
    body = {"league_id": LEAGUE_ID, "team_id": MY_TEAM, "season": 2027}
    with client:  # one event loop for every request, as on the real server
        first = client.post("/api/analyze/job", json=body)
        assert first.status_code == 200
        job = first.json()
        assert job["status"] in {"running", "done"} and 0 <= job["progress"] <= 1
        seen = [job["progress"]]
        deadline = time.monotonic() + 120
        while job["status"] == "running" and time.monotonic() < deadline:
            time.sleep(0.2)
            job = client.post("/api/analyze/job", json=body).json()
            seen.append(job["progress"])
            if job.get("preview"):  # free agents shown while trades are screened
                assert job["preview"]["pending"] and not job["preview"]["trades"]
        assert job["status"] == "done" and job["progress"] == 1.0
        assert job["report"]["pending"] == [] and job.get("preview") is None
        assert job["report"]["team_name"] == "Kucukesat Aspavas"
        assert seen == sorted(seen)  # progress never goes backwards
        # The blocking endpoint now serves the same cached report.
        calls = fake_espn.calls
        assert client.post("/api/analyze", json=body).json()["generated_at"] == job["report"]["generated_at"]
        assert fake_espn.calls == calls


def test_analysis_job_is_only_for_the_verified_manager(client):
    r = client.post("/api/analyze/job", json={"league_id": LEAGUE_ID, "team_id": 8, "season": 2027})
    assert r.status_code == 403 and r.json()["detail"]["code"] == "not_your_team"
    r = TestClient(client.app).post("/api/analyze/job", json={"league_id": LEAGUE_ID, "team_id": MY_TEAM})
    assert r.status_code == 401


def test_failed_analysis_job_reports_the_error_then_retries(client, monkeypatch):
    import app.main as main

    def boom(self):
        raise RuntimeError("secret internals")

    monkeypatch.setattr(main.Analyzer, "report", boom)
    body = {"league_id": LEAGUE_ID, "team_id": MY_TEAM, "season": 2027}
    with client:
        job = client.post("/api/analyze/job", json=body).json()
        deadline = time.monotonic() + 30
        while job["status"] == "running" and time.monotonic() < deadline:
            time.sleep(0.1)
            job = client.post("/api/analyze/job", json=body).json()
        assert job["status"] == "failed" and job["error"]["code"] == "analysis_failed"
        assert "secret" not in str(job)
        # The next call starts a fresh run.
        assert client.post("/api/analyze/job", json=body).json()["status"] in {"running", "failed"}


def test_analysis_is_only_for_the_verified_manager(client, fake_espn):
    app = client.app
    other = client.post("/api/analyze", json={"league_id": LEAGUE_ID, "team_id": 8, "season": 2027})
    assert other.status_code == 403 and other.json()["detail"]["code"] == "not_your_team"
    # A verified team is closed to everyone else, also to an account that only added the league.
    follower = signed_in(app, team_id=None, follows=True)
    r = follower.post("/api/analyze", json={"league_id": LEAGUE_ID, "team_id": MY_TEAM, "season": 2027})
    assert r.status_code == 403 and r.json()["detail"]["code"] == "not_your_team"
    # A team nobody verified needs the league in your leagues first.
    stranger = signed_in(app, team_id=None)
    r = stranger.post("/api/analyze", json={"league_id": LEAGUE_ID, "team_id": 8, "season": 2027})
    assert r.status_code == 403 and r.json()["detail"]["code"] == "not_following"
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


def test_matchup_projection_needs_a_verified_team(client, fake_espn):
    res = client.post("/api/matchup-projection", json={"league_id": LEAGUE_ID})
    assert res.status_code == 200
    body = res.json()
    assert body["wins"] + body["losses"] == len(body["categories"])
    assert {c["winner"] for c in body["categories"]} <= {"me", "them"}
    unverified = signed_in(client.app, team_id=None)
    assert unverified.post("/api/matchup-projection", json={"league_id": LEAGUE_ID}).status_code == 403


def test_a_team_nobody_verified_opens_to_an_account_that_added_the_league(client, fake_espn):
    """Before the ESPN code check a new manager picks their team and sees its analysis."""
    follower = signed_in(client.app, team_id=None, follows=True)
    picked = {"league_id": LEAGUE_ID, "season": 2027, "team_id": 8}
    res = follower.post("/api/matchup-projection", json=picked)
    assert res.status_code == 200 and res.json()["opponent_id"] != 8
    with follower:
        job = follower.post("/api/analyze/job", json=picked)
        assert job.status_code == 200 and job.json()["status"] in {"running", "done"}
    # The team its manager verified stays closed, and so do the things that need a verified seat.
    mine = {**picked, "team_id": MY_TEAM}
    for path in ("/api/matchup-projection", "/api/analyze/job", "/api/shop", "/api/target"):
        r = follower.post(path, json={**mine, "offer_player_ids": [1], "player_id": 1})
        assert r.status_code == 403 and r.json()["detail"]["code"] == "not_your_team", path
    r = follower.post("/api/trade-rules/mine", json=picked)
    assert r.status_code == 403 and r.json()["detail"]["code"] == "not_linked"
    # Verifying a team closes it: the follower loses team 8 once its manager shows up.
    signed_in(client.app, team_id=8)
    assert follower.post("/api/matchup-projection", json=picked).status_code == 403
    # A verified manager cannot look into teams nobody verified either.
    assert client.post("/api/matchup-projection", json={**picked, "team_id": 4}).status_code == 403


def test_projection_reuses_the_analysis_league(client, fake_espn):
    body = {"league_id": LEAGUE_ID, "team_id": MY_TEAM, "season": 2027}
    assert client.post("/api/analyze", json=body).status_code == 200
    calls = fake_espn.calls
    res = client.post("/api/matchup-projection", json={"league_id": LEAGUE_ID, "season": 2027})
    assert res.status_code == 200 and res.json()["categories"]
    assert fake_espn.calls == calls  # no second league load


def test_one_league_load_at_a_time(fake_espn):
    """Two pages asking for the same league at once share one load from ESPN."""
    import asyncio

    from app.accounts import MemoryAccountStore
    from app.store import MemoryStore

    async def slow(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.05)  # like the real network: the second request arrives mid-load
        return fake_espn.handler(request)

    app = create_app(
        EspnClient(transport=httpx.MockTransport(slow)), store=MemoryStore(), accounts=MemoryAccountStore()
    )
    headers = dict(signed_in(app).headers)

    async def both():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t", headers=headers) as c:
            body = {"league_id": LEAGUE_ID, "season": 2027}
            return await asyncio.gather(
                c.post("/api/matchup-projection", json=body), c.post("/api/matchup-projection", json=body)
            )

    a, b = asyncio.run(both())
    assert a.status_code == b.status_code == 200 and a.json() == b.json()
    league_calls = [u for u in fake_espn.seen_urls if f"/leagues/{LEAGUE_ID}" in u and "kona_player_info" not in u]
    assert len(league_calls) == 1

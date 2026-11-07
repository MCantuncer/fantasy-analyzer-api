import copy
import os
import time
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from app.accounts import MemoryAccountStore
from app.cache import credentials_key
from app.espn.client import EspnClient
from app.main import create_app
from app.saved import SAVED_FRESH, changes, fingerprint, read_saved, report_key
from app.store import MemoryStore, PostgresStore
from tests.conftest import LEAGUE_ID, MY_TEAM, FakeEspn, signed_in

BODY = {"league_id": LEAGUE_ID, "team_id": MY_TEAM, "season": 2027}


def raw_entries(raw: dict, team_id: int) -> list[dict]:
    team = next(t for t in raw["teams"] if t["id"] == team_id)
    return team["roster"]["entries"]


def raw_player(entry: dict) -> dict:
    return entry["playerPoolEntry"]["player"]


def healthy_player(raw: dict, team_id: int) -> dict:
    return next(raw_player(e) for e in raw_entries(raw, team_id) if raw_player(e).get("injuryStatus") == "ACTIVE")


def run_job(client, body=BODY) -> list[dict]:
    """Poll the analysis job until it is done; every answer, first to last."""
    seen = [client.post("/api/analyze/job", json=body).json()]
    deadline = time.monotonic() + 120
    while seen[-1]["status"] == "running" and time.monotonic() < deadline:
        time.sleep(0.2)
        seen.append(client.post("/api/analyze/job", json=body).json())
    assert seen[-1]["status"] == "done", seen[-1].get("error")
    return seen


def fresh_app(recorded, pro_schedule, store, now=None):
    """A new server process (nothing cached in memory) sharing ``store`` with the last one."""
    fake = FakeEspn(recorded, schedule=pro_schedule)
    espn = EspnClient(transport=httpx.MockTransport(fake.handler))
    app = create_app(espn, store=store, accounts=MemoryAccountStore())
    if now is not None:
        app.state.now = lambda: now
    return signed_in(app), fake


@pytest.fixture(scope="module")
def first_run(recorded, pro_schedule):
    """One full analysis, saved to a store the tests below reuse."""
    store = MemoryStore()
    client, _ = fresh_app(recorded, pro_schedule, store)
    with client:
        report = run_job(client)[-1]["report"]
    return store, report


@pytest.fixture
def saved(first_run):
    """The first run's store, copied so a test that saves a new report leaves the others alone."""
    store, report = first_run
    return copy.deepcopy(store), report


def test_changes_in_plain_words(league):
    before = fingerprint(league)
    after = copy.deepcopy(before)
    mine = after["teams"][str(MY_TEAM)]["players"]
    hurt_id = next(pid for pid, p in mine.items() if p[1] == "ACTIVE")
    mine[hurt_id][1] = "OUT"
    other = next(tid for tid in after["teams"] if tid != str(MY_TEAM))
    moved_id, moved = next(iter(after["teams"][other]["players"].items()))
    del after["teams"][other]["players"][moved_id]
    after["teams"][str(MY_TEAM)]["players"][moved_id] = moved
    for p in after["teams"][other]["players"].values():
        p[3] += 1
    out = changes(before, after, MY_TEAM)
    hurt = mine[hurt_id][0]
    assert out[0].text == f"{hurt} is out" and out[0].mine and out[0].kind == "status"
    assert out[1].kind == "moved" and out[1].mine and moved[0] in out[1].text
    assert out[-1].kind == "games" and not out[-1].mine
    assert changes(before, before, MY_TEAM) == []


def test_healthy_again_and_ir(league):
    before = fingerprint(league)
    after = copy.deepcopy(before)
    pid, p = next(iter(after["teams"]["1"]["players"].items()))
    before["teams"]["1"]["players"][pid][1] = "DAY_TO_DAY"
    p[1], p[2] = "ACTIVE", not p[2]
    (change,) = changes(before, after, MY_TEAM)
    assert change.text == f"{p[0]} is healthy again" and not change.mine and change.team_id == 1


def test_saved_report_opens_at_once_after_a_restart(recorded, pro_schedule, saved):
    store, report = saved
    client, fake = fresh_app(recorded, pro_schedule, store)
    with client:
        job = client.post("/api/analyze/job", json=BODY).json()
    assert job["status"] == "done"
    assert job["report"]["generated_at"] == report["generated_at"]
    # One light league read to check nothing moved; no free agents, no NBA schedule.
    assert fake.calls == 1


def test_injury_news_starts_a_new_run_and_shows_the_old_report_meanwhile(recorded, pro_schedule, saved):
    store, report = saved
    news = copy.deepcopy(recorded)
    hurt = healthy_player(news["league"], MY_TEAM)
    hurt["injuryStatus"] = "OUT"
    client, _ = fresh_app(news, pro_schedule, store)
    with client:
        seen = run_job(client)
    first = seen[0]
    assert first["status"] == "running"
    assert first["saved"]["generated_at"] == report["generated_at"]
    assert first["changes"][0] == {
        "kind": "status",
        "text": f"{hurt['fullName']} is out",
        "player_id": hurt["id"],
        "team_id": MY_TEAM,
        "mine": True,
    }
    done = seen[-1]
    assert done["report"]["generated_at"] != report["generated_at"]
    assert done.get("saved") is None and done["changes"] == []


def test_an_old_report_is_rebuilt_and_shown_meanwhile(recorded, pro_schedule, saved):
    store, report = saved
    later = datetime.fromisoformat(report["generated_at"]).astimezone(UTC) + SAVED_FRESH + timedelta(minutes=5)
    client, _ = fresh_app(recorded, pro_schedule, store, now=later)
    with client:
        seen = run_job(client)
    assert seen[0]["status"] == "running" and seen[0]["saved"]["generated_at"] == report["generated_at"]
    assert all(s["changes"] == [] for s in seen)  # same rosters and tags, only older


def test_a_store_failure_never_fails_the_analysis(recorded, pro_schedule):
    class Broken(MemoryStore):
        def get_report(self, key):
            raise RuntimeError("database down")

        def save_report(self, *args):
            raise RuntimeError("database down")

    client, _ = fresh_app(recorded, pro_schedule, Broken())
    with client:
        assert run_job(client)[-1]["report"]["team_name"] == "Kucukesat Aspavas"


def test_saved_reports_are_keyed_by_settings(first_run):
    store, _ = first_run
    assert list(store.reports) == [report_key((LEAGUE_ID, 2027, MY_TEAM, (), (), credentials_key(None)))]


@pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="set TEST_DATABASE_URL to test against Postgres")
def test_postgres_keeps_saved_reports(first_run):
    memory, _ = first_run
    ((key, row),) = memory.reports.items()
    store = PostgresStore(os.environ["TEST_DATABASE_URL"])
    store.save_report(key, LEAGUE_ID, 2027, MY_TEAM, row)
    store.save_report(key, LEAGUE_ID, 2027, MY_TEAM, row)  # a rebuild replaces it
    back = store.get_report(key)
    assert back["body"] == row["body"] and back["fingerprint"] == row["fingerprint"]
    built = read_saved(row).built_at
    assert [r["key"] for r in store.recent_reports(built - timedelta(minutes=1))] == [key]
    assert key not in [r["key"] for r in store.recent_reports(built + timedelta(minutes=1))]
    assert read_saved(back).built_at == read_saved(row).built_at
    assert store.get_report("no such key") is None


class WatchedEspn(FakeEspn):
    """Also answers ESPN's league-wide player list (injury tags), read from the league's rosters."""

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/seasons/2027/players"):
            self.calls += 1
            players = [
                {"player": {k: raw_player(e).get(k) for k in ("id", "fullName", "proTeamId", "injuryStatus")}}
                for t in self.recorded["league"]["teams"]
                for e in t["roster"]["entries"]
            ]
            return httpx.Response(200, json=players)
        return super().handler(request)


def test_injury_watch_rebuilds_reports_its_news_touches(recorded, pro_schedule, saved):
    import asyncio

    store, report = saved
    news = copy.deepcopy(recorded)
    fake = WatchedEspn(news, schedule=pro_schedule)
    espn = EspnClient(transport=httpx.MockTransport(fake.handler))
    app = create_app(espn, store=store, accounts=MemoryAccountStore())
    key = (LEAGUE_ID, 2027, MY_TEAM, (), (), credentials_key(None))
    sweep = app.state.injury_sweep

    async def run() -> None:
        assert await sweep() == []  # the first look is only a baseline
        calls = fake.calls
        assert await sweep() == [] and fake.calls == calls + 1  # nothing moved: one light ESPN call
        app.state.reports.set(key, "stale")  # what a manager opened a minute ago
        hurt = healthy_player(news["league"], MY_TEAM)
        hurt["injuryStatus"] = "OUT"
        assert await sweep() == [key]
        fresh = app.state.reports.get(key)
        assert fresh != "stale" and fresh.generated_at != report["generated_at"]
        assert store.reports[report_key(key)]["body"]["generated_at"] == fresh.generated_at
        job = app.state.analysis_jobs[key]
        assert job.changes[0].text == f"{hurt['fullName']} is out" and job.changes[0].mine

    asyncio.run(run())


def test_injury_watch_stays_quiet_without_recent_reports(recorded, pro_schedule):
    import asyncio

    fake = WatchedEspn(recorded, schedule=pro_schedule)
    app = create_app(
        EspnClient(transport=httpx.MockTransport(fake.handler)), store=MemoryStore(), accounts=MemoryAccountStore()
    )
    assert asyncio.run(app.state.injury_sweep()) == [] and fake.calls == 0


def test_recent_reports_skip_old_ones(first_run):
    store, report = first_run
    built = datetime.fromisoformat(next(iter(store.reports.values()))["built_at"])
    (row,) = store.recent_reports(built - timedelta(minutes=1))
    assert row["team_id"] == MY_TEAM and row["fingerprint"] and "body" not in row
    assert store.recent_reports(built + timedelta(minutes=1)) == []

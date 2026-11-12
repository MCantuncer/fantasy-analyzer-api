"""The public sample league: the daily run and the endpoints visitors read before signing in."""

import os
import time
from datetime import UTC, date, datetime, timedelta

import httpx
import pytest
from fastapi.testclient import TestClient

from app.accounts import MemoryAccountStore
from app.analysis.report import Analyzer
from app.espn.client import EspnClient
from app.main import create_app
from app.sample import PLAY_CALLS
from app.schemas import AnalysisReport
from app.store import SAMPLE_INDEX, MemoryStore, PostgresStore

from .conftest import LEAGUE_ID, MY_TEAM

NOW = datetime(2026, 11, 12, 9, 0, tzinfo=UTC)
TODAY = NOW.date().isoformat()


@pytest.fixture(scope="module")
def report(league) -> AnalysisReport:
    return Analyzer(league, MY_TEAM).report()


def stub_analyse(report: AnalysisReport):
    """The real report for every seat (a full run per team takes minutes): only the seat changes."""

    def analyse(lg, team_id: int) -> AnalysisReport:
        return report.model_copy(update={"team_id": team_id, "team_name": lg.team(team_id).name})

    return analyse


def sample_app(fake_espn, report, league_id: int | None = LEAGUE_ID, now: datetime = NOW):
    app = create_app(
        EspnClient(transport=httpx.MockTransport(fake_espn.handler)), store=MemoryStore(), accounts=MemoryAccountStore()
    )
    app.state.sample_league_id = league_id
    app.state.sample_season = 2027
    app.state.now = lambda: now
    app.state.sample_analyse = stub_analyse(report)
    return app


def run_today(client: TestClient) -> None:
    assert client.get("/api/sample/refresh").json() == {"ok": True, "running": True}
    for _ in range(600):
        job = client.app.state.sample_job
        if job is not None and job.done():
            break
        time.sleep(0.05)
    assert client.app.state.sample_job.exception() is None


def test_without_a_sample_league_everything_says_sample_off(fake_espn, report):
    with TestClient(sample_app(fake_espn, report, league_id=None)) as client:
        assert client.get("/api/sample/refresh").json() == {"ok": True, "running": False}
        for path in ("/api/sample", "/api/sample/teams/3", f"/api/sample/teams/3?day={TODAY}"):
            res = client.get(path)
            assert res.status_code == 404
            assert res.json()["detail"]["code"] == "sample_off"
    assert fake_espn.calls == 0


def test_before_the_first_run_there_is_no_sample(fake_espn, report):
    with TestClient(sample_app(fake_espn, report)) as client:
        assert client.get("/api/sample").json()["detail"]["code"] == "sample_off"


def test_the_daily_run_serves_a_masked_public_sample(fake_espn, report, league):
    with TestClient(sample_app(fake_espn, report)) as client:
        run_today(client)
        res = client.get("/api/sample")
        assert res.status_code == 200
        assert res.headers["cache-control"] == "public, max-age=300, s-maxage=3600, stale-while-revalidate=86400"
        assert "set-cookie" not in res.headers
        index = res.json()
        assert index["analysed_on"] == TODAY and index["fresh"] is True and index["phase"] == "season"
        assert index["season"] == 2027
        assert index["league"]["name"] == "Sample league"
        assert index["league"]["size"] == len(league.teams)
        ids = sorted(t.id for t in league.teams)
        assert [t["id"] for t in index["teams"]] == ids
        assert [t["name"] for t in index["teams"]] == [n for n, _ in PLAY_CALLS[: len(ids)]]
        assert sorted(t["projected_rank"] for t in index["teams"]) == list(range(1, len(ids) + 1))
        assert index["default_team_id"] in ids

        team = client.get(f"/api/sample/teams/{MY_TEAM}")
        assert team.status_code == 200 and team.headers["cache-control"].startswith("public")
        body = team.json()
        assert body["team_id"] == MY_TEAM
        assert body["team_name"] == next(t["name"] for t in index["teams"] if t["id"] == MY_TEAM)
        assert body["league"]["id"] == 0
        assert client.get(f"/api/sample/teams/{MY_TEAM}?day={TODAY}").json() == body

        text = res.text + team.text
        for real in league.teams:
            assert real.name not in text
        assert str(LEAGUE_ID) not in text


def test_a_sample_team_that_does_not_exist(fake_espn, report):
    with TestClient(sample_app(fake_espn, report)) as client:
        run_today(client)
        for path in (
            "/api/sample/teams/999",
            f"/api/sample/teams/{SAMPLE_INDEX}",
            "/api/sample/teams/3?day=2026-11-01",
        ):
            res = client.get(path)
            assert res.status_code == 404 and res.json()["detail"]["code"] == "team_not_found", path


def test_the_run_happens_once_a_day_and_needs_no_sign_in(fake_espn, report):
    with TestClient(sample_app(fake_espn, report)) as client:
        run_today(client)
        calls = fake_espn.calls
        assert client.get("/api/sample/refresh").json() == {"ok": True, "running": False}
        assert fake_espn.calls == calls


def test_a_failed_run_is_not_retried_straight_away(fake_espn, report):
    app = sample_app(fake_espn, report)

    def broken(lg, team_id):
        raise RuntimeError("boom")

    app.state.sample_analyse = broken
    with TestClient(app) as client:
        assert client.get("/api/sample/refresh").json()["running"] is True
        for _ in range(200):
            if app.state.sample_job.done():
                break
            time.sleep(0.05)
        assert client.get("/api/sample/refresh").json()["running"] is False
        assert client.get("/api/sample").status_code == 404  # half a run is never served


def test_a_late_run_is_still_served_and_says_so(fake_espn, report):
    app = sample_app(fake_espn, report)
    with TestClient(app) as client:
        run_today(client)
        app.state.now = lambda: NOW + timedelta(days=2)
        app.state.sample_reads.clear()
        index = client.get("/api/sample").json()
        assert index["analysed_on"] == TODAY and index["fresh"] is False


def test_until_the_new_season_runs_last_seasons_sample_shows(fake_espn, report):
    app = sample_app(fake_espn, report)
    with TestClient(app) as client:
        run_today(client)
        app.state.sample_season = None  # follow the calendar: 2027-28 starts in August 2027
        app.state.now = lambda: datetime(2027, 9, 1, tzinfo=UTC)
        app.state.sample_reads.clear()
        index = client.get("/api/sample").json()
        assert index["season"] == 2027 and index["phase"] == "offseason" and index["fresh"] is False
        assert client.get(f"/api/sample/teams/{MY_TEAM}?day={TODAY}").status_code == 200


@pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="set TEST_DATABASE_URL to test against Postgres")
def test_postgres_keeps_sample_runs():
    store = PostgresStore(os.environ["TEST_DATABASE_URL"])
    with store._connect() as conn:
        conn.execute("delete from sample_runs where season = 1999")
    today = date.today()
    month_ago, yesterday = (today - timedelta(days=30)).isoformat(), (today - timedelta(days=1)).isoformat()
    store.save_sample(1999, month_ago, 3, {"team": 3})
    store.save_sample(1999, today.isoformat(), 3, {"team": 3})
    assert store.sample_days(1999) == []  # a day counts once its index is saved
    store.save_sample(1999, yesterday, SAMPLE_INDEX, {"day": 1})
    store.save_sample(1999, today.isoformat(), SAMPLE_INDEX, {"day": 2})
    assert store.sample_days(1999) == [today.isoformat(), yesterday]
    assert store.get_sample(1999, today.isoformat(), 3) == {"team": 3}
    assert store.get_sample(1999, month_ago, 3) is None  # cleared when a newer index was saved

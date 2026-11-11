import logging
import os
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from fastapi.testclient import TestClient

from app import errors
from app.accounts import MemoryAccountStore
from app.errors import COLLECTOR, MemoryErrorStore, Occurrence, PostgresErrorStore, fingerprint, scrub
from app.espn.client import EspnClient
from app.main import create_app
from app.store import MemoryStore
from tests.conftest import signed_in

CRASH = {
    "kind": "crash",
    "name": "TypeError",
    "message": "Cannot read properties of undefined (reading 'name')\nmore lines",
    "stack": "TypeError: Cannot read ...\n    at Moves (https://site/assets/index-abc.js:1:2345)",
    "page": "/?league=744636428&season=2027&tab=moves",
    "build": "75490d4",
}


@pytest.fixture
def app(fake_espn, monkeypatch):
    monkeypatch.setenv("ADMIN_EMAILS", " Boss@Example.com ,other@example.com")
    accounts = MemoryAccountStore()
    app = create_app(
        EspnClient(transport=httpx.MockTransport(fake_espn.handler)),
        store=MemoryStore(),
        accounts=accounts,
        error_store=MemoryErrorStore(),
    )
    return app


def admin(app) -> TestClient:
    from app.social import hash_token

    user = app.state.accounts.create_user("boss@example.com", None)
    app.state.accounts.save_session(hash_token("admin-token"), user.id)
    return TestClient(app, headers={"Authorization": "Bearer admin-token"})


def test_scrub_masks_emails_and_tokens_but_keeps_paths():
    text = scrub(
        "user can@example.com sent AEB0123abcdefghijklmnopqrstuvwxyz%2B9 from "
        "/opt/render/project/src/.venv/lib/python3.12/site-packages/starlette/middleware/errors.py",
        500,
    )
    assert "can@example.com" not in text and "AEB0123" not in text
    assert "[email]" in text and "[token]" in text
    assert "site-packages/starlette/middleware/errors.py" in text
    assert len(scrub("x" * 50, 10)) == 10


def test_fingerprint_ignores_numbers():
    assert fingerprint("api", "KeyError in a.b", "league 12") == fingerprint("api", "KeyError in a.b", "league 99")
    assert fingerprint("api", "KeyError in a.b", "x") != fingerprint("web", "KeyError in a.b", "x")


def test_store_counts_repeats_and_reopens_fixed_ones():
    store = MemoryErrorStore()
    now = datetime.now(UTC)
    store.record(Occurrence("api", "KeyError in x", "x", "first", at=now - timedelta(minutes=5)))
    store.record(Occurrence("api", "KeyError in x", "x", "second", at=now))
    [r] = store.recent()
    assert r.count == 2 and r.message == "second" and r.first_seen < r.last_seen
    assert store.resolve(r.id) and store.recent()[0].resolved_at is not None
    store.record(Occurrence("api", "KeyError in x", "x", "third"))
    assert store.recent()[0].resolved_at is None and store.recent()[0].count == 3
    assert not store.resolve(999)


def test_store_forgets_reports_after_thirty_days():
    store = MemoryErrorStore()
    old = datetime.now(UTC) - timedelta(days=31)
    store.record(Occurrence("api", "Old in x", "x", "old", at=old))
    store.record(Occurrence("api", "New in y", "y", "new"))
    assert [r.title for r in store.recent()] == ["New in y"]


def test_logged_errors_become_reports_with_the_route(app):
    @app.get("/boom")
    async def boom() -> dict:
        try:
            {}["missing"]
        except KeyError:
            logging.getLogger("app.test").exception("lookup for league %s failed", 744636428)
        return {"ok": True}

    TestClient(app).get("/boom")
    TestClient(app).get("/boom")
    logging.getLogger("app.test").error("no traceback, not a report")
    COLLECTOR.wait()
    [r] = app.state.errors.recent()
    assert r.source == "api" and r.title == "KeyError in tests.test_errors.boom" and r.count == 2
    assert r.route == "GET /boom" and "lookup for league 744636428 failed" in r.message
    assert "Traceback" in r.stack and r.build == "dev"


def test_errors_outside_requests_are_background(app):
    try:
        raise ValueError("bad")
    except ValueError:
        logging.getLogger("app.jobs").exception("job failed")
    COLLECTOR.wait()
    assert app.state.errors.recent()[0].route == "background"


def test_uvicorn_errors_are_collected(app):
    """uvicorn's logger doesn't pass records to the root logger, so the collector hooks it too."""
    assert COLLECTOR in logging.getLogger("uvicorn").handlers
    assert logging.getLogger("").handlers.count(COLLECTOR) == 1


def test_website_crash_is_recorded_without_ids_or_query(app):
    client = TestClient(app)
    r = client.post("/api/errors", json=CRASH, headers={"User-Agent": "Mozilla/5.0 (iPhone)"})
    assert r.status_code == 202
    client.post("/api/errors", json=dict(CRASH, page="/?league=1694243943&tab=moves"))
    [rep] = app.state.errors.recent()
    assert rep.source == "web" and rep.count == 2
    assert rep.title == "TypeError: Cannot read properties of undefined (reading 'name')"
    assert rep.route == "/?tab=moves" and "744636428" not in rep.route
    assert rep.build == "75490d4"


def test_website_reports_are_rate_limited(app, monkeypatch):
    import app.main as main

    monkeypatch.setattr(main, "ERROR_POSTS_PER_WINDOW", 3)
    client = TestClient(app)
    codes = [client.post("/api/errors", json=CRASH).status_code for _ in range(4)]
    assert codes == [202, 202, 202, 429]


def test_new_website_errors_a_day_are_capped(app, monkeypatch):
    import app.main as main

    monkeypatch.setattr(main, "NEW_WEB_ERRORS_PER_DAY", 2)
    client = TestClient(app)
    for i in range(3):
        client.post("/api/errors", json=dict(CRASH, name=f"Error{'x' * i}"))
    client.post("/api/errors", json=dict(CRASH, name="Error"))  # a known one still counts
    reports = app.state.errors.recent()
    assert len(reports) == 2 and sum(r.count for r in reports) == 3


def test_only_admins_read_and_resolve_reports(app):
    TestClient(app).post("/api/errors", json=CRASH)
    assert TestClient(app).post("/api/admin/errors").status_code == 401
    someone = signed_in(app)
    assert someone.post("/api/admin/errors").status_code == 403
    assert someone.post("/api/me").json()["admin"] is False

    boss = admin(app)
    assert boss.post("/api/me").json()["admin"] is True
    [report] = boss.post("/api/admin/errors").json()["reports"]
    assert report["source"] == "web" and report["count"] == 1 and report["resolved"] is False
    assert someone.post("/api/admin/errors/resolve", json={"id": report["id"]}).status_code == 403
    assert boss.post("/api/admin/errors/resolve", json={"id": report["id"]}).status_code == 200
    assert boss.post("/api/admin/errors").json()["reports"][0]["resolved"] is True
    assert boss.post("/api/admin/errors/resolve", json={"id": 999}).status_code == 404


def test_no_admins_without_the_setting(fake_espn, monkeypatch):
    monkeypatch.delenv("ADMIN_EMAILS", raising=False)
    app = create_app(
        EspnClient(transport=httpx.MockTransport(fake_espn.handler)),
        store=MemoryStore(),
        accounts=MemoryAccountStore(),
        error_store=MemoryErrorStore(),
    )
    assert admin(app).post("/api/admin/errors").status_code == 403


def test_reporter_failures_are_swallowed(app, monkeypatch):
    def broken(_):
        raise RuntimeError("database down")

    monkeypatch.setattr(app.state.errors, "record", broken)
    try:
        raise ValueError("x")
    except ValueError:
        logging.getLogger("app.jobs").exception("job failed")
    COLLECTOR.wait()  # nothing raised, nothing looped


@pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="set TEST_DATABASE_URL to test against Postgres")
def test_postgres_store():
    store = PostgresErrorStore(os.environ["TEST_DATABASE_URL"])
    with store._connect() as conn:
        conn.execute("delete from error_reports")
    now = datetime.now(UTC)
    store.record(Occurrence("api", "KeyError in x", "x", "first", stack="s", at=now - timedelta(minutes=1)))
    store.record(Occurrence("api", "KeyError in x", "x", "second", at=now))
    store.record(Occurrence("web", "TypeError: y", "y", "web one"))
    assert store.known(fingerprint("api", "KeyError in x", "x")) and not store.known("nope")
    assert store.new_today("web") == 1
    api = next(r for r in store.recent() if r.source == "api")
    assert api.count == 2 and api.message == "second" and api.resolved_at is None
    assert store.resolve(api.id) and not store.resolve(10**9)
    store.record(Occurrence("api", "KeyError in x", "x", "again"))
    api = next(r for r in store.recent() if r.source == "api")
    assert api.count == 3 and api.resolved_at is None
    store.record(Occurrence("api", "Old in z", "z", "old", at=now - timedelta(days=40)))
    store.record(Occurrence("api", "KeyError in x", "x", "prunes"))
    assert all(r.title != "Old in z" for r in store.recent())
    errors.COLLECTOR.store = None

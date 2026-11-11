import os
from datetime import date, timedelta

import httpx
import pytest
from fastapi.testclient import TestClient

from app.accounts import MemoryAccountStore
from app.espn.client import EspnClient
from app.main import create_app
from app.store import MemoryStore
from app.usage import MemoryUsageStore, PostgresUsageStore, Usage
from tests.conftest import signed_in

DAY = date(2026, 11, 11)


@pytest.fixture
def store():
    return MemoryUsageStore()


@pytest.fixture
def app(fake_espn, store):
    return create_app(
        EspnClient(transport=httpx.MockTransport(fake_espn.handler)),
        store=MemoryStore(),
        accounts=MemoryAccountStore(),
        usage_store=store,
    )


def counts(store, day=None):
    rows = store.days(date.min)
    return rows[0] if day is None else next(r for r in rows if r.day == day)


def test_a_visitor_counts_once_a_day_and_hashes_change_daily(store):
    usage = Usage(store)
    usage.visit("1.2.3.4", "Safari", False, day=DAY)
    usage.visit("1.2.3.4", "Safari", True, day=DAY)
    usage.visit("5.6.7.8", "Chrome", False, day=DAY)
    usage.visit("1.2.3.4", "Safari", False, day=DAY + timedelta(days=1))
    usage.flush()
    assert (counts(store, DAY).visitors, counts(store, DAY).signed_in) == (2, 1)
    [(_, today_hash)] = [k for k in store.visitors if k[0] == DAY + timedelta(days=1)]
    yesterday = {k[1] for k in store.visitors if k[0] == DAY}
    assert today_hash not in yesterday  # a new salt every day: days can't be matched up
    assert "1.2.3.4" not in str(store.visitors)
    assert list(store.salts) == [DAY + timedelta(days=1)]  # yesterday's salt is gone


def test_counts_wait_in_memory_until_flushed(store):
    usage = Usage(store)
    usage.count("signup", day=DAY)
    usage.count("signup", day=DAY)
    assert store.days(date.min) == []
    usage.flush()
    usage.count("signup", day=DAY)
    usage.flush()
    assert counts(store).counts == {"signup": 3}


def test_a_failed_save_keeps_the_counts(store, monkeypatch):
    usage = Usage(store)
    usage.count("analysis", day=DAY)
    usage.visit("1.2.3.4", "Safari", False, day=DAY)
    real = store.save

    def down(*_):
        raise RuntimeError("database down")

    monkeypatch.setattr(store, "save", down)
    with pytest.raises(RuntimeError):
        usage.flush()
    monkeypatch.setattr(store, "save", real)
    usage.flush()
    assert counts(store).counts == {"analysis": 1} and counts(store).visitors == 1


def test_page_views_count_known_pages_only(app, store):
    client = TestClient(app, headers={"User-Agent": "Safari"})
    assert client.post("/api/usage", json={"page": "moves"}).status_code == 202
    client.post("/api/usage", json={"page": "moves"})
    client.post("/api/usage", json={"page": "scores"}, headers={"Authorization": "Bearer abc"})
    client.post("/api/usage", json={"page": "made-up-page"})
    assert client.post("/api/usage", json={"page": "x" * 41}).status_code == 422
    app.state.usage.flush()
    day = counts(store)
    assert day.counts == {"view:moves": 2, "view:scores": 1}
    assert (day.visitors, day.signed_in) == (1, 1)


def test_privacy_signals_count_the_view_but_not_the_visitor(app, store):
    client = TestClient(app)
    client.post("/api/usage", json={"page": "news"}, headers={"Sec-GPC": "1"})
    client.post("/api/usage", json={"page": "news"}, headers={"DNT": "1"})
    app.state.usage.flush()
    assert counts(store).counts == {"view:news": 2} and counts(store).visitors == 0


def test_server_events_are_counted(app, store):
    client = TestClient(app)
    client.post("/api/auth/signup", json={"email": "new@example.com", "password": "longenough"})
    client.post("/api/auth/login", json={"email": "new@example.com", "password": "longenough"})
    client.post("/api/auth/login", json={"email": "new@example.com", "password": "wrong-one"})
    signed_in(app).post("/api/analyze/job", json={"league_id": 744636428, "team_id": 3, "season": 2027})
    app.state.usage.flush()
    got = counts(store).counts
    assert got["signup"] == 1 and got["sign_in"] == 1 and got["analysis"] == 1


def test_counts_are_saved_on_shutdown(app, store):
    with TestClient(app) as client:
        client.post("/api/usage", json={"page": "team"})
    assert counts(store).counts == {"view:team": 1}


@pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="set TEST_DATABASE_URL to test against Postgres")
def test_postgres_store():
    store = PostgresUsageStore(os.environ["TEST_DATABASE_URL"])
    with store._connect() as conn:
        for table in ("usage_daily", "usage_visitors", "usage_salts"):
            conn.execute(f"delete from {table}")
    salt = store.salt(DAY)
    assert store.salt(DAY) == salt and store.salt(DAY + timedelta(days=1)) != salt
    with store._connect() as conn:
        assert conn.execute("select count(*) from usage_salts").fetchone()[0] == 1
    store.save({(DAY, "view:moves"): 2, (DAY, "signup"): 1}, {(DAY, "a"): False, (DAY, "b"): True})
    store.save({(DAY, "view:moves"): 3}, {(DAY, "a"): True})
    old = DAY - timedelta(days=400)
    store.save({(old, "view:moves"): 1}, {(old, "z"): False})
    store.save({}, {(DAY, "c"): False})  # prunes visitor hashes past VISITOR_DAYS, keeps counts
    [day, past] = store.days(old)
    assert day.counts == {"view:moves": 5, "signup": 1} and (day.visitors, day.signed_in) == (3, 2)
    assert past.counts == {"view:moves": 1} and past.visitors == 0


def admin_client(app, monkeypatch) -> TestClient:
    from app.social import hash_token

    user = app.state.accounts.create_user("boss@example.com", None)
    user.user_type = "admin"
    app.state.accounts.save_session(hash_token("boss-token"), user.id)
    return TestClient(app, headers={"Authorization": "Bearer boss-token"})


def test_admins_read_usage_with_account_totals(app, monkeypatch):
    TestClient(app).post("/api/auth/signup", json={"email": "new@example.com", "password": "longenough"})
    TestClient(app).post("/api/usage", json={"page": "moves"})
    TestClient(app).post("/api/usage", json={"page": "moves"})
    signed_in(app)  # a verified manager
    boss = admin_client(app, monkeypatch)
    r = boss.post("/api/admin/usage", json={"days": 7})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["accounts"] == 3 and body["verified_teams"] == 1
    [today] = body["days"]
    assert today["views"] == {"moves": 2} and today["events"] == {"signup": 1}  # flushed before reading
    assert today["visitors"] == 1 and today["new_accounts"] == 3


def test_usage_is_admins_only(app):
    assert TestClient(app).post("/api/admin/usage", json={}).status_code == 401
    assert signed_in(app).post("/api/admin/usage", json={}).status_code == 403
    assert signed_in(app).post("/api/admin/usage", json={"days": 0}).status_code == 422


def test_account_totals_in_memory():
    from datetime import UTC, datetime

    from app.accounts import FollowedLeague

    accounts = MemoryAccountStore()
    a = accounts.create_user("a@example.com", None)
    b = accounts.create_user("b@example.com", None)
    accounts.follow(a.id, FollowedLeague(1, 2027, "One"))
    accounts.follow(b.id, FollowedLeague(1, 2027, "One"))
    accounts.follow(b.id, FollowedLeague(2, 2027, "Two"))
    accounts.set_manager(1, 2027, 3, a.id)
    totals = accounts.totals(datetime.now(UTC).date())
    assert (totals.accounts, totals.verified_teams, totals.followed_leagues) == (2, 1, 2)
    assert sum(totals.new_by_day.values()) == 2


@pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="set TEST_DATABASE_URL to test against Postgres")
def test_postgres_account_totals():
    from datetime import UTC, datetime

    from app.accounts import FollowedLeague, PostgresAccountStore

    accounts = PostgresAccountStore(os.environ["TEST_DATABASE_URL"])
    with accounts._connect() as conn:
        conn.execute("truncate users cascade")
        conn.execute("delete from user_leagues")
    a = accounts.create_user("a@example.com", None)
    b = accounts.create_user("b@example.com", None)
    accounts.follow(a.id, FollowedLeague(1, 2027, "One"))
    accounts.follow(b.id, FollowedLeague(1, 2027, "One"))
    accounts.set_manager(1, 2027, 3, a.id)
    totals = accounts.totals(datetime.now(UTC).date())
    assert (totals.accounts, totals.verified_teams, totals.followed_leagues) == (2, 1, 1)
    assert totals.new_by_day == {datetime.now(UTC).date(): 2}

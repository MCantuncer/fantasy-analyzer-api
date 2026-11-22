import os
import typing
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from fastapi.testclient import TestClient

from app import plans, schemas
from app.accounts import MemoryAccountStore, User
from app.espn.client import EspnClient
from app.main import create_app
from app.plans import FREE, PRO, Feature, Subscription, entitlement
from app.social import hash_token
from app.store import MemoryStore

NOW = datetime(2026, 11, 22, 21, 0, tzinfo=UTC)
USER = User(1, "fan@example.com", NOW)
ADMIN = User(2, "boss@example.com", NOW, user_type="admin")


def sub(status, days=10, **kw) -> Subscription:
    return Subscription(1, status=status, provider="paddle", period_end=NOW + timedelta(days=days), **kw)


def test_free_without_a_subscription():
    got = entitlement(USER, None, NOW)
    assert got.plan == FREE and got.features == frozenset() and not got.allows(Feature.NEXT_GAME)


def test_pro_holds_every_feature():
    got = entitlement(USER, sub("active"), NOW)
    assert got.plan == PRO and got.features == frozenset(Feature)
    assert got.renews_at == NOW + timedelta(days=10) and got.ends_at is None and got.trial_ends is None


def test_trial_and_cancelled_at_period_end():
    trial = entitlement(USER, sub("trialing", days=7), NOW)
    assert trial.plan == PRO and trial.trial_ends == NOW + timedelta(days=7)
    leaving = entitlement(USER, sub("active", cancel_at_period_end=True), NOW)
    assert leaving.plan == PRO and leaving.ends_at == NOW + timedelta(days=10) and leaving.renews_at is None


def test_grace_after_the_period_then_free():
    assert entitlement(USER, sub("past_due", days=-2), NOW).plan == PRO  # the provider still retries the card
    assert entitlement(USER, sub("past_due", days=-4), NOW).plan == FREE
    assert entitlement(USER, sub("active", days=-4), NOW).plan == FREE  # a missed webhook can't keep Pro forever


@pytest.mark.parametrize("status", ["canceled", "paused", "expired"])
def test_ended_subscriptions_are_free(status):
    assert entitlement(USER, sub(status), NOW).plan == FREE


def test_gifts_and_admins():
    gift = Subscription(1, comp_until=NOW + timedelta(days=30))
    got = entitlement(USER, gift, NOW)
    assert got.plan == PRO and got.gifted and got.ends_at == NOW + timedelta(days=30)
    assert entitlement(USER, Subscription(1, comp_until=NOW - timedelta(seconds=1)), NOW).plan == FREE
    paying = entitlement(USER, sub("active", comp_until=NOW + timedelta(days=30)), NOW)
    assert paying.plan == PRO and not paying.gifted  # the paid plan is what shows
    assert entitlement(ADMIN, None, NOW).features == frozenset(Feature)


def test_every_feature_is_open_until_plans_are_enforced():
    got = entitlement(USER, None, NOW, enforced=False)
    assert got.plan == FREE and got.features == frozenset(Feature)


def test_schema_lists_every_feature():
    assert set(typing.get_args(schemas.PlanFeature)) == {f.value for f in Feature}


def test_memory_store_keeps_a_gift_through_webhook_writes():
    store = plans.MemorySubscriptionStore()
    until = NOW + timedelta(days=5)
    store.gift(1, until)
    store.save(Subscription(1, status="active", period_end=NOW))
    got = store.get(1)
    assert got.status == "active" and got.comp_until == until


# The API --------------------------------------------------------------------------------------


@pytest.fixture
def app(fake_espn):
    return create_app(
        EspnClient(transport=httpx.MockTransport(fake_espn.handler)),
        store=MemoryStore(),
        accounts=MemoryAccountStore(),
        subscriptions=plans.MemorySubscriptionStore(),
    )


def client_for(app, email, admin=False) -> TestClient:
    user = app.state.accounts.create_user(email, None)
    user.user_type = "admin" if admin else "user"
    app.state.accounts.save_session(hash_token(email), user.id)
    return TestClient(app, headers={"Authorization": f"Bearer {email}"})


def me(client) -> dict:
    r = client.post("/api/me")
    assert r.status_code == 200, r.text
    return r.json()


def test_me_before_launch_opens_everything(app):
    app.state.plans_enforced = False
    got = me(client_for(app, "fan@example.com"))
    assert got["plan"] == "free" and set(got["features"]) == {f.value for f in Feature}


def test_me_once_enforced(app):
    app.state.plans_enforced = True
    fan = client_for(app, "fan@example.com")
    assert me(fan)["plan"] == "free" and me(fan)["features"] == []
    boss = client_for(app, "boss@example.com", admin=True)
    assert me(boss)["plan"] == "pro" and len(me(boss)["features"]) == len(Feature)


def test_admin_gifts_pro(app):
    app.state.plans_enforced = True
    fan = client_for(app, "fan@example.com")
    assert me(fan)["plan"] == "free"  # now cached for half a minute: a gift must clear it
    boss = client_for(app, "boss@example.com", admin=True)
    r = boss.post("/api/admin/gift-pro", json={"email": " Fan@Example.com ", "days": 30})
    assert r.status_code == 200, r.text
    got = me(fan)
    assert got["plan"] == "pro" and got["gifted"] and got["ends_at"] == r.json()["until"]
    assert boss.post("/api/admin/gift-pro", json={"email": "fan@example.com", "days": 0}).json()["until"] is None
    assert me(fan)["plan"] == "free"


def test_only_admins_gift(app):
    fan = client_for(app, "fan@example.com")
    assert fan.post("/api/admin/gift-pro", json={"email": "fan@example.com", "days": 30}).status_code == 403
    assert TestClient(app).post("/api/admin/gift-pro", json={"email": "a@b.co", "days": 1}).status_code == 401
    boss = client_for(app, "boss@example.com", admin=True)
    assert boss.post("/api/admin/gift-pro", json={"email": "nobody@example.com", "days": 1}).status_code == 404


@pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="set TEST_DATABASE_URL to test against Postgres")
def test_postgres_store():
    from app.accounts import PostgresAccountStore

    accounts = PostgresAccountStore(os.environ["TEST_DATABASE_URL"])
    with accounts._connect() as conn:
        conn.execute("truncate users cascade")
    store = plans.PostgresSubscriptionStore(os.environ["TEST_DATABASE_URL"])
    user = accounts.create_user("fan@example.com", None)
    assert store.get(user.id) is None
    store.gift(user.id, NOW)
    store.save(Subscription(user.id, "active", "paddle", "ctm_1", "sub_1", NOW, True))
    got = store.get(user.id)
    assert (got.status, got.subscription_id, got.period_end, got.cancel_at_period_end) == ("active", "sub_1", NOW, True)
    assert got.comp_until == NOW  # a webhook write leaves the gift alone
    plans.PostgresSubscriptionStore(os.environ["TEST_DATABASE_URL"])  # the table is made once; a restart is fine

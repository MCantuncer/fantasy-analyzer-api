import hashlib
import hmac
import json
import os
import time
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from fastapi.testclient import TestClient

from app import billing, plans
from app.accounts import MemoryAccountStore
from app.espn.client import EspnClient
from app.main import create_app
from app.social import hash_token
from app.store import MemoryStore

SECRET = "pdl_ntfset_test_secret"
CONFIG = billing.PaddleConfig("pdl_sdbx_key", SECRET, "test_client_token", {"month": "pri_m", "year": "pri_y"})
FUTURE = datetime.now(UTC) + timedelta(days=7)


def sub_data(user_id=1, status="trialing", sub_id="sub_01", updated=None, cancel=False, ends=FUTURE) -> dict:
    return {
        "id": sub_id,
        "status": status,
        "customer_id": "ctm_01",
        "custom_data": {"user_id": str(user_id)} if user_id else None,
        "current_billing_period": {"starts_at": "2026-11-22T21:00:00Z", "ends_at": ends.isoformat()},
        "scheduled_change": {"action": "cancel", "effective_at": ends.isoformat()} if cancel else None,
        "updated_at": (updated or datetime.now(UTC)).isoformat(),
    }


class FakePaddle:
    """Answers Paddle's API the way its docs show, and keeps what was asked."""

    def __init__(self):
        self.calls = []
        self.subs = {}
        self.txns = {}

    def handler(self, request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer pdl_sdbx_key"
        path = request.url.path
        body = json.loads(request.content) if request.content else None
        self.calls.append((request.method, path, body))
        if request.method == "POST" and path == "/transactions":
            txn = {"id": f"txn_{len(self.txns):012d}", "custom_data": body["custom_data"], "subscription_id": None}
            self.txns[txn["id"]] = txn
            return httpx.Response(201, json={"data": txn})
        if path.startswith("/transactions/"):
            txn = self.txns.get(path.split("/")[-1])
            return httpx.Response(200, json={"data": txn}) if txn else httpx.Response(404, json={})
        if path.startswith("/subscriptions/"):
            data = self.subs.get(path.split("/")[-1])
            return httpx.Response(200, json={"data": data}) if data else httpx.Response(404, json={})
        if path.endswith("/portal-sessions"):
            return httpx.Response(201, json={"data": {"urls": {"general": {"overview": "https://portal.paddle/ctm"}}}})
        return httpx.Response(404, json={})


@pytest.fixture
def fake():
    return FakePaddle()


@pytest.fixture
def app(fake_espn, fake):
    app = create_app(
        EspnClient(transport=httpx.MockTransport(fake_espn.handler)),
        store=MemoryStore(),
        accounts=MemoryAccountStore(),
        subscriptions=plans.MemorySubscriptionStore(),
        billing_events=billing.MemoryEventStore(),
        paddle=billing.Paddle(CONFIG, transport=httpx.MockTransport(fake.handler)),
    )
    app.state.plans_enforced = True
    return app


def client_for(app, email) -> tuple[TestClient, int]:
    user = app.state.accounts.create_user(email, None)
    app.state.accounts.save_session(hash_token(email), user.id)
    return TestClient(app, headers={"Authorization": f"Bearer {email}"}), user.id


def signed(event: dict, secret=SECRET, ts=None) -> tuple[bytes, dict]:
    body = json.dumps(event).encode()
    ts = str(int(ts if ts is not None else time.time()))
    h1 = hmac.new(secret.encode(), f"{ts}:".encode() + body, hashlib.sha256).hexdigest()
    return body, {"Paddle-Signature": f"ts={ts};h1={h1}", "Content-Type": "application/json"}


def deliver(app, data, kind="subscription.created", event_id="evt_01", **kw) -> httpx.Response:
    body, headers = signed({"event_id": event_id, "event_type": kind, "data": data}, **kw)
    return TestClient(app).post("/api/billing/webhook", content=body, headers=headers)


def plan(client) -> str:
    return client.post("/api/me").json()["plan"]


# Signatures ----------------------------------------------------------------------------------------


def test_signature_checks():
    body, headers = signed({"a": 1})
    header = headers["Paddle-Signature"]
    assert billing.verify_signature(header, body, SECRET)
    assert not billing.verify_signature(header, body + b" ", SECRET)
    assert not billing.verify_signature(header, body, "other")
    assert not billing.verify_signature(None, body, SECRET) and not billing.verify_signature("ts=1", body, SECRET)
    old_body, old = signed({"a": 1}, ts=time.time() - 600)
    assert not billing.verify_signature(old["Paddle-Signature"], old_body, SECRET)  # a replay
    ts, h1 = header.split(";")
    assert billing.verify_signature(f"{ts};h1=deadbeef;{h1}", body, SECRET)  # while Paddle rotates the secret


def test_subscription_from_paddle():
    got = billing.subscription_from(sub_data(cancel=True), 7)
    assert (got.user_id, got.status, got.subscription_id, got.customer_id) == (7, "trialing", "sub_01", "ctm_01")
    assert got.cancel_at_period_end and got.period_end is not None and got.provider == "paddle"
    assert billing.subscription_from({"id": "sub_x", "status": "weird"}, 7).status == "expired"
    assert billing.account_of({"custom_data": {"user_id": "12"}}) == 12
    assert billing.account_of({"custom_data": None}) is None and billing.account_of({}) is None


# The webhook ---------------------------------------------------------------------------------------


def test_webhook_turns_pro_on_and_off(app):
    fan, uid = client_for(app, "fan@example.com")
    assert plan(fan) == "free"
    assert deliver(app, sub_data(uid)).status_code == 200
    got = fan.post("/api/me").json()
    assert got["plan"] == "pro" and got["trial_ends"] is not None
    later = datetime.now(UTC) + timedelta(seconds=5)
    # A cancellation that carries no account id is matched by its subscription id.
    assert deliver(app, sub_data(None, "canceled", updated=later), "subscription.canceled", "evt_02").status_code == 200
    assert plan(fan) == "free"


def test_webhook_refuses_unsigned_and_ignores_repeats(app):
    fan, uid = client_for(app, "fan@example.com")
    body, headers = signed({"event_id": "evt_01", "event_type": "subscription.created", "data": sub_data(uid)})
    headers["Paddle-Signature"] = headers["Paddle-Signature"][:-2] + "00"
    assert TestClient(app).post("/api/billing/webhook", content=body, headers=headers).status_code == 401
    assert plan(fan) == "free"
    assert deliver(app, sub_data(uid, "active")).status_code == 200
    app.state.subscriptions.rows[uid].status = "canceled"  # pretend a later state landed
    assert deliver(app, sub_data(uid, "active")).status_code == 200  # the same event again changes nothing
    assert app.state.subscriptions.get(uid).status == "canceled"


def test_webhook_keeps_the_newest_state(app):
    fan, uid = client_for(app, "fan@example.com")
    now = datetime.now(UTC)
    deliver(app, sub_data(uid, "active", updated=now), event_id="evt_new")
    deliver(app, sub_data(uid, "past_due", updated=now - timedelta(minutes=1)), event_id="evt_old")
    assert app.state.subscriptions.get(uid).status == "active"


def test_other_events_and_strangers_are_ignored(app):
    assert deliver(app, {"id": "txn_1"}, "transaction.completed").status_code == 200
    assert deliver(app, sub_data(None, sub_id="sub_unknown")).status_code == 200
    body, headers = signed({"nope": True})
    assert TestClient(app).post("/api/billing/webhook", content=body, headers=headers).status_code == 400


# Checkout, sync, portal --------------------------------------------------------------------------


def test_config_is_public(app):
    got = TestClient(app).get("/api/billing/config").json()
    assert got == {
        "enabled": True,
        "environment": "sandbox",
        "client_token": "test_client_token",
        "prices": {"month": "pri_m", "year": "pri_y"},
    }


def test_checkout_tags_the_account_then_sync_opens_pro(app, fake):
    fan, uid = client_for(app, "fan@example.com")
    r = fan.post("/api/billing/checkout", json={"interval": "month"})
    assert r.status_code == 200, r.text
    txn = r.json()["transaction_id"]
    assert fake.calls[-1] == (
        "POST",
        "/transactions",
        {"items": [{"price_id": "pri_m", "quantity": 1}], "custom_data": {"user_id": str(uid)}},
    )
    assert fan.post("/api/billing/sync", json={"transaction_id": txn}).json()["plan"] == "free"  # not paid yet
    fake.txns[txn]["subscription_id"] = "sub_01"
    fake.subs["sub_01"] = sub_data(uid, "trialing")
    assert fan.post("/api/billing/sync", json={"transaction_id": txn}).json()["plan"] == "pro"
    assert fan.post("/api/billing/checkout", json={"interval": "year"}).status_code == 409


def test_sync_refuses_someone_elses_checkout(app):
    fan, _ = client_for(app, "fan@example.com")
    other, _ = client_for(app, "other@example.com")
    txn = fan.post("/api/billing/checkout", json={}).json()["transaction_id"]
    assert other.post("/api/billing/sync", json={"transaction_id": txn}).status_code == 403
    assert other.post("/api/billing/sync", json={"transaction_id": "nope"}).status_code == 422


def test_gifted_accounts_may_still_subscribe(app):
    fan, uid = client_for(app, "fan@example.com")
    app.state.subscriptions.gift(uid, datetime.now(UTC) + timedelta(days=3))
    assert fan.post("/api/billing/checkout", json={}).status_code == 200


def test_portal(app):
    fan, uid = client_for(app, "fan@example.com")
    assert fan.post("/api/billing/portal").status_code == 404
    deliver(app, sub_data(uid, "active"))
    assert fan.post("/api/billing/portal").json() == {"url": "https://portal.paddle/ctm"}


def test_billing_needs_an_account(app):
    assert TestClient(app).post("/api/billing/checkout", json={}).status_code == 401
    assert TestClient(app).post("/api/billing/portal").status_code == 401


def test_billing_off_without_settings(fake_espn, monkeypatch):
    for name in ("PADDLE_API_KEY", "PADDLE_WEBHOOK_SECRET", "PADDLE_PRICE_MONTHLY", "PADDLE_PRICE_YEARLY"):
        monkeypatch.delenv(name, raising=False)
    app = create_app(EspnClient(transport=httpx.MockTransport(fake_espn.handler)), store=MemoryStore())
    assert TestClient(app).get("/api/billing/config").json()["enabled"] is False
    fan, _ = client_for(app, "fan@example.com")
    assert fan.post("/api/billing/checkout", json={}).status_code == 503
    assert TestClient(app).post("/api/billing/webhook", content=b"{}").status_code == 503


def test_config_from_settings(monkeypatch):
    monkeypatch.setenv("PADDLE_API_KEY", "k")
    monkeypatch.setenv("PADDLE_WEBHOOK_SECRET", "s")
    monkeypatch.setenv("PADDLE_PRICE_MONTHLY", "pri_m")
    monkeypatch.delenv("PADDLE_PRICE_YEARLY", raising=False)
    assert billing.paddle_config() is None
    monkeypatch.setenv("PADDLE_PRICE_YEARLY", "pri_y")
    monkeypatch.setenv("PADDLE_ENV", "production")
    got = billing.paddle_config()
    assert got.api_url == "https://api.paddle.com" and got.client_token is None


# A renewal the webhook never told us about -------------------------------------------------------


def test_a_lapsed_period_is_checked_with_paddle_once(app, fake):
    fan, uid = client_for(app, "fan@example.com")
    ended = datetime.now(UTC) - timedelta(days=4)
    deliver(app, sub_data(uid, "active", ends=ended, updated=ended))
    fake.subs["sub_01"] = sub_data(uid, "active", ends=datetime.now(UTC) + timedelta(days=30))
    assert plan(fan) == "pro"  # Paddle renewed it; the webhook was lost
    asked = [c for c in fake.calls if c[1] == "/subscriptions/sub_01"]
    assert len(asked) == 1


@pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="set TEST_DATABASE_URL to test against Postgres")
def test_postgres_event_store():
    store = billing.PostgresEventStore(os.environ["TEST_DATABASE_URL"])
    with store._connect() as conn:
        conn.execute("delete from billing_events")
    assert store.first_time("evt_1") and not store.first_time("evt_1")
    store.forget("evt_1")
    assert store.first_time("evt_1")

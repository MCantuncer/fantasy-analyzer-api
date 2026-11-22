"""Payments through Paddle (Paddle Billing), the seller of record for COURT Pro.

Paddle sells the subscription itself: it takes the card, charges VAT in the buyer's country, runs the
free trial, retries a failed renewal and pays us out. We never see a card.

1. ``/api/billing/checkout`` asks Paddle for a transaction for the chosen price, with our account id
   in its ``custom_data``; the site opens Paddle's checkout overlay on that transaction id, so the
   account a payment unlocks is set by us, never by the page.
2. Paddle tells us about the subscription through its webhook, signed with our endpoint's secret
   (``Paddle-Signature: ts=...;h1=...``, an HMAC-SHA256 of ``ts:body``). Each event id is kept once,
   so a repeated delivery changes nothing, and an older state never overwrites a newer one.
3. Right after checkout the site calls ``/api/billing/sync``: we read the transaction and its
   subscription from Paddle, so Pro opens at once even if the webhook is still on its way.
4. ``/api/billing/portal`` opens Paddle's customer portal to change the card or cancel.

Everything Paddle-specific is here; ``app/plans.py`` only ever sees a ``Subscription``. Settings:
``PADDLE_API_KEY``, ``PADDLE_WEBHOOK_SECRET``, ``PADDLE_CLIENT_TOKEN`` (public, for Paddle.js),
``PADDLE_PRICE_MONTHLY``, ``PADDLE_PRICE_YEARLY``, and ``PADDLE_ENV`` (``sandbox`` or ``production``).
Billing is off until the key, the secret and both prices are set.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

import httpx

from app.plans import STATUSES, Subscription

PROVIDER = "paddle"
API = {"sandbox": "https://sandbox-api.paddle.com", "production": "https://api.paddle.com"}
SIGNATURE_TOLERANCE = 300  # seconds: an older signed delivery is refused (a replay)


class BillingError(Exception):
    """Paddle refused or could not be reached."""


@dataclass(frozen=True)
class PaddleConfig:
    api_key: str
    webhook_secret: str
    client_token: str | None
    prices: dict[str, str]  # "month" / "year" -> Paddle price id
    environment: str = "sandbox"

    @property
    def api_url(self) -> str:
        return API[self.environment]


def paddle_config() -> PaddleConfig | None:
    get = lambda name: os.getenv(name, "").strip()  # noqa: E731
    env = get("PADDLE_ENV") or "sandbox"
    prices = {"month": get("PADDLE_PRICE_MONTHLY"), "year": get("PADDLE_PRICE_YEARLY")}
    if env not in API or not get("PADDLE_API_KEY") or not get("PADDLE_WEBHOOK_SECRET") or not all(prices.values()):
        return None
    token = get("PADDLE_CLIENT_TOKEN") or None
    return PaddleConfig(get("PADDLE_API_KEY"), get("PADDLE_WEBHOOK_SECRET"), token, prices, env)


def verify_signature(header: str | None, body: bytes, secret: str, now: float | None = None) -> bool:
    """Whether ``body`` came from Paddle: ``header`` is ``ts=<unix>;h1=<hex hmac>`` (several h1 while
    Paddle rotates the secret), and the signed time is recent."""
    parts = [p.split("=", 1) for p in (header or "").split(";") if "=" in p]
    ts = next((v for k, v in parts if k == "ts"), None)
    hashes = [v for k, v in parts if k == "h1"]
    if not ts or not hashes or not ts.isdigit():
        return False
    if abs((now if now is not None else time.time()) - int(ts)) > SIGNATURE_TOLERANCE:
        return False
    want = hmac.new(secret.encode(), ts.encode() + b":" + body, hashlib.sha256).hexdigest()
    return any(hmac.compare_digest(want, h) for h in hashes)


def _time(text: str | None) -> datetime | None:
    return datetime.fromisoformat(text.replace("Z", "+00:00")) if text else None


def account_of(data: dict) -> int | None:
    """Our account id from a subscription's or transaction's ``custom_data``, as checkout set it."""
    raw = (data.get("custom_data") or {}).get("user_id")
    return int(raw) if isinstance(raw, str | int) and str(raw).isdigit() else None


def subscription_from(data: dict, user_id: int) -> Subscription:
    """A Paddle subscription entity as our row."""
    status = data.get("status")
    period = data.get("current_billing_period") or {}
    change = data.get("scheduled_change") or {}
    return Subscription(
        user_id=user_id,
        status=status if status in STATUSES else "expired",
        provider=PROVIDER,
        customer_id=data.get("customer_id"),
        subscription_id=data.get("id"),
        period_end=_time(period.get("ends_at")),
        cancel_at_period_end=change.get("action") == "cancel",
        provider_updated_at=_time(data.get("updated_at")),
    )


class Paddle:
    """The few Paddle API calls COURT needs."""

    def __init__(self, config: PaddleConfig, transport=None):
        self.config = config
        self._transport = transport

    async def _call(self, method: str, path: str, json: dict | None = None) -> dict:
        headers = {"Authorization": f"Bearer {self.config.api_key}"}
        async with httpx.AsyncClient(base_url=self.config.api_url, transport=self._transport, timeout=15) as http:
            try:
                resp = await http.request(method, path, json=json, headers=headers)
            except httpx.HTTPError as exc:
                raise BillingError("Paddle could not be reached") from exc
        if resp.status_code >= 400:
            raise BillingError(f"Paddle answered {resp.status_code} on {path.split('/')[1]}")
        return resp.json().get("data") or {}

    async def start_checkout(self, user_id: int, interval: str) -> str:
        """A transaction for ``interval``'s price, tagged with the account; its id opens the checkout."""
        data = await self._call(
            "POST",
            "/transactions",
            {
                "items": [{"price_id": self.config.prices[interval], "quantity": 1}],
                "custom_data": {"user_id": str(user_id)},
            },
        )
        return data["id"]

    async def transaction(self, transaction_id: str) -> dict:
        return await self._call("GET", f"/transactions/{transaction_id}")

    async def subscription(self, subscription_id: str) -> dict:
        return await self._call("GET", f"/subscriptions/{subscription_id}")

    async def portal_url(self, customer_id: str, subscription_id: str | None) -> str:
        body = {"subscription_ids": [subscription_id]} if subscription_id else {}
        data = await self._call("POST", f"/customers/{customer_id}/portal-sessions", body)
        return data["urls"]["general"]["overview"]


# Webhook deliveries already handled --------------------------------------------------------------


class EventStore(Protocol):
    def first_time(self, event_id: str) -> bool: ...
    def forget(self, event_id: str) -> None: ...


@dataclass
class MemoryEventStore:
    seen: set[str] = field(default_factory=set)

    def first_time(self, event_id: str) -> bool:
        if event_id in self.seen:
            return False
        self.seen.add(event_id)
        return True

    def forget(self, event_id: str) -> None:
        self.seen.discard(event_id)


EVENTS_SCHEMA = """
create table if not exists billing_events (
    event_id text primary key,
    received_at timestamptz not null default now()
);
"""


class PostgresEventStore:
    def __init__(self, url: str):
        from app.db import connector

        self._connect = connector(url)
        with self._connect() as conn:
            conn.execute(EVENTS_SCHEMA)

    def first_time(self, event_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "insert into billing_events (event_id) values (%s) on conflict do nothing returning event_id",
                (event_id,),
            ).fetchone()
        return row is not None

    def forget(self, event_id: str) -> None:
        """An event whose handling failed: Paddle's next delivery of it is handled again."""
        with self._connect() as conn:
            conn.execute("delete from billing_events where event_id=%s", (event_id,))


def create_event_store() -> EventStore:
    url = os.getenv("DATABASE_URL", "").strip()
    return PostgresEventStore(url) if url else MemoryEventStore()

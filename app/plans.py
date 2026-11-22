"""Plans: what a Free and a Pro account may see, and who has Pro.

Free is the fantasy league: rosters, moves, trades, scores, news. Pro adds every estimate we
make of an NBA game (a player's next-game line, Our call, the pre-game box, Our Estimates) and the
sportsbook comparison. The split lives in code, as named features and the plans that hold them, so
a new paid feature is one ``Feature`` member: ``PRO`` holds every feature.

Who has Pro lives in the database, one ``subscriptions`` row per account, written only by the
payment provider's signed webhook (or by an admin gifting Pro until a date). ``entitlement``
turns that row into what the account may see now; the API asks it, never the site.

``PLANS_ENFORCED`` keeps every feature open to everyone until launch day, so this can ship first.
Admins always have every feature.
"""

from __future__ import annotations

import contextvars
import functools
import inspect
import os
import typing
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Protocol

from fastapi import Response
from fastapi.routing import APIRoute
from pydantic import BaseModel
from pydantic.fields import FieldInfo

from app.accounts import User


class Feature(StrEnum):
    NEXT_GAME = "next_game"  # a player's line for his next game (cards, FA Targets, My Players)
    GAME_CALLS = "game_calls"  # Our call for a game's score and total, on Scores
    PREGAME = "pregame"  # the pre-game box: every player's line and How it could go
    ESTIMATES = "estimates"  # the Our Estimates page: picks, Kupons, Why not, lost picks
    SPORTSBOOK = "sportsbook"  # the sportsbook's lines beside ours and the running record
    MATCH_FORM = "match_form"  # the Scouting Report on a match card


FREE = "free"
PRO = "pro"
PLANS: dict[str, frozenset[Feature]] = {FREE: frozenset(), PRO: frozenset(Feature)}

# Statuses the provider reports for a subscription that is paid for, or in its trial or retries.
PAYING = frozenset({"trialing", "active", "past_due"})
STATUSES = PAYING | {"canceled", "paused", "expired"}
# Pro holds this long past the paid period: a renewal the provider is still retrying, or a
# webhook that arrives late, never locks someone out who has paid.
GRACE = timedelta(days=3)

UPGRADE_REQUIRED = "upgrade_required"


@dataclass
class Subscription:
    """An account's subscription as the provider last told us, plus any Pro an admin gifted."""

    user_id: int
    status: str = "expired"
    provider: str | None = None
    customer_id: str | None = None
    subscription_id: str | None = None
    period_end: datetime | None = None  # end of the paid (or trial) period
    cancel_at_period_end: bool = False
    comp_until: datetime | None = None  # Pro gifted by an admin, until this moment
    updated_at: datetime | None = None


@dataclass(frozen=True)
class Entitlement:
    """What an account may see right now, and why."""

    plan: str  # FREE or PRO
    features: frozenset[Feature]  # what the API serves it: every feature while plans are off
    trial_ends: datetime | None = None
    renews_at: datetime | None = None  # the next charge, for a subscription that renews
    ends_at: datetime | None = None  # when Pro stops: cancelled at period end, or a gift's last day
    gifted: bool = False

    def allows(self, feature: Feature) -> bool:
        return feature in self.features


ALL = frozenset(Feature)


def plans_enforced() -> bool:
    return os.getenv("PLANS_ENFORCED", "").strip().lower() in {"1", "true", "yes"}


def entitlement(user: User | None, sub: Subscription | None, now: datetime, enforced: bool = True) -> Entitlement:
    """The plan ``user`` is on at ``now``. Admins are on Pro whatever the row says."""
    plan, trial_ends, renews_at, ends_at, gifted = FREE, None, None, None, False
    if sub is not None:
        paid_until = sub.period_end + GRACE if sub.period_end else None
        if sub.status in PAYING and (paid_until is None or now < paid_until):
            plan = PRO
            if sub.status == "trialing":
                trial_ends = sub.period_end
            if sub.cancel_at_period_end:
                ends_at = sub.period_end
            else:
                renews_at = sub.period_end
        if sub.comp_until is not None and sub.comp_until > now and plan == FREE:
            plan, ends_at, gifted = PRO, sub.comp_until, True
    if user is not None and user.admin:
        plan = PRO
    features = PLANS[plan] if enforced else ALL
    return Entitlement(plan, features, trial_ends, renews_at, ends_at, gifted)


class SubscriptionStore(Protocol):
    def get(self, user_id: int) -> Subscription | None: ...
    def save(self, sub: Subscription) -> None: ...
    def gift(self, user_id: int, until: datetime | None) -> None: ...


SCHEMA = """
create table if not exists subscriptions (
    user_id bigint primary key references users(id) on delete cascade,
    status text not null default 'expired',
    provider text,
    customer_id text,
    subscription_id text unique,
    period_end timestamptz,
    cancel_at_period_end boolean not null default false,
    comp_until timestamptz,
    updated_at timestamptz not null default now()
);
"""


@dataclass
class MemorySubscriptionStore:
    rows: dict[int, Subscription] = field(default_factory=dict)

    def get(self, user_id: int) -> Subscription | None:
        return self.rows.get(user_id)

    def save(self, sub: Subscription) -> None:
        kept = self.rows.get(sub.user_id)
        sub.comp_until = kept.comp_until if kept else sub.comp_until  # a gift is the admin's to change
        sub.updated_at = datetime.now(UTC)
        self.rows[sub.user_id] = sub

    def gift(self, user_id: int, until: datetime | None) -> None:
        sub = self.rows.setdefault(user_id, Subscription(user_id))
        sub.comp_until = until
        sub.updated_at = datetime.now(UTC)


class PostgresSubscriptionStore:
    _COLS = (
        "user_id, status, provider, customer_id, subscription_id, period_end, cancel_at_period_end, "
        "comp_until, updated_at"
    )

    def __init__(self, url: str):
        from app.db import connector

        self._connect = connector(url)
        with self._connect() as conn:
            conn.execute(SCHEMA)

    def get(self, user_id: int) -> Subscription | None:
        with self._connect() as conn:
            row = conn.execute(f"select {self._COLS} from subscriptions where user_id=%s", (user_id,)).fetchone()
        return Subscription(*row) if row else None

    def save(self, sub: Subscription) -> None:
        with self._connect() as conn:
            conn.execute(
                "insert into subscriptions (user_id, status, provider, customer_id, subscription_id, period_end, "
                "cancel_at_period_end) values (%s, %s, %s, %s, %s, %s, %s) on conflict (user_id) do update set "
                "status=excluded.status, provider=excluded.provider, customer_id=excluded.customer_id, "
                "subscription_id=excluded.subscription_id, period_end=excluded.period_end, "
                "cancel_at_period_end=excluded.cancel_at_period_end, updated_at=now()",
                (
                    sub.user_id,
                    sub.status,
                    sub.provider,
                    sub.customer_id,
                    sub.subscription_id,
                    sub.period_end,
                    sub.cancel_at_period_end,
                ),
            )

    def gift(self, user_id: int, until: datetime | None) -> None:
        with self._connect() as conn:
            conn.execute(
                "insert into subscriptions (user_id, comp_until) values (%s, %s) "
                "on conflict (user_id) do update set comp_until=excluded.comp_until, updated_at=now()",
                (user_id, until),
            )


def create_subscription_store() -> SubscriptionStore:
    url = os.getenv("DATABASE_URL", "").strip()
    return PostgresSubscriptionStore(url) if url else MemorySubscriptionStore()


# Pro fields inside free responses ------------------------------------------------------------
# A field that only Pro accounts see carries its feature in the schema:
#     next_game: NextGameLine | None = Field(default=None, json_schema_extra=pro(Feature.NEXT_GAME))
# and ``strip`` hands everyone else a copy with that field back at its default. Only the models on
# the way to a blanked field are copied (shallowly), so a shared cached report is never changed.

FEATURE_KEY = "pro_feature"


def pro(feature: Feature) -> dict:
    """``json_schema_extra`` for a field only accounts with ``feature`` see."""
    return {FEATURE_KEY: feature.value}


def field_feature(info: FieldInfo) -> Feature | None:
    extra = info.json_schema_extra
    return Feature(extra[FEATURE_KEY]) if isinstance(extra, dict) and FEATURE_KEY in extra else None


def _models_in(annotation) -> list[type[BaseModel]]:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return [annotation]
    return [m for arg in typing.get_args(annotation) for m in _models_in(arg)]


@functools.cache
def reaches_pro(model: type[BaseModel]) -> bool:
    """Whether ``model`` holds a Pro field anywhere inside it."""
    _seen.add(model)
    try:
        return any(
            field_feature(info) is not None
            or any(m not in _seen and reaches_pro(m) for m in _models_in(info.annotation))
            for info in model.model_fields.values()
        )
    finally:
        _seen.discard(model)


_seen: set[type[BaseModel]] = set()  # models on the walk's path, so a model that holds itself ends


def strip(value, features: frozenset[Feature]):
    """``value`` with every Pro field outside ``features`` back at its default; ``value`` itself
    when nothing changes."""
    if len(features) == len(Feature):
        return value
    if isinstance(value, BaseModel):
        cls = type(value)
        if not reaches_pro(cls):
            return value
        changes = {}
        for name, info in cls.model_fields.items():
            current = getattr(value, name)
            feature = field_feature(info)
            if feature is not None and feature not in features:
                blank = info.get_default(call_default_factory=True)
                if current != blank:
                    changes[name] = blank
            elif current is not None and any(reaches_pro(m) for m in _models_in(info.annotation)):
                new = strip(current, features)
                if new is not current:
                    changes[name] = new
        return value.model_copy(update=changes) if changes else value
    if isinstance(value, list):
        new = [strip(v, features) for v in value]
        return new if any(a is not b for a, b in zip(new, value, strict=True)) else value
    if isinstance(value, dict):
        new = {k: strip(v, features) for k, v in value.items()}
        return new if any(new[k] is not v for k, v in value.items()) else value
    return value


# Every route whose response can hold a Pro field strips it on the way out ----------------------
# ``ProRoute`` is the app's route class: when a route's response model reaches a Pro field, its
# endpoint's result goes through ``strip`` for the caller's features. New routes are covered the
# moment they return such a model; nothing to remember. The caller is read from the request's
# Authorization header, kept for the request by ``RequestScope``.

_scope: contextvars.ContextVar[dict | None] = contextvars.ContextVar("plans_request_scope", default=None)


class RequestScope:
    """ASGI middleware: keeps the request's scope where ``ProRoute``'s endpoints can read it."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        token = _scope.set(scope)
        try:
            await self.app(scope, receive, send)
        finally:
            _scope.reset(token)


def _authorization(scope: dict) -> str | None:
    for key, value in scope.get("headers", []):
        if key == b"authorization":
            return value.decode("latin-1")
    return None


class ProRoute(APIRoute):
    def __init__(self, path: str, endpoint, **kwargs):
        model = kwargs.get("response_model")
        if model is not None and any(reaches_pro(m) for m in _models_in(model)):
            endpoint = _stripping(endpoint)
        super().__init__(path, endpoint, **kwargs)


def _stripping(endpoint):
    if not inspect.iscoroutinefunction(endpoint):
        raise TypeError(f"{endpoint.__name__} returns Pro fields: make it async so they can be stripped")

    @functools.wraps(endpoint)
    async def stripped(*args, **kwargs):
        result = await endpoint(*args, **kwargs)
        scope = _scope.get()
        if scope is None or isinstance(result, Response):
            return result
        features = await scope["app"].state.features_for(_authorization(scope))
        return strip(result, features)

    return stripped


def request_memo() -> dict | None:
    """A dict that lives as long as the current request (None outside one): the signed-in user is
    read once even when both the endpoint and ``ProRoute`` ask."""
    scope = _scope.get()
    return None if scope is None else scope.setdefault("court_memo", {})

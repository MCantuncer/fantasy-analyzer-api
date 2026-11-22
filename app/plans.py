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

import os
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Protocol

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
    provider_updated_at: datetime | None = None  # when the provider says this state began; older ones lose
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
    def by_subscription(self, subscription_id: str) -> Subscription | None: ...
    def save(self, sub: Subscription) -> bool: ...
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
    provider_updated_at timestamptz,
    comp_until timestamptz,
    updated_at timestamptz not null default now()
);
alter table subscriptions add column if not exists provider_updated_at timestamptz;
"""


@dataclass
class MemorySubscriptionStore:
    rows: dict[int, Subscription] = field(default_factory=dict)

    def get(self, user_id: int) -> Subscription | None:
        return self.rows.get(user_id)

    def by_subscription(self, subscription_id: str) -> Subscription | None:
        return next((s for s in self.rows.values() if s.subscription_id == subscription_id), None)

    def save(self, sub: Subscription) -> bool:
        """Writes the provider's state unless the row already holds a newer one; True when written."""
        kept = self.rows.get(sub.user_id)
        if kept and _older(sub.provider_updated_at, kept.provider_updated_at):
            return False
        sub.comp_until = kept.comp_until if kept else sub.comp_until  # a gift is the admin's to change
        sub.updated_at = datetime.now(UTC)
        self.rows[sub.user_id] = sub
        return True

    def gift(self, user_id: int, until: datetime | None) -> None:
        sub = self.rows.setdefault(user_id, Subscription(user_id))
        sub.comp_until = until
        sub.updated_at = datetime.now(UTC)


def _older(new: datetime | None, kept: datetime | None) -> bool:
    return new is not None and kept is not None and new < kept


class PostgresSubscriptionStore:
    _COLS = (
        "user_id, status, provider, customer_id, subscription_id, period_end, cancel_at_period_end, "
        "provider_updated_at, comp_until, updated_at"
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

    def by_subscription(self, subscription_id: str) -> Subscription | None:
        with self._connect() as conn:
            row = conn.execute(
                f"select {self._COLS} from subscriptions where subscription_id=%s", (subscription_id,)
            ).fetchone()
        return Subscription(*row) if row else None

    def save(self, sub: Subscription) -> bool:
        """Writes the provider's state unless the row already holds a newer one; True when written."""
        with self._connect() as conn:
            row = conn.execute(
                "insert into subscriptions (user_id, status, provider, customer_id, subscription_id, period_end, "
                "cancel_at_period_end, provider_updated_at) values (%s, %s, %s, %s, %s, %s, %s, %s) "
                "on conflict (user_id) do update set "
                "status=excluded.status, provider=excluded.provider, customer_id=excluded.customer_id, "
                "subscription_id=excluded.subscription_id, period_end=excluded.period_end, "
                "cancel_at_period_end=excluded.cancel_at_period_end, "
                "provider_updated_at=excluded.provider_updated_at, updated_at=now() "
                "where subscriptions.provider_updated_at is null or excluded.provider_updated_at is null "
                "or excluded.provider_updated_at >= subscriptions.provider_updated_at returning user_id",
                (
                    sub.user_id,
                    sub.status,
                    sub.provider,
                    sub.customer_id,
                    sub.subscription_id,
                    sub.period_end,
                    sub.cancel_at_period_end,
                    sub.provider_updated_at,
                ),
            ).fetchone()
        return row is not None

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

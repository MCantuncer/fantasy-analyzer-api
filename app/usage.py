"""Usage numbers: how many people use the site each day and which pages they open, without
tracking anyone.

* **Counts.** Each day keeps a count per event: a page view (``view:moves``) or something done
  on the server (``analysis``, ``signup``, ``verify``...). Counts carry nothing about who.
* **Visitors.** A visitor is counted once a day by a hash of a daily random salt, their
  address and their browser. The salt lives only for its day and is then deleted, so a hash
  can't be traced back to an address or matched with another day's (the approach of Plausible
  and Fathom). No cookies, nothing in browser storage, no account ids, so no consent banner.
  Browsers that send Global Privacy Control or Do Not Track are counted in page views only.
* Counts are buffered in memory and written once a minute, so no request waits on them.

**Storage.** The same Postgres as everything else (``DATABASE_URL``), in memory without it.
Daily counts are kept; visitor hashes for ``VISITOR_DAYS`` days.
"""

from __future__ import annotations

import hashlib
import os
import secrets
import threading
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Protocol

VISITOR_DAYS = 120

# Pages the website reports: its tabs and screens. Anything else is ignored, so the table holds
# only names we chose.
PAGES = frozenset(
    {
        "moves", "park", "trade-for", "team", "league", "scores", "news", "chat",
        "overview", "feed", "film", "verify", "my-leagues", "sign-in",
    }
)  # fmt: skip
# Things done on the server, counted where they happen.
EVENTS = frozenset({"signup", "sign_in", "league_added", "verify", "analysis", "trade_feedback", "chat_post"})


def visitor_address(forwarded_for: str, client) -> str:
    """The visitor's address as Render's proxy saw it. Only ever hashed with the day's salt."""
    return forwarded_for.split(",")[0].strip() or (client.host if client else "")


def today() -> date:
    return datetime.now(UTC).date()


@dataclass
class DayUsage:
    day: date
    visitors: int
    signed_in: int
    counts: dict[str, int]


SCHEMA = """
create table if not exists usage_daily (
    day date not null,
    event text not null,
    count bigint not null,
    primary key (day, event)
);
create table if not exists usage_visitors (
    day date not null,
    visitor text not null,
    signed_in boolean not null default false,
    primary key (day, visitor)
);
create table if not exists usage_salts (
    day date primary key,
    salt text not null
);
"""


class UsageStore(Protocol):
    def salt(self, day: date) -> str: ...
    def save(self, counts: dict[tuple[date, str], int], visitors: dict[tuple[date, str], bool]) -> None: ...
    def days(self, since: date) -> list[DayUsage]: ...


class MemoryUsageStore:
    def __init__(self) -> None:
        self.counts: Counter[tuple[date, str]] = Counter()
        self.visitors: dict[tuple[date, str], bool] = {}
        self.salts: dict[date, str] = {}
        self._lock = threading.Lock()

    def salt(self, day: date) -> str:
        with self._lock:
            for old in [d for d in self.salts if d < day]:
                del self.salts[old]
            return self.salts.setdefault(day, secrets.token_hex(16))

    def save(self, counts: dict[tuple[date, str], int], visitors: dict[tuple[date, str], bool]) -> None:
        with self._lock:
            self.counts.update(counts)
            for key, signed_in in visitors.items():
                self.visitors[key] = self.visitors.get(key, False) or signed_in

    def days(self, since: date) -> list[DayUsage]:
        with self._lock:
            out: dict[date, DayUsage] = {}
            for (day, event), n in self.counts.items():
                if day >= since:
                    out.setdefault(day, DayUsage(day, 0, 0, {})).counts[event] = n
            for (day, _), signed_in in self.visitors.items():
                if day >= since:
                    row = out.setdefault(day, DayUsage(day, 0, 0, {}))
                    row.visitors += 1
                    row.signed_in += signed_in
        return sorted(out.values(), key=lambda r: r.day, reverse=True)


class PostgresUsageStore:
    def __init__(self, url: str):
        from app.db import connector

        self._connect = connector(url)
        with self._connect() as conn:
            conn.execute(SCHEMA)

    def salt(self, day: date) -> str:
        with self._connect() as conn:
            conn.execute("delete from usage_salts where day < %s", (day,))
            conn.execute(
                "insert into usage_salts (day, salt) values (%s, %s) on conflict (day) do nothing",
                (day, secrets.token_hex(16)),
            )
            return conn.execute("select salt from usage_salts where day = %s", (day,)).fetchone()[0]

    def save(self, counts: dict[tuple[date, str], int], visitors: dict[tuple[date, str], bool]) -> None:
        with self._connect() as conn, conn.transaction():
            if counts:
                conn.cursor().executemany(
                    "insert into usage_daily (day, event, count) values (%s, %s, %s) "
                    "on conflict (day, event) do update set count = usage_daily.count + excluded.count",
                    [(day, event, n) for (day, event), n in counts.items()],
                )
            if visitors:
                conn.cursor().executemany(
                    "insert into usage_visitors (day, visitor, signed_in) values (%s, %s, %s) "
                    "on conflict (day, visitor) do update "
                    "set signed_in = usage_visitors.signed_in or excluded.signed_in",
                    [(day, visitor, signed_in) for (day, visitor), signed_in in visitors.items()],
                )
                oldest = min(day for day, _ in visitors) - timedelta(days=VISITOR_DAYS)
                conn.execute("delete from usage_visitors where day < %s", (oldest,))

    def days(self, since: date) -> list[DayUsage]:
        with self._connect() as conn:
            counts = conn.execute("select day, event, count from usage_daily where day >= %s", (since,)).fetchall()
            visitors = conn.execute(
                "select day, count(*), count(*) filter (where signed_in) from usage_visitors "
                "where day >= %s group by day",
                (since,),
            ).fetchall()
        out: dict[date, DayUsage] = {}
        for day, event, n in counts:
            out.setdefault(day, DayUsage(day, 0, 0, {})).counts[event] = int(n)
        for day, total, signed_in in visitors:
            row = out.setdefault(day, DayUsage(day, 0, 0, {}))
            row.visitors, row.signed_in = int(total), int(signed_in)
        return sorted(out.values(), key=lambda r: r.day, reverse=True)


def create_usage_store() -> UsageStore:
    url = os.getenv("DATABASE_URL", "").strip()
    return PostgresUsageStore(url) if url else MemoryUsageStore()


class Usage:
    """Counts in memory until ``flush`` writes them; safe to call from any thread."""

    def __init__(self, store: UsageStore) -> None:
        self.store = store
        self._counts: Counter[tuple[date, str]] = Counter()
        self._visitors: dict[tuple[date, str], bool] = {}
        self._salt: tuple[date, str] | None = None
        self._lock = threading.Lock()

    def count(self, event: str, day: date | None = None) -> None:
        with self._lock:
            self._counts[(day or today(), event)] += 1

    def visit(self, address: str, user_agent: str, signed_in: bool, day: date | None = None) -> None:
        """Counts a visitor once a day by a hash that can't be traced back (see the module doc)."""
        day = day or today()
        visitor = hashlib.sha256(f"{self._day_salt(day)}|{address}|{user_agent}".encode()).hexdigest()[:24]
        with self._lock:
            self._visitors[(day, visitor)] = self._visitors.get((day, visitor), False) or signed_in

    def flush(self) -> None:
        with self._lock:
            counts, visitors = dict(self._counts), dict(self._visitors)
            self._counts.clear()
            self._visitors.clear()
        if not counts and not visitors:
            return
        try:
            self.store.save(counts, visitors)
        except Exception:
            with self._lock:  # put them back for the next try
                self._counts.update(counts)
                for key, signed_in in visitors.items():
                    self._visitors[key] = self._visitors.get(key, False) or signed_in
            raise

    def _day_salt(self, day: date) -> str:
        cached = self._salt
        if cached is None or cached[0] != day:
            cached = (day, self.store.salt(day))
            self._salt = cached
        return cached[1]

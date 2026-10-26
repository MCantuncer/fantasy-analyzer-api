"""Small persistent store for the trade of the day, its "Would you take it?" poll, finished NBA days and NBA news.

With ``DATABASE_URL`` set (any Postgres, e.g. a free Neon database) the day's pick
and the votes survive restarts, which matters on a free server that sleeps when
idle. Without it everything lives in memory, which is fine for local runs and tests.

Nothing about a caller is stored: private-league picks are keyed by a hash of the
cookies (see ``credentials_key``), and a voter is a random id the browser makes up.
"""

from __future__ import annotations

import json
import os
from typing import Protocol

# (league_id, season, credentials key, YYYY-MM-DD)
TradeKey = tuple[int, int, str, str]

SCHEMA = """
create table if not exists daily_trades (
    league_id bigint not null,
    season int not null,
    cred_key text not null,
    day date not null,
    body jsonb not null,
    created_at timestamptz not null default now(),
    primary key (league_id, season, cred_key, day)
);
create table if not exists trade_votes (
    league_id bigint not null,
    season int not null,
    cred_key text not null,
    day date not null,
    team_id int not null,
    voter text not null,
    take boolean not null,
    voted_at timestamptz not null default now(),
    primary key (league_id, season, cred_key, day, team_id, voter)
);
-- The pick can change during the day; a vote counts only for the deal it was cast on.
alter table trade_votes add column if not exists pick_id text not null default '';
create table if not exists nba_days (
    season int not null,
    day int not null,
    body jsonb not null,
    saved_at timestamptz not null default now(),
    primary key (season, day)
);
create table if not exists news_items (
    id text primary key,
    published_at timestamptz not null,
    body jsonb not null,
    seen_at timestamptz not null default now()
);
create index if not exists news_items_published on news_items (published_at);
create table if not exists news_state (
    key text primary key,
    body jsonb not null,
    saved_at timestamptz not null default now()
);
"""
# News older than this is deleted when new items are saved.
NEWS_KEEP_DAYS = 14


class Store(Protocol):
    def get_trade(self, key: TradeKey) -> dict | None: ...
    def save_trade(self, key: TradeKey, body: dict) -> None: ...
    def vote(self, key: TradeKey, pick_id: str, team_id: int, voter: str, take: bool) -> None: ...
    def tally(self, key: TradeKey, pick_id: str) -> dict[int, tuple[int, int]]: ...
    def get_day(self, season: int, day: int) -> dict | None: ...
    def save_day(self, season: int, day: int, body: dict) -> None: ...
    def save_news_items(self, items: list[dict]) -> None: ...
    def news_items(self, since: str) -> list[dict]: ...
    def get_news_state(self, key: str) -> dict | None: ...
    def save_news_state(self, key: str, body: dict) -> None: ...


class MemoryStore:
    def __init__(self) -> None:
        self.trades: dict[TradeKey, dict] = {}
        self.votes: dict[TradeKey, dict[tuple[int, str], tuple[str, bool]]] = {}
        self.days: dict[tuple[int, int], dict] = {}
        self.news: dict[str, dict] = {}
        self.news_state: dict[str, dict] = {}

    def get_trade(self, key: TradeKey) -> dict | None:
        return self.trades.get(key)

    def save_trade(self, key: TradeKey, body: dict) -> None:
        self.trades[key] = body

    def vote(self, key: TradeKey, pick_id: str, team_id: int, voter: str, take: bool) -> None:
        self.votes.setdefault(key, {})[(team_id, voter)] = (pick_id, take)

    def tally(self, key: TradeKey, pick_id: str) -> dict[int, tuple[int, int]]:
        out: dict[int, tuple[int, int]] = {}
        for (team_id, _), (voted_on, take) in self.votes.get(key, {}).items():
            if voted_on != pick_id:
                continue
            yes, no = out.get(team_id, (0, 0))
            out[team_id] = (yes + take, no + (not take))
        return out

    def get_day(self, season: int, day: int) -> dict | None:
        return self.days.get((season, day))

    def save_day(self, season: int, day: int, body: dict) -> None:
        self.days[(season, day)] = body

    def save_news_items(self, items: list[dict]) -> None:
        for item in items:
            self.news.setdefault(item["id"], item)  # first seen wins: a feed re-listing an item changes nothing

    def news_items(self, since: str) -> list[dict]:
        return [i for i in self.news.values() if i["published"] >= since]

    def get_news_state(self, key: str) -> dict | None:
        return self.news_state.get(key)

    def save_news_state(self, key: str, body: dict) -> None:
        self.news_state[key] = body


class PostgresStore:
    def __init__(self, url: str):
        import psycopg

        self._connect = lambda: psycopg.connect(url, autocommit=True)
        with self._connect() as conn:
            conn.execute(SCHEMA)

    def get_trade(self, key: TradeKey) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "select body from daily_trades where league_id=%s and season=%s and cred_key=%s and day=%s", key
            ).fetchone()
        return row[0] if row else None

    def save_trade(self, key: TradeKey, body: dict) -> None:
        # The day's pick is re-checked and can be replaced, so the latest save wins.
        with self._connect() as conn:
            conn.execute(
                "insert into daily_trades (league_id, season, cred_key, day, body) values (%s, %s, %s, %s, %s) "
                "on conflict (league_id, season, cred_key, day) do update set body = excluded.body",
                (*key, json.dumps(body)),
            )

    def vote(self, key: TradeKey, pick_id: str, team_id: int, voter: str, take: bool) -> None:
        with self._connect() as conn:
            conn.execute(
                "insert into trade_votes (league_id, season, cred_key, day, team_id, voter, take, pick_id) "
                "values (%s, %s, %s, %s, %s, %s, %s, %s) "
                "on conflict (league_id, season, cred_key, day, team_id, voter) "
                "do update set take = excluded.take, pick_id = excluded.pick_id, voted_at = now()",
                (*key, team_id, voter, take, pick_id),
            )

    def tally(self, key: TradeKey, pick_id: str) -> dict[int, tuple[int, int]]:
        with self._connect() as conn:
            rows = conn.execute(
                "select team_id, count(*) filter (where take), count(*) filter (where not take) from trade_votes "
                "where league_id=%s and season=%s and cred_key=%s and day=%s and pick_id=%s group by team_id",
                (*key, pick_id),
            ).fetchall()
        return {team_id: (yes, no) for team_id, yes, no in rows}

    def get_day(self, season: int, day: int) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("select body from nba_days where season=%s and day=%s", (season, day)).fetchone()
        return row[0] if row else None

    def save_day(self, season: int, day: int, body: dict) -> None:
        # A finished day is final; a later save (a stat fix) replaces it.
        with self._connect() as conn:
            conn.execute(
                "insert into nba_days (season, day, body) values (%s, %s, %s) "
                "on conflict (season, day) do update set body = excluded.body, saved_at = now()",
                (season, day, json.dumps(body)),
            )

    def save_news_items(self, items: list[dict]) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    "insert into news_items (id, published_at, body) values (%s, %s, %s) on conflict (id) do nothing",
                    [(i["id"], i["published"], json.dumps(i)) for i in items],
                )
            conn.execute(f"delete from news_items where published_at < now() - interval '{NEWS_KEEP_DAYS} days'")

    def news_items(self, since: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("select body from news_items where published_at >= %s", (since,)).fetchall()
        return [r[0] for r in rows]

    def get_news_state(self, key: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("select body from news_state where key=%s", (key,)).fetchone()
        return row[0] if row else None

    def save_news_state(self, key: str, body: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                "insert into news_state (key, body) values (%s, %s) "
                "on conflict (key) do update set body = excluded.body, saved_at = now()",
                (key, json.dumps(body)),
            )


def create_store() -> Store:
    url = os.getenv("DATABASE_URL", "").strip()
    return PostgresStore(url) if url else MemoryStore()

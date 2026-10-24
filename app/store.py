"""Small persistent store for the trade of the day, its "Would you take it?" poll, and finished NBA days.

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
create table if not exists nba_days (
    season int not null,
    day int not null,
    body jsonb not null,
    saved_at timestamptz not null default now(),
    primary key (season, day)
);
"""


class Store(Protocol):
    def get_trade(self, key: TradeKey) -> dict | None: ...
    def save_trade(self, key: TradeKey, body: dict) -> None: ...
    def vote(self, key: TradeKey, team_id: int, voter: str, take: bool) -> None: ...
    def tally(self, key: TradeKey) -> dict[int, tuple[int, int]]: ...
    def get_day(self, season: int, day: int) -> dict | None: ...
    def save_day(self, season: int, day: int, body: dict) -> None: ...


class MemoryStore:
    def __init__(self) -> None:
        self.trades: dict[TradeKey, dict] = {}
        self.votes: dict[TradeKey, dict[tuple[int, str], bool]] = {}
        self.days: dict[tuple[int, int], dict] = {}

    def get_trade(self, key: TradeKey) -> dict | None:
        return self.trades.get(key)

    def save_trade(self, key: TradeKey, body: dict) -> None:
        self.trades.setdefault(key, body)

    def vote(self, key: TradeKey, team_id: int, voter: str, take: bool) -> None:
        self.votes.setdefault(key, {})[(team_id, voter)] = take

    def tally(self, key: TradeKey) -> dict[int, tuple[int, int]]:
        out: dict[int, tuple[int, int]] = {}
        for (team_id, _), take in self.votes.get(key, {}).items():
            yes, no = out.get(team_id, (0, 0))
            out[team_id] = (yes + take, no + (not take))
        return out

    def get_day(self, season: int, day: int) -> dict | None:
        return self.days.get((season, day))

    def save_day(self, season: int, day: int, body: dict) -> None:
        self.days[(season, day)] = body


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
        # The first pick of the day stands, so every visitor sees the same trade.
        with self._connect() as conn:
            conn.execute(
                "insert into daily_trades (league_id, season, cred_key, day, body) values (%s, %s, %s, %s, %s) "
                "on conflict do nothing",
                (*key, json.dumps(body)),
            )

    def vote(self, key: TradeKey, team_id: int, voter: str, take: bool) -> None:
        with self._connect() as conn:
            conn.execute(
                "insert into trade_votes (league_id, season, cred_key, day, team_id, voter, take) "
                "values (%s, %s, %s, %s, %s, %s, %s) "
                "on conflict (league_id, season, cred_key, day, team_id, voter) "
                "do update set take = excluded.take, voted_at = now()",
                (*key, team_id, voter, take),
            )

    def tally(self, key: TradeKey) -> dict[int, tuple[int, int]]:
        with self._connect() as conn:
            rows = conn.execute(
                "select team_id, count(*) filter (where take), count(*) filter (where not take) from trade_votes "
                "where league_id=%s and season=%s and cred_key=%s and day=%s group by team_id",
                key,
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


def create_store() -> Store:
    url = os.getenv("DATABASE_URL", "").strip()
    return PostgresStore(url) if url else MemoryStore()

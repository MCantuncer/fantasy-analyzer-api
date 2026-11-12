"""Small persistent store for the trade of the day, its "Would you take it?" poll, finished NBA days, NBA news
and each team's last analysis report.

With ``DATABASE_URL`` set (any Postgres, e.g. a free Neon database) the day's pick
and the votes survive restarts, which matters on a free server that sleeps when
idle. Without it everything lives in memory, which is fine for local runs and tests.

Nothing about a caller is stored: private-league picks are keyed by a hash of the
cookies (see ``credentials_key``), and a voter is a random id the browser makes up.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
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
-- A finished matchup week's recap per league (app/recap.py). Final once saved.
create table if not exists week_recaps (
    league_id bigint not null,
    season int not null,
    period int not null,
    body jsonb not null,
    saved_at timestamptz not null default now(),
    primary key (league_id, season, period)
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
-- A team's last finished analysis (app/saved.py), keyed by its settings, with the league
-- fingerprint it was built on. Only its verified manager can read it back.
create table if not exists saved_reports (
    key text primary key,
    league_id bigint not null,
    season int not null,
    team_id int not null,
    body jsonb not null,
    fingerprint jsonb not null,
    built_at timestamptz not null
);
create index if not exists saved_reports_league on saved_reports (league_id, season);
-- The public sample league (app/sample.py): one day's run, each team's masked report and,
-- under team 0, the day's index. Written by the daily job; read by anyone.
create table if not exists sample_runs (
    season int not null,
    day date not null,
    team_id int not null,
    body jsonb not null,
    saved_at timestamptz not null default now(),
    primary key (season, day, team_id)
);
"""
# News older than this is deleted when new items are saved.
NEWS_KEEP_DAYS = 14
# Saved reports nobody opened again for this long are deleted when another is saved.
REPORT_KEEP_DAYS = 7
# Sample league runs older than this are deleted when a new day's index is saved.
SAMPLE_KEEP_DAYS = 7
# Team 0 in sample_runs is the day's index, not a team.
SAMPLE_INDEX = 0


class Store(Protocol):
    def get_trade(self, key: TradeKey) -> dict | None: ...
    def save_trade(self, key: TradeKey, body: dict) -> None: ...
    def vote(self, key: TradeKey, pick_id: str, team_id: int, voter: str, take: bool) -> None: ...
    def tally(self, key: TradeKey, pick_id: str) -> dict[int, tuple[int, int]]: ...
    def get_day(self, season: int, day: int) -> dict | None: ...
    def save_day(self, season: int, day: int, body: dict) -> None: ...
    def get_recap(self, league_id: int, season: int, period: int) -> dict | None: ...
    def save_recap(self, league_id: int, season: int, period: int, body: dict) -> None: ...
    def save_news_items(self, items: list[dict]) -> None: ...
    def news_items(self, since: str) -> list[dict]: ...
    def get_news_state(self, key: str) -> dict | None: ...
    def save_news_state(self, key: str, body: dict) -> None: ...
    def get_report(self, key: str) -> dict | None: ...
    def save_report(self, key: str, league_id: int, season: int, team_id: int, row: dict) -> None: ...
    def recent_reports(self, since: datetime) -> list[dict]: ...
    def save_sample(self, season: int, day: str, team_id: int, body: dict) -> None: ...
    def get_sample(self, season: int, day: str, team_id: int) -> dict | None: ...
    def sample_days(self, season: int) -> list[str]: ...


class MemoryStore:
    def __init__(self) -> None:
        self.trades: dict[TradeKey, dict] = {}
        self.votes: dict[TradeKey, dict[tuple[int, str], tuple[str, bool]]] = {}
        self.days: dict[tuple[int, int], dict] = {}
        self.recaps: dict[tuple[int, int, int], dict] = {}
        self.news: dict[str, dict] = {}
        self.news_state: dict[str, dict] = {}
        self.reports: dict[str, dict] = {}
        self.samples: dict[tuple[int, str, int], dict] = {}

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

    def get_recap(self, league_id: int, season: int, period: int) -> dict | None:
        return self.recaps.get((league_id, season, period))

    def save_recap(self, league_id: int, season: int, period: int, body: dict) -> None:
        self.recaps[(league_id, season, period)] = body

    def save_news_items(self, items: list[dict]) -> None:
        for item in items:
            self.news.setdefault(item["id"], item)  # first seen wins: a feed re-listing an item changes nothing

    def news_items(self, since: str) -> list[dict]:
        return [i for i in self.news.values() if i["published"] >= since]

    def get_news_state(self, key: str) -> dict | None:
        return self.news_state.get(key)

    def save_news_state(self, key: str, body: dict) -> None:
        self.news_state[key] = body

    def get_report(self, key: str) -> dict | None:
        return self.reports.get(key)

    def save_report(self, key: str, league_id: int, season: int, team_id: int, row: dict) -> None:
        self.reports[key] = {**row, "league_id": league_id, "season": season, "team_id": team_id}

    def recent_reports(self, since: datetime) -> list[dict]:
        return [
            {k: r[k] for k in ("league_id", "season", "team_id", "fingerprint", "built_at")} | {"key": key}
            for key, r in self.reports.items()
            if datetime.fromisoformat(r["built_at"]) >= since
        ]

    def save_sample(self, season: int, day: str, team_id: int, body: dict) -> None:
        self.samples[(season, day, team_id)] = body

    def get_sample(self, season: int, day: str, team_id: int) -> dict | None:
        return self.samples.get((season, day, team_id))

    def sample_days(self, season: int) -> list[str]:
        """Days with a finished run (an index), newest first."""
        return sorted({d for s, d, t in self.samples if s == season and t == SAMPLE_INDEX}, reverse=True)


class PostgresStore:
    def __init__(self, url: str):
        from app.db import connector

        self._connect = connector(url)
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

    def get_recap(self, league_id: int, season: int, period: int) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "select body from week_recaps where league_id=%s and season=%s and period=%s",
                (league_id, season, period),
            ).fetchone()
        return row[0] if row else None

    def save_recap(self, league_id: int, season: int, period: int, body: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                "insert into week_recaps (league_id, season, period, body) values (%s, %s, %s, %s) "
                "on conflict (league_id, season, period) do update set body = excluded.body, saved_at = now()",
                (league_id, season, period, json.dumps(body)),
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

    def get_report(self, key: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("select body, fingerprint, built_at from saved_reports where key=%s", (key,)).fetchone()
        if not row:
            return None
        return {"body": row[0], "fingerprint": row[1], "built_at": row[2].isoformat()}

    def save_report(self, key: str, league_id: int, season: int, team_id: int, row: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                "insert into saved_reports (key, league_id, season, team_id, body, fingerprint, built_at) "
                "values (%s, %s, %s, %s, %s, %s, %s) on conflict (key) do update set body = excluded.body, "
                "fingerprint = excluded.fingerprint, built_at = excluded.built_at",
                (
                    key,
                    league_id,
                    season,
                    team_id,
                    json.dumps(row["body"]),
                    json.dumps(row["fingerprint"]),
                    row["built_at"],
                ),
            )
            conn.execute(f"delete from saved_reports where built_at < now() - interval '{REPORT_KEEP_DAYS} days'")

    def recent_reports(self, since: datetime) -> list[dict]:
        """Reports built since ``since``, without their bodies: what the injury watch checks."""
        with self._connect() as conn:
            rows = conn.execute(
                "select key, league_id, season, team_id, fingerprint, built_at from saved_reports where built_at >= %s",
                (since,),
            ).fetchall()
        return [
            {"key": k, "league_id": lg, "season": s, "team_id": t, "fingerprint": fp, "built_at": b.isoformat()}
            for k, lg, s, t, fp, b in rows
        ]

    def save_sample(self, season: int, day: str, team_id: int, body: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                "insert into sample_runs (season, day, team_id, body) values (%s, %s, %s, %s) "
                "on conflict (season, day, team_id) do update set body = excluded.body, saved_at = now()",
                (season, day, team_id, json.dumps(body)),
            )
            if team_id == SAMPLE_INDEX:
                conn.execute(
                    f"delete from sample_runs where season=%s and day < current_date - {SAMPLE_KEEP_DAYS}", (season,)
                )

    def get_sample(self, season: int, day: str, team_id: int) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "select body from sample_runs where season=%s and day=%s and team_id=%s", (season, day, team_id)
            ).fetchone()
        return row[0] if row else None

    def sample_days(self, season: int) -> list[str]:
        """Days with a finished run (an index), newest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "select day from sample_runs where season=%s and team_id=%s order by day desc", (season, SAMPLE_INDEX)
            ).fetchall()
        return [d.isoformat() for (d,) in rows]


def create_store() -> Store:
    url = os.getenv("DATABASE_URL", "").strip()
    return PostgresStore(url) if url else MemoryStore()

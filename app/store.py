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
-- Each player's injury tag as it stood before his game tipped off (app/pregame.py), per NBA day.
create table if not exists pregame_status (
    season int not null,
    day int not null,
    body jsonb not null,
    saved_at timestamptz not null default now(),
    primary key (season, day)
);
-- Each player's next-game line as it stood at tip-off, and once the night settles what he did
-- (app/linelog.py), one row per NBA day. Never trimmed: it shows over a season who lands near his line.
create table if not exists player_lines (
    season int not null,
    day int not null,
    settled boolean not null default false,
    body jsonb not null,
    saved_at timestamptz not null default now(),
    primary key (season, day)
);
-- iddaa's NBA lines as they stood at tip-off (app/iddaa.py), one row per NBA day. Comparison only:
-- never read by the model.
create table if not exists book_lines (
    season int not null,
    day int not null,
    body jsonb not null,
    saved_at timestamptz not null default now(),
    primary key (season, day)
);
-- Each team's starting five as nba.com listed it before tip-off, every change seen (app/startlog.py),
-- one row per NBA day. Recorded only: nothing reads it into the model yet.
create table if not exists lineup_looks (
    season int not null,
    day int not null,
    body jsonb not null,
    saved_at timestamptz not null default now(),
    primary key (season, day)
);
-- Other ways to call each game (app/tunings.py), as they stood at tip-off, one row per NBA day.
-- Recorded only: nothing reads it into the model or the site's numbers.
create table if not exists tuning_calls (
    season int not null,
    day int not null,
    body jsonb not null,
    saved_at timestamptz not null default now(),
    primary key (season, day)
);
-- Who did not play each night and ESPN's reason (app/benched.py), per NBA day. Final once saved.
create table if not exists sat_out (
    season int not null,
    day int not null,
    body jsonb not null,
    saved_at timestamptz not null default now(),
    primary key (season, day)
);
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
-- Outlet items that name a player, with their emotion tags (app/news/emotion.py). Never trimmed:
-- a later test needs months of them.
create table if not exists news_log (
    id text primary key,
    published_at timestamptz not null,
    body jsonb not null,
    seen_at timestamptz not null default now()
);
create index if not exists news_log_published on news_log (published_at);
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
    def get_pregame(self, season: int, day: int) -> dict | None: ...
    def save_pregame(self, season: int, day: int, body: dict) -> None: ...
    def get_player_lines(self, season: int, day: int) -> dict | None: ...
    def save_player_lines(self, season: int, day: int, body: dict, settled: bool) -> None: ...
    def player_lines(self, season: int, settled: bool) -> list[tuple[int, dict]]: ...
    def get_book_lines(self, season: int, day: int) -> dict | None: ...
    def save_book_lines(self, season: int, day: int, body: dict) -> None: ...
    def book_line_days(self, season: int) -> list[int]: ...
    def get_lineups(self, season: int, day: int) -> dict | None: ...
    def save_lineups(self, season: int, day: int, body: dict) -> None: ...
    def lineup_days(self, season: int) -> list[int]: ...
    def get_tunings(self, season: int, day: int) -> dict | None: ...
    def save_tunings(self, season: int, day: int, body: dict) -> None: ...
    def tuning_days(self, season: int) -> list[int]: ...
    def sat_out(self, season: int, since: int) -> dict[int, dict]: ...
    def save_sat_out(self, season: int, day: int, body: dict) -> None: ...
    def get_recap(self, league_id: int, season: int, period: int) -> dict | None: ...
    def save_recap(self, league_id: int, season: int, period: int, body: dict) -> None: ...
    def save_news_items(self, items: list[dict]) -> None: ...
    def news_items(self, since: str) -> list[dict]: ...
    def save_news_log(self, rows: list[dict]) -> None: ...
    def news_log(self, since: str, limit: int, tagged: bool = False) -> list[dict]: ...
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
        self.pregame: dict[tuple[int, int], dict] = {}
        self.lines: dict[tuple[int, int], tuple[bool, dict]] = {}
        self.book: dict[tuple[int, int], dict] = {}
        self.lineups: dict[tuple[int, int], dict] = {}
        self.tunings: dict[tuple[int, int], dict] = {}
        self.sat: dict[tuple[int, int], dict] = {}
        self.recaps: dict[tuple[int, int, int], dict] = {}
        self.news: dict[str, dict] = {}
        self.log: dict[str, dict] = {}
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

    def get_pregame(self, season: int, day: int) -> dict | None:
        return self.pregame.get((season, day))

    def save_pregame(self, season: int, day: int, body: dict) -> None:
        self.pregame[(season, day)] = body

    def get_player_lines(self, season: int, day: int) -> dict | None:
        got = self.lines.get((season, day))
        return got[1] if got else None

    def save_player_lines(self, season: int, day: int, body: dict, settled: bool) -> None:
        self.lines[(season, day)] = (settled, body)

    def player_lines(self, season: int, settled: bool) -> list[tuple[int, dict]]:
        return sorted((d, b) for (s, d), (done, b) in self.lines.items() if s == season and done == settled)

    def get_book_lines(self, season: int, day: int) -> dict | None:
        return self.book.get((season, day))

    def save_book_lines(self, season: int, day: int, body: dict) -> None:
        self.book[(season, day)] = body

    def book_line_days(self, season: int) -> list[int]:
        return sorted(d for s, d in self.book if s == season)

    def get_lineups(self, season: int, day: int) -> dict | None:
        return self.lineups.get((season, day))

    def save_lineups(self, season: int, day: int, body: dict) -> None:
        self.lineups[(season, day)] = body

    def lineup_days(self, season: int) -> list[int]:
        return sorted(d for s, d in self.lineups if s == season)

    def get_tunings(self, season: int, day: int) -> dict | None:
        return self.tunings.get((season, day))

    def save_tunings(self, season: int, day: int, body: dict) -> None:
        self.tunings[(season, day)] = body

    def tuning_days(self, season: int) -> list[int]:
        return sorted(d for s, d in self.tunings if s == season)

    def sat_out(self, season: int, since: int) -> dict[int, dict]:
        """day -> who did not play and why, for the saved nights of a season from day ``since`` on."""
        return {d: body for (s, d), body in self.sat.items() if s == season and d >= since}

    def save_sat_out(self, season: int, day: int, body: dict) -> None:
        self.sat[(season, day)] = body

    def get_recap(self, league_id: int, season: int, period: int) -> dict | None:
        return self.recaps.get((league_id, season, period))

    def save_recap(self, league_id: int, season: int, period: int, body: dict) -> None:
        self.recaps[(league_id, season, period)] = body

    def save_news_items(self, items: list[dict]) -> None:
        for item in items:
            self.news.setdefault(item["id"], item)  # first seen wins: a feed re-listing an item changes nothing

    def news_items(self, since: str) -> list[dict]:
        return [i for i in self.news.values() if i["published"] >= since]

    def save_news_log(self, rows: list[dict]) -> None:
        for row in rows:
            self.log.setdefault(row["id"], row)  # first seen wins, like news_items

    def news_log(self, since: str, limit: int, tagged: bool = False) -> list[dict]:
        """Logged items published since then, newest first; ``tagged``: only those with an emotion tag."""
        rows = [r for r in self.log.values() if r["published"] >= since and (r["tags"] or not tagged)]
        return sorted(rows, key=lambda r: r["published"], reverse=True)[:limit]

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

        self._connect = connector(url, SCHEMA)  # tables made on first use

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

    def get_pregame(self, season: int, day: int) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("select body from pregame_status where season=%s and day=%s", (season, day)).fetchone()
        return row[0] if row else None

    def save_pregame(self, season: int, day: int, body: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                "insert into pregame_status (season, day, body) values (%s, %s, %s) "
                "on conflict (season, day) do update set body = excluded.body, saved_at = now()",
                (season, day, json.dumps(body)),
            )

    def get_player_lines(self, season: int, day: int) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("select body from player_lines where season=%s and day=%s", (season, day)).fetchone()
        return row[0] if row else None

    def save_player_lines(self, season: int, day: int, body: dict, settled: bool) -> None:
        with self._connect() as conn:
            conn.execute(
                "insert into player_lines (season, day, settled, body) values (%s, %s, %s, %s) "
                "on conflict (season, day) do update set settled = excluded.settled, body = excluded.body, "
                "saved_at = now()",
                (season, day, settled, json.dumps(body, separators=(",", ":"))),
            )

    def player_lines(self, season: int, settled: bool) -> list[tuple[int, dict]]:
        with self._connect() as conn:
            rows = conn.execute(
                "select day, body from player_lines where season=%s and settled=%s order by day", (season, settled)
            ).fetchall()
        return [(day, body) for day, body in rows]

    def get_book_lines(self, season: int, day: int) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("select body from book_lines where season=%s and day=%s", (season, day)).fetchone()
        return row[0] if row else None

    def save_book_lines(self, season: int, day: int, body: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                "insert into book_lines (season, day, body) values (%s, %s, %s) "
                "on conflict (season, day) do update set body = excluded.body, saved_at = now()",
                (season, day, json.dumps(body, separators=(",", ":"))),
            )

    def book_line_days(self, season: int) -> list[int]:
        with self._connect() as conn:
            rows = conn.execute("select day from book_lines where season=%s order by day", (season,)).fetchall()
        return [day for (day,) in rows]

    def get_lineups(self, season: int, day: int) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("select body from lineup_looks where season=%s and day=%s", (season, day)).fetchone()
        return row[0] if row else None

    def save_lineups(self, season: int, day: int, body: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                "insert into lineup_looks (season, day, body) values (%s, %s, %s) "
                "on conflict (season, day) do update set body = excluded.body, saved_at = now()",
                (season, day, json.dumps(body, separators=(",", ":"))),
            )

    def lineup_days(self, season: int) -> list[int]:
        with self._connect() as conn:
            rows = conn.execute("select day from lineup_looks where season=%s order by day", (season,)).fetchall()
        return [day for (day,) in rows]

    def get_tunings(self, season: int, day: int) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("select body from tuning_calls where season=%s and day=%s", (season, day)).fetchone()
        return row[0] if row else None

    def save_tunings(self, season: int, day: int, body: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                "insert into tuning_calls (season, day, body) values (%s, %s, %s) "
                "on conflict (season, day) do update set body = excluded.body, saved_at = now()",
                (season, day, json.dumps(body, separators=(",", ":"))),
            )

    def tuning_days(self, season: int) -> list[int]:
        with self._connect() as conn:
            rows = conn.execute("select day from tuning_calls where season=%s order by day", (season,)).fetchall()
        return [day for (day,) in rows]

    def sat_out(self, season: int, since: int) -> dict[int, dict]:
        with self._connect() as conn:
            rows = conn.execute("select day, body from sat_out where season=%s and day>=%s", (season, since)).fetchall()
        return {day: body for day, body in rows}

    def save_sat_out(self, season: int, day: int, body: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                "insert into sat_out (season, day, body) values (%s, %s, %s) on conflict (season, day) do nothing",
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

    def save_news_log(self, rows: list[dict]) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.executemany(
                "insert into news_log (id, published_at, body) values (%s, %s, %s) on conflict (id) do nothing",
                [(r["id"], r["published"], json.dumps(r)) for r in rows],
            )

    def news_log(self, since: str, limit: int, tagged: bool = False) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "select body from news_log where published_at >= %s "
                "and (not %s or jsonb_array_length(body->'tags') > 0) order by published_at desc limit %s",
                (since, tagged, limit),
            ).fetchall()
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
    if not url:
        return MemoryStore()
    from app.quiet import QuietStore  # the watches' records wait in memory so the database can sleep

    return QuietStore(PostgresStore(url))  # type: ignore[return-value]

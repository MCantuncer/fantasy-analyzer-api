"""Feedback on trade ideas: what a manager thought of a deal we suggested, and the deal as they saw it.

Feedback never hides a card or tunes the model by itself. Each entry keeps a snapshot
of the idea exactly as it was shown (players, odds, category changes, reasons) and
the build of the API that made it, so the checks that verify it can run later on the
same deal even after rosters change.

One entry per manager and deal: giving feedback on the same deal again replaces the
earlier entry, so a changed mind doesn't count twice.

When the manager's checks run on a deal, their verdicts are kept on the entry with the
time they ran, so later work can count which complaints the data backed up.

**Storage.** The same Postgres as the trade of the day (``DATABASE_URL``), in memory without it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, Protocol

Rating = Literal["bad", "good"]
Reason = Literal["fake_gain", "wont_accept", "bad_for_build", "player_concern", "already_tried", "other"]
Source = Literal["trades", "edge", "shop", "trade_for", "trade_of_day"]

TEXT_MAX_CHARS = 500
LIST_LIMIT = 100


def model_version() -> str:
    """The API build that made the idea: Render's commit when deployed, "dev" locally."""
    return os.getenv("RENDER_GIT_COMMIT", "").strip()[:12] or "dev"


def idea_key(source: str, partner_team_id: int, give: list[int], get: list[int]) -> str:
    """Names one deal for one source, whatever order the players were listed in."""
    return f"{source}:{deal_key(partner_team_id, give, get)}"


def deal_key(partner_team_id: int, give: list[int], get: list[int]) -> str:
    """Names one deal whatever list it came from: the end of every idea_key for it."""
    ids = lambda xs: ",".join(str(x) for x in sorted(xs))  # noqa: E731
    return f"{partner_team_id}:{ids(give)}>{ids(get)}"


@dataclass
class Feedback:
    id: int
    user_id: int
    league_id: int
    season: int
    team_id: int
    source: str
    idea_key: str
    rating: str
    reasons: list[str]
    text: str
    idea: dict  # the TradeIdea as it was shown
    model_version: str
    created_at: datetime
    checks: list[dict] | None = None  # CheckVerdicts from the last time the checks ran on this deal
    checked_at: datetime | None = None
    reading: dict | None = None  # how the note was read (app.reader.Reading), None until read


SCHEMA = """
create table if not exists trade_feedback (
    id bigserial primary key,
    user_id bigint not null,
    league_id bigint not null,
    season int not null,
    team_id int not null,
    source text not null,
    idea_key text not null,
    rating text not null,
    reasons jsonb not null,
    text text not null,
    idea jsonb not null,
    model_version text not null,
    created_at timestamptz not null default now(),
    unique (user_id, league_id, season, idea_key)
);
alter table trade_feedback add column if not exists checks jsonb;
alter table trade_feedback add column if not exists checked_at timestamptz;
alter table trade_feedback add column if not exists reading jsonb;
create index if not exists trade_feedback_by_league on trade_feedback (league_id, season, created_at desc);
"""


class FeedbackStore(Protocol):
    def save(self, f: Feedback) -> Feedback: ...
    def mine(self, user_id: int, league_id: int, season: int, limit: int = LIST_LIMIT) -> list[Feedback]: ...
    def record_checks(self, user_id: int, league_id: int, season: int, deal: str, checks: list[dict]) -> int: ...
    def for_deal(self, user_id: int, league_id: int, season: int, deal: str) -> list[Feedback]: ...
    def record_reading(self, feedback_id: int, reading: dict) -> None: ...


class MemoryFeedbackStore:
    def __init__(self) -> None:
        self.rows: dict[tuple[int, int, int, str], Feedback] = {}
        self._next = 1

    def save(self, f: Feedback) -> Feedback:
        key = (f.user_id, f.league_id, f.season, f.idea_key)
        old = self.rows.get(key)
        f.id = old.id if old else self._next
        if old is None:
            self._next += 1
        f.created_at = datetime.now(UTC)
        f.checks, f.checked_at, f.reading = None, None, None  # changed feedback waits for a fresh check and read
        self.rows[key] = f
        return f

    def record_checks(self, user_id: int, league_id: int, season: int, deal: str, checks: list[dict]) -> int:
        now = datetime.now(UTC)
        hits = [
            f
            for (u, lg, s, k), f in self.rows.items()
            if (u, lg, s) == (user_id, league_id, season) and k.split(":", 1)[1] == deal
        ]
        for f in hits:
            f.checks, f.checked_at = checks, now
        return len(hits)

    def for_deal(self, user_id: int, league_id: int, season: int, deal: str) -> list[Feedback]:
        return [
            f
            for (u, lg, s, k), f in self.rows.items()
            if (u, lg, s) == (user_id, league_id, season) and k.split(":", 1)[1] == deal
        ]

    def record_reading(self, feedback_id: int, reading: dict) -> None:
        for f in self.rows.values():
            if f.id == feedback_id:
                f.reading = reading

    def mine(self, user_id: int, league_id: int, season: int, limit: int = LIST_LIMIT) -> list[Feedback]:
        rows = [f for (u, lg, s, _), f in self.rows.items() if (u, lg, s) == (user_id, league_id, season)]
        return sorted(rows, key=lambda f: f.created_at, reverse=True)[:limit]


class PostgresFeedbackStore:
    _COLS = (
        "id, user_id, league_id, season, team_id, source, idea_key, rating, reasons, text, idea, "
        "model_version, created_at, checks, checked_at, reading"
    )

    def __init__(self, url: str):
        import psycopg

        self._connect = lambda: psycopg.connect(url, autocommit=True)
        with self._connect() as conn:
            conn.execute(SCHEMA)

    def save(self, f: Feedback) -> Feedback:
        with self._connect() as conn:
            row = conn.execute(
                "insert into trade_feedback (user_id, league_id, season, team_id, source, idea_key, rating, "
                "reasons, text, idea, model_version) values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
                "on conflict (user_id, league_id, season, idea_key) do update set team_id=excluded.team_id, "
                "rating=excluded.rating, reasons=excluded.reasons, text=excluded.text, idea=excluded.idea, "
                "model_version=excluded.model_version, created_at=now(), checks=null, checked_at=null, reading=null "
                f"returning {self._COLS}",
                (
                    f.user_id,
                    f.league_id,
                    f.season,
                    f.team_id,
                    f.source,
                    f.idea_key,
                    f.rating,
                    json.dumps(f.reasons),
                    f.text,
                    json.dumps(f.idea),
                    f.model_version,
                ),
            ).fetchone()
        return Feedback(*row)

    def mine(self, user_id: int, league_id: int, season: int, limit: int = LIST_LIMIT) -> list[Feedback]:
        with self._connect() as conn:
            rows = conn.execute(
                f"select {self._COLS} from trade_feedback where user_id=%s and league_id=%s and season=%s "
                "order by created_at desc limit %s",
                (user_id, league_id, season, limit),
            ).fetchall()
        return [Feedback(*r) for r in rows]

    def record_checks(self, user_id: int, league_id: int, season: int, deal: str, checks: list[dict]) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "update trade_feedback set checks=%s, checked_at=now() where user_id=%s and league_id=%s "
                "and season=%s and substr(idea_key, strpos(idea_key, ':') + 1)=%s",
                (json.dumps(checks), user_id, league_id, season, deal),
            )
        return cur.rowcount

    def for_deal(self, user_id: int, league_id: int, season: int, deal: str) -> list[Feedback]:
        with self._connect() as conn:
            rows = conn.execute(
                f"select {self._COLS} from trade_feedback where user_id=%s and league_id=%s and season=%s "
                "and substr(idea_key, strpos(idea_key, ':') + 1)=%s order by created_at desc",
                (user_id, league_id, season, deal),
            ).fetchall()
        return [Feedback(*r) for r in rows]

    def record_reading(self, feedback_id: int, reading: dict) -> None:
        with self._connect() as conn:
            conn.execute("update trade_feedback set reading=%s where id=%s", (json.dumps(reading), feedback_id))


def create_feedback_store() -> FeedbackStore:
    url = os.getenv("DATABASE_URL", "").strip()
    return PostgresFeedbackStore(url) if url else MemoryFeedbackStore()

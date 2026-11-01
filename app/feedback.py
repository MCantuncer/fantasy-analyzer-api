"""Feedback on trade ideas: what a manager thought of a deal we suggested, and the deal as they saw it.

Feedback never hides a card or tunes the model by itself. Each entry keeps a snapshot
of the idea exactly as it was shown (players, odds, category changes, reasons) and
the build of the API that made it, so the checks that verify it can run later on the
same deal even after rosters change.

One entry per manager and deal: giving feedback on the same deal again replaces the
earlier entry, so a changed mind doesn't count twice.

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
    ids = lambda xs: ",".join(str(x) for x in sorted(xs))  # noqa: E731
    return f"{source}:{partner_team_id}:{ids(give)}>{ids(get)}"


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
create index if not exists trade_feedback_by_league on trade_feedback (league_id, season, created_at desc);
"""


class FeedbackStore(Protocol):
    def save(self, f: Feedback) -> Feedback: ...
    def mine(self, user_id: int, league_id: int, season: int, limit: int = LIST_LIMIT) -> list[Feedback]: ...


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
        self.rows[key] = f
        return f

    def mine(self, user_id: int, league_id: int, season: int, limit: int = LIST_LIMIT) -> list[Feedback]:
        rows = [f for (u, lg, s, _), f in self.rows.items() if (u, lg, s) == (user_id, league_id, season)]
        return sorted(rows, key=lambda f: f.created_at, reverse=True)[:limit]


class PostgresFeedbackStore:
    _COLS = (
        "id, user_id, league_id, season, team_id, source, idea_key, rating, reasons, text, idea, "
        "model_version, created_at"
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
                "model_version=excluded.model_version, created_at=now() "
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


def create_feedback_store() -> FeedbackStore:
    url = os.getenv("DATABASE_URL", "").strip()
    return PostgresFeedbackStore(url) if url else MemoryFeedbackStore()

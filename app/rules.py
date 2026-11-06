"""A manager's own trade rules: ideas they never want to see again, said once.

Rules come from feedback the checks backed up ("Protect TO" after a confirmed build
conflict, "Skip Los Avokados" after "they'd never accept") or are added by hand. A rule
only skips ideas that break it; it never changes the model's numbers, and the manager
can see and remove every rule.

Kinds:
  * protect_category: skip ideas that cost this category (value: its name, e.g. "TO")
  * skip_team: skip ideas with this team (value: the team id)
  * no_injury_prone: skip ideas bringing in an injury-prone player (value: "")

One rule per manager, league, season, kind and value. **Storage:** the same Postgres as
feedback (``DATABASE_URL``), in memory without it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, Protocol

RuleKind = Literal["protect_category", "skip_team", "no_injury_prone"]
MAX_RULES = 20


@dataclass
class Rule:
    id: int
    user_id: int
    league_id: int
    season: int
    kind: str
    value: str
    label: str  # how the rule reads, e.g. "Protect TO"
    why: str  # where it came from, in a sentence
    created_at: datetime


SCHEMA = """
create table if not exists trade_rules (
    id bigserial primary key,
    user_id bigint not null,
    league_id bigint not null,
    season int not null,
    kind text not null,
    value text not null,
    label text not null,
    why text not null,
    created_at timestamptz not null default now(),
    unique (user_id, league_id, season, kind, value)
);
"""


class RuleStore(Protocol):
    def add(self, r: Rule) -> Rule: ...
    def mine(self, user_id: int, league_id: int, season: int) -> list[Rule]: ...
    def remove(self, user_id: int, rule_id: int) -> bool: ...


class MemoryRuleStore:
    def __init__(self) -> None:
        self.rows: dict[tuple[int, int, int, str, str], Rule] = {}
        self._next = 1

    def add(self, r: Rule) -> Rule:
        key = (r.user_id, r.league_id, r.season, r.kind, r.value)
        if key in self.rows:
            return self.rows[key]
        r.id, r.created_at = self._next, datetime.now(UTC)
        self._next += 1
        self.rows[key] = r
        return r

    def mine(self, user_id: int, league_id: int, season: int) -> list[Rule]:
        rows = [r for r in self.rows.values() if (r.user_id, r.league_id, r.season) == (user_id, league_id, season)]
        return sorted(rows, key=lambda r: r.id)

    def remove(self, user_id: int, rule_id: int) -> bool:
        for key, r in list(self.rows.items()):
            if r.id == rule_id and r.user_id == user_id:
                del self.rows[key]
                return True
        return False


class PostgresRuleStore:
    _COLS = "id, user_id, league_id, season, kind, value, label, why, created_at"

    def __init__(self, url: str):
        from app.db import connector

        self._connect = connector(url)
        with self._connect() as conn:
            conn.execute(SCHEMA)

    def add(self, r: Rule) -> Rule:
        with self._connect() as conn:
            row = conn.execute(
                "insert into trade_rules (user_id, league_id, season, kind, value, label, why) "
                "values (%s, %s, %s, %s, %s, %s, %s) "
                "on conflict (user_id, league_id, season, kind, value) do update set kind=excluded.kind "
                f"returning {self._COLS}",
                (r.user_id, r.league_id, r.season, r.kind, r.value, r.label, r.why),
            ).fetchone()
        return Rule(*row)

    def mine(self, user_id: int, league_id: int, season: int) -> list[Rule]:
        with self._connect() as conn:
            rows = conn.execute(
                f"select {self._COLS} from trade_rules where user_id=%s and league_id=%s and season=%s order by id",
                (user_id, league_id, season),
            ).fetchall()
        return [Rule(*r) for r in rows]

    def remove(self, user_id: int, rule_id: int) -> bool:
        with self._connect() as conn:
            cur = conn.execute("delete from trade_rules where id=%s and user_id=%s", (rule_id, user_id))
        return cur.rowcount > 0


def create_rule_store() -> RuleStore:
    url = os.getenv("DATABASE_URL", "").strip()
    return PostgresRuleStore(url) if url else MemoryRuleStore()

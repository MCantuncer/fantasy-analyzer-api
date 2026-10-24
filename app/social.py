"""League chat: verifying a team's manager, and what managers post.

**Verifying a team.** ESPN has no login for other sites, so a signed-in account
proves it manages a team by setting the team's abbreviation on ESPN to a short code
we hand out. Only the team's owner (and the league manager) can change it, and we
read it back from ESPN's league data. The code travels in a signed, expiring
challenge bound to the account, so nothing is stored until the check passes; then
``app.accounts`` records the account as the team's manager. No ESPN password or
cookie is ever kept.

**Storage.** Posts and reactions live in the same Postgres as the trade of the day
(``DATABASE_URL``), in memory without it.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol

# No 0/O, 1/I/L, 2/Z, 5/S or 8/B: easy to read off a phone and type into ESPN.
CODE_ALPHABET = "ACDEFGHJKMNPQRTUVWXY34679"
CODE_LENGTH = 4
CHALLENGE_SECONDS = 30 * 60

POST_MAX_CHARS = 500
REACTIONS = ("steal", "smart", "overpay", "lol")
# A trade's "Who won it?" poll is a reaction too: won:<team id>, one per voter.
WON_PREFIX = "won:"

_SECRET = (os.getenv("SECRET_KEY") or secrets.token_hex(32)).encode()


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def new_code(avoid: str = "") -> str:
    while True:
        code = "".join(secrets.choice(CODE_ALPHABET) for _ in range(CODE_LENGTH))
        if code != avoid.upper():
            return code


@dataclass(frozen=True)
class Challenge:
    league_id: int
    season: int
    team_id: int
    code: str
    expires: int  # unix seconds
    user_id: int


def sign_challenge(c: Challenge) -> str:
    body = _b64(json.dumps([c.league_id, c.season, c.team_id, c.code, c.expires, c.user_id]).encode())
    mac = _b64(hmac.new(_SECRET, body.encode(), hashlib.sha256).digest())
    return f"{body}.{mac}"


def read_challenge(token: str, now: float | None = None) -> Challenge | None:
    """The challenge if the signature holds and it hasn't expired, else None."""
    try:
        body, mac = token.split(".", 1)
        want = _b64(hmac.new(_SECRET, body.encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(mac, want):
            return None
        league_id, season, team_id, code, expires, user_id = json.loads(_unb64(body))
    except (ValueError, TypeError):
        return None
    if expires < (now if now is not None else time.time()):
        return None
    return Challenge(int(league_id), int(season), int(team_id), str(code), int(expires), int(user_id))


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def valid_reaction(reaction: str, team_ids: list[int]) -> bool:
    """One of the fixed reactions, or a "won:<team>" vote for a team in this league."""
    if reaction in REACTIONS:
        return True
    rest = reaction.removeprefix(WON_PREFIX)
    return rest != reaction and rest.isdigit() and int(rest) in team_ids


@dataclass
class Post:
    id: int
    league_id: int
    season: int
    team_id: int
    target: str | None  # None: the general chat; "tx:<id>" or "post:<id>": a reply
    body: str
    created_at: datetime
    deleted: bool = False


@dataclass
class ReactionCounts:
    counts: dict[str, int] = field(default_factory=dict)
    mine: list[str] = field(default_factory=list)


SCHEMA = """
create table if not exists league_posts (
    id bigserial primary key,
    league_id bigint not null,
    season int not null,
    team_id int not null,
    target text,
    body text not null,
    created_at timestamptz not null default now(),
    deleted boolean not null default false
);
create index if not exists league_posts_by_league on league_posts (league_id, season, created_at desc);
create index if not exists league_posts_by_target on league_posts (league_id, season, target);
create table if not exists league_reactions (
    league_id bigint not null,
    season int not null,
    target text not null,
    team_id int not null,
    reaction text not null,
    created_at timestamptz not null default now(),
    primary key (league_id, season, target, team_id, reaction)
);
"""


class SocialStore(Protocol):
    def add_post(self, league_id: int, season: int, team_id: int, target: str | None, body: str) -> Post: ...
    def post(self, league_id: int, season: int, post_id: int) -> Post | None: ...
    def delete_post(self, league_id: int, season: int, post_id: int) -> None: ...
    def chat(self, league_id: int, season: int, since: datetime, until: datetime | None) -> list[Post]: ...
    def replies(self, league_id: int, season: int, target: str) -> list[Post]: ...
    def reply_counts(self, league_id: int, season: int, targets: list[str]) -> dict[str, int]: ...
    def react(self, league_id: int, season: int, target: str, team_id: int, reaction: str, on: bool) -> None: ...
    def reactions(self, league_id: int, season: int, targets: list[str], team_id: int) -> dict[str, ReactionCounts]: ...


class MemorySocialStore:
    def __init__(self) -> None:
        self.posts: list[Post] = []
        self.marks: set[tuple[int, int, str, int, str]] = set()

    def add_post(self, league_id: int, season: int, team_id: int, target: str | None, body: str) -> Post:
        post = Post(len(self.posts) + 1, league_id, season, team_id, target, body, datetime.now(UTC))
        self.posts.append(post)
        return post

    def post(self, league_id: int, season: int, post_id: int) -> Post | None:
        return next((p for p in self.posts if p.id == post_id and (p.league_id, p.season) == (league_id, season)), None)

    def delete_post(self, league_id: int, season: int, post_id: int) -> None:
        if p := self.post(league_id, season, post_id):
            p.deleted = True

    def _live(self, league_id: int, season: int) -> list[Post]:
        return [p for p in self.posts if (p.league_id, p.season) == (league_id, season) and not p.deleted]

    def chat(self, league_id: int, season: int, since: datetime, until: datetime | None) -> list[Post]:
        return [
            p
            for p in self._live(league_id, season)
            if p.target is None and p.created_at >= since and (until is None or p.created_at < until)
        ]

    def replies(self, league_id: int, season: int, target: str) -> list[Post]:
        return [p for p in self._live(league_id, season) if p.target == target]

    def reply_counts(self, league_id: int, season: int, targets: list[str]) -> dict[str, int]:
        out: dict[str, int] = {}
        for p in self._live(league_id, season):
            if p.target in targets:
                out[p.target] = out.get(p.target, 0) + 1
        return out

    def react(self, league_id: int, season: int, target: str, team_id: int, reaction: str, on: bool) -> None:
        if reaction.startswith(WON_PREFIX):
            self.marks -= {
                m for m in self.marks if m[:4] == (league_id, season, target, team_id) and m[4].startswith(WON_PREFIX)
            }
        mark = (league_id, season, target, team_id, reaction)
        if on:
            self.marks.add(mark)
        else:
            self.marks.discard(mark)

    def reactions(self, league_id: int, season: int, targets: list[str], team_id: int) -> dict[str, ReactionCounts]:
        out: dict[str, ReactionCounts] = {}
        for lid, s, target, tid, reaction in self.marks:
            if (lid, s) != (league_id, season) or target not in targets:
                continue
            rc = out.setdefault(target, ReactionCounts())
            rc.counts[reaction] = rc.counts.get(reaction, 0) + 1
            if tid == team_id:
                rc.mine.append(reaction)
        return out


class PostgresSocialStore:
    _POST_COLS = "id, league_id, season, team_id, target, body, created_at, deleted"

    def __init__(self, url: str):
        import psycopg

        self._connect = lambda: psycopg.connect(url, autocommit=True)
        with self._connect() as conn:
            conn.execute(SCHEMA)

    def add_post(self, league_id: int, season: int, team_id: int, target: str | None, body: str) -> Post:
        with self._connect() as conn:
            row = conn.execute(
                "insert into league_posts (league_id, season, team_id, target, body) values (%s, %s, %s, %s, %s) "
                f"returning {self._POST_COLS}",
                (league_id, season, team_id, target, body),
            ).fetchone()
        return Post(*row)

    def post(self, league_id: int, season: int, post_id: int) -> Post | None:
        with self._connect() as conn:
            row = conn.execute(
                f"select {self._POST_COLS} from league_posts where league_id=%s and season=%s and id=%s",
                (league_id, season, post_id),
            ).fetchone()
        return Post(*row) if row else None

    def delete_post(self, league_id: int, season: int, post_id: int) -> None:
        with self._connect() as conn:
            conn.execute(
                "update league_posts set deleted=true where league_id=%s and season=%s and id=%s",
                (league_id, season, post_id),
            )

    def chat(self, league_id: int, season: int, since: datetime, until: datetime | None) -> list[Post]:
        with self._connect() as conn:
            rows = conn.execute(
                f"select {self._POST_COLS} from league_posts where league_id=%s and season=%s and target is null "
                "and not deleted and created_at >= %s and (%s::timestamptz is null or created_at < %s) "
                "order by created_at desc limit 500",
                (league_id, season, since, until, until),
            ).fetchall()
        return [Post(*r) for r in rows]

    def replies(self, league_id: int, season: int, target: str) -> list[Post]:
        with self._connect() as conn:
            rows = conn.execute(
                f"select {self._POST_COLS} from league_posts where league_id=%s and season=%s and target=%s "
                "and not deleted order by created_at limit 500",
                (league_id, season, target),
            ).fetchall()
        return [Post(*r) for r in rows]

    def reply_counts(self, league_id: int, season: int, targets: list[str]) -> dict[str, int]:
        if not targets:
            return {}
        with self._connect() as conn:
            rows = conn.execute(
                "select target, count(*) from league_posts where league_id=%s and season=%s "
                "and target = any(%s) and not deleted group by target",
                (league_id, season, targets),
            ).fetchall()
        return dict(rows)

    def react(self, league_id: int, season: int, target: str, team_id: int, reaction: str, on: bool) -> None:
        key = (league_id, season, target, team_id)
        with self._connect() as conn:
            if reaction.startswith(WON_PREFIX):
                conn.execute(
                    "delete from league_reactions where league_id=%s and season=%s and target=%s and team_id=%s "
                    "and reaction like 'won:%%'",
                    key,
                )
            if on:
                conn.execute(
                    "insert into league_reactions (league_id, season, target, team_id, reaction) "
                    "values (%s, %s, %s, %s, %s) on conflict do nothing",
                    (*key, reaction),
                )
            else:
                conn.execute(
                    "delete from league_reactions where league_id=%s and season=%s and target=%s and team_id=%s "
                    "and reaction=%s",
                    (*key, reaction),
                )

    def reactions(self, league_id: int, season: int, targets: list[str], team_id: int) -> dict[str, ReactionCounts]:
        if not targets:
            return {}
        with self._connect() as conn:
            rows = conn.execute(
                "select target, reaction, count(*), bool_or(team_id = %s) from league_reactions "
                "where league_id=%s and season=%s and target = any(%s) group by target, reaction",
                (team_id, league_id, season, targets),
            ).fetchall()
        out: dict[str, ReactionCounts] = {}
        for target, reaction, n, mine in rows:
            rc = out.setdefault(target, ReactionCounts())
            rc.counts[reaction] = n
            if mine:
                rc.mine.append(reaction)
        return out


def create_social_store() -> SocialStore:
    url = os.getenv("DATABASE_URL", "").strip()
    return PostgresSocialStore(url) if url else MemorySocialStore()

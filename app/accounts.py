"""Accounts, the leagues each account follows, and who manages which team.

A person signs up with an email and password (or Google), adds leagues by ID, and
proves they manage a team with the ESPN abbreviation check in ``app.social``. Each
team has at most one verified manager. A newer check by another account takes the
team over, because only the team's ESPN owner (or the league manager) can pass it;
the previous account gets a notice. That also makes a fresh check the way back in
after losing an account.

Passwords are stored as scrypt hashes and session tokens as SHA-256 hashes. With
``DATABASE_URL`` everything is in Postgres, otherwise in memory.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

SESSION_DAYS = 90
PASSWORD_MIN = 8
_SCRYPT = {"n": 2**14, "r": 8, "p": 1}


GOOGLE_ISSUERS = {"accounts.google.com", "https://accounts.google.com"}
TOKENINFO_URL = "https://oauth2.googleapis.com/tokeninfo"


class EmailTaken(Exception):
    pass


@dataclass
class GoogleIdentity:
    sub: str
    email: str | None


async def verify_google(credential: str, client_id: str, transport=None) -> GoogleIdentity | None:
    """Checks a Google Identity Services ID token with Google; None unless it is valid and meant for us."""
    import httpx

    async with httpx.AsyncClient(transport=transport, timeout=10) as http:
        try:
            resp = await http.get(TOKENINFO_URL, params={"id_token": credential})
        except httpx.HTTPError:
            return None
    if resp.status_code != 200:
        return None
    info = resp.json()
    if info.get("aud") != client_id or info.get("iss") not in GOOGLE_ISSUERS or not info.get("sub"):
        return None
    verified = str(info.get("email_verified", "")).lower() == "true"
    email = normalize_email(info["email"]) if verified and info.get("email") else None
    return GoogleIdentity(str(info["sub"]), email)


def normalize_email(email: str) -> str:
    return email.strip().lower()


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, dklen=32, **_SCRYPT)
    return "scrypt$" + base64.b64encode(salt).decode() + "$" + base64.b64encode(digest).decode()


def check_password(password: str, stored: str | None) -> bool:
    try:
        scheme, salt, digest = (stored or "").split("$")
    except ValueError:
        return False
    if scheme != "scrypt":
        return False
    got = hashlib.scrypt(password.encode(), salt=base64.b64decode(salt), dklen=32, **_SCRYPT)
    return hmac.compare_digest(got, base64.b64decode(digest))


@dataclass
class User:
    id: int
    email: str | None
    created_at: datetime
    has_google: bool = False


@dataclass
class Seat:
    """A signed-in account acting as the verified manager of one team."""

    user_id: int
    league_id: int
    season: int
    team_id: int


@dataclass
class FollowedLeague:
    league_id: int
    season: int
    name: str


@dataclass
class Notice:
    id: int
    text: str
    created_at: datetime


SCHEMA = """
create table if not exists users (
    id bigserial primary key,
    email text unique,
    password_hash text,
    google_sub text unique,
    created_at timestamptz not null default now()
);
create table if not exists user_sessions (
    token_hash text primary key,
    user_id bigint not null references users(id) on delete cascade,
    created_at timestamptz not null default now()
);
create table if not exists user_leagues (
    user_id bigint not null references users(id) on delete cascade,
    league_id bigint not null,
    season int not null,
    name text not null,
    added_at timestamptz not null default now(),
    primary key (user_id, league_id, season)
);
create table if not exists team_managers (
    league_id bigint not null,
    season int not null,
    team_id int not null,
    user_id bigint not null references users(id) on delete cascade,
    verified_at timestamptz not null default now(),
    primary key (league_id, season, team_id)
);
create unique index if not exists one_team_per_league on team_managers (league_id, season, user_id);
create table if not exists user_notices (
    id bigserial primary key,
    user_id bigint not null references users(id) on delete cascade,
    text text not null,
    created_at timestamptz not null default now()
);
"""


class AccountStore(Protocol):
    def create_user(self, email: str | None, password_hash: str | None, google_sub: str | None = None) -> User: ...
    def login_lookup(self, email: str) -> tuple[User, str | None] | None: ...
    def user_by_google(self, sub: str) -> User | None: ...
    def attach_google(self, user_id: int, sub: str) -> None: ...
    def save_session(self, token_hash: str, user_id: int) -> None: ...
    def session_user(self, token_hash: str) -> User | None: ...
    def drop_session(self, token_hash: str) -> None: ...
    def follow(self, user_id: int, league: FollowedLeague) -> None: ...
    def unfollow(self, user_id: int, league_id: int, season: int) -> None: ...
    def followed(self, user_id: int) -> list[FollowedLeague]: ...
    def set_manager(self, league_id: int, season: int, team_id: int, user_id: int) -> int | None: ...
    def managers(self, league_id: int, season: int) -> dict[int, int]: ...
    def managed_teams(self, user_id: int) -> dict[tuple[int, int], int]: ...
    def add_notice(self, user_id: int, text: str) -> None: ...
    def notices(self, user_id: int) -> list[Notice]: ...
    def clear_notices(self, user_id: int) -> None: ...


def team_of(store: AccountStore, user_id: int, league_id: int, season: int) -> int | None:
    return next((t for t, u in store.managers(league_id, season).items() if u == user_id), None)


class MemoryAccountStore:
    def __init__(self) -> None:
        self.users: dict[int, tuple[User, str | None, str | None]] = {}  # id -> (user, password hash, google sub)
        self.sessions: dict[str, int] = {}
        self.leagues: dict[int, dict[tuple[int, int], FollowedLeague]] = {}
        self.teams: dict[tuple[int, int, int], int] = {}
        self.notes: dict[int, list[Notice]] = {}

    def create_user(self, email: str | None, password_hash: str | None, google_sub: str | None = None) -> User:
        if email and any(u.email == email for u, _, _ in self.users.values()):
            raise EmailTaken(email)
        user = User(len(self.users) + 1, email, datetime.now(UTC), has_google=google_sub is not None)
        self.users[user.id] = (user, password_hash, google_sub)
        return user

    def login_lookup(self, email: str) -> tuple[User, str | None] | None:
        return next(((u, pw) for u, pw, _ in self.users.values() if u.email == email), None)

    def user_by_google(self, sub: str) -> User | None:
        return next((u for u, _, g in self.users.values() if g == sub), None)

    def attach_google(self, user_id: int, sub: str) -> None:
        user, pw, _ = self.users[user_id]
        user.has_google = True
        self.users[user_id] = (user, pw, sub)

    def save_session(self, token_hash: str, user_id: int) -> None:
        self.sessions[token_hash] = user_id

    def session_user(self, token_hash: str) -> User | None:
        uid = self.sessions.get(token_hash)
        return self.users[uid][0] if uid in self.users else None

    def drop_session(self, token_hash: str) -> None:
        self.sessions.pop(token_hash, None)

    def follow(self, user_id: int, league: FollowedLeague) -> None:
        self.leagues.setdefault(user_id, {})[(league.league_id, league.season)] = league

    def unfollow(self, user_id: int, league_id: int, season: int) -> None:
        self.leagues.get(user_id, {}).pop((league_id, season), None)

    def followed(self, user_id: int) -> list[FollowedLeague]:
        return list(self.leagues.get(user_id, {}).values())

    def set_manager(self, league_id: int, season: int, team_id: int, user_id: int) -> int | None:
        # One team per account per league: verifying another team releases the old one.
        for key, uid in list(self.teams.items()):
            if key[:2] == (league_id, season) and uid == user_id:
                del self.teams[key]
        previous = self.teams.get((league_id, season, team_id))
        self.teams[(league_id, season, team_id)] = user_id
        return previous if previous != user_id else None

    def managers(self, league_id: int, season: int) -> dict[int, int]:
        return {t: u for (lid, s, t), u in self.teams.items() if (lid, s) == (league_id, season)}

    def managed_teams(self, user_id: int) -> dict[tuple[int, int], int]:
        return {(lid, s): t for (lid, s, t), u in self.teams.items() if u == user_id}

    def add_notice(self, user_id: int, text: str) -> None:
        notes = self.notes.setdefault(user_id, [])
        notes.append(Notice(sum(len(n) for n in self.notes.values()) + 1, text, datetime.now(UTC)))

    def notices(self, user_id: int) -> list[Notice]:
        return list(self.notes.get(user_id, []))

    def clear_notices(self, user_id: int) -> None:
        self.notes.pop(user_id, None)


class PostgresAccountStore:
    def __init__(self, url: str):
        import psycopg

        self._connect = lambda: psycopg.connect(url, autocommit=True)
        with self._connect() as conn:
            conn.execute(SCHEMA)

    _USER = "id, email, created_at, google_sub is not null"

    def create_user(self, email: str | None, password_hash: str | None, google_sub: str | None = None) -> User:
        import psycopg

        try:
            with self._connect() as conn:
                row = conn.execute(
                    f"insert into users (email, password_hash, google_sub) values (%s, %s, %s) returning {self._USER}",
                    (email, password_hash, google_sub),
                ).fetchone()
        except psycopg.errors.UniqueViolation:
            raise EmailTaken(email) from None
        return User(*row)

    def login_lookup(self, email: str) -> tuple[User, str | None] | None:
        with self._connect() as conn:
            row = conn.execute(f"select {self._USER}, password_hash from users where email=%s", (email,)).fetchone()
        return (User(*row[:4]), row[4]) if row else None

    def user_by_google(self, sub: str) -> User | None:
        with self._connect() as conn:
            row = conn.execute(f"select {self._USER} from users where google_sub=%s", (sub,)).fetchone()
        return User(*row) if row else None

    def attach_google(self, user_id: int, sub: str) -> None:
        with self._connect() as conn:
            conn.execute("update users set google_sub=%s where id=%s", (sub, user_id))

    def save_session(self, token_hash: str, user_id: int) -> None:
        with self._connect() as conn:
            conn.execute("insert into user_sessions (token_hash, user_id) values (%s, %s)", (token_hash, user_id))

    def session_user(self, token_hash: str) -> User | None:
        with self._connect() as conn:
            row = conn.execute(
                f"select u.id, u.email, u.created_at, u.google_sub is not null from user_sessions s "
                f"join users u on u.id = s.user_id where s.token_hash=%s "
                f"and s.created_at > now() - interval '{SESSION_DAYS} days'",
                (token_hash,),
            ).fetchone()
        return User(*row) if row else None

    def drop_session(self, token_hash: str) -> None:
        with self._connect() as conn:
            conn.execute("delete from user_sessions where token_hash=%s", (token_hash,))

    def follow(self, user_id: int, league: FollowedLeague) -> None:
        with self._connect() as conn:
            conn.execute(
                "insert into user_leagues (user_id, league_id, season, name) values (%s, %s, %s, %s) "
                "on conflict (user_id, league_id, season) do update set name = excluded.name",
                (user_id, league.league_id, league.season, league.name),
            )

    def unfollow(self, user_id: int, league_id: int, season: int) -> None:
        with self._connect() as conn:
            conn.execute(
                "delete from user_leagues where user_id=%s and league_id=%s and season=%s", (user_id, league_id, season)
            )

    def followed(self, user_id: int) -> list[FollowedLeague]:
        with self._connect() as conn:
            rows = conn.execute(
                "select league_id, season, name from user_leagues where user_id=%s order by added_at", (user_id,)
            ).fetchall()
        return [FollowedLeague(*r) for r in rows]

    def set_manager(self, league_id: int, season: int, team_id: int, user_id: int) -> int | None:
        with self._connect() as conn, conn.transaction():
            conn.execute(
                "delete from team_managers where league_id=%s and season=%s and user_id=%s and team_id<>%s",
                (league_id, season, user_id, team_id),
            )
            row = conn.execute(
                "select user_id from team_managers where league_id=%s and season=%s and team_id=%s for update",
                (league_id, season, team_id),
            ).fetchone()
            conn.execute(
                "insert into team_managers (league_id, season, team_id, user_id) values (%s, %s, %s, %s) "
                "on conflict (league_id, season, team_id) do update set user_id=excluded.user_id, verified_at=now()",
                (league_id, season, team_id, user_id),
            )
        previous = row[0] if row else None
        return previous if previous != user_id else None

    def managers(self, league_id: int, season: int) -> dict[int, int]:
        with self._connect() as conn:
            rows = conn.execute(
                "select team_id, user_id from team_managers where league_id=%s and season=%s", (league_id, season)
            ).fetchall()
        return dict(rows)

    def managed_teams(self, user_id: int) -> dict[tuple[int, int], int]:
        with self._connect() as conn:
            rows = conn.execute(
                "select league_id, season, team_id from team_managers where user_id=%s", (user_id,)
            ).fetchall()
        return {(lid, s): t for lid, s, t in rows}

    def add_notice(self, user_id: int, text: str) -> None:
        with self._connect() as conn:
            conn.execute("insert into user_notices (user_id, text) values (%s, %s)", (user_id, text))

    def notices(self, user_id: int) -> list[Notice]:
        with self._connect() as conn:
            rows = conn.execute(
                "select id, text, created_at from user_notices where user_id=%s order by id", (user_id,)
            ).fetchall()
        return [Notice(*r) for r in rows]

    def clear_notices(self, user_id: int) -> None:
        with self._connect() as conn:
            conn.execute("delete from user_notices where user_id=%s", (user_id,))


def create_account_store() -> AccountStore:
    url = os.getenv("DATABASE_URL", "").strip()
    return PostgresAccountStore(url) if url else MemoryAccountStore()

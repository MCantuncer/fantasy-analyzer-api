"""One small pool of Postgres connections, shared by every store.

Each query used to open its own connection: a fresh TCP and TLS handshake with the
database host before every read, several times per page. A pool keeps a few
connections open and hands them out, so a query costs one round trip.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager

import psycopg
from psycopg_pool import ConnectionPool

# Enough for the threads that serve requests at once on a small server; more would only
# hold memory and database slots. Idle connections close after a few minutes, which
# also lets a hosted database (Neon) suspend when nobody is using the site.
MAX_SIZE = int(os.getenv("DB_POOL_SIZE", "4"))
MAX_IDLE = float(os.getenv("DB_POOL_IDLE_SECONDS", "240"))

_pools: dict[str, ConnectionPool] = {}
_connects: dict[str, Callable[[], AbstractContextManager[psycopg.Connection]]] = {}
_lock = threading.Lock()

# Each store's tables (idempotent DDL), per database, in the order the stores were made, and how
# many of them are in place. They are created on the first connection that works, not when the
# store is made: a database that is down (or over its hosting quota) must not stop the server
# from starting, since most pages only read ESPN. A later schema may reference an earlier one's
# tables (users), so they always run in order.
_schemas: dict[str, list[str]] = {}
_applied: dict[str, int] = {}
_schema_lock = threading.Lock()


def connector(url: str, schema: str | None = None) -> Callable[[], AbstractContextManager[psycopg.Connection]]:
    """``with connector(url)() as conn:`` borrows a pooled connection (autocommit) and gives it back.

    ``schema`` is the store's tables: made before the first query that reaches the database.
    """
    with _lock:
        connect = _connects.get(url)
        if connect is None:
            pool = ConnectionPool(
                url,
                min_size=0,
                max_size=MAX_SIZE,
                max_idle=MAX_IDLE,
                kwargs={"autocommit": True},
                # A connection the server closed while idle is replaced, not handed out.
                check=ConnectionPool.check_connection,
                open=True,
                name="db",
            )
            _pools[url] = pool
            connect = _connects[url] = _borrower(url, pool)
        if schema is not None and schema not in _schemas.setdefault(url, []):
            _schemas[url].append(schema)
    return connect


def _borrower(url: str, pool: ConnectionPool) -> Callable[[], AbstractContextManager[psycopg.Connection]]:
    @contextmanager
    def connect() -> Iterator[psycopg.Connection]:
        with pool.connection() as conn:
            _make_tables(url, conn)
            yield conn

    return connect


def _make_tables(url: str, conn: psycopg.Connection) -> None:
    if _applied.get(url, 0) >= len(_schemas.get(url, ())):
        return
    with _schema_lock:
        schemas = _schemas.get(url, [])
        while _applied.get(url, 0) < len(schemas):
            conn.execute(schemas[_applied.get(url, 0)])
            _applied[url] = _applied.get(url, 0) + 1

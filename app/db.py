"""One small pool of Postgres connections, shared by every store.

Each query used to open its own connection: a fresh TCP and TLS handshake with the
database host before every read, several times per page. A pool keeps a few
connections open and hands them out, so a query costs one round trip.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from contextlib import AbstractContextManager

import psycopg
from psycopg_pool import ConnectionPool

# Enough for the threads that serve requests at once on a small server; more would only
# hold memory and database slots. Idle connections close after a few minutes, which
# also lets a hosted database (Neon) suspend when nobody is using the site.
MAX_SIZE = int(os.getenv("DB_POOL_SIZE", "4"))
MAX_IDLE = float(os.getenv("DB_POOL_IDLE_SECONDS", "240"))

_pools: dict[str, ConnectionPool] = {}
_lock = threading.Lock()


def connector(url: str) -> Callable[[], AbstractContextManager[psycopg.Connection]]:
    """``with connector(url)() as conn:`` borrows a pooled connection (autocommit) and gives it back."""
    with _lock:
        pool = _pools.get(url)
        if pool is None:
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
    return pool.connection

import os

import psycopg
import pytest

from app.db import connector


@pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="set TEST_DATABASE_URL to test against Postgres")
def test_queries_reuse_one_connection():
    url = os.environ["TEST_DATABASE_URL"]
    connect = connector(url)
    assert connector(url) == connect  # every store shares the same pool
    with connect() as conn:
        first = conn.execute("select pg_backend_pid()").fetchone()[0]
    with connect() as conn:
        again = conn.execute("select pg_backend_pid()").fetchone()[0]
        assert conn.autocommit
    assert again == first  # no new handshake for the second query


@pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="set TEST_DATABASE_URL to test against Postgres")
def test_a_connection_the_server_dropped_is_replaced():
    url = os.environ["TEST_DATABASE_URL"]
    connect = connector(url)
    with connect() as conn:
        pid = conn.execute("select pg_backend_pid()").fetchone()[0]
    # The server ends the idle connection (a restart, or an idle timeout on a hosted database).
    with psycopg.connect(url, autocommit=True) as admin:
        admin.execute("select pg_terminate_backend(%s)", (pid,))
    with connect() as conn:
        assert conn.execute("select 1").fetchone()[0] == 1

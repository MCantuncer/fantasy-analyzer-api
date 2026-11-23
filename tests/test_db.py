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


def test_stores_start_while_the_database_is_down(monkeypatch):
    """2026-11-23: Neon refused connections (free compute quota used up) and every start died
    after 30 s waiting to make tables. The server now starts; tables come with the first query."""
    import time

    from fastapi.testclient import TestClient

    from app.main import create_app

    # Nothing listens on port 1: a connection fails at once instead of hanging.
    monkeypatch.setenv("DATABASE_URL", "postgresql://nobody@127.0.0.1:1/down?connect_timeout=1")
    started = time.monotonic()
    app = create_app()
    assert time.monotonic() - started < 10
    with TestClient(app) as client:
        assert client.get("/api/health").json() == {"ok": True}


@pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="set TEST_DATABASE_URL to test against Postgres")
def test_tables_are_made_on_the_first_query_in_store_order():
    from app.accounts import PostgresAccountStore
    from app.plans import PostgresSubscriptionStore

    base = os.environ["TEST_DATABASE_URL"]
    with psycopg.connect(base, autocommit=True) as admin:  # an empty schema of its own: no tables yet
        admin.execute("drop schema if exists fresh_start cascade")
        admin.execute("create schema fresh_start")
    url = base + ("&" if "?" in base else "?") + "options=-csearch_path%3Dfresh_start"
    accounts = PostgresAccountStore(url)
    plans = PostgresSubscriptionStore(url)  # its table references users, made by the accounts store
    assert plans.get(1) is None
    assert accounts.login_lookup("nobody@example.com") is None

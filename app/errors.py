"""Error reports: what broke, where, and how often, so problems are seen before a manager writes in.

Two sources feed one list:

* **api**: anything logged at ERROR with a traceback. That covers an unhandled exception in a
  request (uvicorn logs it) and every background job that logs its failure (analysis, injury
  watch, news). A logging handler picks them up, so no route needs its own reporting.
* **web**: the website's crashes and unhandled errors, sent to ``POST /api/errors``.

The same problem twice is one report with a count. Reports are grouped by a fingerprint (the
kind of error and where it happened, with numbers taken out) and keep the latest example: the
message, a trimmed stack, the route or page, the build and the browser. No IP addresses,
accounts, request bodies or cookies are kept, and emails and long tokens are masked in
messages. A report not seen for 30 days is deleted. A report marked fixed opens again if it
happens again. Only admin accounts (``users.user_type = 'admin'``) can read them.

**Storage.** The same Postgres as everything else (``DATABASE_URL``), in memory without it.
"""

from __future__ import annotations

import contextvars
import hashlib
import logging
import os
import queue
import re
import threading
import traceback
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Protocol

KEEP = timedelta(days=30)
MESSAGE_MAX = 1000
STACK_MAX = 8000
LIST_LIMIT = 200

# The request being served, for reports logged while serving it (set by RouteContext).
ROUTE: contextvars.ContextVar[str | None] = contextvars.ContextVar("route", default=None)

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)+")
# 32+ characters of letters and digits mixed (session tokens, espn_s2, SWID); paths and names stay.
_TOKEN = re.compile(r"(?=[\w%+=-]*\d)(?=[\w%+=-]*[A-Za-z])[\w%+=-]{32,}")
_NUMBER = re.compile(r"\d+")


def scrub(text: str | None, limit: int) -> str:
    """Masks emails and long tokens (session tokens, cookies), then trims."""
    text = _TOKEN.sub("[token]", _EMAIL.sub("[email]", text or ""))
    return text if len(text) <= limit else text[: limit - 1] + "…"


def fingerprint(source: str, title: str, where: str) -> str:
    """Names one problem, whatever ids or counts its message carried."""
    raw = "|".join(_NUMBER.sub("N", part) for part in (source, title, where))
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def build() -> str:
    """The API build: Render's commit when deployed, "dev" locally."""
    return os.getenv("RENDER_GIT_COMMIT", "").strip()[:12] or "dev"


@dataclass
class Occurrence:
    """One time a problem happened."""

    source: str  # "api" or "web"
    title: str  # e.g. "KeyError in app.analysis.report.build", or the website's "TypeError: x is undefined"
    where: str  # code location for the api, page for the web; part of the fingerprint
    message: str
    stack: str = ""
    route: str = ""  # the request or page it happened on (not part of the fingerprint)
    build: str = ""
    user_agent: str = ""
    at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.source, self.title, self.where)


@dataclass
class Report:
    id: int
    fingerprint: str
    source: str
    title: str
    where: str
    message: str
    stack: str
    route: str
    build: str
    user_agent: str
    count: int
    first_seen: datetime
    last_seen: datetime
    resolved_at: datetime | None = None


SCHEMA = """
create table if not exists error_reports (
    id bigserial primary key,
    fingerprint text not null unique,
    source text not null,
    title text not null,
    location text not null,
    message text not null,
    stack text not null,
    route text not null,
    build text not null,
    user_agent text not null,
    count bigint not null default 1,
    first_seen timestamptz not null,
    last_seen timestamptz not null,
    resolved_at timestamptz
);
create index if not exists error_reports_last_seen on error_reports (last_seen desc);
"""


class ErrorStore(Protocol):
    def record(self, o: Occurrence) -> None: ...
    def recent(self, limit: int = LIST_LIMIT) -> list[Report]: ...
    def resolve(self, report_id: int) -> bool: ...
    def known(self, fingerprint: str) -> bool: ...
    def new_today(self, source: str) -> int: ...


class MemoryErrorStore:
    def __init__(self) -> None:
        self.rows: dict[str, Report] = {}
        self._next = 1
        self._lock = threading.Lock()

    def record(self, o: Occurrence) -> None:
        with self._lock:
            self._prune(o.at)
            r = self.rows.get(o.fingerprint)
            if r is None:
                self.rows[o.fingerprint] = Report(
                    self._next, o.fingerprint, o.source, o.title, o.where, o.message, o.stack, o.route, o.build,
                    o.user_agent, 1, o.at, o.at,
                )  # fmt: skip
                self._next += 1
                return
            r.count += 1
            r.last_seen = o.at
            r.message, r.stack, r.route, r.build, r.user_agent = o.message, o.stack, o.route, o.build, o.user_agent
            r.resolved_at = None

    def recent(self, limit: int = LIST_LIMIT) -> list[Report]:
        with self._lock:
            return sorted(self.rows.values(), key=lambda r: r.last_seen, reverse=True)[:limit]

    def resolve(self, report_id: int) -> bool:
        with self._lock:
            for r in self.rows.values():
                if r.id == report_id:
                    r.resolved_at = datetime.now(UTC)
                    return True
        return False

    def known(self, fingerprint: str) -> bool:
        with self._lock:
            return fingerprint in self.rows

    def new_today(self, source: str) -> int:
        start = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        with self._lock:
            return sum(1 for r in self.rows.values() if r.source == source and r.first_seen >= start)

    def _prune(self, now: datetime) -> None:
        for key in [k for k, r in self.rows.items() if now - r.last_seen > KEEP]:
            del self.rows[key]


class PostgresErrorStore:
    _COLS = (
        "id, fingerprint, source, title, location, message, stack, route, build, user_agent, count, "
        "first_seen, last_seen, resolved_at"
    )

    def __init__(self, url: str):
        from app.db import connector

        self._connect = connector(url)
        with self._connect() as conn:
            conn.execute(SCHEMA)

    def record(self, o: Occurrence) -> None:
        with self._connect() as conn:
            conn.execute(
                "insert into error_reports (fingerprint, source, title, location, message, stack, route, build, "
                "user_agent, first_seen, last_seen) values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
                "on conflict (fingerprint) do update set count = error_reports.count + 1, "
                "last_seen = excluded.last_seen, message = excluded.message, stack = excluded.stack, "
                "route = excluded.route, build = excluded.build, user_agent = excluded.user_agent, resolved_at = null",
                (
                    o.fingerprint, o.source, o.title, o.where, o.message, o.stack, o.route, o.build, o.user_agent,
                    o.at, o.at,
                ),
            )  # fmt: skip
            conn.execute("delete from error_reports where last_seen < %s", (o.at - KEEP,))

    def recent(self, limit: int = LIST_LIMIT) -> list[Report]:
        with self._connect() as conn:
            rows = conn.execute(
                f"select {self._COLS} from error_reports order by last_seen desc limit %s", (limit,)
            ).fetchall()
        return [Report(*r) for r in rows]

    def resolve(self, report_id: int) -> bool:
        with self._connect() as conn:
            cur = conn.execute("update error_reports set resolved_at = now() where id = %s", (report_id,))
        return cur.rowcount > 0

    def known(self, fingerprint: str) -> bool:
        with self._connect() as conn:
            return (
                conn.execute("select 1 from error_reports where fingerprint = %s", (fingerprint,)).fetchone()
                is not None
            )

    def new_today(self, source: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "select count(*) from error_reports where source = %s and first_seen >= date_trunc('day', now())",
                (source,),
            ).fetchone()
        return int(row[0])


def create_error_store() -> ErrorStore:
    url = os.getenv("DATABASE_URL", "").strip()
    return PostgresErrorStore(url) if url else MemoryErrorStore()


def from_record(record: logging.LogRecord) -> Occurrence | None:
    """An api Occurrence from an ERROR log record with a traceback; None without one."""
    if not record.exc_info or record.exc_info[0] is None:
        return None
    kind, exc, tb = record.exc_info
    frames = traceback.extract_tb(tb)
    # Where in our code it broke: the deepest frame in app/ (not this middleware), else the deepest at all.
    ours = [f for f in frames if _module(f.filename).startswith("app.") and _module(f.filename) != __name__]
    at = (ours or frames or [None])[-1]
    where = f"{_module(at.filename)}.{at.name}" if at else record.name
    stack = "".join(traceback.format_exception(kind, exc, tb))
    return Occurrence(
        source="api",
        title=f"{kind.__name__} in {where}",
        where=f"{where}|{record.msg}",
        message=scrub(f"{record.getMessage()}: {exc}", MESSAGE_MAX),
        stack=scrub(stack[-STACK_MAX:], STACK_MAX),
        route=ROUTE.get() or "background",
        build=build(),
    )


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _module(filename: str) -> str:
    """A dotted module name: app.analysis.report from this repo, starlette.routing from a library."""
    path = os.path.abspath(filename)
    if path.startswith(_ROOT + os.sep) and "site-packages" not in path:
        path = os.path.relpath(path, _ROOT)
    elif "site-packages" in path:
        path = path.split("site-packages", 1)[1].lstrip(os.sep)
    return path.removesuffix(".py").replace(os.sep, ".")


class Collector(logging.Handler):
    """Turns ERROR log records with a traceback into reports, written from a thread of its own so
    logging never waits on the database. One per process: create_app points it at its store."""

    QUEUE_MAX = 1000

    def __init__(self) -> None:
        super().__init__(level=logging.ERROR)
        self.store: ErrorStore | None = None
        self._queue: queue.Queue[Occurrence] = queue.Queue(self.QUEUE_MAX)
        self._worker: threading.Thread | None = None
        self._start = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        if self.store is None or threading.current_thread() is self._worker:
            return  # never report the reporter's own failures
        try:
            occurrence = from_record(record)
            if occurrence is not None:
                self._ensure_worker()
                self._queue.put_nowait(occurrence)
        except queue.Full:
            pass  # an error storm: the counts are already high, drop the rest
        except Exception:  # noqa: BLE001 - logging must never raise
            self.handleError(record)

    def wait(self) -> None:
        """Blocks until every queued report is written (for tests)."""
        self._queue.join()

    def _ensure_worker(self) -> None:
        with self._start:
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._run, name="error-reports", daemon=True)
                self._worker.start()

    def _run(self) -> None:
        while True:
            occurrence = self._queue.get()
            try:
                if self.store is not None:
                    self.store.record(occurrence)
            except Exception:  # noqa: BLE001 - a report that can't be saved is dropped
                pass
            finally:
                self._queue.task_done()


COLLECTOR = Collector()


def install(store: ErrorStore) -> None:
    """Points the collector at ``store`` and hooks it to the loggers that carry errors: the root
    logger (our own code) and uvicorn's (unhandled exceptions in requests; it doesn't propagate)."""
    COLLECTOR.store = store
    for name in ("", "uvicorn"):
        logger = logging.getLogger(name)
        if COLLECTOR not in logger.handlers:
            logger.addHandler(COLLECTOR)


class RouteContext:
    """ASGI middleware: remembers which request is being served, so an error logged while serving
    it (or in a job it started) says where it came from. Only the method and path are kept."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "http":
            ROUTE.set(f"{scope['method']} {scope['path']}")
        await self.app(scope, receive, send)

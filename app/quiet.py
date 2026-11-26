"""Lets the hosted database sleep while nobody is on the site.

Neon bills (and on its free plan, counts against 100 hours a month) every minute the database is
awake, and it only sleeps after 5 minutes without a query. The server's background watches used
to read and write it every few minutes all day: the pregame tags every 10 minutes, the line log
and the iddaa log every 15, the news state every 2, the injury watch's recent reports every 5, the
sat-out nights every 10. So it never slept, and on 2026-11-23 the free hours ran out and the site
went down.

``QuietStore`` wraps the Postgres store for exactly those records:

* the latest copy of each one is kept in memory, so a watch reads its own last write instead of
  asking the database again;
* the watches' writes wait in memory and go to the database together (``flush``): at once when
  the database is awake anyway (someone used the site in the last couple of minutes), otherwise
  at most ``FLUSH_EVERY`` later, and always when the server stops;
* a read across many days (the admin line log, the news a page shows) saves what is waiting first,
  so it sees everything.

Everything else (accounts, reports, trades, chat) goes straight to the database as before. This
server is the only writer of these tables, so the memory copies can't go stale. What can be lost:
the writes still waiting when the server crashes without stopping cleanly, at most
``FLUSH_EVERY`` of watch looks; a deploy or restart saves them first.
"""

from __future__ import annotations

import copy
import os
import threading
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

from app import db

FLUSH_EVERY = float(os.getenv("QUIET_FLUSH_SECONDS", "3600"))
# A query this recent means the database is awake: waiting writes ride along for free.
AWAKE_SECONDS = 120.0


def database_requests() -> int:
    """Connections handed out so far by every pool: when it moves, the database is awake."""
    return sum(pool.get_stats().get("requests_num", 0) for pool in list(db._pools.values()))


class QuietStore:
    def __init__(self, inner: Any, clock: Callable[[], float] = time.monotonic) -> None:
        self._inner = inner
        self._clock = clock
        self._lock = threading.RLock()
        self._flush_lock = threading.Lock()
        self._docs: dict[tuple, Any] = {}  # (kind, *key) -> the latest body, read or written
        self._waiting: dict[tuple, tuple[str, tuple]] = {}  # (kind, *key) -> (store method, args)
        self._news_items: list[dict] = []
        self._news_log: list[dict] = []
        self._since: float | None = None  # when the oldest waiting write came in
        self._sat: dict[int, tuple[int, dict[int, dict]]] = {}  # season -> (since, day -> body)
        self._reports: tuple[datetime, list[dict]] | None = None  # (since, rows) of recent_reports
        self._report_saves = 0  # a read that crossed a save isn't kept
        self._seen = database_requests()  # the pools' request count at the last look, and when it last moved
        self._used_at = float("-inf")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    # --- keeping and saving --------------------------------------------------------------------

    def _get(self, key: tuple, read: Callable[[], Any]) -> Any:
        with self._lock:
            if key in self._docs:
                return copy.deepcopy(self._docs[key])
        body = read()
        with self._lock:
            self._docs.setdefault(key, copy.deepcopy(body))
            return copy.deepcopy(self._docs[key])

    def _put(self, key: tuple, body: Any, method: str, args: tuple) -> None:
        with self._lock:
            self._docs[key] = copy.deepcopy(body)
            self._waiting[key] = (method, copy.deepcopy(args))
            self._note_waiting()

    def _note_waiting(self) -> None:
        if self._since is None:
            self._since = self._clock()

    @property
    def waiting(self) -> int:
        with self._lock:
            return len(self._waiting) + len(self._news_items) + len(self._news_log)

    def due(self) -> bool:
        """Time to save what is waiting: the database is awake anyway, or it has waited long enough."""
        with self._lock:
            if self._since is None:
                return False
            now = self._clock()
            seen = database_requests()
            if seen != self._seen:
                self._seen, self._used_at = seen, now
            return now - self._since >= FLUSH_EVERY or now - self._used_at < AWAKE_SECONDS

    def flush(self) -> None:
        """Saves everything waiting, oldest first. What fails stays waiting for the next try."""
        with self._flush_lock:
            with self._lock:
                waiting, self._waiting = list(self._waiting.items()), {}
                items, self._news_items = self._news_items, []
                log, self._news_log = self._news_log, []
                self._since = None
            try:
                self._save(waiting, items, log)
            finally:
                with self._lock:  # its own queries don't count as someone waking the database
                    self._seen = database_requests()

    def _save(self, waiting: list, items: list[dict], log: list[dict]) -> None:
        try:
            if items:
                self._inner.save_news_items(items)
                items = []
            if log:
                self._inner.save_news_log(log)
                log = []
            while waiting:
                method, args = waiting[0][1]
                getattr(self._inner, method)(*args)
                waiting.pop(0)
        except Exception:
            with self._lock:
                for key, call in waiting:
                    self._waiting.setdefault(key, call)  # a newer write that came in meanwhile wins
                self._news_items[:0] = items
                self._news_log[:0] = log
                if waiting or items or log:
                    self._note_waiting()
            raise

    # --- pregame tags, line log, iddaa log (app/pregame.py, app/linelog.py, app/iddaa.py) ---------

    def get_pregame(self, season: int, day: int) -> dict | None:
        return self._get(("pregame", season, day), lambda: self._inner.get_pregame(season, day))

    def save_pregame(self, season: int, day: int, body: dict) -> None:
        self._put(("pregame", season, day), body, "save_pregame", (season, day, body))

    def get_player_lines(self, season: int, day: int) -> dict | None:
        return self._get(("lines", season, day), lambda: self._inner.get_player_lines(season, day))

    def save_player_lines(self, season: int, day: int, body: dict, settled: bool) -> None:
        self._put(("lines", season, day), body, "save_player_lines", (season, day, body, settled))

    def player_lines(self, season: int, settled: bool) -> list[tuple[int, dict]]:
        self.flush()
        return self._inner.player_lines(season, settled)

    def get_book_lines(self, season: int, day: int) -> dict | None:
        return self._get(("book", season, day), lambda: self._inner.get_book_lines(season, day))

    def save_book_lines(self, season: int, day: int, body: dict) -> None:
        self._put(("book", season, day), body, "save_book_lines", (season, day, body))

    def book_line_days(self, season: int) -> list[int]:
        self.flush()
        return self._inner.book_line_days(season)

    # --- starting fives before tip-off (app/startlog.py) ----------------------------------------

    def get_lineups(self, season: int, day: int) -> dict | None:
        return self._get(("lineups", season, day), lambda: self._inner.get_lineups(season, day))

    def save_lineups(self, season: int, day: int, body: dict) -> None:
        self._put(("lineups", season, day), body, "save_lineups", (season, day, body))

    def lineup_days(self, season: int) -> list[int]:
        self.flush()
        return self._inner.lineup_days(season)

    # --- other ways to call each game (app/tunings.py) -----------------------------------------

    def get_tunings(self, season: int, day: int) -> dict | None:
        return self._get(("tunings", season, day), lambda: self._inner.get_tunings(season, day))

    def save_tunings(self, season: int, day: int, body: dict) -> None:
        self._put(("tunings", season, day), body, "save_tunings", (season, day, body))

    def tuning_days(self, season: int) -> list[int]:
        self.flush()
        return self._inner.tuning_days(season)

    # --- who sat out (app/benched.py) -----------------------------------------------------------

    def sat_out(self, season: int, since: int) -> dict[int, dict]:
        with self._lock:
            known = self._sat.get(season)
            if known is not None and since >= known[0]:
                return copy.deepcopy({d: b for d, b in known[1].items() if d >= since})
        rows = self._inner.sat_out(season, since)
        with self._lock:
            for key, (_, args) in self._waiting.items():
                if key[0] == "sat" and key[1] == season and key[2] >= since:
                    rows[key[2]] = copy.deepcopy(args[2])
            self._sat[season] = (since, copy.deepcopy(rows))
            return copy.deepcopy(rows)

    def save_sat_out(self, season: int, day: int, body: dict) -> None:
        with self._lock:
            known = self._sat.get(season)
            if known is not None and day >= known[0]:
                known[1][day] = copy.deepcopy(body)
            self._put(("sat", season, day), body, "save_sat_out", (season, day, body))

    # --- news (app/news/service.py) -----------------------------------------------------------

    def get_news_state(self, key: str) -> dict | None:
        return self._get(("news", key), lambda: self._inner.get_news_state(key))

    def save_news_state(self, key: str, body: dict) -> None:
        self._put(("news", key), body, "save_news_state", (key, body))

    def save_news_items(self, items: list[dict]) -> None:
        with self._lock:
            self._news_items.extend(copy.deepcopy(items))
            self._note_waiting()

    def news_items(self, since: str) -> list[dict]:
        self.flush()
        return self._inner.news_items(since)

    def save_news_log(self, rows: list[dict]) -> None:
        with self._lock:
            self._news_log.extend(copy.deepcopy(rows))
            self._note_waiting()

    def news_log(self, since: str, limit: int, tagged: bool = False) -> list[dict]:
        self.flush()
        return self._inner.news_log(since, limit, tagged)

    # --- the injury watch's look at recent reports --------------------------------------------

    def recent_reports(self, since: datetime) -> list[dict]:
        with self._lock:
            known = self._reports
            if known is not None and since >= known[0]:
                return [dict(r) for r in known[1] if datetime.fromisoformat(r["built_at"]) >= since]
        saves = self._report_saves
        rows = self._inner.recent_reports(since)
        with self._lock:
            if saves == self._report_saves:
                self._reports = (since, [dict(r) for r in rows])
        return rows

    def save_report(self, key: str, league_id: int, season: int, team_id: int, row: dict) -> None:
        self._inner.save_report(key, league_id, season, team_id, row)
        with self._lock:
            self._reports = None  # the next look reads them again
            self._report_saves += 1

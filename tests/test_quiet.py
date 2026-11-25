from datetime import UTC, datetime, timedelta

import pytest

from app import quiet
from app.quiet import QuietStore
from app.store import MemoryStore


class Counting:
    """A store that counts every call that reaches it: each one would wake the database."""

    def __init__(self):
        self.inner, self.calls, self.fail = MemoryStore(), [], False

    def __getattr__(self, name):
        method = getattr(self.inner, name)

        def call(*args):
            if self.fail:
                raise ConnectionError("the database is asleep and won't wake")
            self.calls.append(name)
            return method(*args)

        return call


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@pytest.fixture
def setup(monkeypatch):
    requests = {"n": 0}
    monkeypatch.setattr(quiet, "database_requests", lambda: requests["n"])
    db, clock = Counting(), Clock()
    return QuietStore(db, clock), db, clock, requests


def test_the_watches_read_their_own_writes_without_the_database(setup):
    store, db, _, _ = setup
    assert store.get_pregame(2027, 30) is None
    store.save_pregame(2027, 30, {"teams": {"1": "a"}})
    store.save_pregame(2027, 30, {"teams": {"1": "b"}})
    assert store.get_pregame(2027, 30) == {"teams": {"1": "b"}}
    store.save_player_lines(2027, 30, {"players": {}}, False)
    assert store.get_player_lines(2027, 30) == {"players": {}}
    store.save_news_state("meta", {"refreshed_at": "x"})
    assert store.get_news_state("meta") == {"refreshed_at": "x"}
    assert db.calls == ["get_pregame"]  # one read to learn there was nothing, then memory

    store.flush()
    assert db.calls[1:] == ["save_pregame", "save_player_lines", "save_news_state"]  # the latest of each, once
    assert db.inner.get_pregame(2027, 30) == {"teams": {"1": "b"}}
    assert store.waiting == 0


def test_a_change_to_what_was_read_or_saved_stays_out_of_memory(setup):
    store, _, _, _ = setup
    body = {"teams": {"1": "a"}}
    store.save_book_lines(2027, 30, body)
    body["teams"]["1"] = "changed by the caller"
    got = store.get_book_lines(2027, 30)
    got["teams"]["1"] = "changed again"
    assert store.get_book_lines(2027, 30) == {"teams": {"1": "a"}}


def test_writes_go_out_when_the_database_is_awake_or_after_an_hour(setup):
    store, _, clock, requests = setup
    assert not store.due()  # nothing waiting
    store.save_pregame(2027, 30, {})
    clock.t += 60
    assert not store.due()  # asleep, and not waiting long
    requests["n"] += 1  # someone signed in: the database is awake now
    assert store.due()
    clock.t += quiet.AWAKE_SECONDS + 1
    assert not store.due()  # it has fallen asleep again
    clock.t += quiet.FLUSH_EVERY
    assert store.due()  # an hour is the longest a write waits


def test_a_failed_save_keeps_everything_and_a_newer_write_wins(setup):
    store, db, _, _ = setup
    store.save_pregame(2027, 30, {"v": 1})
    store.save_news_items([{"id": "n1", "published": "2026-11-23"}])
    db.fail = True
    with pytest.raises(ConnectionError):
        store.flush()
    assert store.waiting == 2
    store.save_pregame(2027, 30, {"v": 2})
    db.fail = False
    store.flush()
    assert db.inner.get_pregame(2027, 30) == {"v": 2}
    assert db.inner.news_items("2026") == [{"id": "n1", "published": "2026-11-23"}]


def test_reads_across_days_save_what_waits_first(setup):
    store, db, _, _ = setup
    store.save_player_lines(2027, 30, {"a": 1}, True)
    store.save_news_items([{"id": "n1", "published": "2026-11-23"}])
    assert store.player_lines(2027, True) == [(30, {"a": 1})]
    assert [i["id"] for i in store.news_items("2026")] == ["n1"]


def test_sat_out_nights_are_read_once(setup):
    store, db, _, _ = setup
    db.inner.save_sat_out(2027, 28, {"1": "rest"})
    store.save_sat_out(2027, 29, {"2": "injury"})  # waiting, not in the database yet
    assert store.sat_out(2027, 27) == {28: {"1": "rest"}, 29: {"2": "injury"}}
    store.save_sat_out(2027, 30, {"3": "coach"})
    assert store.sat_out(2027, 29) == {29: {"2": "injury"}, 30: {"3": "coach"}}
    assert db.calls.count("sat_out") == 1


def test_the_injury_watch_reads_recent_reports_again_only_after_a_save(setup):
    store, db, _, _ = setup
    now = datetime(2026, 11, 23, 22, tzinfo=UTC)
    row = {"body": {}, "fingerprint": {}, "built_at": (now - timedelta(hours=1)).isoformat()}
    db.inner.save_report("k1", 1, 2027, 3, row)
    assert [r["key"] for r in store.recent_reports(now - timedelta(days=2))] == ["k1"]
    assert [r["key"] for r in store.recent_reports(now - timedelta(minutes=30))] == []  # later window, memory
    assert db.calls.count("recent_reports") == 1
    store.save_report("k2", 1, 2027, 4, row | {"built_at": now.isoformat()})
    assert {r["key"] for r in store.recent_reports(now - timedelta(days=2))} == {"k1", "k2"}
    assert db.calls.count("recent_reports") == 2


def test_its_own_saves_dont_keep_the_database_awake(setup, monkeypatch):
    store, db, clock, requests = setup
    monkeypatch.setattr(QuietStore, "_save", lambda self, *a: requests.update(n=requests["n"] + 3))
    store.save_pregame(2027, 30, {})
    clock.t += quiet.FLUSH_EVERY
    assert store.due()
    store.flush()  # three queries of its own
    store.save_pregame(2027, 30, {"v": 2})
    clock.t += 60
    assert not store.due()


def test_the_starting_fives_wait_in_memory_and_a_day_list_saves_them_first(setup):
    store, db, _, _ = setup
    assert store.get_lineups(2027, 37) is None
    store.save_lineups(2027, 37, {"teams": {"9": {"looks": []}}})
    assert store.get_lineups(2027, 37) == {"teams": {"9": {"looks": []}}}
    assert db.calls == ["get_lineups"]
    assert store.lineup_days(2027) == [37]  # the admin read sees the waiting night
    assert db.calls[1:] == ["save_lineups", "lineup_days"]

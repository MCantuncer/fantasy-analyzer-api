import os
from datetime import date

import pytest

from app import main
from app.analysis.daily import spotlight_index, trade_of_the_day
from app.store import MemoryStore, PostgresStore
from tests.conftest import LEAGUE_ID

DAY = date(2026, 10, 23)


@pytest.fixture(scope="module")
def pick(league):
    return trade_of_the_day(league, DAY)


def test_pick_is_a_win_win_between_two_teams(pick):
    assert pick.status == "ready"
    a, b = pick.sides
    assert a.team_id != b.team_id
    assert a.cats_gain > 0 and b.cats_gain > 0
    for s in pick.sides:
        assert s.playoff_pct_after >= s.playoff_pct_before
        assert s.sends and s.pitch
    assert not {p.id for p in a.sends} & {p.id for p in b.sends}
    assert a.team_name in pick.headline and b.team_name in pick.headline
    assert pick.win_win_deals >= 1 and pick.team_pairs == 45


def test_reasons_never_call_a_side_the_loser(pick):
    for s in pick.sides:
        assert not any(r.startswith(("Sell high", "Buy low")) for r in s.reasons)


def test_same_day_same_pick(league, pick):
    again = trade_of_the_day(league, DAY)
    assert [[p.id for p in s.sends] for s in again.sides] == [[p.id for p in s.sends] for s in pick.sides]


def test_every_team_gets_a_turn_in_the_spotlight(league):
    n = len(league.teams)
    days = [date.fromordinal(DAY.toordinal() + k) for k in range(n)]
    assert sorted(spotlight_index(league, d) for d in days) == list(range(n))


def test_memory_store_counts_one_vote_per_voter():
    store, key = MemoryStore(), (1, 2027, "public", "2026-10-23")
    store.vote(key, 3, "voter-aaaa", True)
    store.vote(key, 3, "voter-aaaa", False)  # changed their mind
    store.vote(key, 3, "voter-bbbb", True)
    store.vote(key, 7, "voter-aaaa", True)
    assert store.tally(key) == {3: (1, 1), 7: (1, 0)}


@pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="set TEST_DATABASE_URL to test against Postgres")
def test_postgres_store():
    store = PostgresStore(os.environ["TEST_DATABASE_URL"])
    key = (999, 2027, "public", "2026-10-23")
    with store._connect() as conn:
        conn.execute("delete from daily_trades where league_id = 999")
        conn.execute("delete from trade_votes where league_id = 999")
    store.save_trade(key, {"status": "none"})
    store.save_trade(key, {"status": "ready"})  # the first pick of the day stands
    assert store.get_trade(key) == {"status": "none"}
    store.vote(key, 3, "voter-aaaa", True)
    store.vote(key, 3, "voter-aaaa", False)
    store.vote(key, 3, "voter-bbbb", True)
    assert store.tally(key) == {3: (1, 1)}


def test_trade_of_the_day_endpoint(client, fake_espn, monkeypatch):
    monkeypatch.setattr(main, "today", lambda: DAY.isoformat())
    r = client.post("/api/trade-of-the-day", json={"league_id": LEAGUE_ID, "season": 2027})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready" and body["date"] == "2026-10-23"
    calls = fake_espn.calls
    again = client.post("/api/trade-of-the-day", json={"league_id": LEAGUE_ID, "season": 2027}).json()
    assert again["sides"] == body["sides"] and fake_espn.calls == calls

    team = body["sides"][0]["team_id"]
    vote = {"league_id": LEAGUE_ID, "season": 2027, "date": "2026-10-23", "team_id": team}
    r = client.post("/api/trade-of-the-day/vote", json={**vote, "take": True, "voter": "browser-1234"})
    assert r.status_code == 200
    client.post("/api/trade-of-the-day/vote", json={**vote, "take": False, "voter": "browser-5678"})
    tally = client.post("/api/trade-of-the-day", json={"league_id": LEAGUE_ID, "season": 2027}).json()["votes"]
    assert tally[str(team)] == {"yes": 1, "no": 1}


def test_vote_on_other_teams_or_days_is_refused(client, monkeypatch):
    monkeypatch.setattr(main, "today", lambda: DAY.isoformat())
    client.post("/api/trade-of-the-day", json={"league_id": LEAGUE_ID, "season": 2027})
    base = {"league_id": LEAGUE_ID, "season": 2027, "take": True, "voter": "browser-1234"}
    assert (
        client.post("/api/trade-of-the-day/vote", json={**base, "date": "2026-10-22", "team_id": 1}).status_code == 409
    )
    r = client.post("/api/trade-of-the-day/vote", json={**base, "date": "2026-10-23", "team_id": 99})
    assert r.status_code == 409 and r.json()["detail"]["code"] == "poll_closed"


def test_slow_scan_answers_pending(client, monkeypatch):
    monkeypatch.setattr(main, "today", lambda: DAY.isoformat())
    monkeypatch.setattr(main, "DAILY_WAIT", 0.0)
    r = client.post("/api/trade-of-the-day", json={"league_id": LEAGUE_ID, "season": 2027})
    assert r.status_code == 200 and r.json()["status"] == "pending"

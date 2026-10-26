import copy
import os
from datetime import date

import pytest

from app import main
from app.analysis.daily import KEEP_SHARE, DailyTrade, fully_healthy, spotlight_index, trade_of_the_day
from app.store import MemoryStore, PostgresStore
from tests.conftest import LEAGUE_ID

DAY = date(2026, 10, 23)


@pytest.fixture(scope="module")
def pick(league):
    return trade_of_the_day(league, DAY)


_scans: dict[tuple, object] = {}


@pytest.fixture(autouse=True)
def one_scan_per_pick(monkeypatch):
    """The endpoint tests serve the same recorded league, so each distinct scan runs once per test process.

    The scan itself is covered by the tests above that call ``trade_of_the_day`` directly.
    """

    def scan(lg, day, keep_id=None):
        key = (lg.id, lg.season, day, keep_id)
        if key not in _scans:
            _scans[key] = trade_of_the_day(lg, day, keep_id)
        return _scans[key].model_copy(deep=True)

    monkeypatch.setattr(main, "trade_of_the_day", scan)
    monkeypatch.setattr(main, "DAILY_WAIT", 120.0)  # a busy CI runner must not turn the first scan into "pending"


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


def test_only_healthy_players_change_hands(league, pick):
    moving = [p for s in pick.sides for p in s.sends]
    assert all(p.injury_status in ("ACTIVE", "NORMAL") for p in moving)
    # A player in today's deal turns day-to-day: the deal is out and the next pick leaves him alone.
    hurt = copy.deepcopy(league)
    star = next(p for t in hurt.teams for p in t.roster if p.id == moving[0].id)
    star.injury_status = "DAY_TO_DAY"
    assert not fully_healthy(star)
    again = trade_of_the_day(hurt, DAY, keep_id=pick.pick_id)
    assert again.pick_id != pick.pick_id
    assert star.id not in {p.id for s in again.sides for p in s.sends}


def test_recheck_keeps_the_pick_unless_something_clearly_beats_it(league, pick):
    assert trade_of_the_day(league, DAY, keep_id=pick.pick_id).pick_id == pick.pick_id
    daily = DailyTrade(league, DAY)
    weak = next(p for p in daily.pairs() if p.score < KEEP_SHARE * pick.score - 1e-3)
    assert daily.report(keep_id=weak.pick_id(daily.teams)).pick_id != weak.pick_id(daily.teams)


def test_every_team_gets_a_turn_in_the_spotlight(league):
    n = len(league.teams)
    days = [date.fromordinal(DAY.toordinal() + k) for k in range(n)]
    assert sorted(spotlight_index(league, d) for d in days) == list(range(n))


def test_memory_store_counts_one_vote_per_voter():
    store, key = MemoryStore(), (1, 2027, "public", "2026-10-23")
    store.vote(key, "deal-1", 3, "voter-aaaa", True)
    store.vote(key, "deal-1", 3, "voter-aaaa", False)  # changed their mind
    store.vote(key, "deal-1", 3, "voter-bbbb", True)
    store.vote(key, "deal-1", 7, "voter-aaaa", True)
    assert store.tally(key, "deal-1") == {3: (1, 1), 7: (1, 0)}
    store.vote(key, "deal-2", 3, "voter-bbbb", True)  # the pick changed; votes start over
    assert store.tally(key, "deal-2") == {3: (1, 0)}
    assert store.tally(key, "deal-1") == {3: (0, 1), 7: (1, 0)}


@pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="set TEST_DATABASE_URL to test against Postgres")
def test_postgres_store():
    store = PostgresStore(os.environ["TEST_DATABASE_URL"])
    key = (999, 2027, "public", "2026-10-23")
    with store._connect() as conn:
        conn.execute("delete from daily_trades where league_id = 999")
        conn.execute("delete from trade_votes where league_id = 999")
    store.save_trade(key, {"status": "none"})
    store.save_trade(key, {"status": "ready"})  # a re-check replaced the pick
    assert store.get_trade(key) == {"status": "ready"}
    store.vote(key, "deal-1", 3, "voter-aaaa", True)
    store.vote(key, "deal-1", 3, "voter-aaaa", False)
    store.vote(key, "deal-1", 3, "voter-bbbb", True)
    assert store.tally(key, "deal-1") == {3: (1, 1)}
    store.vote(key, "deal-2", 3, "voter-bbbb", True)
    assert store.tally(key, "deal-2") == {3: (1, 0)}


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


def test_vote_on_a_replaced_pick_is_refused(client, monkeypatch):
    monkeypatch.setattr(main, "today", lambda: DAY.isoformat())
    pick = client.post("/api/trade-of-the-day", json={"league_id": LEAGUE_ID, "season": 2027}).json()
    vote = {"league_id": LEAGUE_ID, "season": 2027, "date": "2026-10-23", "take": True, "voter": "browser-1234"}
    vote["team_id"] = pick["sides"][0]["team_id"]
    r = client.post("/api/trade-of-the-day/vote", json={**vote, "pick_id": "an-earlier-deal"})
    assert r.status_code == 409 and r.json()["detail"]["code"] == "pick_changed"
    assert client.post("/api/trade-of-the-day/vote", json={**vote, "pick_id": pick["pick_id"]}).status_code == 200


def test_stale_pick_is_served_while_it_is_rechecked(client, fake_espn, monkeypatch):
    monkeypatch.setattr(main, "today", lambda: DAY.isoformat())
    first = client.post("/api/trade-of-the-day", json={"league_id": LEAGUE_ID, "season": 2027}).json()
    monkeypatch.setattr(main, "DAILY_RECHECK", 0.0)
    again = client.post("/api/trade-of-the-day", json={"league_id": LEAGUE_ID, "season": 2027}).json()
    assert again["pick_id"] == first["pick_id"]  # answered from the current pick, not a fresh wait
    assert again["checked_at"] is not None


def test_slow_scan_answers_pending(client, monkeypatch):
    monkeypatch.setattr(main, "today", lambda: DAY.isoformat())
    monkeypatch.setattr(main, "DAILY_WAIT", 0.0)
    r = client.post("/api/trade-of-the-day", json={"league_id": LEAGUE_ID, "season": 2027})
    assert r.status_code == 200 and r.json()["status"] == "pending"

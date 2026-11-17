import asyncio
import gzip
import json
from datetime import UTC, datetime, timedelta

import httpx

from app import pregame
from app.accounts import MemoryAccountStore
from app.espn.client import EspnClient
from app.main import create_app
from app.news.facts import PlayerState
from app.scores import Schedule, ScheduledGame, parse_schedule
from app.store import MemoryStore
from tests.test_saved import WatchedEspn
from tests.test_scores import FIXTURES

T0 = datetime(2026, 11, 17, 0, 0, tzinfo=UTC)


def schedule() -> Schedule:
    early = ScheduledGame(1, 30, T0, away=19, home=2, official=False)
    late = ScheduledGame(2, 30, T0 + timedelta(hours=2), away=4, home=5, official=False)
    later = ScheduledGame(3, 31, T0 + timedelta(days=1), away=19, home=4, official=False)
    return Schedule(teams={}, games={30: [early, late], 31: [later]})


def states(**status: str) -> dict[int, PlayerState]:
    teams = {"suggs": 19, "banchero": 19, "queta": 2, "tatum": 2, "white": 4}
    ids = {name: i for i, name in enumerate(teams, start=1)}
    return {ids[n]: PlayerState(ids[n], n, teams[n], status.get(n, "ACTIVE")) for n in teams}


def test_days_with_a_game_still_to_start_within_a_day_are_watched():
    sched = schedule()
    assert pregame.days_to_watch(sched, T0 - timedelta(hours=3)) == [30]
    assert pregame.days_to_watch(sched, T0 + timedelta(hours=1)) == [30, 31]  # one game of day 30 still to go
    assert pregame.days_to_watch(sched, T0 + timedelta(hours=3)) == [31]


def test_a_team_keeps_its_last_reading_before_tip_off():
    sched = schedule()
    before = T0 - timedelta(minutes=10)
    rec = pregame.update(None, sched, 30, states(banchero="OUT", queta="DAY_TO_DAY"), before)
    assert rec["players"]["2"] == {"team": 19, "status": "OUT"} and rec["players"]["3"]["status"] == "DAY_TO_DAY"
    assert rec["teams"]["19"] == {"tip": T0.isoformat(), "checked": before.isoformat()}
    # After ORL-BOS tips off, Queta is ruled out: too late for that game, so the record keeps day-to-day.
    after = T0 + timedelta(minutes=5)
    rec = pregame.update(rec, sched, 30, states(banchero="OUT", queta="OUT", white="OUT"), after)
    assert rec["players"]["3"]["status"] == "DAY_TO_DAY" and rec["teams"]["2"]["checked"] == before.isoformat()
    assert rec["players"]["5"] == {"team": 4, "status": "OUT"}  # CHI hasn't tipped off yet


def test_a_player_who_left_a_waiting_team_leaves_its_list():
    sched = schedule()
    rec = pregame.update(None, sched, 30, states(), T0 - timedelta(hours=1))
    moved = states()
    moved[1] = PlayerState(1, "suggs", 9, "ACTIVE")  # traded to a team not playing
    rec = pregame.update(rec, sched, 30, moved, T0 - timedelta(minutes=5))
    assert "1" not in rec["players"] and "2" in rec["players"]


def test_the_watch_saves_tonights_tags_in_the_store(recorded):
    with gzip.open(FIXTURES / "scores_schedule_2027.json.gz", "rt") as fh:
        pro_schedule = json.load(fh)
    fake = WatchedEspn(recorded, schedule=pro_schedule)
    store = MemoryStore()
    app = create_app(
        EspnClient(transport=httpx.MockTransport(fake.handler)), store=store, accounts=MemoryAccountStore()
    )
    sched = parse_schedule(pro_schedule)
    day = min(d for d, gs in sched.games.items() if gs)
    tip = min(g.start for g in sched.games[day])
    app.state.now = lambda: tip - timedelta(hours=2)
    assert asyncio.run(app.state.pregame_sweep()) == [day]
    rec = store.get_pregame(2027, day)
    playing = {t for g in sched.games[day] for t in (g.away, g.home)}
    assert rec["players"] and all(v["team"] in playing for v in rec["players"].values())
    assert set(rec["teams"]) == {str(t) for t in playing}

import asyncio
import copy
import gzip
import json
import os
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from app import linelog
from app.accounts import MemoryAccountStore
from app.analysis import gamebox, gamescore
from app.analysis.form import MINUTES, PTS, REB
from app.analysis.gamebox import GameRow
from app.espn.client import EspnClient
from app.main import create_app
from app.schemas import NextGameLine
from app.scores import Schedule, ScheduledGame, parse_schedule
from app.store import MemoryStore, PostgresStore
from tests.conftest import LEAGUE_ID, signed_in
from tests.test_news import boss
from tests.test_saved import WatchedEspn
from tests.test_scores import FIXTURES

T0 = datetime(2026, 11, 19, 0, 0, tzinfo=UTC)


def schedule() -> Schedule:
    early = ScheduledGame(1, 30, T0, away=19, home=2, official=False)
    late = ScheduledGame(2, 30, T0 + timedelta(hours=2), away=4, home=5, official=False)
    later = ScheduledGame(3, 31, T0 + timedelta(days=1), away=19, home=4, official=False)
    return Schedule(teams={}, games={30: [early, late], 31: [later]})


def line(pts: float, minutes: float = 30.0) -> NextGameLine:
    return NextGameLine(
        minutes=minutes, pts=pts, reb=5, ast=3, stl=1, blk=0.5, three_pm=2, to=1.5, fgm=7, fga=15, ftm=3, fta=4
    )


def test_days_with_a_game_tipping_off_soon_are_looked_at():
    sched = schedule()
    assert linelog.days_to_watch(sched, T0 - timedelta(hours=19)) == []
    assert linelog.days_to_watch(sched, T0 - timedelta(hours=12)) == [30]  # the whole night, from the morning
    assert linelog.days_to_watch(sched, T0 - timedelta(hours=1)) == [30]
    assert linelog.days_to_watch(sched, T0 + timedelta(hours=1)) == [30]  # CHI-CLE still to go
    assert linelog.days_to_watch(sched, T0 + timedelta(hours=3)) == []


def test_a_team_keeps_its_last_line_before_tip_off():
    sched = schedule()
    before = T0 - timedelta(minutes=10)
    rec = linelog.update(None, sched, 30, {1: (19, line(20)), 2: (2, line(14)), 3: (4, line(9))}, before)
    assert rec["lines"]["1"] == [19, 30.0, 20.0, 5.0, 3.0, 1.0, 0.5, 2.0, 1.5, 7.0, 15.0, 3.0, 4.0]
    assert rec["teams"]["19"] == [T0.isoformat(), before.isoformat()]
    # ORL-BOS has tipped off: its lines stay as they were; CHI's player is ruled out since, so he leaves.
    after = T0 + timedelta(minutes=5)
    rec = linelog.update(rec, sched, 30, {1: (19, line(30)), 2: (2, line(1))}, after)
    assert rec["lines"]["1"][2] == 20.0 and rec["lines"]["2"][2] == 14.0
    assert "3" not in rec["lines"] and rec["teams"]["4"][1] == after.isoformat()
    assert rec["teams"]["19"][1] == before.isoformat()


def row(player: int, team: int, pts: float, minutes: float = 31.0) -> GameRow:
    return GameRow(player, 99, 30, team, {MINUTES: minutes, PTS: pts, REB: 7.0})


def test_settling_adds_what_each_player_did():
    sched = schedule()
    rec = linelog.update(
        None, sched, 30, {1: (19, line(20)), 2: (19, line(14)), 3: (4, line(9))}, T0 - timedelta(hours=1)
    )
    done = linelog.settle(rec, [row(1, 19, 31), row(7, 19, 5)])
    assert done["results"]["1"][:3] == [31.0, 31.0, 7.0]
    assert done["results"]["2"] == []  # ORL played without him
    assert "3" not in done["results"]  # no rows for CHI: postponed, or not in yet
    assert done["lines"] == rec["lines"]


def test_the_summary_tells_steady_players_from_jumpers():
    sched = schedule()
    steady, jumper = 1, 2
    records = []
    for pts_steady, pts_jumper in ((19, 34), (21, 6), (20, 30)):
        rec = linelog.update(None, sched, 30, {steady: (19, line(20)), jumper: (19, line(20))}, T0 - timedelta(hours=1))
        records.append(linelog.settle(rec, [row(steady, 19, pts_steady), row(jumper, 19, pts_jumper)]))
    benched = linelog.update(None, sched, 30, {jumper: (19, line(20))}, T0 - timedelta(hours=1))
    records.append(linelog.settle(benched, [row(steady, 19, 20)]))
    got = {r["player_id"]: r for r in linelog.summary(records)}
    assert got[steady]["games"] == 3 and got[steady]["dnp"] == 0
    assert got[jumper]["games"] == 3 and got[jumper]["dnp"] == 1
    assert got[steady]["stats"]["pts"] == {"line": 20.0, "actual": 20.0, "miss": 0.0, "swing": pytest.approx(0.82)}
    assert got[jumper]["stats"]["pts"]["miss"] == pytest.approx(3.33)
    assert got[jumper]["stats"]["pts"]["swing"] == pytest.approx(12.81)  # misses +14, -14, +10


def test_a_saved_day_is_small():
    sched = schedule()
    lines = {pid: (19 if pid % 2 else 2, line(17.3, 28.4)) for pid in range(4_000_000, 4_000_300)}
    rec = linelog.update(None, sched, 30, lines, T0 - timedelta(hours=1))
    rows = [GameRow(pid, 1, 30, team, {MINUTES: 27.5, PTS: 18.0, REB: 6.0}) for pid, (team, _) in lines.items()]
    size = len(json.dumps(linelog.settle(rec, rows), separators=(",", ":")))
    assert size < 40_000  # 300 players, line and result: ~35 KB before Postgres compresses it


def _app(recorded):
    with gzip.open(FIXTURES / "scores_schedule_2027.json.gz", "rt") as fh:
        pro_schedule = json.load(fh)
    ids = iter(range(1, 10**6))  # the league box reads game ids and sides; fill in any missing
    for team in pro_schedule["settings"]["proTeams"]:
        for games in (team.get("proGamesByScoringPeriod") or {}).values():
            for g in games:
                g.setdefault("id", next(ids))
                g.setdefault("awayProTeamId", team["id"])
                g.setdefault("homeProTeamId", team["id"])
    recorded = copy.deepcopy(recorded)  # the recorded league predates minutes: give everyone 30 a game
    entries = [e["playerPoolEntry"]["player"] for t in recorded["league"]["teams"] for e in t["roster"]["entries"]]
    for player in entries + [e["player"] for e in recorded["free_agents"]]:
        for block in player.get("stats") or []:
            for key in ("averageStats", "stats"):
                if block.get(key):
                    block[key].setdefault("40", 30.0 if key == "averageStats" else 30.0 * block[key].get("42", 1))
    fake = WatchedEspn(recorded, schedule=pro_schedule)
    store = MemoryStore()
    app = create_app(
        EspnClient(transport=httpx.MockTransport(fake.handler)), store=store, accounts=MemoryAccountStore()
    )
    # The recorded ESPN answers have no team game logs: a league box from the calendar alone.
    box = gamebox.Box(2027, {}, {}, {}, schedule=gamebox.parse_schedule(pro_schedule), scores=gamescore.build([]))
    app.state.game_boxes.set(2027, box)
    return app, store, parse_schedule(pro_schedule)


def test_the_watch_saves_the_lines_and_settling_fills_in_results(recorded):
    app, store, sched = _app(recorded)
    assert asyncio.run(app.state.line_log_sweep()) == []  # no league to read lines from
    app.state.line_log_league_id = LEAGUE_ID
    day = min(d for d, gs in sched.games.items() if gs and d >= 3)  # the recorded league's today
    tip = min(g.start for g in sched.games[day])
    app.state.now = lambda: tip - timedelta(hours=1)
    assert asyncio.run(app.state.line_log_sweep()) == [day]
    rec = store.get_player_lines(2027, day)
    assert len(rec["lines"]) > 20 and store.player_lines(2027, False) == [(day, rec)]
    playing = {t for g in sched.games[day] for t in (g.away, g.home)}
    assert all(v[0] in playing for v in rec["lines"].values())

    # Before the night settles nothing moves; once it has, the rows close it.
    pid, (team, *_) = next(iter(rec["lines"].items()))
    rows = [GameRow(int(pid), 1, day, team, {MINUTES: 33.0, PTS: 25.0, REB: 4.0})]
    asyncio.run(app.state.settle_line_log(2027, sched, rows))
    assert store.player_lines(2027, True) == []
    app.state.now = lambda: max(g.start for g in sched.games[day]) + timedelta(hours=8)
    asyncio.run(app.state.settle_line_log(2027, sched, rows))
    [(got_day, done)] = store.player_lines(2027, True)
    assert got_day == day and done["lines"] == rec["lines"]
    assert done["results"][pid][:3] == [33.0, 25.0, 4.0]
    assert all(v == [] for k, v in done["results"].items() if k != pid and rec["lines"][k][0] == team)


def test_admins_read_back_who_lands_near_his_line(recorded):
    app, store, _ = _app(recorded)
    rec = linelog.update(None, schedule(), 30, {1: (19, line(20)), 2: (19, line(20))}, T0 - timedelta(hours=1))
    store.save_player_lines(2027, 30, linelog.settle(rec, [row(1, 19, 21), row(2, 19, 35)]), True)
    client = signed_in(app, team_id=None)
    assert client.post("/api/admin/player-lines", json={"season": 2027}).status_code == 403
    admin = boss(app)
    # The player index isn't in the recorded answers: names may be blank, ids still read.
    got = admin.post("/api/admin/player-lines", json={"season": 2027}).json()
    assert got["days"] == 1
    assert [p["player_id"] for p in got["players"]] == [2, 1]  # biggest points swing first
    assert got["players"][0]["stats"]["pts"] == {"line": 20.0, "actual": 35.0, "miss": 15.0, "swing": 15.0}
    assert got["players"][0]["team_abbrev"] == "ORL"
    assert admin.post("/api/admin/player-lines", json={"season": 2027, "min_games": 2}).json()["players"] == []


@pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="set TEST_DATABASE_URL to test against Postgres")
def test_postgres_keeps_the_line_log():
    store = PostgresStore(os.environ["TEST_DATABASE_URL"])
    with store._connect() as conn:
        conn.execute("delete from player_lines where season = 1999")
    rec = linelog.update(None, schedule(), 30, {1: (19, line(20))}, T0 - timedelta(hours=1))
    store.save_player_lines(1999, 30, rec, False)
    store.save_player_lines(1999, 31, rec, False)
    assert store.get_player_lines(1999, 30) == rec and store.get_player_lines(1999, 32) is None
    done = linelog.settle(rec, [row(1, 19, 24)])
    store.save_player_lines(1999, 30, done, True)
    assert store.player_lines(1999, True) == [(30, done)]
    assert store.player_lines(1999, False) == [(31, rec)]
    with store._connect() as conn:
        conn.execute("delete from player_lines where season = 1999")

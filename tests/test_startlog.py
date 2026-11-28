import asyncio
from datetime import UTC, date, datetime, timedelta

import httpx
import pytest

from app import linelog, startlog
from app.analysis import gamebox
from app.cache import credentials_key
from app.main import LINE_LOG_FREE_AGENTS
from app.scores import Schedule, ScheduledGame
from tests.conftest import LEAGUE_ID, signed_in
from tests.test_linelog import _app
from tests.test_news import boss

TIP = datetime(2026, 11, 26, 0, 30, tzinfo=UTC)  # 7:30 pm in New York
GSW, NOP, BOS = 9, 3, 2


def schedule() -> Schedule:
    return Schedule(
        teams={
            GSW: ("GS", "Golden State Warriors"),
            NOP: ("NO", "New Orleans Pelicans"),
            BOS: ("BOS", "Boston Celtics"),
        },
        games={
            37: [
                ScheduledGame(501, 37, TIP, away=GSW, home=NOP, official=False),
                ScheduledGame(502, 37, TIP + timedelta(hours=3), away=BOS, home=GSW, official=False),
            ]
        },
    )


def player(pid: int, name: str, position: str | None, status: str = "Expected", roster: str = "Active") -> dict:
    return {
        "personId": pid,
        "playerName": name,
        "position": position,
        "lineupStatus": status,
        "rosterStatus": roster,
        "timestamp": "2026-11-25T18:00:00",
    }


def team(code: str, status: str = "Expected", swap: bool = False, inactive: bool = False) -> dict:
    five = [player(i, f"{code} {i}", pos, status) for i, pos in enumerate(("PG", "SG", "SF", "PF", "C"), start=1)]
    if swap:
        five[0] = player(9, f"{code} 9", "PG", status)
    bench = (
        [player(20, f"{code} 20", "", status), player(21, f"{code} 21", None, status, "Inactive")] if inactive else []
    )
    return {"teamAbbreviation": code, "players": five + bench}


def lineups(away: dict, home: dict) -> dict:
    return {"games": [{"gameId": "0022600070", "gameStatus": 1, "awayTeam": away, "homeTeam": home}]}


def test_parse_reads_the_five_its_status_and_who_is_inactive_in_espn_codes():
    got = startlog.parse(lineups(team("GSW"), team("NOP", "Confirmed", inactive=True)))
    assert set(got) == {"GS", "NO"}  # nba.com's codes in ESPN's spelling
    assert got["GS"]["status"] == "Expected" and len(got["GS"]["starters"]) == 5
    assert got["GS"]["starters"][0] == [1, "GSW 1", "PG"]
    assert got["NO"]["status"] == "Confirmed"
    assert len(got["NO"]["starters"]) == 5  # a bench player has no position
    assert got["NO"]["inactive"] == [[21, "NOP 21"]]
    assert got["GS"]["game"] == "0022600070"


def test_a_look_is_kept_only_on_a_change_and_the_first_confirmed_time_stays():
    sched = schedule()
    t1 = TIP - timedelta(minutes=80)
    rec = startlog.update(None, sched, 37, startlog.parse(lineups(team("GSW"), team("NOP"))), t1)
    assert set(rec["teams"]) == {str(GSW), str(NOP)}  # BOS@GSW tips off 3 hours later: not yet
    assert len(rec["teams"][str(GSW)]["looks"]) == 1 and rec["teams"][str(GSW)]["confirmed_at"] is None

    t2 = TIP - timedelta(minutes=75)
    rec = startlog.update(rec, sched, 37, startlog.parse(lineups(team("GSW"), team("NOP"))), t2)
    gs = rec["teams"][str(GSW)]
    assert len(gs["looks"]) == 1 and gs["checked"] == t2.isoformat(timespec="seconds")  # same five: no new look

    t3 = TIP - timedelta(minutes=30)
    confirmed = startlog.parse(lineups(team("GSW", "Confirmed", swap=True), team("NOP")))
    rec = startlog.update(rec, sched, 37, confirmed, t3)
    gs = rec["teams"][str(GSW)]
    assert [lk["status"] for lk in gs["looks"]] == ["Expected", "Confirmed"]
    assert gs["looks"][1]["starters"][0][0] == 9 and gs["confirmed_at"] == t3.isoformat(timespec="seconds")

    t4 = TIP - timedelta(minutes=10)
    rec = startlog.update(rec, sched, 37, confirmed, t4)
    assert rec["teams"][str(GSW)]["confirmed_at"] == t3.isoformat(timespec="seconds")

    after = startlog.update(
        rec, sched, 37, startlog.parse(lineups(team("GSW"), team("NOP"))), TIP + timedelta(minutes=1)
    )
    assert after == rec  # tipped off: what was seen before tip-off stays as it was


def test_the_evening_look_reads_every_team_of_the_day():
    sched = schedule()
    cut = linelog.early_cut(sched, 37)  # 23:00 in Istanbul, four and a half hours before the first tip
    assert cut == TIP - timedelta(hours=4, minutes=30)
    assert startlog.evening_days(sched, cut - timedelta(minutes=25)) == []
    assert startlog.evening_days(sched, cut - timedelta(minutes=15)) == [37]
    assert startlog.evening_days(sched, cut + timedelta(minutes=1)) == []
    listed = startlog.parse(lineups(team("GSW"), team("NOP")))
    at = cut - timedelta(minutes=15)
    assert startlog.update(None, sched, 37, listed, at)["teams"] == {}  # no game within 90 minutes
    rec = startlog.update(None, sched, 37, listed, at, evening=True)
    assert set(rec["teams"]) == {str(GSW), str(NOP), str(BOS)}  # BOS isn't in this file: marked as looked at
    assert rec["teams"][str(NOP)]["looks"][0]["at"] == at.isoformat(timespec="seconds")
    assert rec["teams"][str(BOS)]["looks"] == []
    # The same five 80 minutes before tip-off adds no look; a swap does, after the evening one.
    rec = startlog.update(rec, sched, 37, listed, TIP - timedelta(minutes=80))
    assert len(rec["teams"][str(GSW)]["looks"]) == 1
    swapped = startlog.parse(lineups(team("GSW", "Confirmed", swap=True), team("NOP")))
    rec = startlog.update(rec, sched, 37, swapped, TIP - timedelta(minutes=20))
    assert [lk["status"] for lk in rec["teams"][str(GSW)]["looks"]] == ["Expected", "Confirmed"]


def test_the_watch_takes_the_evening_look(recorded):
    app, store, sched = _app(recorded)
    day = min(d for d, gs in sched.games.items() if gs and d >= 3)
    game = min(sched.games[day], key=lambda g: g.start)
    away, home = sched.teams[game.away][0].upper(), sched.teams[game.home][0].upper()
    cut = linelog.early_cut(sched, day)
    assert cut < game.start - startlog.LEAD  # an ordinary night: hours before the first tip
    app.state.lineup_transport = httpx.MockTransport(
        lambda _: httpx.Response(200, json=lineups(team(away), team(home)))
    )
    app.state.now = lambda: cut - timedelta(hours=1)
    assert asyncio.run(app.state.lineup_sweep()) == []
    app.state.now = lambda: cut - timedelta(minutes=10)
    assert asyncio.run(app.state.lineup_sweep()) == [day]
    saved = store.get_lineups(2027, day)["teams"]
    assert len(saved) == 2 * len(sched.games[day])
    assert saved[str(game.home)]["looks"][0]["status"] == "Expected"
    assert not app.state.starting_fives.get((2027, day))  # an Expected five moves no line


def test_a_team_missing_from_the_file_is_still_marked_as_looked_at():
    sched = schedule()
    rec = startlog.update(None, sched, 37, {}, TIP - timedelta(minutes=60))
    assert rec["teams"][str(GSW)]["looks"] == [] and rec["teams"][str(GSW)]["checked"]
    assert startlog.days_to_watch(sched, TIP - timedelta(minutes=60)) == [37]
    assert startlog.days_to_watch(sched, TIP - timedelta(hours=3)) == []  # nothing within the lead yet


def test_fetch_asks_for_the_new_york_date_like_a_page_on_nba_com():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=lineups(team("GSW"), team("NOP")))

    raw = asyncio.run(startlog.fetch(date(2026, 11, 25), httpx.MockTransport(handler)))
    assert raw["games"][0]["gameId"] == "0022600070"
    assert seen[0].url.path.endswith("00_daily_lineups_20261125.json")
    assert seen[0].headers["Referer"] == "https://www.nba.com/"


def test_the_watch_saves_the_fives_and_admins_read_them_back(recorded):
    app, store, sched = _app(recorded)
    day = min(d for d, gs in sched.games.items() if gs and d >= 3)
    game = sched.games[day][0]
    away, home = sched.teams[game.away][0].upper(), sched.teams[game.home][0].upper()
    app.state.now = lambda: game.start - timedelta(minutes=45)

    app.state.lineup_transport = httpx.MockTransport(lambda _: httpx.Response(403, text="Forbidden"))
    try:
        asyncio.run(app.state.lineup_sweep())
    except httpx.HTTPError:
        pass
    assert app.state.lineup_log["ok"] is False and "403" in app.state.lineup_log["error"]

    app.state.lineup_transport = httpx.MockTransport(
        lambda _: httpx.Response(200, json=lineups(team(away), team(home, "Confirmed", inactive=True)))
    )
    assert asyncio.run(app.state.lineup_sweep()) == [day]
    saved = store.get_lineups(2027, day)["teams"]
    assert saved[str(game.home)]["confirmed_at"] and saved[str(game.away)]["confirmed_at"] is None
    assert app.state.lineup_log["ok"] is True and app.state.lineup_log["teams"] == 2

    assert signed_in(app, team_id=None).post("/api/admin/lineups", json={"season": 2027}).status_code == 403
    got = boss(app).post("/api/admin/lineups", json={"season": 2027}).json()
    assert got["day"] == day and got["days"] == [day] and got["look"]["ok"] is True
    by_team = {t["team"]: t for t in got["teams"]}
    assert by_team[home]["looks"][0]["status"] == "Confirmed"
    assert len(by_team[home]["looks"][0]["starters"]) == 5 and by_team[home]["looks"][0]["inactive"][0]["nba_id"] == 21


def test_a_confirmed_five_is_the_last_look_with_every_starter_named():
    ids = {f"GSW {i}": 100 + i for i in range(1, 6)} | {f"NOP {i}": 200 + i for i in range(1, 5)}
    first = startlog.update(None, schedule(), 37, startlog.parse(lineups(team("GSW"), team("NOP"))), TIP - LATE)
    assert startlog.confirmed_fives(first, lambda name, _team: ids.get(name)) == {}  # Expected: recorded only
    listed = startlog.parse(lineups(team("GSW", "Confirmed"), team("NOP", "Confirmed")))
    rec = startlog.update(first, schedule(), 37, listed, TIP - timedelta(minutes=20))
    # New Orleans' fifth starter isn't a player we know: half a five would bench a starter, so none of it
    assert startlog.confirmed_fives(rec, lambda name, _team: ids.get(name)) == {GSW: frozenset(range(101, 106))}
    assert startlog.confirmed_fives(None, lambda *_: 1) == {}


LATE = timedelta(minutes=80)


def test_a_confirmed_five_reaches_tonights_lines_and_the_line_kept_at_tip_off(recorded):
    app, store, sched = _app(recorded)
    app.state.line_log_league_id = LEAGUE_ID
    day = min(d for d, gs in sched.games.items() if gs and d >= 3)
    game = sched.games[day][0]
    away, home = sched.teams[game.away][0].upper(), sched.teams[game.home][0].upper()
    app.state.now = lambda: game.start - LATE

    assert asyncio.run(app.state.line_log_sweep()) == [day]
    league = app.state.leagues.get((LEAGUE_ID, 2027, credentials_key(None), True, True, LINE_LOG_FREE_AGENTS))
    pool = [*(p for t in league.teams for p in t.roster), *league.free_agents]
    squad = [p for p in pool if p.pro_team_id == game.home and str(p.id) in store.get_player_lines(2027, day)["lines"]]
    assert len(squad) >= 6
    five, sixth = squad[:5], squad[5]
    # the box knows their games: the five have come off the bench lately, the sixth has started
    box = app.state.game_boxes.get(2027)
    for p in squad[:6]:
        base = dict(rate={}, minutes=30.0, minutes_long=30.0, usage=0.0, last_day=day - 1, team=game.home)
        box.form[p.id] = gamebox.Form(**base, starts_lately=1.0 if p is sixth else 0.0)
    app.state.leagues.clear()
    asyncio.run(app.state.line_log_sweep())
    before = store.get_player_lines(2027, day)["lines"]

    def nba(status: str) -> dict:
        players = [player(i, p.name, "G", status) for i, p in enumerate(five, start=1)]
        return lineups(team(away), {"teamAbbreviation": home, "players": players})

    # a first listing is recorded and moves nothing
    app.state.lineup_transport = httpx.MockTransport(lambda _: httpx.Response(200, json=nba("Expected")))
    asyncio.run(app.state.lineup_sweep())
    assert app.state.starting_fives.get((2027, day), {}) == {}
    assert store.get_player_lines(2027, day)["lines"] == before

    # confirmed: the fives are kept by our player ids, and the line log looks again at once
    app.state.now = lambda: game.start - timedelta(minutes=20)
    app.state.lineup_transport = httpx.MockTransport(lambda _: httpx.Response(200, json=nba("Confirmed")))
    asyncio.run(app.state.lineup_sweep())
    assert app.state.starting_fives[(2027, day)] == {game.home: frozenset(p.id for p in five)}
    after = store.get_player_lines(2027, day)["lines"]
    for p in five:
        assert after[str(p.id)][1] == pytest.approx(before[str(p.id)][1] + gamebox.STARTING_MINUTES, abs=0.11)
    assert after[str(sixth.id)][1] == pytest.approx(before[str(sixth.id)][1] - gamebox.STARTING_MINUTES, abs=0.11)
    others = [k for k, v in before.items() if v[0] != game.home]
    assert others and all(after[k] == before[k] for k in others)  # the away five is only Expected

    # a league read from the cache carries it too
    cached = app.state.leagues.get((LEAGUE_ID, 2027, credentials_key(None), True, True, LINE_LOG_FREE_AGENTS))
    by_id = {p.id: p for p in [*(p for t in cached.teams for p in t.roster), *cached.free_agents]}
    assert all(by_id[p.id].game_context.starts is True for p in five)
    assert by_id[sixth.id].game_context.starts is False
    elsewhere = [p for p in by_id.values() if p.pro_team_id == game.away and p.game_context is not None]
    assert elsewhere and all(p.game_context.starts is None for p in elsewhere)

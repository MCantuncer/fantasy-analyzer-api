import asyncio
from datetime import UTC, date, datetime, timedelta

import httpx

from app import startlog
from app.scores import Schedule, ScheduledGame
from tests.conftest import signed_in
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

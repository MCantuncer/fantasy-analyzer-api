import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

from app import iddaa
from app.scores import Schedule, ScheduledGame
from tests.conftest import signed_in
from tests.test_linelog import _app
from tests.test_news import boss

# MIL @ PHI, 2026-11-21 18:00 UTC, as iddaa answered that morning (trimmed to a few players).
EVENT = json.loads((Path(__file__).parent / "fixtures" / "iddaa_event_mil_phi.json").read_text())
TIP = datetime.fromtimestamp(EVENT["d"], UTC)
MIL, PHI, BOS, SAC = 15, 20, 2, 23


def schedule() -> Schedule:
    return Schedule(
        teams={
            MIL: ("MIL", "Milwaukee Bucks"),
            PHI: ("PHI", "Philadelphia 76ers"),
            BOS: ("BOS", "Boston Celtics"),
            SAC: ("SAC", "Sacramento Kings"),
            12: ("LAC", "LA Clippers"),
            13: ("LAL", "Los Angeles Lakers"),
        },
        games={
            33: [
                ScheduledGame(401, 33, TIP, away=MIL, home=PHI, official=False),
                ScheduledGame(402, 33, TIP + timedelta(hours=4), away=BOS, home=SAC, official=False),
            ]
        },
    )


def listed(**over) -> dict:
    return {**{k: EVENT[k] for k in ("i", "hn", "an", "ci", "d")}, **over}


def test_team_names_match_by_city_or_city_and_name():
    assert iddaa.team_matches("Philadelphia", "Philadelphia 76ers")
    assert iddaa.team_matches("LA Clippers", "LA Clippers")
    assert iddaa.team_matches("LA Lakers", "Los Angeles Lakers")
    assert not iddaa.team_matches("LA Clippers", "Los Angeles Lakers")
    assert not iddaa.team_matches("", "Boston Celtics")


def test_an_event_matches_its_espn_game_by_teams_and_start():
    sched = schedule()
    assert iddaa.match_game(listed(), sched).id == 401
    assert iddaa.match_game(listed(hn="Milwaukee", an="Philadelphia"), sched) is None  # home and away matter
    assert iddaa.match_game(listed(d=EVENT["d"] + 6 * 3600), sched) is None  # another night's game


def test_a_game_reads_into_main_lines_and_player_ladders():
    got = iddaa.parse_event(EVENT)
    assert got["margin"] == [-11.5, 1.6, 1.7]  # of seven handicaps, the one with its odds closest
    assert got["total"] == [228.5, 1.68, 1.62]
    assert got["home_pts"][0] == 120.5 and got["away_pts"][0] == 108.5
    assert got["win"] == [0.0, 4.71]  # iddaa closed the home side
    assert all(odd > 1.0 for rungs in got["players"]["Nate Ament"].values() for _, odd in rungs)
    read = iddaa.readout(got)
    assert read["home_chance"] is None  # home closed (a near-certainty): the book isn't pricing the winner
    embiid, maxey = read["players"]["Joel Embiid"], read["players"]["Tyrese Maxey"]
    assert 25 < embiid["pts"] < 29 and 24 < maxey["pts"] < 27 and 2 < maxey["three_pm"] < 4


def test_a_ladder_line_is_where_it_crosses_half_after_the_margin():
    rungs = [[10, 1.25], [12, 1.65], [14, 2.4]]  # 10+: 66%, 12+: 50%, 14+: 34% once 21% is taken out
    assert iddaa.ladder_line(rungs, 1.21) == 11.5
    assert iddaa.ladder_line([[10, 1.5], [12, 2.25]], 1.0) == 11.0  # 67% at 9.5, 44% at 11.5: drawn straight
    # Can: a rung at or under 1.33 isn't the book saying anything, so a ladder that crosses only there has no line.
    assert iddaa.ladder_line([[10, 1.33], [12, 2.5]], 1.0) is None


def test_a_winner_priced_at_1_33_or_under_is_not_the_book_s_pick():
    assert iddaa.home_chance({"win": [1.6, 2.3]}) == 0.59
    assert iddaa.home_chance({"win": [1.33, 3.4]}) is None
    assert iddaa.home_chance({"win": [1.34, 3.4]}) is not None
    assert iddaa.ladder_line([[20, 3.0], [22, 5.0]], 1.21) is None  # never reaches half


def test_each_game_keeps_its_last_look_before_tip_off():
    sched = schedule()
    early = TIP - timedelta(hours=3)
    rec = iddaa.update({}, sched, [(listed(), EVENT)], early)[33]
    assert rec["games"]["401"]["checked"] == early.isoformat(timespec="seconds")
    later = dict(EVENT, m=[m for m in EVENT["m"] if m["st"] != 114])
    rec = iddaa.update({33: rec}, sched, [(listed(), later)], TIP - timedelta(minutes=5))[33]
    assert "total" not in rec["games"]["401"]  # the newer look replaces the old
    assert iddaa.update({33: rec}, sched, [(listed(), EVENT)], TIP + timedelta(minutes=1)) == {}  # tipped off
    assert iddaa.update({}, sched, [(listed(), EVENT)], TIP - timedelta(hours=19)) == {}  # too early


def test_a_look_keeps_our_last_call_until_a_new_one_is_set():
    sched = schedule()
    rec = iddaa.update({}, sched, [(listed(), EVENT)], TIP - timedelta(hours=3))[33]
    rec["games"]["401"]["ours"] = [15, 20, 0.8]
    rec = iddaa.update({33: rec}, sched, [(listed(), EVENT)], TIP - timedelta(hours=2))[33]
    assert rec["games"]["401"]["ours"] == [15, 20, 0.8]  # a look without the box (just after a deploy) keeps it


def bulletin(events: list[dict], status: int = 200, broken: frozenset = frozenset()):
    """iddaa's bulletin: the list of games, then each game's markets (EVENT's, under the listed id)."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Referer"] == "https://www.iddaa.com/"
        assert request.url.host == "sportsbookv2.iddaa.com"  # Can 2026-11-22: iddaa only, no nesine
        if status != 200:
            return httpx.Response(status, text="Forbidden")
        if request.url.path == "/sportsbook/events":
            return httpx.Response(200, json={"isSuccess": True, "data": {"events": events}, "message": ""})
        event_id = int(request.url.path.rsplit("/", 1)[1])
        if event_id in broken:
            return httpx.Response(500, text="oops")
        listed = next(e for e in events if e["i"] == event_id)
        return httpx.Response(200, json={"isSuccess": True, "data": {**EVENT, **listed}, "message": ""})

    return httpx.MockTransport(handler)


def test_the_watch_saves_iddaa_and_admins_read_it_back(recorded):
    app, store, sched = _app(recorded)
    day = min(d for d, gs in sched.games.items() if gs and d >= 3)
    game = sched.games[day][0]
    away, home = sched.teams[game.away][1], sched.teams[game.home][1]
    event = {"i": 9, "hn": home, "an": away, "ci": iddaa.NBA, "d": int(game.start.timestamp())}
    app.state.now = lambda: game.start - timedelta(hours=1)

    app.state.iddaa_transport = bulletin([], status=403)
    try:
        asyncio.run(app.state.book_log_sweep())
    except httpx.HTTPError:
        pass
    assert app.state.book_log["ok"] is False and "403" in app.state.book_log["error"]

    other = {**event, "i": 10, "ci": 1}  # a non-NBA game is skipped
    later = {**event, "i": 11, "d": event["d"] + 3600}  # its markets fail to read: left for the next look
    app.state.iddaa_transport = bulletin([event, other, later], broken=frozenset({11}))
    assert asyncio.run(app.state.book_log_sweep()) == [day]
    saved = store.get_book_lines(2027, day)["games"]
    assert list(saved) == [str(game.id)] and saved[str(game.id)]["source"] == "iddaa"
    assert app.state.book_log == {
        "checked": app.state.book_log["checked"],
        "ok": True,
        "error": None,
        "games": 1,
        "matched": 1,
    }

    assert signed_in(app, team_id=None).post("/api/admin/book-lines", json={"season": 2027}).status_code == 403
    got = boss(app).post("/api/admin/book-lines", json={"season": 2027}).json()
    assert got["day"] == day and got["days"] == [day] and got["look"]["ok"] is True
    [g] = got["games"]
    assert g["game_id"] == game.id and g["margin"] == -11.5 and g["total"] == 228.5
    assert g["ours"]["away"] != g["ours"]["home"] and 0.5 <= g["ours"]["chance"] <= 1  # our call, same look
    assert {"Joel Embiid", "Tyrese Maxey"} <= {p["name"] for p in g["players"]}


def test_our_call_beside_iddaa_counts_who_is_ruled_out(recorded):
    """Can 2026-11-22: a player ruled out during the day must move our game call on Our Estimates, as
    it does on Scores: each look calls the game with the OUT tags saved for that day."""
    app, store, sched = _app(recorded)
    day = min(d for d, gs in sched.games.items() if gs and d >= 3)
    game = sched.games[day][0]
    away, home = sched.teams[game.away][1], sched.teams[game.home][1]
    event = {"i": 9, "hn": home, "an": away, "ci": iddaa.NBA, "d": int(game.start.timestamp())}
    app.state.now = lambda: game.start - timedelta(hours=1)
    app.state.iddaa_transport = bulletin([event])
    asyncio.run(app.state.book_log_sweep())  # builds the box on its first look
    box = app.state.game_boxes.get(2027) or app.state.last_boxes.get(2027)
    assert box is not None
    store.save_pregame(2027, day, {"teams": {}, "players": {"77": {"team": game.home, "status": "OUT"}}})
    seen = []
    real = box.call
    box.call = lambda d, a, h, out=None: seen.append(out) or real(d, a, h, out)
    asyncio.run(app.state.book_log_sweep())
    assert seen and all(out == {77} for out in seen)

import httpx
import pytest

from app import liveline
from app.gameclock import Clock, clock_seconds, elapsed_minutes, parse_scoreboard
from tests.conftest import LEAGUE_ID
from tests.test_scores import FRIDAY_NIGHT, day4_players, schedule_raw, scores_client  # noqa: F401


def test_plain_line_when_nothing_unusual_happened():
    # 20 points in 32 minutes, on his usual pace at halftime with a close score: half of it is left.
    got = liveline.live_line(10, 16, 20.0, 32.0, 24.0, 3)
    assert got is not None
    assert got.final == pytest.approx(20.0, abs=0.01) and got.minutes_left == pytest.approx(16.0, abs=0.01)


def test_a_quiet_start_is_not_made_up_later():
    # 2 points in 16 minutes from a 20-point line: the rest comes at a little under his usual rate.
    got = liveline.live_line(2, 16, 20.0, 32.0, 24.0, 0)
    usual_rest = 20.0 / 32.0 * 16.0
    assert got.rest < usual_rest and got.rest > usual_rest - 1.0
    assert got.final < 13  # nowhere near 20


def test_a_hot_night_carries_a_little():
    hot = liveline.live_line(18, 16, 20.0, 32.0, 24.0, 0)
    assert 10.0 < hot.rest < 11.0  # usual rest is 10


def test_a_blowout_takes_minutes_from_starters_and_gives_them_to_the_bench():
    star_close = liveline.live_line(14, 26, 28.0, 36.0, 36.0, 2)
    star_blowout = liveline.live_line(14, 26, 28.0, 36.0, 36.0, 28)
    assert star_blowout.minutes_left < star_close.minutes_left - 1
    bench_close = liveline.live_line(4, 11, 8.0, 16.0, 36.0, -2)
    bench_blowout = liveline.live_line(4, 11, 8.0, 16.0, 36.0, -28)
    assert bench_blowout.minutes_left > bench_close.minutes_left


def test_hot_role_player_gets_a_little_more_time():
    cold = liveline.live_line(0, 8, 8.0, 18.0, 20.0, 0)
    hot = liveline.live_line(10, 8, 8.0, 18.0, 20.0, 0)
    assert hot.minutes_left > cold.minutes_left


def test_nothing_left_at_the_end_and_no_line_without_minutes():
    end = liveline.live_line(22, 34, 20.0, 32.0, 48.0, 5)
    assert end.rest == 0 and end.final == 22 and end.half_within == 0
    assert liveline.live_line(5, 10, 10.0, 0.0, 20.0, 0) is None


def test_the_miss_shrinks_as_the_game_goes():
    assert liveline.within(0) == (4.4, 8.3)
    h, m = liveline.within(27)
    assert h == pytest.approx(2.9) and m == pytest.approx(5.1)
    assert liveline.within(60) == (0.0, 0.0)


def test_clock_reading():
    assert clock_seconds("2:31") == 151 and clock_seconds("45.2") == 45.2 and clock_seconds("") is None
    assert elapsed_minutes(Clock("in", "Q3 6:00", frozenset(), 3, 360.0)) == 30.0
    assert elapsed_minutes(Clock("in", "Half", frozenset(), 2, 0.0)) == 24.0
    assert elapsed_minutes(Clock("in", "OT 1:00", frozenset(), 5, 60.0)) == 48.0
    assert elapsed_minutes(Clock("pre", None, frozenset())) is None
    raw = {
        "events": [
            {
                "id": "7",
                "status": {"period": 2, "displayClock": "5:10", "type": {"state": "in", "shortDetail": "5:10 - 2nd"}},
                "competitions": [{"competitors": []}],
            }
        ]
    }
    c = parse_scoreboard(raw)[7]
    assert c.period == 2 and c.seconds == 310.0 and c.label == "Q2 5:10"


def _live_players(game_id: int, away: int, home: int) -> list[dict]:
    def player(pid: int, name: str, team: int, minutes: int, pts: int) -> dict:
        stats = {"0": float(pts), "40": float(minutes)}
        split = {"externalId": str(game_id), "proTeamId": team, "scoringPeriodId": 5, "seasonId": 2027}
        split |= {"statSourceId": 0, "statSplitTypeId": 5, "stats": stats}
        return {"id": pid, "player": {"id": pid, "fullName": name, "proTeamId": team, "stats": [split]}}

    return [
        player(1, "Away Star", away, 18, 6),  # quiet start from a 24-point line
        player(2, "Away Bench", away, 8, 6),
        player(3, "Home Star", home, 17, 14),
        player(4, "No Line", home, 10, 4),  # no tip-off line saved: left out
    ]


def test_live_lines_endpoint(schedule_raw):  # noqa: F811
    client, _, _ = scores_client({}, schedule_raw, [], FRIDAY_NIGHT)
    games = client.post("/api/scores/games", json={"league_id": LEAGUE_ID, "season": 2027}).json()["games"]
    live = next(g for g in games if g["status"] == "live")
    away, home = live["away"]["pro_team_id"], live["home"]["pro_team_id"]

    def scoreboard(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/summary"):
            return httpx.Response(404)
        event = {
            "id": str(live["id"]),
            "status": {"period": 2, "displayClock": "0.0", "type": {"state": "in", "shortDetail": "Halftime"}},
            "competitions": [{"competitors": []}],
        }
        return httpx.Response(200, json={"events": [event]})

    client, _, store = scores_client({}, schedule_raw, _live_players(live["id"], away, home), FRIDAY_NIGHT, scoreboard)
    store.save_player_lines(
        2027,
        5,
        {
            "teams": {},
            "lines": {
                "1": [away, 34.0, 24.0, 5, 5, 1, 0, 2, 2, 9, 18, 4, 5],
                "2": [away, 16.0, 7.0, 3, 1, 0, 0, 1, 1, 3, 6, 1, 1],
                "3": [home, 33.0, 21.0, 6, 4, 1, 1, 2, 2, 8, 17, 3, 4],
            },
            "results": {},
        },
        False,
    )
    r = client.post("/api/scores/live-lines", json={"league_id": LEAGUE_ID, "season": 2027, "game_id": live["id"]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "live" and body["clock"] == "Half" and body["elapsed"] == 24.0
    assert body["half_within"] == 3.1 and body["most_within"] == 5.5
    rows = {x["player_id"]: x for x in body["lines"]}
    assert set(rows) == {1, 2, 3}
    star = rows[1]
    assert star["line_pts"] == 24.0 and star["pts"] == 6
    assert 6 + 10 < star["final"] < 6 + 12.5  # about his usual rest, a touch under: no catch-up
    assert rows[3]["final"] > 14

    # Before tip-off or after the final: no lines.
    done = next(g for g in games if g["status"] == "scheduled")
    r = client.post("/api/scores/live-lines", json={"league_id": LEAGUE_ID, "season": 2027, "game_id": done["id"]})
    assert r.status_code == 200 and r.json()["lines"] == [] and r.json()["elapsed"] is None
    assert (
        client.post("/api/scores/live-lines", json={"league_id": LEAGUE_ID, "season": 2027, "game_id": 1}).status_code
        == 404
    )

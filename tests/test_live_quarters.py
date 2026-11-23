import httpx
import pytest

from app import quarters
from app.analysis import gamebox, gamescore
from app.gameclock import parse_fouls, parse_linescores
from tests.conftest import LEAGUE_ID
from tests.test_scores import FRIDAY_NIGHT, schedule_raw, scores_client  # noqa: F401

PRE = [28.0, 28.0, 28.0, 27.0]


def test_on_the_called_margin_the_rest_is_our_number():
    # Level with our call at halftime: Q3 and Q4 stay at the pre-game numbers.
    got = quarters.live_quarters(PRE, PRE, [30, 26], [30, 26], 2, 0.0)
    assert got == [30, 26, 28.0, 27.0]


def test_a_team_past_the_called_margin_gives_a_little_back():
    # Ahead by 12 at halftime against a level call: +6 a quarter past it, 6.5% of it back per quarter.
    ahead = quarters.live_quarters(PRE, PRE, [34, 32], [28, 26], 2, 0.0)
    behind = quarters.live_quarters(PRE, PRE, [28, 26], [34, 32], 2, 0.0)
    assert ahead[2] == pytest.approx(28 + quarters.GIVE_BACK * 6, abs=1e-6) and behind[2] == pytest.approx(
        28 - quarters.GIVE_BACK * 6, abs=1e-6
    )


def test_the_quarter_in_progress_is_real_so_far_plus_the_rest():
    # 6 minutes into Q3 (half of it left), even with the call: 15 so far + half of 28, and the rest of a
    # quarter scores a little more than its start.
    got = quarters.live_quarters(PRE, PRE, [28, 28, 15], [28, 28, 14], 3, 360.0)
    assert got[:2] == [28, 28]
    move = quarters.GIVE_BACK * 1 / 2.5  # one point past a level call over 2.5 quarters
    shape = quarters.IN_QUARTER * 0.5 * 0.5 * 12
    assert got[2] == pytest.approx(15 + (28 + move) * 0.5 + shape)
    assert got[3] == pytest.approx(27, abs=0.1)


def test_the_penalty_adds_to_the_rest_of_the_quarter_only():
    plain = quarters.live_quarters(PRE, PRE, [28, 28, 15], [28, 28, 14], 3, 360.0)
    bonus = quarters.live_quarters(PRE, PRE, [28, 28, 15], [28, 28, 14], 3, 360.0, bonus=True)
    assert bonus[2] - plain[2] == pytest.approx(quarters.BONUS * 6)  # six minutes left
    assert bonus[3] == plain[3]


def test_when_a_side_is_in_the_penalty():
    assert not quarters.in_penalty([600, 500, 400], 300)  # three fouls
    assert quarters.in_penalty([600, 500, 400, 350], 300)  # four: the next one gives free throws
    assert not quarters.in_penalty([600, 500, 400, 350], 360)  # the fourth hasn't happened yet
    assert not quarters.in_penalty([600], 100)  # none in the last two minutes yet
    assert quarters.in_penalty([600, 110], 100)  # one in the last two minutes: the second gives them
    assert not quarters.in_penalty([], None)


def test_fouls_from_the_summary():
    def a(aid, pf):
        return {"athlete": {"id": str(aid)}, "stats": ["20", str(pf)]}

    def foul(per, clock, kind, team):
        return {
            "type": {"text": kind},
            "period": {"number": per},
            "clock": {"displayValue": clock},
            "team": {"id": team},
        }

    raw = {
        "boxscore": {
            "players": [
                {"team": {"id": "14"}, "statistics": [{"keys": ["minutes", "fouls"], "athletes": [a(1, 5), a(2, 0)]}]},
                {"team": {"id": "30"}, "statistics": [{"keys": ["minutes", "fouls"], "athletes": [a(3, 6)]}]},
            ]
        },
        "plays": [
            foul(2, "1:00", "Shooting Foul", "14"),  # last quarter: not counted
            foul(3, "9:00", "Personal Foul", "14"),
            foul(3, "8:00", "Offensive Charge", "14"),  # offensive: no team foul
            foul(3, "7:00", "Loose Ball Foul", "14"),
            foul(3, "6:30", "Technical Foul", "30"),  # technical: no team foul
            foul(3, "6:10", "Personal Take Foul", "30"),
        ],
    }
    got = parse_fouls(raw)
    assert got.players == {1: 5, 2: 0, 3: 6} and got.period == 3
    assert got.teams == {14: (540.0, 420.0), 30: (370.0,)}
    assert parse_fouls({"boxscore": {}}) is None


def test_before_tip_off_and_in_overtime():
    assert quarters.live_quarters(PRE, PRE, [], [], 0, None) == PRE
    assert quarters.live_quarters(PRE, PRE, [25, 30, 27, 28, 9], [26, 29, 28, 27, 7], 5, 120.0) == [25, 30, 27, 28]


def test_linescores_from_the_scoreboard():
    comp = {
        "competitors": [
            {"homeAway": "home", "linescores": [{"value": 30}, {"value": 12}]},
            {"homeAway": "away", "linescores": [{"value": 25}, {"value": 9}]},
        ]
    }
    assert parse_linescores(comp) == ((25.0, 9.0), (30.0, 12.0))
    assert parse_linescores({"competitors": [{"homeAway": "home", "linescores": []}]}) is None


def test_live_lines_carry_both_sides_quarters(schedule_raw):  # noqa: F811
    client, _, _ = scores_client({}, schedule_raw, [], FRIDAY_NIGHT)
    games = client.post("/api/scores/games", json={"league_id": LEAGUE_ID, "season": 2027}).json()["games"]
    live = next(g for g in games if g["status"] == "live")

    def scoreboard(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/summary"):
            if not with_fouls:
                return httpx.Response(404)
            fouls = [
                {"type": {"text": "Personal Foul"}, "period": {"number": 3}, "clock": {"displayValue": c}}
                | {"team": {"id": str(away_id)}}
                for c in ("11:00", "10:00", "9:00", "8:00")
            ]
            box = {"keys": ["fouls"], "athletes": [{"athlete": {"id": "1"}, "stats": ["2"]}]}
            return httpx.Response(200, json={"boxscore": {"players": [{"statistics": [box]}]}, "plays": fouls})
        event = {
            "id": str(live["id"]),
            "status": {"period": 3, "displayClock": "6:00", "type": {"state": "in", "shortDetail": "6:00 - 3rd"}},
            "competitions": [
                {
                    "competitors": [
                        {
                            "homeAway": "away",
                            "team": {"abbreviation": "MIA"},
                            "linescores": [{"value": v} for v in (31, 29, 16)],
                        },
                        {
                            "homeAway": "home",
                            "team": {"abbreviation": "CHA"},
                            "linescores": [{"value": v} for v in (22, 24, 12)],
                        },
                    ]
                }
            ],
        }
        return httpx.Response(200, json={"events": [event]})

    away_id = live["away"]["pro_team_id"]
    with_fouls = False
    client, _, _ = scores_client({}, schedule_raw, [], FRIDAY_NIGHT, scoreboard)
    client.app.state.game_boxes.set(2027, gamebox.Box(2027, {}, {}, {}, scores=gamescore.build([])))
    r = client.post("/api/scores/live-lines", json={"league_id": LEAGUE_ID, "season": 2027, "game_id": live["id"]})
    assert r.status_code == 200, r.text
    q = r.json()["quarters"]
    assert q["away"]["bonus"] is None and q["away"]["fouls"] is None  # no summary: not known
    plain = q["home"]["live"][2]
    assert q["period"] == 3
    away, home = q["away"], q["home"]
    assert away["real"] == [31, 29, 16] and home["real"] == [22, 24, 12]
    assert len(away["pregame"]) == 4 and len(away["live"]) == 4
    assert away["live"][:2] == [31, 29] and away["live"][2] > 16
    # Miami is well past the called margin, so it gives a little back in the 4th; Charlotte gains it.
    assert away["live"][3] < away["pregame"][3] + 0.5 and home["live"][3] > home["pregame"][3] - 0.5
    assert away["live"][3] - (away["pregame"][3]) < home["live"][3] - home["pregame"][3]

    # Miami has four fouls in the 3rd: Charlotte is in the bonus, and its 3rd gains it.
    with_fouls = True
    client, _, _ = scores_client({}, schedule_raw, [], FRIDAY_NIGHT, scoreboard)
    client.app.state.game_boxes.set(2027, gamebox.Box(2027, {}, {}, {}, scores=gamescore.build([])))
    q = client.post(
        "/api/scores/live-lines", json={"league_id": LEAGUE_ID, "season": 2027, "game_id": live["id"]}
    ).json()["quarters"]
    assert q["away"]["fouls"] == 4 and q["home"]["fouls"] == 0
    assert q["home"]["bonus"] is True and q["away"]["bonus"] is False
    assert q["home"]["live"][2] == pytest.approx(plain + quarters.BONUS * 6, abs=0.06)

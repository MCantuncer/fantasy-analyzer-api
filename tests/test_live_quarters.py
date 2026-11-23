import httpx
import pytest

from app import quarters
from app.analysis import gamebox, gamescore
from app.gameclock import parse_linescores
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
    # 6 minutes into Q3 (half of it left), even with the call: 15 so far + half of 28.
    got = quarters.live_quarters(PRE, PRE, [28, 28, 15], [28, 28, 14], 3, 360.0)
    assert got[:2] == [28, 28]
    move = quarters.GIVE_BACK * 1 / 2.5  # one point past a level call over 2.5 quarters
    assert got[2] == pytest.approx(15 + (28 + move) * 0.5)
    assert got[3] == pytest.approx(27, abs=0.1)


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
            return httpx.Response(404)
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

    client, _, _ = scores_client({}, schedule_raw, [], FRIDAY_NIGHT, scoreboard)
    client.app.state.game_boxes.set(2027, gamebox.Box(2027, {}, {}, {}, scores=gamescore.build([])))
    r = client.post("/api/scores/live-lines", json={"league_id": LEAGUE_ID, "season": 2027, "game_id": live["id"]})
    assert r.status_code == 200, r.text
    q = r.json()["quarters"]
    assert q["period"] == 3
    away, home = q["away"], q["home"]
    assert away["real"] == [31, 29, 16] and home["real"] == [22, 24, 12]
    assert len(away["pregame"]) == 4 and len(away["live"]) == 4
    assert away["live"][:2] == [31, 29] and away["live"][2] > 16
    # Miami is well past the called margin, so it gives a little back in the 4th; Charlotte gains it.
    assert away["live"][3] < away["pregame"][3] + 0.5 and home["live"][3] > home["pregame"][3] - 0.5
    assert away["live"][3] - (away["pregame"][3]) < home["live"][3] - home["pregame"][3]

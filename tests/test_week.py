import gzip
import json
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pytest

from app.accounts import MemoryAccountStore
from app.espn.client import EspnClient
from app.espn.parse import IR_SLOT
from app.main import create_app
from app.scores import parse_schedule
from app.store import MemoryStore
from app.week import category_scores, live_score, roster_pro_teams, week_bounds, week_games
from tests.conftest import LEAGUE_ID, MY_TEAM, FakeEspn, signed_in

FIXTURES = Path(__file__).parent / "fixtures"
FRIDAY_NIGHT = datetime(2026, 10, 24, 1, 0, tzinfo=UTC)  # Fri Oct 23, 9pm ET: day 4 under way
HOU, SA, MIA = 10, 24, 14


@pytest.fixture(scope="module")
def schedule_raw() -> dict:
    with gzip.open(FIXTURES / "scores_schedule_2027.json.gz", "rt") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def schedule(schedule_raw):
    return parse_schedule(schedule_raw)


def test_week_runs_monday_to_sunday_on_us_dates():
    assert week_bounds(FRIDAY_NIGHT) == (date(2026, 10, 19), date(2026, 10, 25))
    # 3am UTC on a Monday is still Sunday evening in the US.
    assert week_bounds(datetime(2026, 10, 26, 3, 0, tzinfo=UTC)) == (date(2026, 10, 19), date(2026, 10, 25))
    assert week_bounds(datetime(2026, 10, 26, 12, 0, tzinfo=UTC))[0] == date(2026, 10, 26)


def test_week_games_count_one_per_player_per_game(schedule):
    week = [g for d, gs in schedule.games.items() if schedule.date_of(d) <= date(2026, 10, 25) for g in gs]
    hou = sum(1 for g in week if HOU in (g.home, g.away))
    one = week_games([HOU], schedule, FRIDAY_NIGHT)
    assert one.total == hou
    two = week_games([HOU, HOU, SA], schedule, FRIDAY_NIGHT)
    assert two.total == 2 * hou + sum(1 for g in week if SA in (g.home, g.away))
    assert 0 <= two.left <= two.total


def test_games_left_drop_once_a_game_tips_off(schedule):
    friday = [g for g in schedule.games[4]]
    team = friday[0].home
    before = week_games([team], schedule, friday[0].start.replace(hour=friday[0].start.hour - 1))
    after = week_games([team], schedule, friday[0].start)
    assert before.today == after.today == 1
    assert before.left == after.left + 1


def test_no_games_outside_the_week(schedule):
    late = datetime(2027, 8, 1, tzinfo=UTC)
    assert week_games([HOU], schedule, late).total == 0


class _P:
    def __init__(self, pro: int, slot: int | None):
        self.pro_team_id, self.lineup_slot = pro, slot


def test_ir_and_unsigned_players_do_not_count():
    assert roster_pro_teams([_P(HOU, 0), _P(SA, IR_SLOT), _P(0, 12), _P(MIA, None)]) == [HOU, MIA]


def test_category_scores_follow_the_league_order():
    side = {
        "cumulativeScore": {
            "scoreByStat": {
                "0": {"score": 562.0, "result": "LOSS"},
                "19": {"score": 0.4977, "result": "WIN"},
                "11": {"score": 67.0, "result": "TIE"},
                "20": {"score": 0.7, "result": "LOSS", "ineligible": True},
            }
        }
    }
    cats = category_scores(side, [19, 0, 11, 20, 6])
    assert [(c.stat, c.value, c.result) for c in cats] == [("FG%", 0.4977, "W"), ("PTS", 562.0, "L"), ("TO", 67.0, "T")]
    assert category_scores({"cumulativeScore": {}}, [0]) == []


def test_live_scores_win_over_the_banked_ones():
    # Real League A numbers on the morning of 2026-11-02: ESPN had not banked Sunday's games yet.
    side = {
        "cumulativeScore": {"wins": 1, "losses": 8, "scoreByStat": {"6": {"score": 224.0, "result": "LOSS"}}},
        "cumulativeScoreLive": {"wins": 0, "losses": 9, "scoreByStat": {"6": {"score": 246.0, "result": "LOSS"}}},
    }
    assert live_score(side)["losses"] == 9
    assert [c.value for c in category_scores(side, [6])] == [246.0]
    # Past and future periods carry no live score, or an empty one: keep the banked score.
    banked = {"cumulativeScore": side["cumulativeScore"], "cumulativeScoreLive": {"wins": 0, "scoreByStat": None}}
    assert live_score(banked)["wins"] == 1
    assert live_score({"cumulativeScore": side["cumulativeScore"]})["wins"] == 1
    assert live_score({}) == {}


def test_overview_shows_the_live_matchup_score(recorded, schedule_raw):
    league = json.loads(json.dumps(recorded["league"]))
    period = league["status"]["currentMatchupPeriod"]
    m = next(x for x in league["schedule"] if x.get("matchupPeriodId") == period and "away" in x)
    m["home"]["cumulativeScore"] = {"wins": 1, "losses": 8, "ties": 0, "scoreByStat": {"0": {"score": 525.0}}}
    m["home"]["cumulativeScoreLive"] = {
        "wins": 0,
        "losses": 9,
        "ties": 0,
        "scoreByStat": {"0": {"score": 563.0, "result": "LOSS"}},
    }
    espn = FakeEspn({**recorded, "league": league}, schedule=schedule_raw)
    app = create_app(
        EspnClient(transport=httpx.MockTransport(espn.handler)), store=MemoryStore(), accounts=MemoryAccountStore()
    )
    app.state.now = lambda: FRIDAY_NIGHT
    body = signed_in(app).post("/api/league-overview", json={"league_id": LEAGUE_ID, "season": 2027}).json()
    home = next(mu["home"] for mu in body["matchups"] if mu["home"]["team_id"] == m["home"]["teamId"])
    assert (home["wins"], home["losses"]) == (0, 9)
    assert home["categories"] == [{"stat": "PTS", "value": 563.0, "result": "L"}]


def test_overview_asks_espn_for_live_scores():
    from app.espn.client import OVERVIEW_VIEWS

    assert "mLiveScoring" in OVERVIEW_VIEWS


def test_overview_sends_categories_games_and_standings(recorded, schedule_raw):
    league = json.loads(json.dumps(recorded["league"]))
    period = league["status"]["currentMatchupPeriod"]
    m = next(x for x in league["schedule"] if x.get("matchupPeriodId") == period and "away" in x)
    m["home"].setdefault("cumulativeScore", {})["scoreByStat"] = {"0": {"score": 500.0, "result": "WIN"}}
    league["teams"][0]["playoffSeed"] = 4
    espn = FakeEspn({**recorded, "league": league}, schedule=schedule_raw)
    app = create_app(
        EspnClient(transport=httpx.MockTransport(espn.handler)), store=MemoryStore(), accounts=MemoryAccountStore()
    )
    app.state.now = lambda: FRIDAY_NIGHT
    client = signed_in(app)
    body = client.post("/api/league-overview", json={"league_id": LEAGUE_ID, "season": 2027}).json()
    assert body["week_start"] == "2026-10-19" and body["week_end"] == "2026-10-25"
    sides = [s for mu in body["matchups"] for s in (mu["home"], mu["away"])]
    home = next(s for s in sides if s["team_id"] == m["home"]["teamId"])
    assert home["categories"] == [{"stat": "PTS", "value": 500.0, "result": "W"}]
    assert all(s["games"] is not None and s["games"]["total"] > 0 for s in sides)
    mine = next(s for s in sides if s["team_id"] == MY_TEAM)
    assert mine["games"]["left"] <= mine["games"]["total"]
    first = next(t for t in body["teams"] if t["id"] == league["teams"][0]["id"])
    assert first["seed"] == 4 and first["wins"] + first["losses"] + first["ties"] >= 0


def test_overview_still_works_without_game_counts(recorded, pro_schedule):
    espn = FakeEspn(recorded, schedule={"settings": {"proTeams": [{"id": 1, "proGamesByScoringPeriod": {"1": [{}]}}]}})
    app = create_app(
        EspnClient(transport=httpx.MockTransport(espn.handler)), store=MemoryStore(), accounts=MemoryAccountStore()
    )
    body = signed_in(app).post("/api/league-overview", json={"league_id": LEAGUE_ID, "season": 2027}).json()
    assert all(mu["home"]["games"] is None for mu in body["matchups"])

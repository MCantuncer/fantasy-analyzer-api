import copy
import gzip
import json
from pathlib import Path

import httpx
import pytest

from app.accounts import MemoryAccountStore
from app.espn.client import EspnClient
from app.main import create_app
from app.recap import VERSION, Week, build_recap, is_final
from app.store import MemoryStore
from tests.conftest import LEAGUE_ID, FakeEspn, signed_in

SCORES_SCHEDULE = Path(__file__).parent / "fixtures" / "scores_schedule_2027.json.gz"
BENCH, IR, PG = 12, 13, 0
CATS = [0, 6, 3, 2, 1, 17, 19, 20, 11]  # PTS REB AST STL BLK 3PM FG% FT% TO


def line(pid: int, name: str, pts: int, **kw) -> dict:
    base = dict(minutes=30, pts=pts, reb=5, ast=3, stl=1, blk=0, tpm=1, to=2, fgm=8, fga=17, ftm=3, fta=4)
    return {"player_id": pid, "name": name, "pro_team_id": 1, **base, **kw}


def side(team_id: int, wins: int, losses: int) -> dict:
    return {
        "teamId": team_id,
        "cumulativeScore": {
            "wins": wins,
            "losses": losses,
            "ties": 0,
            "scoreByStat": {"0": {"score": 500.0, "result": "WIN"}, "19": {"score": 0.47, "result": "LOSS"}},
        },
    }


@pytest.fixture
def week() -> Week:
    # Players 1-3 rostered all week, 9 a free agent added by team 20 on day 3, 8 free all week.
    days = [1, 2, 3]
    lines = {
        1: [line(1, "Star", 40, reb=12), line(2, "Starter", 15), line(3, "Benched", 25), line(8, "Free", 20)],
        2: [line(1, "Star", 35), line(9, "Hot Pickup", 30, stl=4), line(8, "Free", 22)],
        3: [line(9, "Hot Pickup", 28, stl=3), line(3, "Benched", 18)],
    }
    seats = {1: (10, PG), 2: (10, PG), 3: (20, BENCH)}
    rosters = {1: dict(seats), 2: dict(seats), 3: {**seats, 9: (20, PG)}}
    return Week(
        period=2,
        days=days,
        day_lines=lines,
        rosters=rosters,
        owners_now={1: 10, 2: 10, 3: 20, 9: 20},
        categories=CATS,
        matchups=[
            {"matchupPeriodId": 2, "winner": "HOME", "home": side(10, 5, 4), "away": side(20, 4, 5)},
            {"matchupPeriodId": 2, "winner": "AWAY", "home": side(30, 1, 8), "away": side(40, 8, 1)},
        ],
    )


def test_players_of_the_week_rank_by_value_and_name_their_owner(week):
    recap = build_recap(week)
    assert recap["players"][0]["name"] == "Star"
    assert recap["players"][0]["owner_team_id"] == 10
    assert recap["players"][0]["totals"]["pts"] == 75 and recap["players"][0]["games"] == 2
    assert [p["rank"] for p in recap["players"]] == list(range(1, len(recap["players"]) + 1))
    assert set(recap["players"][0]["z"]) == {"PTS", "REB", "AST", "STL", "BLK", "3PM", "FG%", "FT%", "TO"}


def test_free_agents_are_those_free_when_the_week_began(week):
    fas = {p["name"]: p for p in build_recap(week)["free_agents"]}
    assert set(fas) == {"Hot Pickup", "Free"}
    picked = fas["Hot Pickup"]["picked_up"]
    assert (picked["team_id"], picked["day"], picked["weekday"]) == (20, 3, 2)
    assert fas["Hot Pickup"]["owner_now_team_id"] == 20
    assert fas["Free"]["picked_up"] is None and fas["Free"]["owner_now_team_id"] is None


def test_team_mvp_counts_started_games_and_bench_counts_the_rest(week):
    teams = {t["team_id"]: t for t in build_recap(week)["teams"]}
    assert teams[10]["mvp"]["name"] == "Star"
    # Team 20 started only the pickup; its benched player's games count as left on the bench.
    assert teams[20]["mvp"]["name"] == "Hot Pickup"
    assert teams[20]["bench"]["games"] == 2 and teams[20]["bench"]["pts"] == 43
    assert teams[20]["best_add"]["name"] == "Hot Pickup" and teams[20]["best_add"]["weekday"] == 2
    assert teams[10]["best_add"] is None


def test_an_add_counts_only_the_games_he_played_for_his_new_team(week):
    recap = build_recap(week)
    pickup = next(p for p in recap["free_agents"] if p["name"] == "Hot Pickup")
    # He played days 2 and 3, but team 20 only had him on day 3.
    assert pickup["games"] == 2 and pickup["totals"]["pts"] == 58
    after = pickup["picked_up"]["after"]
    assert (after["games"], after["started"], after["totals"]["pts"], after["totals"]["stl"]) == (1, 1, 28, 3)
    assert recap["teams"][1]["best_add"]["after"] == after


def test_a_late_add_with_no_games_yet_is_not_the_best_add(week):
    # Team 20 also adds "Free" on the last day, after his games: a bigger week, nothing for them.
    week.rosters[3][8] = (20, BENCH)
    teams = {t["team_id"]: t for t in build_recap(week)["teams"]}
    assert teams[20]["best_add"]["name"] == "Hot Pickup"
    free = next(p for p in build_recap(week)["free_agents"] if p["name"] == "Free")
    assert free["picked_up"]["after"]["games"] == 0 and free["picked_up"]["after"]["value"] == 0.0


def test_matchups_biggest_margin_first(week):
    matchups = build_recap(week)["matchups"]
    assert [m["margin"] for m in matchups] == [7, 1]
    assert matchups[0]["winner_team_id"] == 40
    assert matchups[1]["home"]["values"] == {"PTS": 500.0, "FG%": 0.47}


def test_a_week_is_final_only_once_every_matchup_has_a_winner():
    assert not is_final([])
    assert not is_final([{"winner": "HOME"}, {"winner": "UNDECIDED"}])
    assert is_final([{"winner": "HOME"}, {"winner": "TIE"}])


@pytest.fixture(scope="module")
def nba_schedule() -> dict:
    with gzip.open(SCORES_SCHEDULE, "rt") as fh:
        return json.load(fh)


def finished_week_one(recorded: dict) -> dict:
    """The recorded league with week 1 settled."""
    data = copy.deepcopy(recorded)
    for m in data["league"]["schedule"]:
        if m["matchupPeriodId"] == 1:
            m["winner"] = "HOME"
    return data


def test_recap_endpoint_builds_once_then_serves_the_stored_week(recorded, nba_schedule):

    fake = FakeEspn(finished_week_one(recorded), schedule=nba_schedule)
    store = MemoryStore()
    app = create_app(
        EspnClient(transport=httpx.MockTransport(fake.handler)), store=store, accounts=MemoryAccountStore()
    )
    client = signed_in(app, team_id=None)
    r = client.post("/api/recap", json={"league_id": LEAGUE_ID})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["period"] == 1 and [w["period"] for w in body["weeks"]] == [1]
    assert body["start"] == "2026-10-20"
    assert {t["team_id"] for t in body["teams"]} and all(t["name"] for t in body["teams"])
    assert store.get_recap(LEAGUE_ID, 2027, 1) is not None
    calls = fake.calls
    app.state.recaps.clear()
    assert client.post("/api/recap", json={"league_id": LEAGUE_ID, "period": 1}).status_code == 200
    rosters_calls = [u for u in fake.seen_urls[calls:] if "scoringPeriodId" in u and "mRoster" in u]
    assert rosters_calls == []  # served from the store


def test_recap_endpoint_404s_before_any_week_is_over(client):
    r = client.post("/api/recap", json={"league_id": LEAGUE_ID})
    assert r.status_code == 404
    assert r.json()["detail"]["code"] == "week_not_finished"


def test_recap_endpoint_needs_cookies_for_a_private_league(recorded, nba_schedule):

    fake = FakeEspn(finished_week_one(recorded), private=True, schedule=nba_schedule)
    store = MemoryStore()
    store.save_recap(LEAGUE_ID, 2027, 1, {"period": 1})
    app = create_app(
        EspnClient(transport=httpx.MockTransport(fake.handler)), store=store, accounts=MemoryAccountStore()
    )
    r = signed_in(app, team_id=None).post("/api/recap", json={"league_id": LEAGUE_ID})
    assert r.status_code in (401, 403)


def test_a_recap_stored_by_an_older_version_is_rebuilt(recorded, nba_schedule):
    fake = FakeEspn(finished_week_one(recorded), schedule=nba_schedule)
    store = MemoryStore()
    store.save_recap(LEAGUE_ID, 2027, 1, {"period": 1, "days": [1], "matchups": [], "players": [], "teams": []})
    app = create_app(
        EspnClient(transport=httpx.MockTransport(fake.handler)), store=store, accounts=MemoryAccountStore()
    )
    r = signed_in(app, team_id=None).post("/api/recap", json={"league_id": LEAGUE_ID, "period": 1})
    assert r.status_code == 200, r.text
    assert store.get_recap(LEAGUE_ID, 2027, 1)["version"] == VERSION

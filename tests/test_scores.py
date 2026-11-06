import gzip
import json
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pytest

from app.espn.client import EspnClient
from app.main import create_app
from app.scores import build_day, game_status, parse_lines, parse_schedule, settled, week_fixtures
from app.store import MemoryStore
from tests.conftest import LEAGUE_ID, MY_TEAM, FakeEspn, signed_in

FIXTURES = Path(__file__).parent / "fixtures"
# Day 4 is Thu 23 Oct 2026 (12 games, all final); day 5 is Fri 24 Oct (8 games).
MORNING_AFTER = datetime(2026, 10, 24, 8, 0, tzinfo=UTC)
FRIDAY_NIGHT = datetime(2026, 10, 24, 23, 0, tzinfo=UTC)  # MIA @ CHA tipped off at 22:00 UTC
HOU, SA = 10, 24
AMEN_THOMPSON = 4684740


@pytest.fixture(scope="module")
def schedule_raw() -> dict:
    with gzip.open(FIXTURES / "scores_schedule_2027.json.gz", "rt") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def day4_players() -> list[dict]:
    with gzip.open(FIXTURES / "day_stats_2027_day4.json.gz", "rt") as fh:
        return json.load(fh)["players"]


@pytest.fixture(scope="module")
def schedule(schedule_raw):
    return parse_schedule(schedule_raw)


def test_schedule_days_and_dates(schedule):
    assert schedule.opening == date(2026, 10, 20)
    assert schedule.date_of(4) == date(2026, 10, 23)
    assert len(schedule.games[4]) == 12
    assert len(schedule.games[5]) == 8
    assert schedule.latest_started(MORNING_AFTER) == 4
    assert schedule.latest_started(FRIDAY_NIGHT) == 5


def test_scores_add_up_from_player_lines(schedule, day4_players):
    day = build_day(schedule, 4, day4_players, MORNING_AFTER)
    game = next(g for g in day["games"] if g["away"]["pro_team_id"] == HOU and g["home"]["pro_team_id"] == SA)
    assert game["status"] == "final"
    assert (game["away"]["score"], game["home"]["score"]) == (114, 110)
    assert game["away"]["abbrev"] == "HOU" and game["home"]["name"] == "San Antonio Spurs"
    amen = next(ln for ln in game["lines"] if ln["player_id"] == AMEN_THOMPSON)
    assert (amen["pts"], amen["reb"], amen["fgm"], amen["fga"], amen["minutes"]) == (20, 10, 9, 15, 36)
    assert all(g["away"]["score"] > 70 and g["home"]["score"] > 70 for g in day["games"])


def test_players_who_did_not_play_are_left_out(day4_players):
    lines = parse_lines(day4_players, 4)
    assert all(ln.minutes > 0 for game in lines.values() for ln in game)
    assert parse_lines(day4_players, 5) == {}  # another day's lines never leak in


def test_game_status_follows_the_clock(schedule):
    game = next(g for g in schedule.games[5] if g.start.hour == 22)
    assert game_status(game, datetime(2026, 10, 24, 21, 59, tzinfo=UTC)) == "scheduled"
    assert game_status(game, datetime(2026, 10, 24, 22, 30, tzinfo=UTC)) == "live"
    assert game_status(game, datetime(2026, 10, 25, 1, 0, tzinfo=UTC)) == "final"


def test_a_day_settles_only_after_its_last_game(schedule):
    assert settled(schedule, 4, MORNING_AFTER)
    assert not settled(schedule, 5, FRIDAY_NIGHT)
    assert not settled(schedule, 4, datetime(2026, 10, 24, 3, 0, tzinfo=UTC))


class ScoresEspn(FakeEspn):
    def __init__(self, recorded, schedule, players):
        super().__init__(recorded, schedule=schedule)
        self.players = players
        self.stat_calls = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.params.get("view") == "kona_player_info" and request.url.params.get("scoringPeriodId"):
            self.calls += 1
            self.stat_calls += 1
            return httpx.Response(200, json={"players": self.players})
        return super().handler(request)


def no_scoreboard(request: httpx.Request) -> httpx.Response:
    return httpx.Response(503)


def scores_client(recorded, schedule_raw, players, now: datetime, scoreboard=no_scoreboard):
    fake = ScoresEspn(recorded, schedule_raw, players)
    store = MemoryStore()
    app = create_app(
        EspnClient(transport=httpx.MockTransport(fake.handler)),
        store=store,
        clock_transport=httpx.MockTransport(scoreboard),
    )
    app.state.now = lambda: now
    return signed_in(app), fake, store


def test_scores_endpoint_tags_owners_and_saves_finished_days(recorded, schedule_raw, day4_players):
    client, fake, store = scores_client(recorded, schedule_raw, day4_players, MORNING_AFTER)
    r = client.post("/api/scores", json={"league_id": LEAGUE_ID, "season": 2027})
    assert r.status_code == 200
    body = r.json()
    assert body["day"] == 4 and body["date"] == "2026-10-23" and body["live"] is False
    assert {d["day"]: d["games"] for d in body["days"]}[4] == 12
    lines = [ln for g in body["games"] for ln in g["lines"]]
    assert next(ln for ln in lines if ln["player_id"] == AMEN_THOMPSON)["owner_team_id"] == MY_TEAM
    assert any(ln["owner_team_id"] is None for ln in lines)  # free agents who played
    assert store.get_day(2027, 4) is not None

    # A finished day is served from the store, with no new stats call.
    client.app.state.settled_days.clear()
    assert client.post("/api/scores", json={"league_id": LEAGUE_ID, "season": 2027, "day": 4}).status_code == 200
    assert fake.stat_calls == 1


def test_live_day_is_cached_briefly_and_not_saved(recorded, schedule_raw, day4_players):
    client, fake, store = scores_client(recorded, schedule_raw, day4_players, FRIDAY_NIGHT)
    body = client.post("/api/scores", json={"league_id": LEAGUE_ID, "season": 2027}).json()
    assert body["day"] == 5 and body["live"] is True
    statuses = [g["status"] for g in body["games"]]
    assert statuses.count("live") == 1 and statuses.count("scheduled") == 7
    assert all(g["home"]["score"] is None for g in body["games"] if g["status"] == "scheduled")
    assert store.get_day(2027, 5) is None
    client.post("/api/scores", json={"league_id": LEAGUE_ID, "season": 2027})
    assert fake.stat_calls == 1  # the second visitor within a minute shares the first call


def test_future_day_needs_no_stats_call(recorded, schedule_raw, day4_players):
    client, fake, _ = scores_client(recorded, schedule_raw, day4_players, MORNING_AFTER)
    body = client.post("/api/scores", json={"league_id": LEAGUE_ID, "season": 2027, "day": 5}).json()
    assert {g["status"] for g in body["games"]} == {"scheduled"}
    assert fake.stat_calls == 0


def test_day_without_games(recorded, schedule_raw, day4_players):
    client, _, _ = scores_client(recorded, schedule_raw, day4_players, MORNING_AFTER)
    r = client.post("/api/scores", json={"league_id": LEAGUE_ID, "season": 2027, "day": 2})
    assert r.status_code == 404 and r.json()["detail"]["code"] == "no_games"


def test_week_fixtures_stop_at_sunday(schedule):
    days = {d: len(gs) for d, gs in week_fixtures(schedule, MORNING_AFTER)}
    assert days == {5: 8, 6: 7}  # Sat 24 and Sun 25; Monday 26 starts the next week
    tonight = dict(week_fixtures(schedule, FRIDAY_NIGHT))
    assert len(tonight[5]) == 7  # MIA @ CHA has tipped off


def test_scores_list_this_weeks_fixtures_with_my_players(recorded, schedule_raw, day4_players, league):
    client, _, _ = scores_client(recorded, schedule_raw, day4_players, MORNING_AFTER)
    body = client.post("/api/scores", json={"league_id": LEAGUE_ID, "season": 2027, "team_id": MY_TEAM}).json()
    assert [d["day"] for d in body["week"]] == [5, 6]
    assert body["week"][0]["date"] == "2026-10-24"
    mine = {p["name"] for d in body["week"] for g in d["games"] for p in g["my_players"]}
    roster = {p.name for p in league.team(MY_TEAM).roster}
    assert mine and mine <= roster
    # Each of your players is on one of the two teams, so the other one is his opponent.
    for d in body["week"]:
        for g in d["games"]:
            for p in g["my_players"]:
                assert p["pro_team_id"] in (g["away"]["pro_team_id"], g["home"]["pro_team_id"])
    # Who sits is one of your players with a game that day.
    for d in body["week"]:
        playing = {p["player_id"] for g in d["games"] for p in g["my_players"]}
        assert set(d["sits"]) <= playing
    # Without a team, fixtures still come back, just without names.
    body = client.post("/api/scores", json={"league_id": LEAGUE_ID, "season": 2027}).json()
    assert all(not g["my_players"] for d in body["week"] for g in d["games"])
    assert all(not d["sits"] for d in body["week"])


def test_live_games_get_the_quarter_and_clock(recorded, schedule_raw, day4_players):
    client, _, _ = scores_client(recorded, schedule_raw, day4_players, FRIDAY_NIGHT)
    live = next(
        g
        for g in client.post("/api/scores", json={"league_id": LEAGUE_ID, "season": 2027}).json()["games"]
        if g["status"] == "live"
    )
    assert live["clock"] is None  # scoreboard down: no clock, status from the fantasy data

    asked = []

    def scoreboard(request: httpx.Request) -> httpx.Response:
        asked.append(request.url.params["dates"])
        event = {
            "id": str(live["id"]),
            "status": {"period": 4, "displayClock": "2:31", "type": {"state": "in", "shortDetail": "2:31 - 4th"}},
            "competitions": [{"competitors": []}],
        }
        return httpx.Response(200, json={"events": [event]})

    client, _, _ = scores_client(recorded, schedule_raw, day4_players, FRIDAY_NIGHT, scoreboard)
    body = client.post("/api/scores", json={"league_id": LEAGUE_ID, "season": 2027}).json()
    game = next(g for g in body["games"] if g["id"] == live["id"])
    assert game["status"] == "live" and game["clock"] == "Q4 2:31"
    assert asked == [body["date"].replace("-", "")]


def test_a_game_espn_calls_final_ends_right_away(recorded, schedule_raw, day4_players):
    client, _, _ = scores_client(recorded, schedule_raw, day4_players, FRIDAY_NIGHT)
    live = next(
        g
        for g in client.post("/api/scores", json={"league_id": LEAGUE_ID, "season": 2027}).json()["games"]
        if g["status"] == "live"
    )

    def scoreboard(request: httpx.Request) -> httpx.Response:
        # No id match: found by its two teams instead.
        teams = [{"team": {"abbreviation": live[s]["abbrev"]}} for s in ("away", "home")]
        event = {
            "id": "1",
            "status": {"period": 4, "displayClock": "0.0", "type": {"state": "post", "shortDetail": "Final"}},
            "competitions": [{"competitors": teams}],
        }
        return httpx.Response(200, json={"events": [event]})

    client, _, _ = scores_client(recorded, schedule_raw, day4_players, FRIDAY_NIGHT, scoreboard)
    body = client.post("/api/scores", json={"league_id": LEAGUE_ID, "season": 2027}).json()
    game = next(g for g in body["games"] if g["id"] == live["id"])
    assert game["status"] == "final" and game["clock"] is None
    assert body["live"] is False


def test_games_first_needs_no_stats_call(recorded, schedule_raw, day4_players):
    client, fake, _ = scores_client(recorded, schedule_raw, day4_players, MORNING_AFTER)
    r = client.post("/api/scores/games", json={"league_id": LEAGUE_ID, "season": 2027, "team_id": MY_TEAM})
    assert r.status_code == 200
    body = r.json()
    assert body["day"] == 4 and body["complete"] is False and len(body["games"]) == 12
    assert all(g["status"] == "final" for g in body["games"])
    # Unknown, not zero: the scores come with /api/scores.
    assert all(g["away"]["score"] is None and g["home"]["score"] is None and not g["lines"] for g in body["games"])
    assert {d["day"] for d in body["days"]} >= {4, 5}
    assert any(p for d in body["week"] for g in d["games"] for p in g["my_players"])
    assert all(not d["sits"] for d in body["week"])
    assert fake.stat_calls == 0


def test_games_first_sends_scores_already_in_memory(recorded, schedule_raw, day4_players):
    client, fake, _ = scores_client(recorded, schedule_raw, day4_players, MORNING_AFTER)
    full = client.post("/api/scores", json={"league_id": LEAGUE_ID, "season": 2027}).json()
    body = client.post("/api/scores/games", json={"league_id": LEAGUE_ID, "season": 2027}).json()
    assert body["complete"] is True and body["games"] == full["games"]
    assert fake.stat_calls == 1


def test_games_first_for_a_day_not_started_is_complete(recorded, schedule_raw, day4_players):
    client, fake, _ = scores_client(recorded, schedule_raw, day4_players, MORNING_AFTER)
    body = client.post("/api/scores/games", json={"league_id": LEAGUE_ID, "season": 2027, "day": 5}).json()
    assert body["complete"] is True and {g["status"] for g in body["games"]} == {"scheduled"}
    assert fake.stat_calls == 0
    r = client.post("/api/scores/games", json={"league_id": LEAGUE_ID, "season": 2027, "day": 2})
    assert r.status_code == 404


def test_sits_on_their_own(recorded, schedule_raw, day4_players):
    client, _, _ = scores_client(recorded, schedule_raw, day4_players, MORNING_AFTER)
    full = client.post("/api/scores", json={"league_id": LEAGUE_ID, "season": 2027, "team_id": MY_TEAM}).json()
    sits = client.post("/api/scores/sits", json={"league_id": LEAGUE_ID, "season": 2027, "team_id": MY_TEAM}).json()
    assert [(d["day"], d["sits"]) for d in sits["days"]] == [(d["day"], d["sits"]) for d in full["week"]]
    quick = client.post(
        "/api/scores", json={"league_id": LEAGUE_ID, "season": 2027, "team_id": MY_TEAM, "with_sits": False}
    ).json()
    assert all(not d["sits"] for d in quick["week"])
    assert [d["day"] for d in quick["week"]] == [d["day"] for d in full["week"]]

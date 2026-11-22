import asyncio
import gzip
import json
import time
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pytest

from app.analysis import gamebox, gamescore
from app.espn.client import EspnClient
from app.lineup import Spot, next_game, open_slots, season_average, slot_name, team_days, week_of
from app.main import create_app
from app.scores import build_day, game_status, parse_lines, parse_schedule, settled, week_fixtures
from app.store import MemoryStore
from tests.conftest import LEAGUE_ID, MY_TEAM, FakeEspn, signed_in

FIXTURES = Path(__file__).parent / "fixtures"
# Day 4 is Thu 23 Oct 2026 (12 games, all final); day 5 is Fri 24 Oct (8 games).
MORNING_AFTER = datetime(2026, 10, 24, 8, 0, tzinfo=UTC)
FRIDAY_NIGHT = datetime(2026, 10, 24, 23, 0, tzinfo=UTC)  # MIA @ CHA tipped off at 22:00 UTC
FRIDAY_LATE_MORNING = datetime(2026, 10, 24, 15, 30, tzinfo=UTC)  # 11:30 in New York
FRIDAY_AFTERNOON = datetime(2026, 10, 24, 18, 0, tzinfo=UTC)  # 14:00 in New York, before the first tip-off
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


def test_scores_open_on_last_night_until_noon_then_tonight(schedule):
    assert schedule.latest_started(FRIDAY_AFTERNOON) == 4  # nothing has tipped off yet tonight
    assert schedule.default_day(MORNING_AFTER) == 4
    assert schedule.default_day(FRIDAY_LATE_MORNING) == 4
    assert schedule.default_day(FRIDAY_AFTERNOON) == 5
    assert schedule.default_day(FRIDAY_NIGHT) == 5


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


def test_scores_endpoint_opens_on_tonights_slate_in_the_afternoon(recorded, schedule_raw, day4_players):
    client, _, _ = scores_client(recorded, schedule_raw, day4_players, FRIDAY_AFTERNOON)
    body = client.post("/api/scores", json={"league_id": LEAGUE_ID, "season": 2027}).json()
    assert body["day"] == 5 and body["date"] == "2026-10-24" and body["live"] is False
    assert len(body["games"]) == 8 and all(g["status"] == "scheduled" for g in body["games"])


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
        if request.url.path.endswith("/summary"):
            return httpx.Response(404)
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


def test_live_games_ask_espn_who_is_on_the_floor(recorded, schedule_raw, day4_players):
    client, _, _ = scores_client(recorded, schedule_raw, day4_players, FRIDAY_NIGHT)
    live = next(
        g
        for g in client.post("/api/scores", json={"league_id": LEAGUE_ID, "season": 2027}).json()["games"]
        if g["status"] == "live"
    )
    boxes = []

    def espn_site(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/summary"):
            boxes.append(request.url.params["event"])
            teams = [
                {"statistics": [{"athletes": [{"athlete": {"id": str(b + i)}, "active": i < 5} for i in range(9)]}]}
                for b in (9000, 9100)
            ]
            return httpx.Response(200, json={"boxscore": {"players": teams}})
        event = {
            "id": str(live["id"]),
            "status": {"period": 2, "displayClock": "5:10", "type": {"state": "in", "shortDetail": "5:10 - 2nd"}},
            "competitions": [{"competitors": []}],
        }
        return httpx.Response(200, json={"events": [event]})

    client, _, _ = scores_client(recorded, schedule_raw, day4_players, FRIDAY_NIGHT, espn_site)
    for _ in range(2):
        assert client.post("/api/scores", json={"league_id": LEAGUE_ID, "season": 2027}).status_code == 200
    assert boxes == [str(live["id"])]  # only the game in progress, and once while the scores are fresh


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


def _with_starters(players: list[dict], day: int, starters: set[int]) -> list[dict]:
    """The recorded day trimmed the games-started stat (41); put it back for the given players."""
    out = json.loads(json.dumps(players))
    for entry in out:
        p = entry.get("player") or entry
        for st in p.get("stats") or []:
            if st.get("scoringPeriodId") == day and st.get("statSplitTypeId") == 5 and st.get("stats"):
                st["stats"]["41"] = 1.0 if int(p["id"]) in starters else 0.0
    return out


def test_lines_say_who_started(day4_players):
    players = _with_starters(day4_players, 4, {AMEN_THOMPSON})
    lines = [ln for game in parse_lines(players, 4).values() for ln in game]
    assert [ln.player_id for ln in lines if ln.starter] == [AMEN_THOMPSON]
    # A line without the stat (a day saved before) is no starter.
    assert not any(ln.starter for game in parse_lines(day4_players, 4).values() for ln in game)


def test_a_day_saved_before_starters_is_fetched_once_more(recorded, schedule_raw, day4_players):
    players = _with_starters(day4_players, 4, {AMEN_THOMPSON})
    client, fake, store = scores_client(recorded, schedule_raw, players, MORNING_AFTER)
    old = build_day(parse_schedule(schedule_raw), 4, day4_players, MORNING_AFTER)
    for g in old["games"]:
        for ln in g["lines"]:
            del ln["starter"]
    store.save_day(2027, 4, old)

    body = client.post("/api/scores", json={"league_id": LEAGUE_ID, "season": 2027, "day": 4}).json()
    lines = [ln for g in body["games"] for ln in g["lines"]]
    assert next(ln for ln in lines if ln["player_id"] == AMEN_THOMPSON)["starter"] is True
    assert fake.stat_calls == 1
    assert all("starter" in ln for g in store.get_day(2027, 4)["games"] for ln in g["lines"])

    # Saved again with starters, so the next read stays in the store.
    client.app.state.settled_days.clear()
    client.post("/api/scores", json={"league_id": LEAGUE_ID, "season": 2027, "day": 4})
    assert fake.stat_calls == 1


def _lineup_client(recorded, schedule_raw, day4_players, now):
    client, fake, _ = scores_client(recorded, schedule_raw, day4_players, now)
    return client, fake


def _lineup_calls(fake) -> list[str]:
    return [u for u in fake.seen_urls if "forTeamId=" in u]


def test_lineup_reads_one_teams_slots_for_the_day(recorded, schedule_raw, day4_players):
    client, fake = _lineup_client(recorded, schedule_raw, day4_players, MORNING_AFTER)
    r = client.post("/api/scores/lineup", json={"league_id": LEAGUE_ID, "season": 2027, "team_id": MY_TEAM, "day": 4})
    assert r.status_code == 200
    body = r.json()
    assert body["day"] == 4 and body["date"] == "2026-10-23" and body["team_id"] == MY_TEAM
    calls = _lineup_calls(fake)
    assert len(calls) == 1 and f"forTeamId={MY_TEAM}" in calls[0] and "scoringPeriodId=4" in calls[0]
    team = next(t for t in recorded["league"]["teams"] if t["id"] == MY_TEAM)
    assert {s["player_id"] for s in body["spots"]} == {
        e["playerPoolEntry"]["player"]["id"] for e in team["roster"]["entries"]
    }
    # ESPN's slot order: starting slots first, then the bench, then IR.
    ids = [s["slot_id"] for s in body["spots"]]
    assert ids == sorted(ids)
    assert all(s["starting"] == (s["slot"] not in ("BE", "IR")) for s in body["spots"])
    # Each player's game that day, by his NBA team.
    sched = parse_schedule(schedule_raw)
    playing = {t for g in sched.games[4] for t in (g.away, g.home)}
    assert all((s["game_id"] is not None) == (s["pro_team_id"] in playing) for s in body["spots"])
    assert any(s["game_id"] for s in body["spots"])
    # Slots each can fill, among the league's starting slots: UTIL fits everyone, never BE or IR.
    assert all("UTIL" in s["eligible"] and not {"BE", "IR"} & set(s["eligible"]) for s in body["spots"])
    # The week the day falls in (opening night Tue 20 to Sun 25 Oct), each player's game days and his next game.
    assert [w["date"] for w in body["week"]] == [f"2026-10-{d}" for d in range(20, 26)]
    week = [w["day"] for w in body["week"]]
    for s in body["spots"]:
        assert s["week_games"] == [
            d for d in week if any(s["pro_team_id"] in (g.away, g.home) for g in sched.games.get(d, []))
        ]
        nxt = s["next_game"]
        assert nxt is not None and nxt["day"] > 4 and nxt["opponent"]["pro_team_id"] != s["pro_team_id"]


def test_lineup_follows_espn_within_its_cache(recorded, schedule_raw, day4_players):
    client, fake = _lineup_client(recorded, schedule_raw, day4_players, FRIDAY_NIGHT)
    req = {"league_id": LEAGUE_ID, "season": 2027, "team_id": MY_TEAM}
    first = client.post("/api/scores/lineup", json=req).json()
    assert first["day"] == 5
    client.post("/api/scores/lineup", json=req)
    assert len(_lineup_calls(fake)) == 1  # a refresh within half a minute shares the read
    client.app.state.lineups.clear()
    client.post("/api/scores/lineup", json=req)
    assert len(_lineup_calls(fake)) == 2  # then ESPN is asked again, so a change there shows


def test_finished_days_lineup_is_kept(recorded, schedule_raw, day4_players):
    client, fake = _lineup_client(recorded, schedule_raw, day4_players, MORNING_AFTER)
    req = {"league_id": LEAGUE_ID, "season": 2027, "team_id": MY_TEAM, "day": 4}
    client.post("/api/scores/lineup", json=req)
    client.app.state.lineups.clear()
    client.post("/api/scores/lineup", json=req)
    assert len(_lineup_calls(fake)) == 1


def test_lineup_for_an_unknown_team(recorded, schedule_raw, day4_players):
    client, _ = _lineup_client(recorded, schedule_raw, day4_players, MORNING_AFTER)
    r = client.post("/api/scores/lineup", json={"league_id": LEAGUE_ID, "season": 2027, "team_id": 99, "day": 4})
    assert r.status_code == 404


def test_open_starting_slots():
    spots = [
        Spot(player_id=1, name="A", pro_team_id=1, slot_id=0, injury_status="ACTIVE", game_id=None),
        Spot(player_id=2, name="B", pro_team_id=1, slot_id=11, injury_status="ACTIVE", game_id=None),
        Spot(player_id=3, name="C", pro_team_id=1, slot_id=12, injury_status="ACTIVE", game_id=None),
    ]
    assert open_slots(spots, {0: 1, 4: 1, 11: 3}) == [4, 11, 11]
    assert [slot_name(s) for s in (0, 11, 12, 13, 99)] == ["PG", "UTIL", "BE", "IR", "Slot 99"]


def test_season_average_is_his_real_line_this_season():
    player = {
        "stats": [
            {"seasonId": 2027, "statSourceId": 1, "statSplitTypeId": 0, "averageStats": {"0": 30.0, "42": 82}},
            {"seasonId": 2026, "statSourceId": 0, "statSplitTypeId": 0, "averageStats": {"0": 20.0, "42": 70}},
            {"seasonId": 2027, "statSourceId": 0, "statSplitTypeId": 1, "averageStats": {"0": 40.0, "42": 1}},
            {
                "seasonId": 2027,
                "statSourceId": 0,
                "statSplitTypeId": 0,
                "averageStats": {"0": 27.111, "6": 6.0, "13": 8.22, "14": 19.56, "40": 33.78, "42": 9},
            },
        ]
    }
    avg = season_average(player, 2027)
    assert avg is not None
    assert (avg["gp"], avg["pts"], avg["reb"], avg["fgm"], avg["fga"], avg["min"], avg["ast"]) == (
        9,
        27.11,
        6.0,
        8.22,
        19.56,
        33.78,
        0.0,
    )
    # No games yet, or no line this season: nothing rather than a projection.
    none_yet = {"stats": [{"seasonId": 2027, "statSourceId": 0, "statSplitTypeId": 0, "averageStats": {"42": 0}}]}
    assert season_average(none_yet, 2027) is None
    assert season_average({"stats": player["stats"][:2]}, 2027) is None


def test_next_game_and_week_days(schedule_raw):
    sched = parse_schedule(schedule_raw)
    hou_days = sorted(d for d, gs in sched.games.items() if any(HOU in (g.away, g.home) for g in gs))
    assert hou_days == [4, 5, 7]  # the recorded first week
    g = next_game(sched, HOU, 5)
    assert g is not None and g.day == 7 and HOU in (g.away, g.home)
    assert next_game(sched, HOU, 7) is None  # the record ends there
    # Day 4 is Fri 23 Oct: its week runs from opening night (Tue 20, day 1) to Sun 25 Oct (day 6).
    week = week_of(sched, 4)
    assert week == [1, 2, 3, 4, 5, 6]
    assert team_days(sched, HOU, week) == [4, 5]
    assert week_of(sched, 7) == list(range(7, 14))  # Mon 26 Oct to Sun 1 Nov


def test_nba_team_endpoint(recorded, schedule_raw, day4_players):
    client, fake, store = scores_client(recorded, schedule_raw, day4_players, MORNING_AFTER)
    r = client.post("/api/nba-team", json={"league_id": LEAGUE_ID, "season": 2027, "team": "hou"})
    assert r.status_code == 200
    body = r.json()
    assert body["abbrev"] == "HOU" and body["name"].endswith("Rockets")
    # The fixture has player lines for day 4 only: that's the one game with a score.
    assert body["wins"] + body["losses"] == len(body["latest"]) == 1
    game = body["latest"][0]
    assert game["date"] == "2026-10-23" and game["won"] == (game["points"] > game["opponent_points"])
    assert game["top"]["pts"] > 0
    assert body["next_game"] is not None and body["next_game"]["start"] > "2026-10-24"
    assert store.get_day(2027, 4) is not None

    calls = fake.stat_calls
    client.app.state.settled_days.clear()
    assert client.post("/api/nba-team", json={"league_id": LEAGUE_ID, "season": 2027, "team": "HOU"}).json() == body
    assert fake.stat_calls == calls  # finished days come from the store

    r = client.post("/api/nba-team", json={"league_id": LEAGUE_ID, "season": 2027, "team": "XYZ"})
    assert r.status_code == 404 and r.json()["detail"]["code"] == "no_team"


def test_match_form_endpoint_tags_kinds_and_reads_through_the_public_league(recorded, schedule_raw, day4_players):
    sched = parse_schedule(schedule_raw)
    game = next(g for g in sched.games[4] if HOU in (g.away, g.home))
    other = sched.teams[game.home if game.away == HOU else game.away][0]
    asked = []

    def scoreboard(request: httpx.Request) -> httpx.Response:
        asked.append(request.url.params.get("dates"))
        cup = {"notes": [{"headline": "NBA Cup - Group Play"}]}
        return httpx.Response(
            200, json={"events": [{"id": str(game.id), "season": {"type": 2}, "competitions": [cup]}]}
        )

    client, _, _ = scores_client(recorded, schedule_raw, day4_players, MORNING_AFTER, scoreboard)
    body = {"season": 2027, "away": "hou", "home": other}
    assert client.post("/api/match-form", json=body).status_code == 503  # no public league to read with
    client.app.state.sample_league_id = LEAGUE_ID
    r = client.post("/api/match-form", json=body)
    assert r.status_code == 200
    hou, opp = r.json()["away"], r.json()["home"]
    assert hou["abbrev"] == "HOU" and opp["abbrev"] == other
    assert len(hou["games"]) == len(opp["games"]) == 1 and hou["wins"] + opp["wins"] == 1
    g = hou["games"][0]
    assert (g["game_id"], g["kind"], g["note"], g["neutral"]) == (game.id, "cup", "NBA Cup - Group Play", False)
    assert g["won"] == (g["points"] > g["opponent_points"]) and g["opponent"] == other
    winner = "HOU" if g["won"] else other
    assert g["mvp"]["team"] == winner and g["best"]["team"] == "HOU" and g["mvp"]["minutes"] > 0
    assert opp["games"][0]["mvp"] == g["mvp"]
    assert "20261023" in asked and len(asked) == len(set(asked))  # one look per day either team played
    seen = list(asked)
    client.post("/api/match-form", json=body)
    assert asked == seen  # a finished day's kinds are kept

    r = client.post("/api/match-form", json={"season": 2027, "away": "HOU", "home": "XYZ"})
    assert r.status_code == 404 and r.json()["detail"]["code"] == "no_team"


def test_match_form_without_the_scoreboard_has_no_kinds(recorded, schedule_raw, day4_players):
    client, _, _ = scores_client(recorded, schedule_raw, day4_players, MORNING_AFTER)
    client.app.state.sample_league_id = LEAGUE_ID
    sched = parse_schedule(schedule_raw)
    game = next(g for g in sched.games[4] if HOU in (g.away, g.home))
    other = sched.teams[game.home if game.away == HOU else game.away][0]
    r = client.post("/api/match-form", json={"season": 2027, "away": other, "home": "HOU"})
    assert r.status_code == 200
    g = r.json()["home"]["games"][0]
    assert g["kind"] is None and g["note"] is None


def test_match_form_names_their_next_meeting_with_its_kind(recorded, schedule_raw, day4_players):
    sched = parse_schedule(schedule_raw)
    game = next(g for g in sched.games[5] if HOU in (g.away, g.home))
    away, home = sched.teams[game.away][0], sched.teams[game.home][0]

    def scoreboard(request: httpx.Request) -> httpx.Response:
        event = {"id": str(game.id), "season": {"type": 2}, "competitions": [{"neutralSite": True}]}
        return httpx.Response(200, json={"events": [event]})

    client, _, _ = scores_client(recorded, schedule_raw, day4_players, MORNING_AFTER, scoreboard)
    client.app.state.sample_league_id = LEAGUE_ID
    nxt = client.post("/api/match-form", json={"season": 2027, "away": away, "home": home}).json()["next"]
    assert nxt["game_id"] == game.id and nxt["kind"] == "regular" and nxt["neutral"] is True
    # The other way round is the return game, a different one.
    back = client.post("/api/match-form", json={"season": 2027, "away": home, "home": away}).json()["next"]
    assert back is None or (back["game_id"] != game.id and back["start"] > nxt["start"])


def test_nba_team_before_tonights_games(recorded, schedule_raw, day4_players):
    client, _, _ = scores_client(recorded, schedule_raw, day4_players, FRIDAY_AFTERNOON)
    body = client.post("/api/nba-team", json={"league_id": LEAGUE_ID, "season": 2027, "team": "SA"}).json()
    assert body["next_game"]["start"] > FRIDAY_AFTERNOON.isoformat()[:10]


def test_scores_carry_our_call_once_the_league_box_is_built(recorded, schedule_raw, day4_players):
    client, _, _ = scores_client(recorded, schedule_raw, day4_players, MORNING_AFTER)
    body = client.post("/api/scores", json={"league_id": LEAGUE_ID, "season": 2027}).json()
    assert all(g["call"] is None for g in body["games"])  # no box yet
    client.app.state.game_boxes.set(2027, gamebox.Box(2027, {}, {}, {}, scores=gamescore.build([])))
    client.app.state.settled_days.clear()
    body = client.post("/api/scores", json={"league_id": LEAGUE_ID, "season": 2027}).json()
    for g in body["games"]:
        c = g["call"]
        assert c["away"] != c["home"] and 0.5 <= c["chance"] <= 1
        assert c["home"] > c["away"]  # even teams: home court


def _box(played: dict[int, list[int]] | None = None) -> gamebox.Box:
    return gamebox.Box(2027, {}, {}, {}, played=played or {}, scores=gamescore.build([]))


def test_our_call_stays_on_scores_after_the_box_expires(recorded, schedule_raw, day4_players):
    client, _, _ = scores_client(recorded, schedule_raw, day4_players, MORNING_AFTER)
    client.app.state.last_boxes[2027] = _box()  # built earlier; gone from game_boxes after GAME_BOX_TTL
    body = client.post("/api/scores", json={"league_id": LEAGUE_ID, "season": 2027}).json()
    assert body["games"] and all(g["call"] is not None for g in body["games"])
    quick = client.post("/api/scores/games", json={"league_id": LEAGUE_ID, "season": 2027, "day": 5}).json()
    assert quick["games"] and all(g["call"] is not None for g in quick["games"])


def test_box_watch_builds_the_box_when_there_is_none_or_a_finished_night_is_missing(recorded, schedule_raw):
    fake = FakeEspn(recorded, schedule=schedule_raw)
    app = create_app(EspnClient(transport=httpx.MockTransport(fake.handler)), store=MemoryStore())
    app.state.now = lambda: MORNING_AFTER  # day 4 is over, day 5 not started
    asyncio.run(app.state.box_sweep())
    assert 2027 in app.state.box_tried  # no box: a build was started (ESPN's logs aren't in the fake)

    app.state.box_tried.clear()
    app.state.game_boxes.set(2027, _box({HOU: [1, 4], SA: [2, 4]}))
    assert asyncio.run(app.state.box_sweep()) is False
    assert 2027 not in app.state.box_tried  # last night is in the box: nothing to do

    app.state.game_boxes.set(2027, _box({HOU: [1, 3], SA: [2, 3]}))
    asyncio.run(app.state.box_sweep())
    assert 2027 in app.state.box_tried  # last night is missing: built again
    tried = app.state.box_tried[2027] = time.monotonic()
    calls = fake.calls
    assert asyncio.run(app.state.box_sweep()) is False
    assert app.state.box_tried[2027] == tried and fake.calls == calls  # tried within the hour: wait

import copy
import gzip
import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from app import playchance
from app.playchance import (
    AVAILABLE,
    DAY_TO_DAY,
    DOUBTFUL,
    OUT,
    PROBABLE,
    QUESTIONABLE,
    FeedNote,
    comeback_chance,
    estimate,
    parse_feed,
    play_chance,
    tag_of,
)
from app.scores import parse_schedule
from tests.conftest import LEAGUE_ID, MY_TEAM
from tests.test_scores import AMEN_THOMPSON, FIXTURES, scores_client

# Wednesday 18 Nov 2026, 7 pm in New York; the team's last game was Monday night.
TIP = datetime(2026, 11, 19, 0, 0, tzinfo=UTC)
LAST_TIP = datetime(2026, 11, 17, 0, 30, tzinfo=UTC)
MORNING = TIP - timedelta(hours=8, minutes=30)
DAY5 = {"league_id": LEAGUE_ID, "season": 2027, "team_id": MY_TEAM, "day": 5}


@pytest.fixture(scope="module")
def schedule_raw() -> dict:
    with gzip.open(FIXTURES / "scores_schedule_2027.json.gz", "rt") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def day4_players() -> list[dict]:
    with gzip.open(FIXTURES / "day_stats_2027_day4.json.gz", "rt") as fh:
        return json.load(fh)["players"]


def note(comment: str, at: datetime = datetime(2026, 11, 17, 20, 49, tzinfo=UTC), status="Day-To-Day") -> FeedNote:
    return FeedNote(player_id=1, status=status, comment=comment, at=at, return_date=None)


def test_the_tag_comes_from_the_comments_words():
    assert tag_of("Thompson (knee) is questionable for Wednesday's game against the Pacers.") == QUESTIONABLE
    assert tag_of("Aldama (ankle) is probable for Wednesday's game against the Warriors.") == PROBABLE
    assert tag_of("Lawal is doubtful for Wednesday's game due to tightness in his right groin.") == DOUBTFUL
    assert tag_of("Prince (knee) has been ruled out for Tuesday's game against the Spurs.") == OUT
    assert tag_of("Phillips is available for Wednesday's game against Dallas.") == AVAILABLE
    assert tag_of("Coach JJ Redick said he expects Walker (leg) to be available for Wednesday's game.") == AVAILABLE
    assert tag_of("Green will be a game-time decision Friday.") == QUESTIONABLE
    # The last tag named is the news: an upgrade names the old one first.
    assert tag_of("Smith has been upgraded from questionable to probable for Friday.") == PROBABLE
    assert tag_of("Jones is recovering from a sprained ankle.") is None


def test_a_comment_counts_only_for_the_game_it_is_about():
    # Named weekday: the game's day in New York, whenever ESPN dated it.
    assert note("is questionable for Wednesday's game").about(TIP, LAST_TIP)
    assert note("is available for Wednesday's game", at=LAST_TIP - timedelta(hours=10)).about(TIP, LAST_TIP)
    # ESPN re-dates old comments: Tuesday's news stamped Wednesday morning is not about Wednesday.
    assert not note("has been ruled out for Tuesday's game", at=MORNING).about(TIP, LAST_TIP)
    # No weekday: only news written after his team's last game.
    assert note("is questionable tonight", at=MORNING).about(TIP, LAST_TIP)
    assert not note("is questionable tonight", at=LAST_TIP - timedelta(hours=2)).about(TIP, LAST_TIP)
    # Never news from after the tip.
    assert not note("is questionable for Wednesday", at=TIP + timedelta(minutes=5)).about(TIP, LAST_TIP)


def test_questionable_amen_eight_hours_out():
    got = estimate("DAY_TO_DAY", note("is questionable for Wednesday's game"), TIP, LAST_TIP, MORNING, 33.5)
    assert got is not None and got.tag == QUESTIONABLE and got.chance == 0.56
    assert "questionable" in got.basis and "56%" in got.basis


def test_the_chance_moves_with_the_hours_and_the_role():
    # A starter still questionable half an hour out has usually been told he's playing.
    assert play_chance(QUESTIONABLE, 0.6, 33) > play_chance(QUESTIONABLE, 8, 33)
    # A bench player listed questionable sits more often.
    assert play_chance(QUESTIONABLE, 8, 10) < play_chance(QUESTIONABLE, 8, 20) < play_chance(QUESTIONABLE, 8, 30)
    assert play_chance(DOUBTFUL, 3, 30) < 0.1 < play_chance(PROBABLE, 3, 30)
    # Unknown minutes: the middle role.
    assert play_chance(QUESTIONABLE, 8, None) == play_chance(QUESTIONABLE, 8, 20)
    # Every number is a share.
    assert all(0 <= p <= 1 for rows in playchance.PLAY.values() for row in rows for p in row)


def test_day_to_day_with_no_word_and_no_tag():
    stale = note("has been ruled out for Tuesday's game", at=MORNING)
    got = estimate("DAY_TO_DAY", stale, TIP, LAST_TIP, MORNING, 30)
    assert got is not None and got.tag == DAY_TO_DAY and got.chance == 0.62
    assert estimate("DAY_TO_DAY", None, TIP, LAST_TIP, MORNING, 30).chance == 0.62
    assert estimate("ACTIVE", None, TIP, LAST_TIP, MORNING, 30) is None
    # Once the game has started there's nothing to estimate.
    assert estimate("DAY_TO_DAY", None, TIP, LAST_TIP, TIP, 30) is None


def test_out_players_coming_back():
    # Upgraded: ESPN still says out, the news names a tag for the game.
    up = estimate("OUT", note("is questionable for Wednesday's game", status="Out"), TIP, LAST_TIP, MORNING, 30, 4)
    assert up is not None and up.upgraded and up.tag == QUESTIONABLE and up.missed_in_a_row == 4
    # Ruled out for this game.
    gone = estimate("OUT", note("has been ruled out for Wednesday's game", status="Out"), TIP, LAST_TIP, MORNING, 30, 4)
    assert gone is not None and gone.chance == 0 and gone.tag == OUT
    # No word yet: the games missed in a row decide.
    one = estimate("OUT", None, TIP, LAST_TIP, MORNING, 30, 1)
    many = estimate("OUT", None, TIP, LAST_TIP, MORNING, 30, 9)
    assert one is not None and many is not None and one.chance == 0.47 and many.chance == 0.10
    assert "1 game missed" in one.basis and "9 games missed" in many.basis
    # Just hurt in his last game, or no game log: nothing to go on.
    assert estimate("OUT", None, TIP, LAST_TIP, MORNING, 30, 0) is None
    assert estimate("OUT", None, TIP, LAST_TIP, MORNING, 30, None) is None
    assert [comeback_chance(k) for k in (1, 2, 3, 4, 6, 7, 30)] == [0.47, 0.2, 0.2, 0.14, 0.14, 0.1, 0.1]


FEED = {
    "injuries": [
        {
            "injuries": [
                {
                    "status": "Day-To-Day",
                    "date": "2026-11-17T20:49Z",
                    "shortComment": "Thompson (knee) is questionable for Wednesday's game against the Pacers.",
                    "athlete": {
                        "displayName": "Amen Thompson",
                        "links": [{"href": "https://www.espn.com/nba/player/_/id/4684740/amen-thompson"}],
                    },
                    "details": {"returnDate": "2026-11-18"},
                },
                {"status": "Out", "date": "bad", "shortComment": "", "athlete": {"id": "6475"}},
                {"status": "Out", "athlete": {"links": []}},
            ]
        }
    ]
}


def test_parse_feed():
    got = parse_feed(FEED)
    assert set(got) == {4684740, 6475}
    amen = got[4684740]
    assert amen.status == "Day-To-Day" and amen.tag() == QUESTIONABLE and amen.return_date == "2026-11-18"
    assert amen.at == datetime(2026, 11, 17, 20, 49, tzinfo=UTC)
    assert got[6475].at is None and got[6475].tag() is None


def _amen_day_to_day(recorded, schedule_raw):
    """The recorded league with Amen Thompson tagged day-to-day, and his game on day 5."""
    league = copy.deepcopy(recorded)
    player = next(
        e["playerPoolEntry"]["player"]
        for t in league["league"]["teams"]
        for e in t["roster"]["entries"]
        if e["playerPoolEntry"]["player"]["id"] == AMEN_THOMPSON
    )
    player["injuryStatus"] = "DAY_TO_DAY"
    sched = parse_schedule(schedule_raw)
    return league, next(g for g in sched.games[5] if player["proTeamId"] in (g.away, g.home))


def test_lineup_shows_the_chance_for_a_tagged_player(recorded, schedule_raw, day4_players):
    league, game = _amen_day_to_day(recorded, schedule_raw)
    now = game.start - timedelta(hours=6)
    weekday = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")[
        game.start.astimezone(playchance.NEW_YORK).weekday()
    ]
    feed = copy.deepcopy(FEED)
    item = feed["injuries"][0]["injuries"][0]
    item["shortComment"] = f"Thompson (knee) is probable for {weekday}'s game."
    item["date"] = (now - timedelta(hours=1)).isoformat().replace("+00:00", "Z")

    def site(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/injuries"):
            return httpx.Response(200, json=feed)
        return httpx.Response(503)

    client, _, _ = scores_client(league, schedule_raw, day4_players, now, scoreboard=site)
    body = client.post("/api/scores/lineup", json=DAY5)
    assert body.status_code == 200
    spots = {s["player_id"]: s for s in body.json()["spots"]}
    play = spots[AMEN_THOMPSON]["play"]
    assert play["tag"] == PROBABLE and 0.7 <= play["chance"] <= 1
    assert play["news"].startswith("Thompson (knee) is probable") and play["return_date"] == "2026-11-18"
    # Healthy players get no chance.
    assert all(s["play"] is None for s in spots.values() if s["injury_status"] == "ACTIVE")
    # Once his game tips off the chance goes away.
    client.app.state.now = lambda: game.start + timedelta(minutes=1)
    client.app.state.lineups.clear()
    after = {s["player_id"]: s for s in client.post("/api/scores/lineup", json=DAY5).json()["spots"]}
    assert after[AMEN_THOMPSON]["play"] is None


def test_lineup_without_the_feed_falls_back_to_the_tag(recorded, schedule_raw, day4_players):
    league, game = _amen_day_to_day(recorded, schedule_raw)
    client, _, _ = scores_client(league, schedule_raw, day4_players, game.start - timedelta(hours=10))
    spots = client.post("/api/scores/lineup", json=DAY5).json()["spots"]
    play = next(s for s in spots if s["player_id"] == AMEN_THOMPSON)["play"]
    assert play["tag"] == DAY_TO_DAY and play["news"] is None and 0.4 < play["chance"] < 0.8

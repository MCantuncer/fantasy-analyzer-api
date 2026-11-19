import asyncio
import gzip
import json
import os

import httpx
import pytest

from app import benched
from app.espn.client import EspnClient
from app.main import create_app
from app.scores import parse_schedule
from app.store import MemoryStore, PostgresStore
from tests.conftest import FakeEspn
from tests.test_scores import FIXTURES, MORNING_AFTER


def athlete(pid: int, reason: str | None = None) -> dict:
    """One box-score row, shaped like ESPN's: a player who sat has no stats and a reason."""
    sat = reason is not None
    return {"athlete": {"id": str(pid)}, "didNotPlay": sat, "reason": reason or "", "stats": [] if sat else ["30"]}


def summary(away: list[dict], home: list[dict]) -> dict:
    return {"boxscore": {"players": [{"statistics": [{"athletes": side}]} for side in (away, home)]}}


GAME = summary(
    [athlete(1), athlete(2, "COACH'S DECISION"), athlete(3, "Right Knee Contusion")],
    [athlete(4), athlete(5, "PERSONAL"), athlete(6, "DNP - Coach's Decision")],
)


def test_a_box_score_gives_everyone_who_sat_and_why():
    assert benched.parse_sat(GAME) == {
        2: "COACH'S DECISION",
        3: "RIGHT KNEE CONTUSION",
        5: "PERSONAL",
        6: "DNP - COACH'S DECISION",
    }
    assert benched.parse_sat(summary([athlete(1)], [athlete(4)])) == {}  # everyone played
    # A game not played yet, or half a box score, says nothing.
    assert benched.parse_sat({}) is None
    assert benched.parse_sat({"boxscore": {"players": [{"statistics": [{"athletes": [athlete(1)]}]}]}}) is None
    assert benched.parse_sat(summary([athlete(1)], [])) is None


def test_coach_days_keeps_only_the_coachs_calls():
    nights = {7: {"2": "COACH'S DECISION", "3": "RIGHT KNEE CONTUSION"}, 5: {"2": "COACH'S DECISION"}, 6: {"6": "REST"}}
    assert benched.coach_days(nights) == {2: [5, 7]}


def site(games: dict[int, dict | None]):
    """ESPN's summary endpoint for these games; None answers 503."""
    asked: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        gid = int(request.url.params["event"])
        asked.append(gid)
        body = games.get(gid)
        return httpx.Response(200, json=body) if body is not None else httpx.Response(503)

    return httpx.MockTransport(handler), asked


def test_a_night_counts_only_when_every_game_answers():
    transport, _ = site({10: GAME, 11: summary([athlete(7, "COACH'S DECISION")], [athlete(8)])})
    night = asyncio.run(benched.fetch_night([10, 11], transport))
    assert night[2] == "COACH'S DECISION" and night[7] == "COACH'S DECISION" and 8 not in night
    transport, _ = site({10: GAME, 11: None})
    assert asyncio.run(benched.fetch_night([10, 11], transport)) is None


def test_the_watch_reads_each_finished_night_once(recorded):
    with gzip.open(FIXTURES / "scores_schedule_2027.json.gz", "rt") as fh:
        raw = json.load(fh)
    sched = parse_schedule(raw)
    done = sorted(d for d in sched.games if d <= 4)  # MORNING_AFTER: day 4 is over, day 5 not started
    ids = {d: [g.id for g in sched.games[d]] for d in done}
    answers = {gid: summary([athlete(1)], [athlete(4)]) for gids in ids.values() for gid in gids}
    answers[ids[4][0]] = GAME
    broken = ids[3][0]
    answers[broken] = None  # ESPN can't give one game of day 3 yet
    transport, asked = site(answers)
    store = MemoryStore()
    app = create_app(
        EspnClient(transport=httpx.MockTransport(FakeEspn(recorded, schedule=raw).handler)),
        store=store,
        clock_transport=transport,
    )
    app.state.now = lambda: MORNING_AFTER

    wrote = asyncio.run(app.state.sat_sweep())
    assert wrote == [d for d in done if d != 3]
    nights = store.sat_out(2027, 0)
    assert nights[4]["2"] == "COACH'S DECISION" and 3 not in nights
    assert benched.coach_days(nights) == {2: [4], 6: [4]}

    # The next look asks only for the night still missing, and saves it once ESPN has it.
    asked.clear()
    answers[broken] = summary([athlete(1)], [athlete(9, "COACH'S DECISION")])
    assert asyncio.run(app.state.sat_sweep()) == [3]
    assert sorted(asked) == sorted(ids[3])
    assert benched.coach_days(store.sat_out(2027, 0)) == {2: [4], 6: [4], 9: [3]}
    asked.clear()
    assert asyncio.run(app.state.sat_sweep()) == [] and asked == []


@pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="set TEST_DATABASE_URL to test against Postgres")
def test_postgres_keeps_sat_out_nights():
    store = PostgresStore(os.environ["TEST_DATABASE_URL"])
    with store._connect() as conn:
        conn.execute("delete from sat_out where season = 1999")
    store.save_sat_out(1999, 4, {"2": "COACH'S DECISION"})
    store.save_sat_out(1999, 4, {"2": "REST"})  # a saved night is final
    store.save_sat_out(1999, 9, {})
    assert store.sat_out(1999, 0) == {4: {"2": "COACH'S DECISION"}, 9: {}}
    assert store.sat_out(1999, 5) == {9: {}}
    with store._connect() as conn:
        conn.execute("delete from sat_out where season = 1999")


class NoGamesEspn(FakeEspn):
    """A season with a schedule and nobody's game lines yet: enough to build the league box."""

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/players"):
            return httpx.Response(200, json={"players": []})
        return super().handler(request)


def test_the_league_box_carries_the_saved_nights(recorded):
    with gzip.open(FIXTURES / "scores_schedule_2027.json.gz", "rt") as fh:
        raw = json.load(fh)
    store = MemoryStore()
    store.save_sat_out(2027, 4, {"2": "COACH'S DECISION", "3": "RIGHT KNEE CONTUSION"})
    store.save_sat_out(2027, 2, {"2": "COACH'S DECISION"})
    store.save_sat_out(2026, 150, {"7": "COACH'S DECISION"})  # last season's nights stay out
    app = create_app(
        EspnClient(transport=httpx.MockTransport(NoGamesEspn(recorded, schedule=raw).handler)), store=store
    )
    app.state.now = lambda: MORNING_AFTER
    assert asyncio.run(app.state.box_sweep()) is True
    assert app.state.game_boxes.get(2027).sat == {2: [2, 4]}

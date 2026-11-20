import copy
from datetime import UTC, datetime

import httpx
import pytest

from app.analysis import gamebox, gamescore, preview
from app.analysis.gamebox import Box, Form
from app.espn.client import EspnClient
from app.main import create_app
from app.scores import parse_schedule
from app.store import MemoryStore
from tests.conftest import LEAGUE_ID, signed_in
from tests.test_scores import FRIDAY_AFTERNOON, HOU, ScoresEspn, day4_players, schedule_raw  # noqa: F401


def test_largest_remainder_adds_up_to_the_target():
    assert preview.largest_remainder([5.51, -1.85, 0.59, 0.0], 5) == [6, -2, 1, 0]
    assert preview.largest_remainder([7.86, 1.85, -4.32, 0.0], 6) == [8, 2, -4, 0]
    assert preview.largest_remainder([2.4, 2.4, 2.4], 6) == [2, 2, 2]  # floors pass the target: lose one each
    assert sum(preview.largest_remainder([0.3, 0.3, 0.3], 1)) == 1


def _parts(ratings: float, home: float, opp: float = 0.0, own: float = 0.0, b2b: float = 0.0) -> dict[str, float]:
    return {
        "ratings": ratings,
        "level": gamescore.LEVEL,
        "home": home,
        "opp_three_luck": opp,
        "own_three_luck": own,
        "back_to_back": b2b,
    }


def test_the_margin_is_told_from_the_picked_side_and_adds_up():
    away = _parts(111.5, -0.925, 0.4, 0.2)  # 111.7 -> away picked
    home = _parts(106.0, 0.925, -0.3, 0.1)  # 107.2
    m = preview.margin(away, home, 112, 107)
    assert m.pick == "away" and m.points == 5
    assert sum(t.points for t in m.terms) == 5
    terms = {t.kind: t for t in m.terms}
    assert terms["matchup"].raw == pytest.approx(5.5)
    assert terms["home_court"].points == -2  # the picked side is on the road
    assert terms["three_point_luck"].raw == pytest.approx(0.8)
    assert m.raw == pytest.approx(sum(away.values()) - sum(home.values()), abs=0.01)


def _form(team: int, last_day: int, minutes: float) -> Form:
    return Form(rate={}, minutes=minutes, minutes_long=minutes, usage=0.0, last_day=last_day, team=team)


def test_the_rotation_is_who_played_lately_with_minutes():
    box = Box(
        2027,
        {},
        {},
        {
            1: _form(5, 20, 30.0),  # a regular
            2: _form(5, 10, 25.0),  # hasn't played in his team's last five games
            3: _form(5, 20, 4.0),  # end of the bench
            4: _form(6, 20, 30.0),  # another team
            5: _form(5, 18, 9.0),
            6: _form(5, 19, 20.0),  # known already
        },
        played={5: [10, 12, 14, 16, 18, 20]},
    )
    assert preview.rotation(box, 5, 21, known={6}) == [1, 5]
    assert preview.rotation(box, 5, 10, known=set()) == []  # no games before the day


def test_team_line_is_the_lines_plus_the_rest_against_the_call():
    tl = preview.team_line(shown=100.0, season=90.0, team_points=112.0, call=115.0)
    assert (tl.rest, tl.total, tl.gap) == (22.0, 122.0, 7.0)
    assert preview.team_line(130.0, 120.0, 112.0, 115.0).rest == 0.0


def _with_minutes(recorded: dict) -> dict:
    """The recorded league predates minutes: give everyone 30 a game (as test_linelog does)."""
    recorded = copy.deepcopy(recorded)
    entries = [e["playerPoolEntry"]["player"] for t in recorded["league"]["teams"] for e in t["roster"]["entries"]]
    for player in entries + [e["player"] for e in recorded["free_agents"]]:
        for block in player.get("stats") or []:
            for key in ("averageStats", "stats"):
                if block.get(key):
                    block[key].setdefault("40", 30.0 if key == "averageStats" else 30.0 * block[key].get("42", 1))
    return recorded


def _client(recorded, schedule_raw, day4_players, box: Box | None):  # noqa: F811
    fake = ScoresEspn(_with_minutes(recorded), schedule_raw, day4_players)
    app = create_app(EspnClient(transport=httpx.MockTransport(fake.handler)), store=MemoryStore())
    app.state.now = lambda: FRIDAY_AFTERNOON
    if box is not None:
        app.state.game_boxes.set(2027, box)
    return signed_in(app)


def test_pregame_box_needs_our_call(recorded, schedule_raw, day4_players):  # noqa: F811
    client = _client(recorded, schedule_raw, day4_players, None)
    game = parse_schedule(schedule_raw).games[5][0]
    r = client.post("/api/scores/pregame", json={"league_id": LEAGUE_ID, "season": 2027, "game_id": game.id})
    assert r.status_code == 409 and r.json()["detail"]["code"] == "not_ready"


def test_pregame_box_for_tonights_game(recorded, schedule_raw, day4_players):  # noqa: F811
    sched = parse_schedule(schedule_raw)
    box = Box(2027, {}, {}, {}, schedule=gamebox.parse_schedule(schedule_raw), scores=gamescore.build([]))
    client = _client(recorded, schedule_raw, day4_players, box)
    r = client.post("/api/scores/pregame", json={"league_id": LEAGUE_ID, "season": 2027, "game_id": 1})
    assert r.status_code == 404 and r.json()["detail"]["code"] == "no_game"

    game = next(g for g in sched.games[5] if HOU in (g.away, g.home))
    body = client.post(
        "/api/scores/pregame", json={"league_id": LEAGUE_ID, "season": 2027, "game_id": game.id, "day": 5}
    ).json()
    call = box.call(5, game.away, game.home)
    assert (body["away"]["call"], body["home"]["call"]) == (call.away, call.home)
    assert body["status"] == "scheduled" and body["pick"] == "home"  # even teams: home court
    assert sum(t["points"] for t in body["terms"]) == body["margin"] == call.home - call.away
    for side in ("away", "home"):
        s = body[side]
        assert s["call_raw"] == pytest.approx(sum(v for k, v in s["parts"].items() if k in _parts(0, 0)), abs=0.05)
        assert s["players"], "the league's players on this team are listed"
        mins = [p["line"]["minutes"] for p in s["players"]]
        assert mins == sorted(mins, reverse=True)
        assert all(p["in_pool"] for p in s["players"])  # no game logs in the box: no rotation to read
        assert s["team_line"]["call"] == pytest.approx(s["call_raw"], abs=0.1)
    hou = body["away"] if game.away == HOU else body["home"]
    assert any(p["owner_team_id"] for p in hou["players"])

    # Kept a few minutes: the same box again without a new league load.
    again = client.post("/api/scores/pregame", json={"league_id": LEAGUE_ID, "season": 2027, "game_id": game.id}).json()
    assert again == body
    assert datetime.fromisoformat(body["start"].replace("Z", "+00:00")) > FRIDAY_AFTERNOON.replace(tzinfo=UTC)

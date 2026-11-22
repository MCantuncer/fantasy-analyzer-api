import asyncio
from datetime import date

import httpx
import pytest

from app import quarters
from app.quarters import QuarterGame, Shapes


def _competitor(team: int, side: str, lines: list[int]) -> dict:
    return {"homeAway": side, "team": {"id": str(team)}, "linescores": [{"value": float(x)} for x in lines]}


def _event(gid: int, away: int, home: int, a: list[int], h: list[int], state="post", season_type=2) -> dict:
    return {
        "id": str(gid),
        "season": {"type": season_type},
        "status": {"type": {"state": state}},
        "competitions": [
            {
                "status": {"type": {"state": state}},
                "competitors": [_competitor(home, "home", h), _competitor(away, "away", a)],
            }
        ],
    }


def test_scoreboard_quarters_keep_finished_regular_season_games_only():
    raw = {
        "events": [
            _event(1, 2, 5, [30, 25, 28, 27], [28, 26, 30, 24]),
            _event(2, 3, 4, [30, 25, 28, 27, 10], [28, 26, 30, 27, 8]),  # overtime: four quarters kept
            _event(3, 6, 7, [30, 25], [20, 22], state="in"),  # in progress
            _event(4, 8, 9, [30, 25, 28, 27], [28, 26, 30, 24], season_type=1),  # preseason
            _event(5, 10, 11, [30, 25, 28, 27], [28, 26, 30, 24], season_type=3),  # playoffs
            {"id": "6", "competitions": [{"competitors": [{"homeAway": "home"}]}]},  # broken
        ]
    }
    got = quarters.parse_quarters(raw)
    assert [g.game_id for g in got] == [1, 2]
    assert got[0] == QuarterGame(1, 2, 5, (30, 25, 28, 27), (28, 26, 30, 24))
    assert got[1].home_points == (28, 26, 30, 27)


def test_fetch_quarters_reads_the_day_and_gives_none_when_it_cannot():
    seen = []

    def ok(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.params["dates"])
        return httpx.Response(200, json={"events": [_event(1, 2, 5, [30, 25, 28, 27], [28, 26, 30, 24])]})

    got = asyncio.run(quarters.fetch_quarters(date(2026, 11, 21), httpx.MockTransport(ok)))
    assert seen == ["20261121"] and len(got) == 1
    down = httpx.MockTransport(lambda r: httpx.Response(503))
    assert asyncio.run(quarters.fetch_quarters(date(2026, 11, 21), down)) is None


def test_whole_points_add_up_to_the_rounded_sum():
    assert quarters.whole([28.4, 28.4, 28.4, 27.3]) == [29, 28, 28, 27]
    assert sum(quarters.whole([28.6, 28.1, 28.7, 27.2])) == 113
    assert quarters.whole([25.0, 25.0, 25.0, 25.0]) == [25, 25, 25, 25]


def test_an_even_game_without_a_shape_is_the_league_split():
    s = Shapes()
    q = quarters.quarter_line(1999, s, 1, 2, 112.0, 112.0)  # no last season on file
    assert q.raw == pytest.approx(
        [112 * sh + lv for sh, lv in zip(quarters.LEAGUE_SHARE, quarters.LEVEL, strict=True)], abs=0.01
    )
    assert q.shape == (0.0, 0.0, 0.0, 0.0) and q.games == 0
    assert q.raw[3] < q.raw[0] - 1  # the fourth runs about a point lower
    assert 110.5 < sum(q.raw) < 111.5  # about 0.75% of the points are overtime


def test_the_favourite_starts_fast_and_fades_in_the_fourth():
    s = Shapes()
    fav = quarters.quarter_line(1999, s, 1, 2, 120.0, 105.0)
    dog = quarters.quarter_line(1999, s, 2, 1, 105.0, 120.0)
    even = quarters.quarter_line(1999, s, 1, 2, 120.0, 120.0)
    assert fav.raw[0] - even.raw[0] == pytest.approx(15 * 0.0386, abs=0.01)
    assert fav.raw[3] < even.raw[3] and fav.favourite[3] < 0 < fav.favourite[0]
    assert dog.favourite[0] < 0 < dog.favourite[3]


def test_a_third_quarter_team_gets_more_in_the_third():
    s = Shapes()
    for gid in range(40):  # team 1 scores 34 in every third, 28 elsewhere; everyone else is even
        s.add(QuarterGame(gid, 1, 2 + gid % 5, (28, 28, 34, 28), (29, 29, 29, 29)))
    s.add(QuarterGame(99, 8, 9, (28, 28, 28, 28), (28, 28, 28, 28)))
    q = quarters.quarter_line(1999, s, 1, 9, 115.0, 115.0)
    assert q.games == 40
    assert q.shape[2] > 0.4 and all(x < 0 for x in (q.shape[0], q.shape[1], q.shape[3]))
    assert sum(q.shape) == pytest.approx(0, abs=0.01)  # a shape moves points between quarters, adds none
    # Team 1 allows an even game while the league's thirds run high: its opponents get less in the third.
    facing = quarters.quarter_line(1999, s, 9, 1, 115.0, 115.0)
    assert facing.shape[2] < 0


def test_last_season_starts_the_shape_and_this_season_takes_over():
    team = 28  # TOR scored 25.8% of its points in the third last season
    start = quarters.team_shape(2027, Shapes(), team, allowed=False)
    assert start[2] > 0.003
    s = Shapes()
    for gid in range(60):
        s.add(QuarterGame(gid, team, 1, (30, 30, 25, 30), (28, 28, 28, 28)))
    later = quarters.team_shape(2027, s, team, allowed=False)
    assert later[2] < 0 < later[0]  # this season's own games now outweigh last season's

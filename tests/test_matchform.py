import asyncio
from datetime import date

import httpx
import pytest

from app.matchform import fetch_kinds, form_game, game_score, kind_of, parse_kinds, team_form


def _line(pid: int, team: int, **stats) -> dict:
    return {"player_id": pid, "name": f"P{pid}", "pro_team_id": team, "minutes": 30} | stats


def _game(gid: int, start: str, away: tuple[int, str, int], home: tuple[int, str, int], lines: list[dict]) -> dict:
    side = lambda t: {"pro_team_id": t[0], "abbrev": t[1], "name": f"{t[1]} Club", "score": t[2]}  # noqa: E731
    return {"id": gid, "start": start, "status": "final", "away": side(away), "home": side(home), "lines": lines}


def test_game_score_is_hollingers_with_blended_rebounds():
    # 30 pts on 11/20, 6/8 FT, 10 reb, 5 ast, 2 stl, 1 blk, 3 TO
    line = {"pts": 30, "fgm": 11, "fga": 20, "ftm": 6, "fta": 8, "reb": 10, "ast": 5, "stl": 2, "blk": 1, "to": 3}
    assert game_score(line) == pytest.approx(30 + 4.4 - 14 - 0.8 + 4 + 2 + 3.5 + 0.7 - 3)
    assert game_score({}) == 0


def test_kinds_drop_preseason_and_tag_cup_play_in_and_playoffs():
    assert kind_of(1, []) == "preseason"
    assert kind_of(2, ["NBA Cup - Group Play"]) == "cup"
    assert kind_of(2, ["NBA Mexico City Game"]) == "regular"
    assert kind_of(5, []) == "play_in" and kind_of(3, ["East 1st Round - Game 2"]) == "playoff"
    raw = {
        "events": [
            {"id": "1", "season": {"type": 2}, "competitions": [{"notes": [{"headline": "NBA Cup - Group Play"}]}]},
            {
                "id": "2",
                "season": {"type": 2},
                "competitions": [{"neutralSite": True, "notes": [{"headline": "NBA Mexico City Game"}]}],
            },
            {"id": "3", "season": {"type": 1}, "competitions": [{}]},
            {"id": "bad"},
        ]
    }
    kinds = parse_kinds(raw)
    assert (kinds[1].kind, kinds[1].note, kinds[1].neutral) == ("cup", "NBA Cup - Group Play", False)
    assert (kinds[2].kind, kinds[2].neutral) == ("regular", True)
    assert kinds[3].kind == "preseason" and set(kinds) == {1, 2, 3}


def test_fetch_kinds_reads_the_day_and_gives_none_when_espn_fails():
    seen = []

    def ok(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.params["dates"])
        return httpx.Response(200, json={"events": [{"id": "7", "season": {"type": 2}, "competitions": [{}]}]})

    got = asyncio.run(fetch_kinds(date(2026, 11, 13), httpx.MockTransport(ok)))
    assert seen == ["20261113"] and got is not None and got[7].kind == "regular"
    assert asyncio.run(fetch_kinds(date(2026, 11, 13), httpx.MockTransport(lambda r: httpx.Response(403)))) is None


def test_mvp_comes_from_the_winners_and_best_from_the_team():
    lines = [
        _line(1, 11, pts=18, reb=7, ast=1, fgm=7, fga=12),  # IND, the winners' best
        _line(2, 11, pts=12, reb=2),
        _line(3, 18, pts=30, reb=8, ast=5, fgm=11, fga=22),  # NY's best: more than the MVP, but lost
        _line(4, 18, pts=0, minutes=0),  # did not play
    ]
    g = _game(9, "2026-11-14T00:00:00Z", (18, "NY", 104), (11, "IND", 106), lines)
    ny = form_game(g, 18, "2026-11-13")
    assert ny is not None and not ny.won and ny.margin == -2 and not ny.home and ny.opponent == "IND"
    assert ny.mvp is not None and (ny.mvp.player_id, ny.mvp.team) == (1, "IND")
    assert ny.mvp.stats["pts"] == 18 and ny.mvp.stats["fga"] == 12 and ny.mvp.stats["stl"] == 0
    assert ny.best is not None and ny.best.player_id == 3
    assert ny.opp_best == ny.mvp  # after a loss the other side's best is the MVP
    ind = form_game(g, 11, "2026-11-13")
    assert ind is not None and ind.won and ind.home and ind.mvp == ind.best
    assert ind.opp_best is not None and ind.opp_best.player_id == 3
    assert form_game(g, 5, None) is None  # not in this game
    assert form_game({**g, "status": "live"}, 11, None) is None


def test_team_form_is_newest_first_and_skips_ties_and_half_boxes():
    a = _game(1, "2026-11-01T23:00:00Z", (11, "IND", 90), (2, "BKN", 98), [_line(1, 11, pts=20)])
    b = _game(2, "2026-11-03T23:00:00Z", (11, "IND", 104), (3, "ATL", 104), [])  # no tie in the NBA: half a box
    c = _game(3, "2026-11-05T00:00:00Z", (11, "IND", 107), (4, "MEM", 103), [_line(1, 11, pts=20)])
    boxes = {12: {"date": "2026-11-01", "games": [a]}, 14: {"date": "2026-11-03", "games": [b]}}
    boxes[16] = {"date": "2026-11-04", "games": [c]}
    games = team_form(11, boxes)
    assert [g.game_id for g in games] == [3, 1]
    assert [g.date for g in games] == ["2026-11-04", "2026-11-01"]

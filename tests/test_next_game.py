import copy

import pytest

from app.analysis.form import BLK, MINUTES, PTS, REB, TO
from app.analysis.model import RECENT_MINUTES_WEIGHTS
from app.analysis.report import NEXT_GAME_LEVEL, Analyzer


def _player(league, recent=(None, None)):
    lg = copy.deepcopy(league)
    p = lg.team(3).roster[0]
    p.per_game = {**p.per_game, MINUTES: 22.0, PTS: 11.0, REB: 4.0, BLK: 0.5, TO: 1.2}
    p.minute_shifts = ()
    p.recent_minutes = recent
    p.injury_status = "ACTIVE"
    return lg, p


def test_next_game_is_his_line_at_the_level_players_reach(league):
    lg, p = _player(league)
    ng = Analyzer(lg, 3).ref(p).next_game
    assert ng is not None
    assert ng.minutes == 22.0
    assert ng.pts == pytest.approx(round(11.0 * NEXT_GAME_LEVEL[PTS], 1))
    assert ng.reb == pytest.approx(round(4.0 * NEXT_GAME_LEVEL[REB], 1))


def test_next_game_follows_his_recent_minutes(league):
    lg, p = _player(league, recent=(33.0, 30.0))
    ng = Analyzer(lg, 3).ref(p).next_game
    minutes = 22.0 + RECENT_MINUTES_WEIGHTS[0] * 11.0 + RECENT_MINUTES_WEIGHTS[1] * 8.0
    assert ng.minutes == pytest.approx(round(minutes, 1))
    assert ng.pts == pytest.approx(round(11.0 * minutes / 22.0 * NEXT_GAME_LEVEL[PTS], 1))


def test_no_next_game_while_he_is_out(league):
    lg, p = _player(league)
    p.injury_status = "OUT"
    assert Analyzer(lg, 3).ref(p).next_game is None

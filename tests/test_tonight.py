import copy

import pytest

from app.analysis.report import NEXT_GAME_LEVEL, Analyzer
from app.analysis.tonight import OTHER_ROLE, SAME_ROLE, minutes_gained, role
from app.espn.parse import MINUTES

AST, REB, PTS = 3, 6, 0
GUARD = {MINUTES: 30.0, PTS: 15.0, AST: 6.0, REB: 3.0}
BIG = {MINUTES: 28.0, PTS: 12.0, AST: 1.5, REB: 9.0}


def _setup(league, mate_line, status="OUT", played=True):
    lg = copy.deepcopy(league)
    roster = lg.team(3).roster
    p, mate = roster[0], roster[1]
    for q, line in ((p, GUARD), (mate, mate_line)):
        q.per_game = dict(line)
        q.pro_team_id = 99
        q.minute_shifts = ()
        q.recent_minutes = (line[MINUTES], line[MINUTES]) if played else (None, None)
        q.injury_status = "ACTIVE"
    mate.injury_status = status
    return lg, p, mate


def test_roles_come_from_assists_and_rebounds_per_36():
    assert role(GUARD) == 0
    assert role(BIG) == 2
    assert role({MINUTES: 30.0, AST: 2.0, REB: 5.0}) == 1


def test_a_same_role_teammate_ruled_out_gives_more_than_another_role(league):
    lg, p, mate = _setup(league, {**GUARD, MINUTES: 32.0})
    assert minutes_gained(p, [p, mate]) == pytest.approx(SAME_ROLE * 32.0)
    lg, p, mate = _setup(league, BIG)
    assert minutes_gained(p, [p, mate]) == pytest.approx(OTHER_ROLE * 28.0)


def test_only_a_fresh_out_counts(league):
    _, p, mate = _setup(league, GUARD, status="DAY_TO_DAY")
    assert minutes_gained(p, [mate]) == 0.0
    _, p, mate = _setup(league, GUARD, played=False)  # out for a while: his teammates' recent minutes hold it
    assert minutes_gained(p, [mate]) == 0.0


def test_the_next_game_line_takes_the_minutes(league):
    lg, p, _ = _setup(league, {**GUARD, MINUTES: 32.0})
    ng = Analyzer(lg, 3).ref(p).next_game
    minutes = 30.0 + SAME_ROLE * 32.0
    assert ng.minutes == pytest.approx(round(minutes, 1))
    assert ng.pts == pytest.approx(round(15.0 * minutes / 30.0 * NEXT_GAME_LEVEL[PTS], 1))

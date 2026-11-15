import copy

import pytest

from app.analysis.model import RECENT_MINUTES_WEIGHTS, CategoryModel, recent_shift
from app.espn.parse import GAMES_PLAYED, LAST_7_DAYS, LAST_15_DAYS, MINUTES, _recent_minutes


def _split(season: int, split: int, minutes: float, games: int) -> dict:
    return {
        "seasonId": season,
        "statSourceId": 0,
        "statSplitTypeId": split,
        "averageStats": {str(MINUTES): minutes, str(GAMES_PLAYED): games},
    }


def test_recent_minutes_come_from_the_7_and_15_day_splits_he_played_in():
    player = {
        "stats": [
            _split(2027, LAST_7_DAYS, 33.0, 3),
            _split(2027, LAST_15_DAYS, 30.0, 6),
            _split(2026, LAST_7_DAYS, 12.0, 3),  # another season
            _split(2027, 0, 22.0, 12),  # the season's own average
        ]
    }
    assert _recent_minutes(player, 2027) == (33.0, 30.0)
    no_games = {"stats": [_split(2027, LAST_7_DAYS, 0.0, 0), _split(2027, LAST_15_DAYS, 28.0, 2)]}
    assert _recent_minutes(no_games, 2027) == (None, 28.0)


def _bigger_role(league):
    lg = copy.deepcopy(league)
    p = lg.team(3).roster[0]
    p.per_game = {**p.per_game, MINUTES: 22.0}  # the recorded fixture keeps no minutes
    p.minute_shifts = ()
    p.recent_minutes = (33.0, 30.0)
    return lg, p


def test_this_week_leans_on_his_recent_minutes(league):
    lg, p = _bigger_role(league)
    model = CategoryModel(lg)
    pts = next(i for i, c in enumerate(model.cats) if c.stat_id == 0)
    week = model.horizon_row(p, lg.current_period)
    plain = model.row(p)
    shift = RECENT_MINUTES_WEIGHTS[0] * 11.0 + RECENT_MINUTES_WEIGHTS[1] * 8.0
    assert recent_shift(p) == pytest.approx(shift)
    # Counting stats follow the minutes.
    assert week[0, pts] / plain[0, pts] == pytest.approx((22.0 + shift) / 22.0)


def test_later_weeks_and_the_season_keep_the_blend(league):
    lg, p = _bigger_role(league)
    model = CategoryModel(lg)
    plain = model.row(p)
    assert model.horizon_row(p, "season") is plain
    later = next(k for k in model.horizon if k not in ("season", lg.current_period))
    assert model.horizon_row(p, later) is plain


def test_no_recent_games_or_a_measured_teammate_gain_leave_recent_minutes_out(league):
    lg, p = _bigger_role(league)
    p.recent_minutes = (None, None)
    model = CategoryModel(lg)
    assert model.horizon_row(p, lg.current_period) is model.row(p)
    lg, p = _bigger_role(league)
    p.minute_shifts = ((0.0, 10_000),)  # a gain while a teammate is out, measured from his games
    model = CategoryModel(lg)
    pts = next(i for i, c in enumerate(model.cats) if c.stat_id == 0)
    assert model.horizon_row(p, lg.current_period)[0, pts] == pytest.approx(model.row(p)[0, pts])

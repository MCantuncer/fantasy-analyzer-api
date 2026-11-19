"""The week under way, built game by game (CategoryModel.week_line, gamebox.week_line)."""

import copy

import pytest

from app.analysis import gamebox
from app.analysis.gamebox import GameContext
from app.analysis.model import CategoryModel
from app.espn.parse import MINUTES

PTS = 0


def _game(day: int, **kw) -> GameContext:
    base = dict(day=day, opponent=2, home=False, back_to_back=False, margin=0.0, defense={}, form=None, games_missed=0)
    return GameContext(**{**base, **kw})


def _player(league):
    lg = copy.deepcopy(league)
    p = lg.team(3).roster[0]
    p.per_game = {**p.per_game, MINUTES: 30.0}  # the recorded fixture keeps no minutes
    p.minute_shifts = ()
    p.recent_minutes = (None, None)
    p.injury_status = "ACTIVE"
    p.games_out = None
    return lg, p


def _points(model: CategoryModel, row) -> float:
    return row[0, next(i for i, c in enumerate(model.cats) if c.stat_id == PTS)]


def test_this_week_counts_his_games_one_by_one(league):
    lg, p = _player(league)
    today = lg.calendar.today
    p.week_contexts = (_game(today, defense={PTS: 0.3}), _game(today + 1, defense={PTS: 0.3}))
    model = CategoryModel(lg)
    week = model.horizon_row(p, lg.current_period)
    expected = model._row_of(gamebox.week_line(p.per_game, p.week_contexts))
    assert week == pytest.approx(expected)
    # Two soft defenses: more points than the same week with no opponent in it.
    p.week_contexts = (_game(today), _game(today + 1))
    assert _points(model, week) > _points(CategoryModel(lg), CategoryModel(lg).horizon_row(p, lg.current_period))


def test_the_season_and_later_weeks_keep_one_line(league):
    lg, p = _player(league)
    p.week_contexts = (_game(lg.calendar.today, defense={PTS: 0.3}),)
    model = CategoryModel(lg)
    plain = model.row(p)
    assert model.horizon_row(p, "season") is plain
    later = next(k for k in model.horizon if k not in ("season", lg.current_period))
    assert model.horizon_row(p, later) is plain


def test_without_the_league_box_the_week_is_as_before(league):
    lg, p = _player(league)
    assert p.week_contexts == ()
    model = CategoryModel(lg)
    assert model.horizon_row(p, lg.current_period) is model.row(p)


def test_a_player_listed_out_starts_from_his_first_game_back(league):
    lg, p = _player(league)
    today = lg.calendar.today
    soft, tough = _game(today, defense={PTS: 0.3}), _game(today + 3, defense={PTS: -0.3})
    p.week_contexts = (soft, tough)
    model = CategoryModel(lg)
    model.return_day = lambda _p: today + 2  # back for the second game only
    assert model.week_line(p, p.per_game) == gamebox.week_line(p.per_game, [tough])
    model.return_day = lambda _p: today + 9  # out all week: his games as they are, he starts none
    assert model.week_line(p, p.per_game) == gamebox.week_line(p.per_game, [soft, tough])

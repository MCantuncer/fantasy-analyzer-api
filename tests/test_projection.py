import numpy as np

from app.analysis.model import CategoryModel, Dist
from app.analysis.projection import crowded_days, full_week, project_matchup
from tests.conftest import MY_TEAM


def test_projects_every_category_of_your_matchup(league):
    proj = project_matchup(league, MY_TEAM)
    assert proj is not None
    assert len(proj.categories) == len(league.categories)
    assert proj.wins + proj.losses == len(proj.categories)
    assert proj.wins == sum(c.winner == "me" for c in proj.categories)
    m = next(x for x in league.schedule if x.period == league.current_period and MY_TEAM in (x.home, x.away))
    assert proj.opponent_id == (m.away if m.home == MY_TEAM else m.home)


def test_no_projection_without_a_matchup(league):
    assert project_matchup(league, 999) is None


def test_crowded_days_bench_the_weakest_players(league):
    model = CategoryModel(league)
    period = league.current_period
    # Two rosters together: far more players with games than lineup slots.
    roster = league.teams[0].roster + league.teams[1].roster
    days = crowded_days(model, roster, period)
    assert days
    by_id = {p.id: p for p in roster}
    best = max(roster, key=model.value)
    for d in days:
        assert d.slots == len(model.seats)
        assert d.playing > d.slots
        assert len(d.sit) >= d.playing - d.slots
        assert all(pid in by_id for pid, _ in d.sit)
        assert best.id not in {pid for pid, _ in d.sit}


def test_no_crowded_days_when_everyone_fits(league):
    model = CategoryModel(league)
    roster = sorted(league.teams[0].roster, key=model.value, reverse=True)[: len(model.seats)]
    assert crowded_days(model, roster, league.current_period) == []


def test_full_week_adds_what_is_banked(league):
    model = CategoryModel(league)
    cats = model.cats
    n = len(cats)
    rest = Dist(np.full(n, 0.5), np.full(n, 0.1), np.where(model.is_ratio, 100.0, 0.0))
    for k, c in enumerate(cats):
        if c.ratio is None:
            rest.mu[k] = 100.0
    banked = {}
    for c in cats:
        if c.ratio is None:
            banked[c.stat_id] = 50.0
        else:
            made, tried = c.ratio
            banked[made], banked[tried] = 90.0, 100.0  # .900 so far
    week = full_week(model, rest, banked)
    for k, c in enumerate(cats):
        if c.ratio is None:
            assert week.mu[k] == 150.0
        else:
            # 90/100 banked plus .500 on 100 attempts left: .700, and less spread than the days left alone.
            assert abs(week.mu[k] - 0.7) < 1e-9
            assert week.sd[k] < rest.sd[k]

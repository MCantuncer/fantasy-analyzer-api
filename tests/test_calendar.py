from dataclasses import replace

from app.analysis.model import CategoryModel
from app.analysis.trades import TradeFinder
from tests.conftest import MY_TEAM


def test_calendar_periods_cover_the_season(league):
    cal = league.calendar
    assert cal is not None
    periods = sorted(cal.period_days)
    assert periods[0] == 1
    days = [d for p in periods for d in cal.period_days[p]]
    assert days == sorted(days) and len(days) == len(set(days))
    # The All-Star break is folded into the week before it: one two-week period.
    assert max(len(cal.period_days[p]) for p in periods) == 14


def test_starts_respect_games_and_slots(league):
    model = CategoryModel(league)
    for team in league.teams:
        starts = model.starts(team.roster)
        games = [model.plays(p, "season").sum() for p in team.roster]
        assert all(0 <= s <= g for s, g in zip(starts, games, strict=True))
        # Never more starters on a day than the lineup has seats.
        days = len(model.horizon["season"][0])
        assert starts.sum() <= days * len(model.seats)


def test_injured_player_comes_back(league):
    model = CategoryModel(league)
    p = next(p for t in league.teams for p in t.roster if p.injury_status == "ACTIVE" and p.pro_team_id)
    healthy = model.plays(p, "season").sum()
    out = replace(p, id=-p.id, injury_status="OUT")
    missed = healthy - model.plays(out, "season").sum()
    # Misses a couple of weeks, not the rest of the season.
    assert 0 < missed < healthy / 2


def test_open_roster_spot_is_filled_from_free_agency(league):
    finder = TradeFinder(
        league, CategoryModel(league), next(i for i, t in enumerate(league.teams) if t.id == MY_TEAM), []
    )
    assert finder.filler is not None
    assert finder.filler.pro_team_id and finder.filler.available

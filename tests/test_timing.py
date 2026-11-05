from dataclasses import replace
from datetime import UTC, datetime, time

import pytest

from app.analysis.model import CategoryModel
from app.analysis.report import Analyzer
from app.analysis.timing import WAIT_MIN_GAIN, MatchupTiming
from tests.conftest import MY_TEAM


def noon_ms(league, day: int) -> int:
    """Midday in New York on that scoring day: before any NBA game tips off."""
    d = league.calendar.date_of(day)
    return int(datetime.combine(d, time(16, 0), tzinfo=UTC).timestamp() * 1000)


@pytest.fixture
def setup(league):
    cal = league.calendar
    model = CategoryModel(league)
    me = league.team(MY_TEAM)
    today = cal.today
    days = cal.days_left(league.current_period)
    # The best player of yours who plays today: he surely starts tonight.
    drop = max((p for p in me.roster if p.available and today in cal.team_days[p.pro_team_id]), key=model.value)
    # The same player on a team with no game today but one later this week.
    idle = next(t for t, d in cal.team_days.items() if today not in d and any(x in d for x in days[1:]))
    add = replace(drop, id=-1, pro_team_id=idle, on_team_id=0, status="FREEAGENT", lineup_slot=None)
    return model, drop, add


def test_waits_rather_than_throw_away_tonights_game(league, setup):
    model, drop, add = setup
    t = MatchupTiming(league, MY_TEAM, model, noon_ms(league, league.calendar.today)).best(add, drop)
    assert t is not None
    assert t.earliest_day == league.calendar.today
    assert t.best_day > t.earliest_day
    assert t.drop_games_kept >= 1
    assert t.cats_best >= t.cats_earliest + WAIT_MIN_GAIN


def test_after_tip_off_the_game_is_already_yours(league, setup):
    model, drop, add = setup
    tip = league.calendar.tips[drop.pro_team_id][league.calendar.today]
    t = MatchupTiming(league, MY_TEAM, model, tip + 60_000).best(add, drop)
    # His game tonight counts either way, and the new man has none today: no reason to wait.
    assert t is not None and t.best_day == t.earliest_day


def test_waiver_claim_counts_from_when_it_processes(league, setup):
    model, drop, add = setup
    today = league.calendar.today
    claim = replace(add, status="WAIVERS", waiver_ms=noon_ms(league, today + 1) - 3600_000)
    t = MatchupTiming(league, MY_TEAM, model, noon_ms(league, today)).best(claim, drop)
    assert t is not None
    assert t.earliest_day == today + 1  # the dropped player keeps tonight's game anyway


def test_same_player_same_games_never_waits(league, setup):
    model, drop, _ = setup
    twin = replace(drop, id=-2, on_team_id=0, status="FREEAGENT", lineup_slot=None)
    t = MatchupTiming(league, MY_TEAM, model, noon_ms(league, league.calendar.today)).best(twin, drop)
    assert t is not None and t.best_day == t.earliest_day
    assert abs(t.cats_earliest) < 1e-9


def test_no_timing_without_the_calendar(flat_league):
    model = CategoryModel(flat_league)
    p = flat_league.team(MY_TEAM).roster[0]
    assert MatchupTiming(flat_league, MY_TEAM, model, 0).best(p, p) is None


def test_pickups_say_when_to_make_them(league):
    now = datetime.fromtimestamp(noon_ms(league, league.calendar.today) / 1000, UTC)
    ideas = Analyzer(league, MY_TEAM, now=now).pickups()
    assert ideas
    days = {league.calendar.date_of(d).isoformat() for d in league.calendar.days_left(league.current_period)}
    for idea in ideas:
        t = idea.timing
        assert t is not None
        assert t.make_on in days or t.next_week
        assert t.wait == (t.make_on != league.calendar.date_of(league.calendar.today).isoformat())
        if t.wait:
            assert t.cats_best >= t.cats_now + WAIT_MIN_GAIN - 1e-3

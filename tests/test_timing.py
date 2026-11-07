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


def test_timing_carries_its_categories_and_week(league, setup):
    model, drop, add = setup
    t = MatchupTiming(league, MY_TEAM, model, noon_ms(league, league.calendar.today)).best(add, drop)
    assert abs(float(t.cat_changes.sum()) - t.cats_best) < 1e-9
    assert len(t.rest.mu) == len(league.categories)


def test_adds_allowed_per_day_and_per_matchup(league):
    from app.espn.parse import _adds_limit

    week = league.current_period
    days = len(league.calendar.game_days(week))
    per_day = replace(league, **_adds_limit({"matchupAcquisitionLimit": 5 / 7, "matchupLimitPerScoringPeriod": True}))
    assert per_day.adds_allowed(week) == (5 if days == 7 else int(round(5 / 7 * days, 6)))
    flat = replace(league, **_adds_limit({"matchupAcquisitionLimit": 4}))
    assert flat.adds_allowed(week) == 4
    assert replace(league, **_adds_limit({"matchupAcquisitionLimit": -1})).adds_allowed(week) is None


def test_adds_allowed_skips_days_without_games(league):
    """ESPN earns the per-day pickup allowance only on days with NBA games: League A's week of
    2026-11-02 had no games on Election Day, so ESPN allowed 4 pickups, not 5."""
    from app.espn.parse import _adds_limit

    week = league.current_period
    limit = _adds_limit({"matchupAcquisitionLimit": 5 / 7, "matchupLimitPerScoringPeriod": True})

    def with_games(game_days: set[int]):
        cal = replace(league.calendar, period_days={week: list(range(15, 22))}, team_days={0: frozenset(game_days)})
        return replace(league, calendar=cal, **limit)

    assert with_games(set(range(15, 22))).adds_allowed(week) == 5
    election_week = with_games(set(range(15, 22)) - {16})
    assert election_week.calendar.game_days(week) == [15, 17, 18, 19, 20, 21]
    assert election_week.adds_allowed(week) == 4


def test_week_override_with_todays_roster_changes_nothing(league):
    a = Analyzer(league, MY_TEAM)
    today = a.model.dist(a.me.roster, league.current_period)
    assert abs(a.playoff_pct(a.dists, week=today) - a.playoff_pct(a.dists)) < 1e-9


@pytest.fixture(scope="module")
def ideas(league):
    now = datetime.fromtimestamp(noon_ms(league, league.calendar.today) / 1000, UTC)
    a = Analyzer(league, MY_TEAM, now=now)
    return a, a.pickups()


def test_streams_only_release_players_without_trade_value(ideas):
    from app.analysis.report import STREAM_MIN_CATS
    from app.analysis.tiers import has_market

    a, found = ideas
    roster = {p.id: p for p in a.me.roster}
    for idea in found:
        if idea.stream:
            assert not idea.open_spot_by_trade
            assert not has_market(roster[idea.drop.id])
            assert idea.timing is not None and idea.my_cats_gain >= STREAM_MIN_CATS - 1e-3
        assert idea.playoff_pct_after >= idea.playoff_pct_before


def test_no_adds_left_means_no_streams_and_no_timing(league):
    me = league.team(MY_TEAM)
    full = replace(me, period_adds={league.current_period: 99})
    capped = replace(
        league,
        teams=[full if t.id == MY_TEAM else t for t in league.teams],
        adds_limit=5,
        adds_limit_per_day=False,
    )
    now = datetime.fromtimestamp(noon_ms(league, league.calendar.today) / 1000, UTC)
    a = Analyzer(capped, MY_TEAM, now=now)
    assert a.adds_left() == 0
    assert a.week_matchup().adds_allowed == 5
    found = a.pickups()
    assert not any(i.stream for i in found)
    assert all(i.timing is None for i in found)


def test_timing_lists_who_plays_each_day_left(ideas, league):
    _, found = ideas
    days = [league.calendar.date_of(d).isoformat() for d in league.calendar.days_left(league.current_period)]
    for idea in found:
        if idea.timing is not None:
            assert [d.date for d in idea.timing.days] == days

import asyncio
import copy
import time

import pytest

from app.analysis import teammates
from app.analysis.injury import Beneficiary, GameLine, Impact, Season
from app.analysis.model import CategoryModel
from app.espn.parse import MINUTES, MINUTES_WEIGHT_GAMES
from tests.conftest import LEAGUE_ID


def _setup(league, games_out=6.0, played_without=5, played=10):
    lg = copy.deepcopy(league)
    absent, mate = lg.team(3).roster[:2]
    absent.injury_status, absent.games_out = "OUT", games_out
    mate.pro_team_id = absent.pro_team_id  # make them NBA teammates
    mate.per_game = {**mate.per_game, MINUTES: 25.0}  # the recorded fixture keeps no minutes
    team = absent.pro_team_id
    lines = []
    for g in range(played):
        lines.append(GameLine(mate.id, 1000 + g, g + 1, team, 25.0))
        if g < played - played_without:
            lines.append(GameLine(absent.id, 1000 + g, g + 1, team, 32.0))
    impact = Impact(
        minutes=32.0,
        games_missed=played_without,
        beneficiaries=[Beneficiary(mate.id, mate.name, 25.0, 31.0, 6.0, 1.0, played_without, {}, 0.5, "some")],
    )
    return lg, absent, mate, impact, Season(lines)


def _shifts(lg, absent, impact, season):
    return teammates.shifts(lg, absent, teammates.measure(impact, season, absent.pro_team_id, absent.id))


def test_gain_counts_only_while_he_is_out_net_of_what_the_blend_already_holds(league):
    lg, absent, mate, impact, season = _setup(league)
    [s] = _shifts(lg, absent, impact, season)
    left = [d for d in teammates.season_days(lg) if d in lg.calendar.team_days[absent.pro_team_id]]
    ahead = sum(d < s.back for d in left) / len(left)
    already = 10 / (10 + MINUTES_WEIGHT_GAMES) * 5 / 10
    assert s.change == 6.0
    assert s.season == pytest.approx(6.0 * (ahead - already))
    assert 0 < ahead < 1


def test_short_absences_and_bench_players_move_nobody(league):
    lg, absent, mate, impact, season = _setup(league, games_out=1.0)
    assert _shifts(lg, absent, impact, season) == []
    lg, absent, mate, impact, season = _setup(league)
    impact.minutes = 12.0
    assert _shifts(lg, absent, impact, season) == []
    assert absent in teammates.absent_regulars(lg)


def test_apply_moves_minutes_and_counting_stats_together(league):
    lg, absent, mate, impact, season = _setup(league, played_without=0)
    before = dict(mate.per_game)
    moved = _shifts(lg, absent, impact, season)
    teammates.apply(lg, moved)
    assert mate.per_game[MINUTES] == pytest.approx(before[MINUTES] + moved[0].season)
    assert mate.per_game[0] / before[0] == pytest.approx(mate.per_game[MINUTES] / before[MINUTES])
    assert mate.minute_shifts == ((6.0, moved[0].back),)


def test_weekly_projection_gets_the_full_gain_while_he_is_out(league):
    lg, absent, mate, impact, season = _setup(league, games_out=20.0, played_without=0)
    teammates.apply(lg, _shifts(lg, absent, impact, season))
    model = CategoryModel(lg)
    week = model.horizon_row(mate, lg.current_period)
    rest = model.horizon_row(mate, "season")
    pts = next(i for i, c in enumerate(model.cats) if c.stat_id == 0)
    assert week[0, pts] > rest[0, pts]  # out all of this week, back later in the season


def _full_league(client):
    """The league the last full load cached."""
    return max((v for _, v in client.app.state.leagues._items.values()), key=lambda lg: len(lg.free_agents))


def _one_out(monkeypatch):
    """Every load lists team 3's first player out for 6 more games; the second plays 25 minutes."""

    def out(league):
        x, mate = league.team(3).roster[:2]
        x.injury_status, x.games_out = "OUT", 6.0
        mate.per_game = {**mate.per_game, MINUTES: 25.0}  # the recorded fixture keeps no minutes
        return [x]

    monkeypatch.setattr(teammates, "absent_regulars", out)


def test_a_league_load_never_waits_for_measuring(client, monkeypatch):
    _one_out(monkeypatch)
    asked = []

    async def never(season, team):
        asked.append(team)
        await asyncio.sleep(3600)  # ESPN never answers

    monkeypatch.setattr(client.app.state.espn, "team_gamelogs", never)
    with client:
        started = time.perf_counter()
        res = client.post("/api/matchup-projection", json={"league_id": LEAGUE_ID, "season": 2027})
        assert res.status_code == 200 and time.perf_counter() - started < 30
        job = client.app.state.absence_jobs[2027]
        assert not job.done()  # still measuring in the background
        assert all(not p.minute_shifts for t in _full_league(client).teams for p in t.roster)
        job.cancel()


def test_a_later_load_uses_the_measured_absences(client, monkeypatch):
    _one_out(monkeypatch)
    with client:
        client.post("/api/matchup-projection", json={"league_id": LEAGUE_ID, "season": 2027})
        lg = _full_league(client)
        x = lg.team(3).roster[0]
        mate = lg.team(3).roster[1]
        for job in client.app.state.absence_jobs.values():
            job.cancel()
        client.app.state.absence_jobs.clear()
        gain = teammates.Gain(mate.id, 6.0, 0, 10)
        client.app.state.absence_minutes.set((2027, x.id), teammates.Measured(32.0, (gain,)))
        client.app.state.leagues.clear()
        client.post("/api/matchup-projection", json={"league_id": LEAGUE_ID, "season": 2027})
        mate = _full_league(client).team(3).roster[1]
        assert mate.minute_shifts == ((6.0, mate.minute_shifts[0][1]),)
        assert 2027 not in client.app.state.absence_jobs  # nothing left to measure

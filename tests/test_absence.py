from dataclasses import replace

import pytest

from app.analysis.absence import (
    REMAINING,
    attach,
    expected_missed,
    is_out,
    missed_in_a_row,
    played_days,
    season_end_run,
)
from app.analysis.model import CategoryModel
from app.espn.parse import IR_SLOT


def _line(season, day, played, source=0):
    return {
        "seasonId": season,
        "statSplitTypeId": 5,
        "statSourceId": source,
        "scoringPeriodId": day,
        "stats": {"42": 1.0} if played else {},
    }


def test_the_longer_out_the_longer_still_out():
    misses = [expected_missed(k) for k in range(1, 60)]
    assert misses == sorted(misses)
    assert expected_missed(1) == REMAINING[0][1]
    assert expected_missed(12) > 3 * expected_missed(1)
    assert expected_missed(1000) == REMAINING[-1][1]


def test_out_after_his_last_game_misses_at_least_the_next():
    assert expected_missed(0) == pytest.approx(1 + expected_missed(1))


def test_games_missed_in_a_row_count_team_games_since_his_last():
    team_days = [1, 3, 5, 7, 9]
    assert missed_in_a_row(team_days, {1, 3}, today=9) == 2
    assert missed_in_a_row(team_days, {1, 3, 7}, today=9) == 0
    # Days from today on haven't happened yet.
    assert missed_in_a_row(team_days, {1}, today=6) == 2
    # No game yet this season: last season's closing run counts too.
    assert missed_in_a_row(team_days, set(), today=9, before=10) == 14
    assert missed_in_a_row(team_days, {3}, today=9, before=10) == 2


def test_game_lines_give_days_played_and_last_seasons_closing_run():
    logs = [
        {"player": {"id": 1, "stats": [_line(2027, 2, True), _line(2027, 4, False), _line(2027, 6, True, source=1)]}},
        {"id": 2, "stats": [_line(2026, 170, True), _line(2026, 172, False), _line(2026, 174, False)]},
    ]
    assert played_days(logs, 2027) == {1: {2}, 2: set()}
    assert season_end_run(logs, 2026) == {1: 0, 2: 2}


def test_long_absence_brings_him_back_later(league):
    p = next(p for t in league.teams for p in t.roster if p.injury_status == "ACTIVE" and p.pro_team_id)
    cal = replace(league.calendar, today=40)  # the fixture is from before opening night
    league = replace(league, calendar=cal)
    team_days = sorted(cal.team_days[p.pro_team_id])
    before = [d for d in team_days if d < cal.today]
    assert before, "the fixture has games played"

    def back(played_on: set[int], last_run: int = 0):
        out = replace(p, injury_status="OUT")
        lg = replace(league, teams=[replace(league.teams[0], roster=[out])], free_agents=[])
        attach(lg, {out.id: played_on}, {out.id: last_run})
        return out, CategoryModel(lg)

    just_hurt, model = back({before[-1]})
    short = model.plays(just_hurt, "season").sum()
    assert just_hurt.games_out == expected_missed(0)
    upcoming = [d for d in team_days if d >= cal.today]
    assert model.return_day(just_hurt) == upcoming[round(expected_missed(0))]
    assert model.back_date(just_hurt) == cal.date_of(model.return_day(just_hurt)).isoformat()

    long_out, model = back(set(), last_run=30)
    assert long_out.games_out == REMAINING[-1][1]
    assert model.plays(long_out, "season").sum() < short


def test_only_players_listed_out_get_games_out(league):
    p = next(p for t in league.teams for p in t.roster if p.injury_status == "ACTIVE")
    assert not is_out(p)
    assert is_out(replace(p, injury_status="OUT"))
    assert is_out(replace(p, lineup_slot=IR_SLOT))
    assert not is_out(replace(p, injury_status="DAY_TO_DAY"))
    healthy = replace(p)
    lg = replace(league, teams=[replace(league.teams[0], roster=[healthy])], free_agents=[])
    attach(lg, {healthy.id: set()})
    assert healthy.games_out is None
    # Without game lines the flat guess stays.
    assert CategoryModel(lg).back_date(replace(p, injury_status="OUT")) is None

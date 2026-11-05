from dataclasses import replace

import pytest

from app.analysis.durability import (
    LEAGUE_SHARE,
    ONE_OFF_LENGTH,
    PRONE_SHARE,
    PastSeason,
    attach,
    expected_share,
    games_played,
    keep_share,
    kept,
    long_absences,
    missed_runs,
    one_off_candidates,
    season_length,
    set_aside_one_off,
)
from app.analysis.model import CategoryModel

FULL = PastSeason(82, 82)
LAVINE = [PastSeason(39, 82), PastSeason(74, 82), PastSeason(25, 82)]


def test_no_record_means_a_typical_share():
    assert expected_share([]) == pytest.approx(LEAGUE_SHARE)
    assert expected_share([None, None, None]) == pytest.approx(LEAGUE_SHARE)


def test_missed_seasons_lower_the_share_and_recent_ones_count_more():
    prone = expected_share(LAVINE)
    assert prone < LEAGUE_SHARE < expected_share([FULL, FULL, FULL])
    # A bad season last year weighs more than the same season three years ago.
    assert expected_share([PastSeason(25, 82), FULL, FULL]) < expected_share([FULL, FULL, PastSeason(25, 82)])


def test_shortened_seasons_count_as_shares():
    assert expected_share([PastSeason(72, 72)]) == pytest.approx(expected_share([FULL]))


def test_healthy_start_eases_the_risk_only_partway():
    before = expected_share(LAVINE)
    after = expected_share(LAVINE, played=10, team_games=10)
    durable = expected_share([FULL, FULL, FULL], played=10, team_games=10)
    assert before < after < durable
    assert expected_share(LAVINE, played=4, team_games=10) < before


def test_games_played_and_season_length_from_espn_totals():
    players = [
        {"id": 1, "stats": [{"seasonId": 2026, "statSourceId": 0, "statSplitTypeId": 0, "stats": {"42": 39.0}}]},
        {
            "player": {
                "id": 2,
                "stats": [{"seasonId": 2026, "statSourceId": 0, "statSplitTypeId": 0, "stats": {"42": 84}}],
            }
        },
        {"id": 3, "stats": [{"seasonId": 2026, "statSourceId": 1, "statSplitTypeId": 0, "stats": {"42": 70.0}}]},
    ]
    played = games_played(players, 2026)
    assert played == {1: 39, 2: 84}
    assert season_length(played) == 82


@pytest.mark.parametrize("share", [0.3, 0.5, 0.72, 0.9])
def test_missed_games_are_spread_evenly(share):
    for phase in (0.0, 0.37, 0.99):
        games = [kept(i, share, phase) for i in range(80)]
        assert sum(games) == pytest.approx(80 * share, abs=1)
        assert sum(games[:20]) == pytest.approx(20 * share, abs=1)


def test_typical_or_better_players_play_every_game():
    p = object.__new__(type("P", (), {}))
    p.durability = None
    assert keep_share(p) == 1.0
    p.durability = 0.9
    assert keep_share(p) == 1.0
    p.durability = LEAGUE_SHARE / 2
    assert keep_share(p) == pytest.approx(0.5)


def test_attach_sets_every_player(league):
    league = replace(
        league,
        teams=[replace(t, roster=[replace(p) for p in t.roster]) for t in league.teams],
        free_agents=[replace(p) for p in league.free_agents],
    )
    star = league.teams[0].roster[0]
    attach(league, [{star.id: 25, 0: 82}, {star.id: 30, 0: 82}, {0: 82}])
    assert star.durability < LEAGUE_SHARE
    assert all(p.durability is not None for t in league.teams for p in t.roster)
    assert all(p.durability is not None for p in league.free_agents)


def test_prone_player_starts_fewer_games_and_his_team_projects_less(league):
    model = CategoryModel(league)
    team = league.teams[0]
    p = max(team.roster, key=model.value)
    prone = replace(p, id=-p.id, durability=LEAGUE_SHARE / 2)
    roster = [q for q in team.roster if q is not p]
    sound = CategoryModel(league)
    healthy_starts = sound.starts(roster + [p])[-1]
    prone_starts = sound.starts(roster + [prone])[-1]
    assert 0.4 * healthy_starts < prone_starts < 0.6 * healthy_starts
    with_him = sound.dist(roster + [p]).mu
    without = sound.dist(roster).mu
    halfway = sound.dist(roster + [prone]).mu
    pts = [c.name for c in sound.cats].index("PTS")
    assert without[pts] < halfway[pts] < with_him[pts]


def test_player_refs_carry_the_expected_share(league):
    from app.analysis.report import Analyzer
    from tests.conftest import MY_TEAM

    lg = replace(league, teams=[replace(t, roster=list(t.roster)) for t in league.teams])
    star = lg.teams[0].roster[0]
    lg.teams[0].roster[0] = prone = replace(star, durability=0.5)
    ref = Analyzer(lg, MY_TEAM).ref(prone)
    assert ref.games_share == 0.5 and ref.injury_prone


# Tatum 2025-26: a torn Achilles kept him out 62 games, after years of 72-76.
ACHILLES = [PastSeason(16, 82), PastSeason(72, 82), PastSeason(74, 82)]


def test_missed_runs():
    assert missed_runs([True, False, False, True, False]) == [2, 1]
    assert missed_runs([]) == []
    assert missed_runs([False] * 3) == [3]


def test_a_single_major_absence_is_set_aside():
    record = set_aside_one_off(ACHILLES, [[62], [], []])
    assert record[0] == PastSeason(16, 20)
    assert expected_share(ACHILLES) < PRONE_SHARE < LEAGUE_SHARE < expected_share(record)


def test_several_long_absences_still_count():
    # Zion-like: a major absence plus other long ones in the window keeps every game counted.
    assert set_aside_one_off(ACHILLES, [[62], [12], []]) == ACHILLES
    assert set_aside_one_off(ACHILLES, [[45, 17], [], []]) == ACHILLES
    # A long absence shorter than half a season is wear and tear, not a one-off.
    short = [PastSeason(50, 82), PastSeason(82, 82)]
    assert set_aside_one_off(short, [[ONE_OFF_LENGTH - 1], []]) == short


def test_missing_game_logs_keep_the_record():
    assert set_aside_one_off(ACHILLES, [[62], None, []]) == ACHILLES
    # No record that season, so no logs needed for it.
    rookie = [PastSeason(16, 82), None, None]
    assert set_aside_one_off(rookie, [[62], None, None])[0] == PastSeason(16, 20)


def test_only_players_with_half_a_season_missed_need_logs():
    past = [{1: 16, 2: 70, 0: 82}, {1: 72, 2: 30, 0: 82}, {0: 82}]
    assert one_off_candidates(past, {1, 2, 3}) == {1, 2}
    assert one_off_candidates([{1: 43, 0: 82}], {1}) == set()


def test_long_absences_from_espn_game_lines():
    def line(day, played, season=2026):
        return {
            "seasonId": season,
            "statSplitTypeId": 5,
            "statSourceId": 0,
            "scoringPeriodId": day,
            "stats": {"42": played},
        }

    games = [line(d, 0) for d in range(1, 13)] + [line(13, 1), line(14, 0), line(15, 1)]
    other = [line(1, 0, 2025), {"seasonId": 2026, "statSplitTypeId": 0, "statSourceId": 0, "stats": {"42": 3}}]
    assert long_absences([{"player": {"id": 7, "stats": list(reversed(games)) + other}}], 2026) == {7: [12]}


def test_attach_sets_aside_a_one_off(league):
    league = replace(
        league,
        teams=[replace(t, roster=[replace(p) for p in t.roster]) for t in league.teams],
        free_agents=[replace(p) for p in league.free_agents],
    )
    star = league.teams[0].roster[0]
    past = [{star.id: 16, 0: 82}, {star.id: 72, 0: 82}, {star.id: 74, 0: 82}]
    attach(league, past)
    counted = star.durability
    attach(league, past, [{star.id: [62]}, {star.id: []}, {star.id: []}])
    assert counted < PRONE_SHARE < LEAGUE_SHARE < star.durability


def test_player_refs_name_the_nba_team(league):
    from app.analysis.report import Analyzer
    from app.espn.parse import PRO_TEAM_ABBREV
    from tests.conftest import MY_TEAM

    rostered = league.teams[0].roster[0]
    ref = Analyzer(league, MY_TEAM).ref(rostered)
    assert ref.pro_team == PRO_TEAM_ABBREV.get(rostered.pro_team_id, "")
    assert Analyzer(league, MY_TEAM).ref(replace(rostered, pro_team_id=0)).pro_team == ""

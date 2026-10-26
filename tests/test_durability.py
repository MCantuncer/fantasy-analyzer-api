from dataclasses import replace

import pytest

from app.analysis.durability import (
    LEAGUE_SHARE,
    PastSeason,
    attach,
    expected_share,
    games_played,
    keep_share,
    kept,
    season_length,
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

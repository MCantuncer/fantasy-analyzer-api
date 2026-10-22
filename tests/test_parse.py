def test_league_settings(league):
    assert league.name == "Chill Primera Division"
    assert league.scoring_type == "H2H_CATEGORY"
    assert sorted(c.name for c in league.categories) == sorted(
        ["PTS", "REB", "AST", "STL", "BLK", "3PM", "FG%", "FT%", "TO"]
    )
    assert next(c for c in league.categories if c.name == "TO").reverse
    assert len(league.teams) == 10
    assert league.starter_slots == 10
    assert league.playoff_teams == 6
    assert league.regular_season_periods == 18


def test_rosters_and_free_agents(league):
    me = league.team(3)
    assert me.name == "Kucukesat Aspavas"
    assert len(me.roster) == 13
    assert me.faab_remaining is not None
    names = {p.name for p in me.roster}
    assert "Giannis Antetokounmpo" in names
    assert league.free_agents and all(p.on_team_id == 0 for p in league.free_agents)
    giannis = next(p for p in me.roster if p.name == "Giannis Antetokounmpo")
    assert giannis.per_game[0] > 20  # points per game
    assert giannis.per_game[14] > giannis.per_game[13]  # FGA > FGM


def test_schedule_covers_regular_season(league):
    periods = {m.period for m in league.schedule}
    assert periods == set(range(1, 19))
    assert all(len([m for m in league.schedule if m.period == p]) == 5 for p in periods)

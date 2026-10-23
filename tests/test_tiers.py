from app.analysis.model import CategoryModel
from app.analysis.report import Analyzer
from app.analysis.tiers import droppable, has_market, tier
from app.analysis.trades import TradeFinder


def test_well_owned_players_are_never_droppable(league):
    for team in league.teams:
        for p in team.roster:
            if p.percent_owned >= 50:
                assert tier(p) in ("tradeable", "injured")
                assert p not in droppable(team.roster)


def test_kept_players_are_core(league):
    p = league.team(3).roster[0]
    assert tier(p, {p.id}) == "core"


def test_pickups_release_only_droppable_players(league):
    a = Analyzer(league, 3)
    for idea in a.pickups():
        drop = next(p for p in a.me.roster if p.id == idea.drop.id)
        assert idea.open_spot_by_trade == has_market(drop)
        assert idea.drop.tier in (("tradeable",) if idea.open_spot_by_trade else ("droppable",))


def test_trades_never_cut_my_players_with_a_market(league):
    model = CategoryModel(league)
    me = next(i for i, t in enumerate(league.teams) if t.id == 3)
    finder = TradeFinder(league, model, me, keep=set())
    for d in finder.candidates():
        if d.my_drop is not None:
            assert not has_market(d.my_drop)


def test_small_samples_without_a_prior_are_not_suggested(league):
    a = Analyzer(league, 3)
    fas = {p.id: p for p in league.free_agents}
    for idea in a.pickups():
        fa = fas[idea.add.id]
        assert fa.has_prior or fa.games_played >= 5

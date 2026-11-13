import dataclasses

from app.analysis.badges import as_of_ms
from app.analysis.model import CategoryModel
from app.analysis.report import Analyzer
from app.analysis.tiers import WAIVER_FRESH_DAYS, droppable, has_market, off_the_wire, tier
from app.analysis.trades import TradeFinder

DAY_MS = 86_400_000


def test_well_owned_players_are_never_droppable(league):
    for team in league.teams:
        for p in team.roster:
            if p.percent_owned >= 50:
                assert tier(p) in ("tradeable", "injured")
                assert p not in droppable(team.roster)


def test_good_season_totals_alone_are_not_a_market(league):
    p = league.team(3).roster[0]
    hot_start = dataclasses.replace(p, percent_owned=30.0, espn_rank=150, season_rank=20, games_played=12)
    assert not has_market(hot_start)
    assert tier(hot_start) == "droppable"
    assert has_market(dataclasses.replace(hot_start, espn_rank=60))
    assert has_market(dataclasses.replace(hot_start, percent_owned=65.0))


def test_a_recent_free_agent_pickup_is_off_the_wire_whatever_his_ranking(league):
    now = as_of_ms(league)
    p = dataclasses.replace(league.team(3).roster[0], percent_owned=90.0, espn_rank=40)
    added = dataclasses.replace(p, acquisition_type="ADD", acquired_ms=now - 3 * DAY_MS)
    assert off_the_wire(added, now)
    assert not off_the_wire(dataclasses.replace(added, acquired_ms=now - (WAIVER_FRESH_DAYS + 1) * DAY_MS), now)
    # Drafted or traded for: someone paid for him.
    assert not off_the_wire(dataclasses.replace(added, acquisition_type="DRAFT"), now)
    assert not off_the_wire(dataclasses.replace(added, acquisition_type="TRADE"), now)
    # No market at all is waiver level however he was acquired.
    assert off_the_wire(dataclasses.replace(p, percent_owned=20.0, espn_rank=180, acquisition_type="DRAFT"), now)


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

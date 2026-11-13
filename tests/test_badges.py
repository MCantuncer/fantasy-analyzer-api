import dataclasses

import pytest

from app.analysis.badges import (
    FRESH_DAYS,
    JEWEL_RANK,
    KEYSTONE_MIN_DOLLARS,
    MAX_PREMIUM,
    OwnerPremium,
)
from app.analysis.model import CategoryModel
from app.analysis.perceived import PerceivedValue
from app.analysis.trades import TradeFinder
from app.espn.parse import _trade_block

ME = 3


@pytest.fixture(scope="module")
def model(league):
    return CategoryModel(league)


@pytest.fixture(scope="module")
def premium(league, model):
    return OwnerPremium(league, model, PerceivedValue(league, model))


def player(league, name):
    return next(p for t in league.teams for p in t.roster if p.name == name)


def kinds(premium, p):
    return {b.kind for b in premium.badges(p)}


def test_star_names_are_jewels(league, premium):
    tatum = player(league, "Jayson Tatum")
    assert "jewel" in kinds(premium, tatum)
    assert premium.rate(tatum) >= 0.2
    jewels = [p for t in league.teams for p in t.roster if "jewel" in kinds(premium, p)]
    assert JEWEL_RANK <= len(jewels) <= 20


def test_category_carriers_are_keystones(league, premium):
    haliburton = player(league, "Tyrese Haliburton")
    assert premium.keystone(haliburton) == "AST"
    assert premium.keystone(player(league, "Victor Wembanyama")) == "BLK"
    assert premium.keystone(player(league, "Walker Kessler")) == "BLK"
    reason = next(b.reason for b in premium.badges(haliburton) if b.kind == "keystone")
    assert reason.startswith("Carries ") and "AST" in reason
    # The best free agent in this league is a steals specialist, so a steals carrier can be replaced.
    assert premium.keystone(player(league, "Dyson Daniels")) is None
    # About one per team, not every starter.
    keystones = [p for t in league.teams for p in t.roster if premium.keystone(p)]
    assert 5 <= len(keystones) <= 2 * len(league.teams)


def test_cheap_keystone_still_costs_something(league, premium):
    kessler = player(league, "Walker Kessler")
    assert premium.perceived.value(kessler) * 0.15 < KEYSTONE_MIN_DOLLARS
    assert premium.dollars(kessler) == pytest.approx(KEYSTONE_MIN_DOLLARS)


def test_premium_is_capped(league, premium):
    for t in league.teams:
        for p in t.roster:
            assert premium.rate(p) <= MAX_PREMIUM + 1e-9


def test_fresh_signing_fades(league, model):
    pv = PerceivedValue(league, model)
    base = OwnerPremium(league, model, pv)
    p = max(league.team(ME).roster, key=lambda q: 0 if base.is_attached(q) else pv.value(q))
    day = 86_400_000
    now = base.now_ms
    for team in league.teams:
        if p in team.roster:
            idx = team.roster.index(p)
            team.roster[idx] = dataclasses.replace(p, acquisition_type="TRADE", acquired_ms=now - 2 * day)
    try:
        fresh = OwnerPremium(league, model, pv, now_ms=now)
        q = next(x for x in league.team(ME).roster if x.id == p.id)
        assert "fresh_signing" in kinds(fresh, q)
        later = OwnerPremium(league, model, pv, now_ms=now + FRESH_DAYS * day)
        assert "fresh_signing" not in kinds(later, q)
        assert fresh.rate(q) > later.rate(q)
    finally:
        team = league.team(ME)
        team.roster[team.roster.index(q)] = p


def test_trade_block_is_parsed():
    assert _trade_block({"players": {"123": "ON_THE_BLOCK", "456": "UNTOUCHABLE"}}) == {
        123: "ON_THE_BLOCK",
        456: "UNTOUCHABLE",
    }
    assert _trade_block(None) == {} and _trade_block({}) == {}


def test_on_the_block_is_cheaper(league, model):
    pv = PerceivedValue(league, model)
    other = next(t for t in league.teams if t.id != ME)
    p = max(other.roster, key=pv.value)
    before = OwnerPremium(league, model, pv).rate(p)
    other.trade_block[p.id] = "ON_THE_BLOCK"
    try:
        after = OwnerPremium(league, model, pv)
        assert "on_the_block" in kinds(after, p)
        assert after.rate(p) < before
    finally:
        other.trade_block.pop(p.id)


def test_not_for_sale_players_are_never_targeted(league, model):
    me = next(i for i, t in enumerate(league.teams) if t.id == ME)
    pv = PerceivedValue(league, model)
    finder = TradeFinder(league, model, me, keep=set(), premium=OwnerPremium(league, model, pv))
    targets = {p.id for d in finder.candidates() for p in d.get}
    assert targets
    blocked = set(list(targets)[:3])
    finder = TradeFinder(league, model, me, keep=set(), premium=OwnerPremium(league, model, pv, blocked))
    assert not blocked & {p.id for d in finder.candidates() for p in d.get}


def test_their_badged_players_cost_the_premium(league, model, premium):
    me = next(i for i, t in enumerate(league.teams) if t.id == ME)
    finder = TradeFinder(league, model, me, keep=set(), premium=premium)
    deals = finder.candidates()
    badged = [d for d in deals if any(premium.rate(p) > 0 for p in d.get)]
    assert badged
    for d in badged:
        assert d.premium > 0
    assert all(d.premium == 0 for d in deals if all(premium.dollars(p) == 0 for p in d.get))


def test_my_attached_players_need_a_spectacular_deal(league, model, premium):
    me = next(i for i, t in enumerate(league.teams) if t.id == ME)
    finder = TradeFinder(league, model, me, keep=set(), premium=premium)
    bar = finder.spectacular(finder.candidates())
    picked = finder.queue()
    assert picked
    for d in picked:
        if any(premium.is_attached(p) for p in d.give):
            assert d.my_gain >= bar
            assert d.warnings


def _swap(league, p, **changes):
    """Replace player p on his team with a changed copy; returns the copy."""
    for team in league.teams:
        if p in team.roster:
            q = dataclasses.replace(p, **changes)
            team.roster[team.roster.index(p)] = q
            return team, q
    raise AssertionError(p.name)


def _restore(team, q, p):
    team.roster[team.roster.index(q)] = p


def _cheap_other_player(league, pv):
    other = next(t for t in league.teams if t.id != ME)
    return min((p for p in other.roster if p.espn_rank and p.espn_rank > 120), key=pv.value)


def test_a_breakout_is_hot_and_priced_at_current_form(league, model):
    # Darius Acuff Jr. on 2 Nov 2026: #138 before the season, #68 on ESPN after six games.
    pv = PerceivedValue(league, model)
    p = _cheap_other_player(league, pv)
    team, q = _swap(league, p, espn_rank=138, recent_rank=68, season_rank=68, games_played=6)
    try:
        premium = OwnerPremium(league, model, PerceivedValue(league, model))
        assert premium.is_hot(q) and "hot" in kinds(premium, q)
        reason = next(b.reason for b in premium.badges(q) if b.kind == "hot")
        assert "#68" in reason and "#138 before the season" in reason
        assert premium.price(q) == pytest.approx(premium.perceived.dollars_at_rank(68))
        assert premium.price(q) > premium.perceived.value(q)
    finally:
        _restore(team, q, p)


def test_not_hot_without_a_real_jump(league, model):
    pv = PerceivedValue(league, model)
    p = _cheap_other_player(league, pv)
    cases = [
        dict(espn_rank=8, recent_rank=2, season_rank=2, games_played=6),  # a star staying a star
        dict(espn_rank=138, recent_rank=68, season_rank=68, games_played=2),  # two games
        dict(espn_rank=138, recent_rank=100, season_rank=100, games_played=6),  # better, not a breakout
        dict(espn_rank=None, recent_rank=400, season_rank=400, games_played=6),  # outside the rostered pool
    ]
    for changes in cases:
        team, q = _swap(league, p, **changes)
        try:
            premium = OwnerPremium(league, model, PerceivedValue(league, model))
            assert not premium.is_hot(q), changes
        finally:
            _restore(team, q, p)


def test_their_hot_player_is_no_longer_a_throw_in(league, model):
    me = next(i for i, t in enumerate(league.teams) if t.id == ME)
    pv = PerceivedValue(league, model)
    finder = TradeFinder(league, model, me, keep=set(), premium=OwnerPremium(league, model, pv))
    every = finder.candidates()
    deals = [d for d in every if len(d.get) == 2]
    assert deals
    throw_in = min(deals[0].get, key=pv.value)
    team, q = _swap(league, throw_in, espn_rank=200, recent_rank=20, season_rank=20, games_played=6)
    try:
        pv2 = PerceivedValue(league, model)
        finder = TradeFinder(league, model, me, keep=set(), premium=OwnerPremium(league, model, pv2))
        after = [d for d in finder.candidates() if q.id in {p.id for p in d.get}]
        before = [d for d in every if throw_in.id in {p.id for p in d.get}]
        assert len(after) < len(before)
        assert all(d.premium > 0 for d in after)
    finally:
        _restore(team, q, p=throw_in)

import copy

import pytest

from app.analysis.model import CategoryModel
from app.analysis.perceived import PerceivedValue
from app.analysis.report import Analyzer, league_summary
from app.analysis.tiers import droppable
from app.analysis.tiers import droppable as no_trade_value
from app.analysis.trades import (
    MAX_VALUE_RATIO,
    MIN_THEIR_GAIN,
    MIN_VALUE_RATIO,
    VISION_MIN_GAMES,
    VISION_MIN_PLAYOFF_GAIN,
    VISION_MIN_SEASON_GAIN,
    TradeFinder,
    acceptance_label,
    acceptance_odds,
    mix_shapes,
    shape_odds,
)
from app.espn.parse import parse_league


def test_draft_prices_parsed(league):
    assert league.is_auction
    giannis = next(p for p in league.team(3).roster if p.name == "Giannis Antetokounmpo")
    assert league.draft_prices[giannis.id] == 62


def test_perceived_value_orders_stars_above_role_players(league):
    pv = PerceivedValue(league)
    roster = {p.name: p for p in league.team(3).roster}
    assert pv.value(roster["Jayson Tatum"]) > pv.value(roster["Jarrett Allen"]) > pv.value(roster["Davion Mitchell"])
    assert all(pv.value(p) >= 0 for p in roster.values())


@pytest.fixture(scope="module")
def snake_league(recorded):
    """The recorded league, drafted snake style, with ESPN's ADP for a few players."""
    raw = copy.deepcopy(recorded["league"])
    raw["settings"]["draftSettings"]["type"] = "SNAKE"
    for pk in raw["draftDetail"]["picks"]:
        pk.pop("bidAmount", None)
    adp = {p["id"]: 1.5 if p["fullName"] == "Davion Mitchell" else 150.0 for p in _rostered(raw)}
    return parse_league(raw, recorded["free_agents"], market_adp=adp)


def _rostered(raw):
    return [e["playerPoolEntry"]["player"] for t in raw["teams"] for e in t["roster"]["entries"]]


def test_snake_league_prices_picks_and_adp_not_auction_dollars(league, snake_league):
    assert not snake_league.is_auction and not snake_league.draft_prices
    assert snake_league.draft_rounds == 13
    pv = PerceivedValue(snake_league)
    mitchell = next(p for p in snake_league.team(3).roster if p.name == "Davion Mitchell")
    signals = {name: dollars for name, _, dollars in pv.signals(mitchell)}
    assert "avg_auction" not in signals
    assert signals["adp"] == pytest.approx(pv.dollars_at_rank(1.5))
    # An auction league ignores ADP, even when ESPN sends it.
    auction_mitchell = next(p for p in league.team(3).roster if p.name == "Davion Mitchell")
    assert "adp" not in {name for name, _, _ in PerceivedValue(league).signals(auction_mitchell)}


def test_round_values_only_in_snake_leagues(league, snake_league):
    assert league_summary(league).draft_type == "auction"
    assert league_summary(league).round_values is None
    summary = league_summary(snake_league)
    assert summary.draft_type == "snake"
    values = summary.round_values
    assert values is not None and len(values) == 13
    assert all(a >= b for a, b in zip(values, values[1:], strict=False))
    assert values[0] > values[-1]


def test_values_read_as_rounds_in_snake_leagues(league, snake_league):
    assert PerceivedValue(league).describe(12.4) == "$12"
    pv = PerceivedValue(snake_league)
    top = pv.round_values()[0]
    assert pv.describe(top + 5) == "a 1st-round pick"
    assert pv.describe(pv.round_values()[2]) == "a 3rd-round pick"
    lowest = min(v for v in pv.round_values() if v >= 0.5)
    assert pv.describe(lowest) != "a late-round pick"
    if lowest > 0.6:
        assert pv.describe(0.55) == "a late-round pick"
    assert pv.describe(0.1) == "waiver level"


def test_snake_trade_reasons_never_quote_auction_dollars(snake_league):
    model = CategoryModel(snake_league)
    me = next(i for i, t in enumerate(snake_league.teams) if t.id == 3)
    deals = TradeFinder(snake_league, model, me, keep=set()).candidates()
    reasons = [r for d in deals for r in d.reasons]
    assert reasons and not any("$" in r for r in reasons)


def test_dollars_at_rank_is_non_increasing(league):
    pv = PerceivedValue(league)
    values = [pv.dollars_at_rank(r) for r in range(1, 200, 5)]
    assert all(a >= b for a, b in zip(values, values[1:], strict=False))


def test_acceptance_odds_follow_value_and_need():
    assert acceptance_odds(1.2, 0.2) > acceptance_odds(0.9, 0.2)
    assert acceptance_odds(1.0, 0.2) > acceptance_odds(1.0, 0.03)
    assert acceptance_label(0.7) == "likely" and acceptance_label(0.45) == "possible"
    assert acceptance_label(0.2) == "long_shot"


def test_shape_odds_follow_real_offers():
    # At the same value: 1-for-1 and 1-for-2 as is, 2-for-2 a bit lower, and a 2-for-1 that makes
    # the other owner cut someone the lowest (scripts/trade_shape_backtest.py).
    p = 0.5
    assert shape_odds(p, 1, 1) == shape_odds(p, 1, 2) == pytest.approx(p)
    assert shape_odds(p, 2, 1) < shape_odds(p, 2, 2) < p
    assert 0.05 <= shape_odds(0.05, 2, 1) and shape_odds(0.9, 2, 2) <= 0.9


def test_two_for_one_odds_are_lower_than_the_same_deal_would_get_as_a_one_for_one(league):
    model = CategoryModel(league)
    me = next(i for i, t in enumerate(league.teams) if t.id == 3)
    finder = TradeFinder(league, model, me, keep=set(), mode="edge")
    deals = [d for d in finder.candidates() if len(d.give) == 2 and len(d.get) == 1]
    assert deals
    for d in deals:
        assert any("cut a player to fit two of yours" in r for r in d.reasons)
        assert d.acceptance <= shape_odds(0.9, 2, 1)


def test_two_for_two_ideas_are_real_two_for_twos(league):
    model = CategoryModel(league)
    me = next(i for i, t in enumerate(league.teams) if t.id == 3)
    finder = TradeFinder(league, model, me, keep=set(), mode="edge")
    deals = finder.candidates()
    pairs = [d for d in deals if len(d.give) == len(d.get) == 2]
    assert pairs
    mine = finder.teams[me].roster
    for d in pairs:
        assert d.my_drop is None and d.their_drop is None and d.my_add is None
        theirs = finder.teams[d.partner_idx].roster
        # No free-agent-level throw-ins: those deals are really 2-for-1s.
        assert not set(map(id, d.give)) & set(map(id, no_trade_value(mine)))
        assert not set(map(id, d.get)) & set(map(id, no_trade_value(theirs)))
        # Each one does more for me than any 1-for-1 inside it.
        inside = [finder._gains.get((d.partner_idx, g.id, t.id)) for g in d.give for t in d.get]
        assert all(g is None or d.my_gain > g for g in inside)
    assert not [d for d in finder.candidates(two_for_two=False) if len(d.give) == len(d.get) == 2]


def test_every_candidate_is_fair_on_paper_and_helps_both(league):
    model = CategoryModel(league)
    me = next(i for i, t in enumerate(league.teams) if t.id == 3)
    finder = TradeFinder(league, model, me, keep=set())
    deals = finder.candidates()
    assert deals
    for d in deals:
        ratio = d.give_value / d.get_value
        assert MIN_VALUE_RATIO <= ratio <= MAX_VALUE_RATIO
        assert d.their_gain >= MIN_THEIR_GAIN
        assert 1 <= len(d.give) <= 2 and 1 <= len(d.get) <= 2
        if len(d.get) > len(d.give):
            assert d.my_drop is not None and d.my_drop not in d.get
    assert any(len(d.give) == 2 for d in deals) or any(len(d.get) == 2 for d in deals)


def test_shortlist_limits_repeats(league):
    model = CategoryModel(league)
    me = next(i for i, t in enumerate(league.teams) if t.id == 3)
    finder = TradeFinder(league, model, me, keep=set())
    picked = finder.shortlist()
    partners = [d.partner_idx for d in picked]
    assert all(partners.count(p) <= 4 for p in set(partners))
    assert all(finder.pitch(d).endswith(".") for d in picked)


def test_package_value_has_a_star_premium(league):
    pv = PerceivedValue(league)
    roster = sorted(league.team(3).roster, key=pv.value, reverse=True)
    a, b = roster[3], roster[4]
    assert max(pv.value(a), pv.value(b)) < pv.package([a, b]) < pv.total([a, b])
    assert pv.package([a]) == pytest.approx(pv.value(a))


def test_their_gain_ignores_their_punts(league):
    model = CategoryModel(league)
    me = next(i for i, t in enumerate(league.teams) if t.id == 3)
    finder = TradeFinder(league, model, me, keep=set())
    for d in finder.candidates()[:50]:
        j = d.partner_idx
        base = finder.their_free_cats(j) if len(d.get) > len(d.give) else finder.base_cats[j]
        change = finder._avg_win(j, d.dists) - base
        assert d.their_gain == pytest.approx(change[~finder.punts[d.partner_idx]].sum())
        assert all(r.endswith(".") for r in d.reasons)


def test_struggling_teams_are_more_open():
    assert acceptance_odds(1.0, 0.1, struggling=True) > acceptance_odds(1.0, 0.1)


def test_draft_dollars_fade_as_games_are_played(league):
    import dataclasses

    model = CategoryModel(league)
    pv = PerceivedValue(league, model)
    p = max(league.team(3).roster, key=pv.value)
    draft = {"espn", "avg_auction", "paid"}

    def draft_share(player):
        sig = pv.signals(player)
        return sum(w for n, w, _ in sig if n in draft) / sum(w for _, w, _ in sig)

    early = dataclasses.replace(p, games_played=0, season_rank=None, recent_rank=None)
    late = dataclasses.replace(p, games_played=40, season_rank=5, recent_rank=5)
    assert draft_share(early) <= 0.55
    assert draft_share(late) <= 0.15
    assert {n for n, _, _ in pv.signals(late)} >= {"owned", "projection", "season", "recent"}


def test_edge_deals_win_on_paper_for_them_and_on_the_court_for_me(league):
    from app.analysis.trades import EDGE_MIN_MARGIN, EDGE_MIN_RATIO, EDGE_MIN_THEIR_GAIN

    model = CategoryModel(league)
    me = next(i for i, t in enumerate(league.teams) if t.id == 3)
    finder = TradeFinder(league, model, me, keep=set(), mode="edge")
    deals = finder.candidates()
    assert deals
    for d in deals:
        assert d.give_value / d.get_value >= EDGE_MIN_RATIO
        assert d.their_gain >= EDGE_MIN_THEIR_GAIN
        assert d.edge >= EDGE_MIN_MARGIN - 1e-9
        assert len(finder.rivals) <= 4 and finder.me_idx not in finder.rivals


def test_their_side_of_a_one_for_two_is_measured_against_their_free_pickup(league):
    """The spot a 1-for-2 opens on their roster takes the best free agent, which they could add today."""
    model = CategoryModel(league)
    me = next(i for i, t in enumerate(league.teams) if t.id == 3)
    finder = TradeFinder(league, model, me, keep=set(), mode="edge")
    deals = [d for d in finder.candidates() if len(d.get) > len(d.give)]
    assert deals
    for d in deals:
        free = finder.their_free_cats(d.partner_idx)
        assert d.their_gain == pytest.approx(finder.fit_gain(d.partner_idx, d.dists, free))
        assert d.their_gain <= finder.fit_gain(d.partner_idx, d.dists) + 1e-9


def test_two_for_one_is_measured_against_the_free_pickup(league):
    """The spot a 2-for-1 opens takes the best free agent, but that pickup needs no trade."""
    model = CategoryModel(league)
    me = next(i for i, t in enumerate(league.teams) if t.id == 3)
    finder = TradeFinder(league, model, me, keep=set())
    assert finder.free_move is not None
    add, drop = finder.free_move
    assert add is finder.filler and drop in droppable(league.teams[me].roster)
    assert finder.free_base > finder.base[me]
    # Open thresholds: the point is how the gain is measured, not which deals pass.
    j = (me + 1) % len(league.teams)
    mine, theirs = league.teams[me].roster, league.teams[j].roster
    loose = {"min_my_gain": float("-inf"), "min_their_gain": float("-inf"), "min_ratio": 0.0}
    filled = [
        d
        for pair in zip(mine, mine[1:], strict=False)
        for get in theirs
        if (d := finder._evaluate(j, list(pair), [get], **loose)) is not None
    ]
    assert filled
    deals = finder.candidates() + filled
    for d in filled:
        assert d.my_add is finder.filler
        assert d.vs_pickup == finder.free_move
        after = model.strength(me, d.dists)
        assert d.my_gain == pytest.approx(after - finder.free_base)
        assert any(add.name in r and "needs no trade" in r for r in d.reasons)
    assert all(d.vs_pickup is None for d in deals if d.my_add is None)


def test_trade_odds_start_from_the_free_pickup_when_the_deal_uses_it(league):
    a = Analyzer(league, 3)
    today = a.playoff_pct(a.dists)
    ideas = a.trades("fair") + a.trades("edge")
    assert ideas
    for t in ideas:
        if t.my_add is None:
            assert t.playoff_pct_before == pytest.approx(today, abs=1e-3)
        else:
            assert t.playoff_pct_before > today
        assert t.playoff_pct_after >= t.playoff_pct_before


def _slow_starter(league, name="Walker Kessler"):
    """A copy of the league where a rival's player has had a bad first 10 games, and owners see it."""
    lg = copy.deepcopy(league)
    model = CategoryModel(lg)
    me = next(i for i, t in enumerate(lg.teams) if t.id == 3)
    star = next(p for t in lg.teams for p in t.roster if p.name == name)
    star.games_played = 10
    star.season_line = {k: 0.6 * v for k, v in star.per_game.items()}
    star.season_rank = star.recent_rank = 150
    return lg, model, me, star


def test_vision_signal_is_a_slow_start_the_projection_expects_to_fade(league):
    lg, model, me, star = _slow_starter(league)
    finder = TradeFinder(lg, model, me, keep=set(), mode="vision")
    assert finder.vision_signals(star) and "Slow start" in finder.vision_signals(star)[0]
    # Playing to his projection, too few games, or out: no signal.
    steady = copy.copy(star)
    steady.season_line = dict(star.per_game)
    assert finder.vision_signals(steady) == []
    early = copy.copy(star)
    early.games_played = VISION_MIN_GAMES - 1
    assert finder.vision_signals(early) == []
    hurt = copy.copy(star)
    hurt.injury_status = "OUT"
    assert finder.vision_signals(hurt) == []


def test_vision_deals_buy_the_target_and_pay_in_the_playoffs(league):
    lg, model, me, star = _slow_starter(league)
    finder = TradeFinder(lg, model, me, keep=set(), mode="vision")
    deals = finder.candidates()
    assert deals
    for d in deals:
        assert d.get == [star] and d.vision
        assert all(p.available for p in d.give) and 1 <= len(d.give) <= 2
        assert d.playoff_gain >= VISION_MIN_PLAYOFF_GAIN
        assert d.my_gain >= VISION_MIN_SEASON_GAIN
    ranked = finder.shortlist()
    keys = [finder.rank_key(d) for d in ranked]
    assert keys == sorted(keys, reverse=True)


def test_uneven_trades_are_a_small_part_of_the_list():
    ranked = ["1-2a", "1-2b", "1-2c", "2-2a", "1-1a", "2-1a", "1-1b"]
    uneven = lambda s: s[0] != s[2]  # noqa: E731
    # 10 slots: at most 2 uneven, in rank order; no padding with more uneven ones.
    assert mix_shapes(ranked, 10, uneven) == ["1-2a", "1-2b", "2-2a", "1-1a", "1-1b"]
    assert mix_shapes(ranked, 3, uneven) == ["1-2a", "2-2a", "1-1a"]
    assert mix_shapes(["2-2a", "1-1a", "1-1b"], 2, uneven) == ["2-2a", "1-1a"]

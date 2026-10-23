import pytest

from app.analysis.model import CategoryModel
from app.analysis.perceived import PerceivedValue
from app.analysis.trades import (
    MAX_VALUE_RATIO,
    MIN_THEIR_GAIN,
    MIN_VALUE_RATIO,
    TradeFinder,
    acceptance_label,
    acceptance_odds,
)


def test_draft_prices_parsed(league):
    assert league.is_auction
    giannis = next(p for p in league.team(3).roster if p.name == "Giannis Antetokounmpo")
    assert league.draft_prices[giannis.id] == 62


def test_perceived_value_orders_stars_above_role_players(league):
    pv = PerceivedValue(league)
    roster = {p.name: p for p in league.team(3).roster}
    assert pv.value(roster["Jayson Tatum"]) > pv.value(roster["Jarrett Allen"]) > pv.value(roster["Davion Mitchell"])
    assert all(pv.value(p) >= 0 for p in roster.values())


def test_dollars_at_rank_is_non_increasing(league):
    pv = PerceivedValue(league)
    values = [pv.dollars_at_rank(r) for r in range(1, 200, 5)]
    assert all(a >= b for a, b in zip(values, values[1:], strict=False))


def test_acceptance_odds_follow_value_and_need():
    assert acceptance_odds(1.2, 0.2, False) > acceptance_odds(0.9, 0.2, False)
    assert acceptance_odds(1.0, 0.2, False) > acceptance_odds(1.0, 0.03, False)
    assert acceptance_odds(1.0, 0.1, True) < acceptance_odds(1.0, 0.1, False)
    assert acceptance_label(0.7) == "likely" and acceptance_label(0.45) == "possible"
    assert acceptance_label(0.2) == "long_shot"


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
        change = finder._avg_win(d.partner_idx, d.dists) - finder.base_cats[d.partner_idx]
        assert d.their_gain == pytest.approx(change[~finder.punts[d.partner_idx]].sum())
        assert all(r.endswith(".") for r in d.reasons)


def test_struggling_teams_are_more_open():
    assert acceptance_odds(1.0, 0.1, False, struggling=True) > acceptance_odds(1.0, 0.1, False)


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

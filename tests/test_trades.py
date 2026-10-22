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

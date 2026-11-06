import numpy as np
import pytest

import app.analysis.sim as sim
from app.analysis.model import CategoryModel, norm_cdf
from app.analysis.report import PAPER_MAX_PARTNER_DROP, Analyzer, i_gain, partner_gains, worth_a_try
from app.analysis.sim import SeasonSim, category_links, simulate_season
from app.analysis.trades import SCREEN_MIN_THEIR_GAIN


def test_norm_cdf_matches_known_values():
    x = np.array([-1.96, 0.0, 1.0, 2.5])
    assert np.allclose(norm_cdf(x), [0.0249979, 0.5, 0.8413447, 0.9937903], atol=1e-6)


def test_win_probabilities_are_symmetric(league):
    model = CategoryModel(league)
    dists = [model.dist(t.roster) for t in league.teams]
    m = model.win_matrix(dists)
    assert np.allclose(m + m.transpose(1, 0, 2), 1.0, atol=1e-6)
    assert np.allclose(model.cat_win_prob(dists[0], dists[1]), m[0, 1])


def test_better_lineup_is_stronger(league):
    model = CategoryModel(league)
    me = league.team(3)
    dists = [model.dist(t.roster) for t in league.teams]
    idx = league.teams.index(me)
    weaker = list(dists)
    best = max(me.roster, key=model.value)
    weaker[idx] = model.dist([p for p in me.roster if p.id != best.id])
    assert model.strength(idx, weaker) < model.strength(idx, dists)


def test_season_sim_conserves_games(league):
    model = CategoryModel(league)
    dists = [model.dist(t.roster) for t in league.teams]
    res = simulate_season(league, dists, model.reverse, n_sims=500)
    cats = len(league.categories)
    undecided = len({m.period for m in league.schedule if m.winner == "UNDECIDED"})
    played = sum(t.wins + t.losses + t.ties for t in league.teams)
    total = res.wins.sum() + res.losses.sum() + res.ties.sum()
    assert np.isclose(total, played + undecided * len(league.teams) * cats)
    assert np.isclose(res.playoff_pct.sum(), league.playoff_teams)
    assert np.isclose(res.first_seed_pct.sum(), 1.0)


def test_season_sim_is_reproducible(league):
    model = CategoryModel(league)
    dists = [model.dist(t.roster) for t in league.teams]
    a = simulate_season(league, dists, model.reverse, n_sims=300)
    b = simulate_season(league, dists, model.reverse, n_sims=300)
    assert np.array_equal(a.playoff_pct, b.playoff_pct)


def test_what_if_matches_a_full_simulation(league):
    model = CategoryModel(league)
    dists = [model.dist(t.roster) for t in league.teams]
    base = SeasonSim(league, dists, model.reverse, n_sims=400)
    full = simulate_season(league, dists, model.reverse, n_sims=400)
    assert np.allclose(base.base.playoff_pct, full.playoff_pct, atol=0.005)
    # Swap the two teams' best players: only their matchups are re-scored.
    a, b = league.teams[0], league.teams[1]
    star_a, star_b = max(a.roster, key=model.value), max(b.roster, key=model.value)
    changed = list(dists)
    changed[0] = model.dist([p for p in a.roster if p is not star_a] + [star_b])
    changed[1] = model.dist([p for p in b.roster if p is not star_b] + [star_a])
    fast = base.what_if({0, 1}, changed)
    slow = simulate_season(league, changed, model.reverse, n_sims=400)
    # The what-if keeps the baseline league average; the full run recomputes it, which shifts
    # every team a hair. Continuous draws ignore a shared shift, but whole-number counting stats
    # don't quite, so a few of the 400 seasons round the other way.
    assert np.allclose(fast.playoff_pct, slow.playoff_pct, atol=0.015)
    assert np.allclose(fast.wins, slow.wins, atol=0.1)
    assert base.what_if(set(), changed) is base.base


def test_category_links_are_a_valid_correlation_matrix():
    names = ["PTS", "REB", "AST", "STL", "BLK", "3PM", "TO", "FG%", "FT%", "OREB"]
    m = category_links(names)
    assert np.allclose(m, m.T) and np.allclose(np.diag(m), 1.0)
    assert np.linalg.eigvalsh(m).min() > 0
    assert m[0, 1] > 0.5  # points and rebounds move together
    assert abs(m[7, 8]) < 0.05 and abs(m[0, 9]) < 0.05  # FG% vs FT%, and a pair with no link


def test_drift_and_links_spread_the_simulated_records(league, monkeypatch):
    model = CategoryModel(league)
    dists = [model.dist(t.roster) for t in league.teams]
    linked = simulate_season(league, dists, model.reverse, n_sims=1000)
    monkeypatch.setattr(sim, "TEAM_DRIFT", 0.0)
    monkeypatch.setattr(sim, "category_links", lambda names: np.eye(len(names)))
    independent = simulate_season(league, dists, model.reverse, n_sims=1000)
    # Less certain: the favourite's odds come down, the long shots' go up.
    assert linked.playoff_pct.max() < independent.playoff_pct.max()
    assert linked.playoff_pct.min() > independent.playoff_pct.min()


def test_every_trade_idea_raises_the_partners_odds(league):
    report = Analyzer(league, 3).report()
    assert report.trades
    for t in report.trades:
        assert t.partner_playoff_pct_after - t.partner_playoff_pct_before >= 0.0099


def test_partner_gains_needs_a_visible_rise():
    assert not partner_gains(0.04, 0.04)
    assert not partner_gains(0.40, 0.405)
    assert partner_gains(0.40, 0.41)
    assert partner_gains(0.99, 0.99)  # a near-lock can't rise a point: it just must not drop
    assert not partner_gains(0.99, 0.985)


def test_worth_a_try_is_a_paper_win_that_costs_them_little():
    from types import SimpleNamespace

    paper = SimpleNamespace(give_value=12.0, get_value=10.0)  # they get more trade value than they send
    short = SimpleNamespace(give_value=9.0, get_value=10.0)
    assert worth_a_try(short, 0.40, 0.41)  # their odds go up: a reason to say yes on its own
    assert worth_a_try(paper, 0.40, 0.40 - PAPER_MAX_PARTNER_DROP)
    assert not worth_a_try(paper, 0.40, 0.40 - PAPER_MAX_PARTNER_DROP - 0.005)  # sinks their team
    assert not worth_a_try(short, 0.40, 0.39)  # no paper win and their odds drop


def test_edge_trades_win_on_paper_and_cost_them_little(league):
    for t in Analyzer(league, 3).trades("edge"):
        assert t.playoff_pct_after - t.playoff_pct_before >= 0.0099
        partner = t.partner_playoff_pct_after - t.partner_playoff_pct_before
        assert partner >= 0.0099 or (
            t.value_they_get >= t.value_they_give - 0.05 and partner >= -PAPER_MAX_PARTNER_DROP - 0.001
        )


def test_my_side_needs_a_visible_rise_too():
    assert not i_gain(0.52, 0.52)  # 52% -> 52% ±0 is no reason to trade
    assert not i_gain(0.52, 0.525)
    assert i_gain(0.52, 0.53)


def test_teams_drift_back_toward_the_average(league, monkeypatch):
    model = CategoryModel(league)
    dists = [model.dist(t.roster) for t in league.teams]
    drifting = simulate_season(league, dists, model.reverse, n_sims=1000)
    fixed_sim = SeasonSim(league, dists, model.reverse, n_sims=1000)
    monkeypatch.setattr(sim, "REVERSION", 0.0)
    fixed = simulate_season(league, dists, model.reverse, n_sims=1000)
    # Today's best team is less of a lock and today's worst less hopeless when teams can change.
    assert drifting.playoff_pct.max() < fixed.playoff_pct.max()
    assert drifting.playoff_pct.min() > fixed.playoff_pct.min()
    # The fast what-if simulation drifts the same way.
    assert np.allclose(fixed_sim.base.playoff_pct, drifting.playoff_pct, atol=0.005)


def test_report_sections(league):
    keep = [p.id for p in league.team(3).roster if p.name == "Kyshawn George"]
    report = Analyzer(league, 3, keep_player_ids=keep).report()
    assert {c.category for c in report.categories} == {c.name for c in league.categories}
    assert len(report.builds) == 10
    assert len(report.season) == 10
    assert report.trades and report.pickups
    for t in report.trades:
        assert t.my_cats_gain > 0
        # Their average week may stay about flat; the playoff-odds check is what they must pass.
        assert t.their_cats_gain >= SCREEN_MIN_THEIR_GAIN
        assert t.partner_playoff_pct_after - t.partner_playoff_pct_before >= 0.0099
        assert t.playoff_pct_after - t.playoff_pct_before >= 0.0099
        assert all(p.id not in keep for p in t.give)
        assert t.pitch
    assert all(p.drop.id not in keep for p in report.pickups)
    assert all(not p.add.injury_status == "OUT" for p in report.pickups)


def test_banked_stats_count_in_the_week_being_played(league):
    from dataclasses import replace

    a = Analyzer(league, 3)
    pdists = a.period_dists()
    now = league.current_period
    game = next(m for m in league.schedule if m.period == now and m.winner == "UNDECIDED")
    i = next(k for k, t in enumerate(league.teams) if t.id == game.home)
    # The home team has banked a crushing lead in every counting stat and shot 60% / 90%.
    lead = {c.stat_id: 500.0 for c in league.categories if c.ratio is None and not c.reverse}
    lead.update({c.stat_id: 0.0 for c in league.categories if c.reverse})
    lead.update({13: 300.0, 14: 500.0, 15: 180.0, 16: 200.0})
    banked = replace(league, schedule=[replace(m, home_totals=lead) if m is game else m for m in league.schedule])
    before = simulate_season(league, a.dists, a.model.reverse, n_sims=1000, period_dists=pdists)
    after = simulate_season(banked, a.dists, a.model.reverse, n_sims=1000, period_dists=pdists)
    j = next(k for k, t in enumerate(league.teams) if t.id == game.away)
    # It was winning about 6 of 9 this week; with the lead it sweeps.
    assert after.wins[i] - before.wins[i] > 1.5
    assert before.wins[j] - after.wins[j] > 1.5
    assert after.playoff_pct[i] > before.playoff_pct[i]
    # Without a per-period projection the week is a full average week, so banked stats are not added.
    flat = simulate_season(banked, a.dists, a.model.reverse, n_sims=1000)
    plain = simulate_season(league, a.dists, a.model.reverse, n_sims=1000)
    assert np.allclose(flat.wins, plain.wins)


def test_playoff_weeks_count_a_player_back_by_then(league):
    from dataclasses import replace as dc_replace

    a = Analyzer(league, 3)
    assert a.playoff_periods() and min(a.playoff_periods()) > league.regular_season_periods
    me = a.teams[a.idx]
    star = max(me.roster, key=a.model.value)
    hurt = dc_replace(star, id=-star.id, injury_status="OUT", games_out=20.0)  # new id: models cache by id
    roster = [hurt if p is star else p for p in me.roster]
    b = Analyzer(league, 3)
    season_drop = b.model.strength(b.idx, b.dists) - b.model.strength(
        b.idx, [b.model.dist(roster) if i == b.idx else d for i, d in enumerate(b.dists)]
    )
    playoff_drop = b.playoff_strength(b.idx) - b.playoff_strength(b.idx, {b.idx: roster})
    # Out for 20 games costs regular-season weeks, not the playoff weeks in March.
    assert season_drop > 0.01
    assert abs(playoff_drop) < 1e-9
    assert all(r.playoff_week_cats is not None for r in a.season())


def test_lineup_caches_stay_bounded(league, monkeypatch):
    """A league-wide trade scan tries huge numbers of rosters; the caches must not keep them all."""
    import app.analysis.model as model_mod

    rosters = [t.roster[:k] + t.roster[k + 1 :] for t in league.teams for k in range(len(t.roster))]
    full = CategoryModel(league)
    expected = [full.dist(r) for r in rosters]
    assert len(full._fills) > 20 and len(full._dists) > 3  # so the caps below are hit
    monkeypatch.setattr(model_mod, "FILL_CACHE_SIZE", 20)
    monkeypatch.setattr(model_mod, "DIST_CACHE_SIZE", 3)
    small = CategoryModel(league)
    for r, want in zip(rosters, expected, strict=True):
        got = small.dist(r)
        assert np.allclose(got.mu, want.mu) and np.allclose(got.sd, want.sd)
        assert len(small._fills) <= 20 and len(small._dists) <= 3


def test_trades_simulate_down_the_queue_within_budget(league, monkeypatch):
    from app.analysis import report as report_module

    monkeypatch.setitem(report_module.TRADE_SIMS, "fair", 3)
    analyzer = Analyzer(league, 3)
    calls = []
    real = analyzer.deal_odds
    monkeypatch.setattr(analyzer, "deal_odds", lambda d: calls.append(d) or real(d))
    analyzer.trades("fair")
    assert len(calls) <= 3


def test_season_sim_projects_category_ties(league):
    """Counting stats are whole numbers, so some categories end level, about as often as in real weeks."""
    model = CategoryModel(league)
    dists = [model.dist(t.roster) for t in league.teams]
    res = simulate_season(league, dists, model.reverse, n_sims=400)
    left = [m for m in league.schedule if m.winner == "UNDECIDED"]
    weeks = {t.id: sum(t.id in (m.home, m.away) for m in left) for t in league.teams}
    n_cats = len(league.categories)
    for i, t in enumerate(league.teams):
        new = res.wins[i] + res.losses[i] + res.ties[i] - (t.wins + t.losses + t.ties)
        assert new == pytest.approx(weeks[t.id] * n_cats)
    # League A's real weeks, 2022-2027: 0.13 tied categories a matchup.
    per_matchup = (res.ties.sum() - sum(t.ties for t in league.teams)) / 2 / len(left)
    assert 0.05 < per_matchup < 0.25


def test_trade_screen_reports_progress_while_pairing_players(league):
    """Pairing players with every team is half the step: the bar moves before the first simulation."""
    from app.analysis.report import FAIR_SPAN, PAIRING_SHARE

    seen: list[float] = []
    analyzer = Analyzer(league, 3, progress=lambda stage, frac: seen.append(frac))
    analyzer.trades(span=FAIR_SPAN)
    _, lo, hi = FAIR_SPAN
    mid = lo + (hi - lo) * PAIRING_SHARE
    pairing = [f for f in seen if lo < f <= mid]
    assert len(pairing) >= len(league.teams) - 2  # one step per partner team
    assert seen == sorted(seen)

import numpy as np

import app.analysis.sim as sim
from app.analysis.model import CategoryModel, norm_cdf
from app.analysis.report import Analyzer
from app.analysis.sim import category_links, simulate_season


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


def test_report_sections(league):
    keep = [p.id for p in league.team(3).roster if p.name == "Kyshawn George"]
    report = Analyzer(league, 3, keep_player_ids=keep).report()
    assert {c.category for c in report.categories} == {c.name for c in league.categories}
    assert len(report.builds) == 10
    assert len(report.season) == 10
    assert report.trades and report.pickups
    for t in report.trades:
        assert t.my_cats_gain > 0
        assert t.their_cats_gain > 0
        assert t.playoff_pct_after >= t.playoff_pct_before
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

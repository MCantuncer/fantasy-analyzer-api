"""Monte Carlo season simulation over the remaining regular-season schedule."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.analysis.model import Dist
from app.espn.parse import League

EACH_CATEGORY = "H2H_CATEGORY"


@dataclass
class SeasonResult:
    team_ids: list[int]
    wins: np.ndarray  # (T,) mean final wins
    losses: np.ndarray
    ties: np.ndarray
    playoff_pct: np.ndarray  # (T,) 0..1
    first_seed_pct: np.ndarray
    mean_rank: np.ndarray


def simulate_season(
    league: League,
    dists: list[Dist],
    reverse: np.ndarray,
    n_sims: int = 4000,
    seed: int = 7,
) -> SeasonResult:
    """Simulate every undecided regular-season matchup.

    Uses a fixed seed so that two runs differing only in one roster share the same
    random draws, which makes "before vs after" comparisons much less noisy.
    """
    rng = np.random.default_rng(seed)
    ids = [t.id for t in league.teams]
    index = {tid: i for i, tid in enumerate(ids)}
    n_teams = len(ids)
    mu = np.array([d.mu for d in dists])
    sd = np.array([d.sd for d in dists])
    sign = np.where(reverse, -1.0, 1.0)

    wins = np.tile(np.array([t.wins for t in league.teams], float), (n_sims, 1))
    losses = np.tile(np.array([t.losses for t in league.teams], float), (n_sims, 1))
    ties = np.tile(np.array([t.ties for t in league.teams], float), (n_sims, 1))
    each_cat = league.scoring_type == EACH_CATEGORY

    periods = sorted({m.period for m in league.schedule if m.winner == "UNDECIDED"})
    for period in periods:
        games = [m for m in league.schedule if m.period == period and m.winner == "UNDECIDED"]
        if not games:
            continue
        draws = rng.normal(mu, sd, size=(n_sims, n_teams, mu.shape[1])) * sign
        for m in games:
            h, a = index.get(m.home), index.get(m.away)
            if h is None or a is None:
                continue
            h_cats = (draws[:, h, :] > draws[:, a, :]).sum(1)
            a_cats = (draws[:, a, :] > draws[:, h, :]).sum(1)
            if each_cat:
                wins[:, h] += h_cats
                losses[:, h] += a_cats
                wins[:, a] += a_cats
                losses[:, a] += h_cats
            else:
                hw, aw, tie = h_cats > a_cats, a_cats > h_cats, h_cats == a_cats
                wins[:, h] += hw
                losses[:, h] += aw
                wins[:, a] += aw
                losses[:, a] += hw
                ties[:, h] += tie
                ties[:, a] += tie

    games_played = np.maximum(wins + losses + ties, 1.0)
    pct = (wins + 0.5 * ties) / games_played
    # Random tie-breaker; ESPN's real tie-breakers are not modelled.
    order = np.argsort(-(pct + rng.uniform(0, 1e-6, size=pct.shape)), axis=1)
    rank = np.empty_like(order)
    rank[np.arange(n_sims)[:, None], order] = np.arange(n_teams)[None, :]
    playoff_teams = min(league.playoff_teams, n_teams)

    return SeasonResult(
        team_ids=ids,
        wins=wins.mean(0),
        losses=losses.mean(0),
        ties=ties.mean(0),
        playoff_pct=(rank < playoff_teams).mean(0),
        first_seed_pct=(rank == 0).mean(0),
        mean_rank=rank.mean(0) + 1,
    )

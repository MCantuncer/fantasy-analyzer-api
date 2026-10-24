"""Monte Carlo season simulation over the remaining regular-season schedule."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.analysis.model import Dist
from app.espn.parse import League

EACH_CATEGORY = "H2H_CATEGORY"

# How a team's weekly surprises line up across categories, measured on League A's 2025 and
# 2026 seasons (actual minus projected, in projected sds). Counting stats move together
# because they all follow games played; FG% and FT% barely do. Pairs not listed are 0.
CATEGORY_LINKS = {
    ("PTS", "REB"): 0.66,
    ("PTS", "AST"): 0.60,
    ("PTS", "STL"): 0.46,
    ("PTS", "BLK"): 0.37,
    ("PTS", "3PM"): 0.67,
    ("PTS", "TO"): 0.55,
    ("PTS", "FG%"): 0.31,
    ("PTS", "FT%"): 0.18,
    ("REB", "AST"): 0.49,
    ("REB", "STL"): 0.40,
    ("REB", "BLK"): 0.42,
    ("REB", "3PM"): 0.36,
    ("REB", "TO"): 0.50,
    ("REB", "FG%"): 0.13,
    ("AST", "STL"): 0.35,
    ("AST", "BLK"): 0.26,
    ("AST", "3PM"): 0.42,
    ("AST", "TO"): 0.54,
    ("AST", "FT%"): 0.13,
    ("STL", "BLK"): 0.25,
    ("STL", "3PM"): 0.29,
    ("STL", "TO"): 0.35,
    ("BLK", "3PM"): 0.22,
    ("BLK", "TO"): 0.23,
    ("3PM", "TO"): 0.39,
    ("3PM", "FG%"): 0.13,
    ("3PM", "FT%"): 0.10,
    ("TO", "FT%"): 0.13,
}
# Injuries, trades and role changes last: each simulated season gives every team a shock
# per category, in sds of one week, that holds for the rest of that season.
TEAM_DRIFT = 0.5


def category_links(names: list[str]) -> np.ndarray:
    """Correlation matrix for these categories, from CATEGORY_LINKS."""
    n = len(names)
    m = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            m[i, j] = m[j, i] = CATEGORY_LINKS.get((names[i], names[j]), CATEGORY_LINKS.get((names[j], names[i]), 0.0))
    # Keep it a valid correlation matrix whatever categories a league uses.
    vals, vecs = np.linalg.eigh(m)
    m = vecs @ np.diag(np.clip(vals, 1e-6, None)) @ vecs.T
    d = np.sqrt(np.diag(m))
    return m / d[:, None] / d[None, :]


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
    period_dists: dict[int, list[Dist]] | None = None,
) -> SeasonResult:
    """Simulate every undecided regular-season matchup.

    `dists` is each team's average week; `period_dists`, when given, has each team's
    projection for a specific matchup period (its real games and who is injured then).

    Categories are drawn together (CATEGORY_LINKS), and each team carries a lasting
    shock per category for the whole simulated season (TEAM_DRIFT): without them the
    simulated records spread too little and the odds come out too sure.

    Uses a fixed seed so that two runs differing only in one roster share the same
    random draws, which makes "before vs after" comparisons much less noisy.
    """
    rng = np.random.default_rng(seed)
    ids = [t.id for t in league.teams]
    index = {tid: i for i, tid in enumerate(ids)}
    n_teams = len(ids)
    sign = np.where(reverse, -1.0, 1.0)

    wins = np.tile(np.array([t.wins for t in league.teams], float), (n_sims, 1))
    losses = np.tile(np.array([t.losses for t in league.teams], float), (n_sims, 1))
    ties = np.tile(np.array([t.ties for t in league.teams], float), (n_sims, 1))
    each_cat = league.scoring_type == EACH_CATEGORY
    chol = np.linalg.cholesky(category_links([c.name for c in league.categories]))
    drift = rng.normal(0.0, TEAM_DRIFT, size=(n_sims, n_teams, len(league.categories)))

    periods = sorted({m.period for m in league.schedule if m.winner == "UNDECIDED"})
    for period in periods:
        games = [m for m in league.schedule if m.period == period and m.winner == "UNDECIDED"]
        if not games:
            continue
        week = (period_dists or {}).get(period, dists)
        mu = np.array([d.mu for d in week])
        sd = np.array([d.sd for d in week])
        z = rng.standard_normal(size=(n_sims, n_teams, mu.shape[1])) @ chol.T
        draws = (mu + sd * (z + drift)) * sign
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

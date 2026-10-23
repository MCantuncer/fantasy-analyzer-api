"""Weekly category model: what a lineup produces in a week and how often it wins each category.

Each team's weekly total in a category is modelled as a normal distribution.
Counting stats use an over-dispersed Poisson variance per game plus schedule
noise. Percentage stats are makes over attempts with binomial variance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.espn.parse import Category, League, Player

GAMES_PER_WEEK = 3.4
# Per-game variance as a multiple of the per-game mean, by stat id.
DISPERSION = {0: 7.0, 1: 1.3, 2: 1.0, 3: 2.0, 4: 1.5, 5: 2.0, 6: 2.5, 11: 1.2, 17: 1.3}
DEFAULT_DISPERSION = 1.5
SCHEDULE_CV = 0.08  # weekly games played varies by about 8%
RATIO_OVERDISPERSION = 1.5


def norm_cdf(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF (Abramowitz-Stegun 7.1.26, error < 1.5e-7)."""
    z = np.abs(x) / np.sqrt(2.0)
    t = 1.0 / (1.0 + 0.3275911 * z)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    erf = 1.0 - poly * np.exp(-z * z)
    return 0.5 * (1.0 + np.sign(x) * erf)


@dataclass
class Dist:
    mu: np.ndarray  # (C,) weekly mean per category (percentages as fractions)
    sd: np.ndarray  # (C,)


class CategoryModel:
    def __init__(self, league: League, games_per_week: float = GAMES_PER_WEEK):
        self.league = league
        self.cats: list[Category] = league.categories
        self.reverse = np.array([c.reverse for c in self.cats])
        self.is_ratio = np.array([c.ratio is not None for c in self.cats])
        self.g = games_per_week
        self.n_starters = league.starter_slots or 10
        self._rows: dict[int, np.ndarray] = {}
        self._values: dict[int, float] = {}
        pool = [p for t in league.teams for p in t.roster]
        self._fit_values(pool)

    # ---- per-player weekly contribution -------------------------------------------------

    def row(self, p: Player) -> np.ndarray:
        """(3, C): makes-or-total, attempts, variance contribution for one week."""
        cached = self._rows.get(p.id)
        if cached is not None:
            return cached
        r = np.zeros((3, len(self.cats)))
        for i, c in enumerate(self.cats):
            if c.ratio:
                m, a = c.ratio
                r[0, i] = self.g * p.per_game.get(m, 0.0)
                r[1, i] = self.g * p.per_game.get(a, 0.0)
            else:
                avg = max(p.per_game.get(c.stat_id, 0.0), 0.0)
                r[0, i] = self.g * avg
                r[2, i] = self.g * DISPERSION.get(c.stat_id, DEFAULT_DISPERSION) * avg
        self._rows[p.id] = r
        return r

    # ---- generic player value (z-scores), used for lineups and "on paper" checks -------

    def _fit_values(self, pool: list[Player]) -> None:
        rows = np.array([self.row(p) for p in pool]) if pool else np.zeros((1, 3, len(self.cats)))
        totals = rows[:, 0, :]
        atts = rows[:, 1, :]
        with np.errstate(invalid="ignore", divide="ignore"):
            lg_pct = np.where(atts.sum(0) > 0, totals.sum(0) / atts.sum(0), 0.0)
        impact = np.where(self.is_ratio, totals - lg_pct * atts, totals)
        self._lg_pct = lg_pct
        self._z_mean = impact.mean(0)
        self._z_sd = impact.std(0) + 1e-9

    def z_scores(self, p: Player) -> np.ndarray:
        r = self.row(p)
        impact = np.where(self.is_ratio, r[0] - self._lg_pct * r[1], r[0])
        z = (impact - self._z_mean) / self._z_sd
        return np.where(self.reverse, -z, z)

    def value(self, p: Player) -> float:
        v = self._values.get(p.id)
        if v is None:
            v = self._values[p.id] = float(self.z_scores(p).sum())
        return v

    # ---- lineups and team distributions -------------------------------------------------

    def lineup(self, roster: list[Player]) -> list[Player]:
        healthy = [p for p in roster if p.available]
        return sorted(healthy, key=self.value, reverse=True)[: self.n_starters]

    def dist(self, roster: list[Player]) -> Dist:
        lineup = self.lineup(roster)
        if not lineup:
            return Dist(np.zeros(len(self.cats)), np.ones(len(self.cats)))
        s = np.sum([self.row(p) for p in lineup], axis=0)
        total, att, var = s
        with np.errstate(invalid="ignore", divide="ignore"):
            pct = np.where(att > 0, total / np.maximum(att, 1e-9), 0.0)
            ratio_sd = np.sqrt(RATIO_OVERDISPERSION * pct * (1 - pct) / np.maximum(att, 1.0))
        count_sd = np.sqrt(var + (SCHEDULE_CV * total) ** 2) + 1e-6
        mu = np.where(self.is_ratio, pct, total)
        sd = np.where(self.is_ratio, ratio_sd, count_sd)
        return Dist(mu, sd)

    # ---- win probabilities --------------------------------------------------------------

    def cat_win_prob(self, a: Dist, b: Dist) -> np.ndarray:
        """(C,) probability team a beats team b in each category in one week."""
        p = norm_cdf((a.mu - b.mu) / np.sqrt(a.sd**2 + b.sd**2))
        return np.where(self.reverse, 1.0 - p, p)

    def win_matrix(self, dists: list[Dist]) -> np.ndarray:
        """(T, T, C) category win probabilities for every pair of teams."""
        mu = np.array([d.mu for d in dists])
        sd = np.array([d.sd for d in dists])
        diff = mu[:, None, :] - mu[None, :, :]
        scale = np.sqrt(sd[:, None, :] ** 2 + sd[None, :, :] ** 2)
        p = norm_cdf(diff / scale)
        return np.where(self.reverse, 1.0 - p, p)

    def strength(self, idx: int, dists: list[Dist]) -> float:
        """Expected categories won per week against an average opponent in the league."""
        return float(self.avg_cat_win(idx, dists).sum())

    def avg_cat_win(self, idx: int, dists: list[Dist]) -> np.ndarray:
        """(C,) average weekly win probability per category against every other team."""
        me = dists[idx]
        mu = np.array([d.mu for j, d in enumerate(dists) if j != idx])
        sd = np.array([d.sd for j, d in enumerate(dists) if j != idx])
        p = norm_cdf((me.mu - mu) / np.sqrt(me.sd**2 + sd**2))
        return np.where(self.reverse, 1.0 - p, p).mean(0)

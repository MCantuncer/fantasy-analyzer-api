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


def _add_banked(draws: np.ndarray, i: int, banked: dict[int, float], league: League, rest: Dist) -> None:
    """Turn team i's draws for the days left into full-week totals."""
    for k, c in enumerate(league.categories):
        if c.ratio is None:
            draws[:, i, k] += banked.get(c.stat_id, 0.0)
            continue
        makes, attempts = c.ratio
        made, tried = banked.get(makes, 0.0), banked.get(attempts, 0.0)
        left = float(rest.att[k]) if rest.att is not None else 0.0
        if tried + left > 0:
            draws[:, i, k] = (made + draws[:, i, k] * left) / (tried + left)


@dataclass
class SeasonResult:
    team_ids: list[int]
    wins: np.ndarray  # (T,) mean final wins
    losses: np.ndarray
    ties: np.ndarray
    playoff_pct: np.ndarray  # (T,) 0..1
    first_seed_pct: np.ndarray
    mean_rank: np.ndarray
    win_pct: np.ndarray  # (T,) mean final winning percentage
    win_pct_sd: np.ndarray  # (T,) its spread across simulated seasons


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
    For the matchup period being played, `period_dists` covers only the days left, so the
    stats each team has already banked are added on top.

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
        draws = mu + sd * (z + drift)
        if period == league.current_period and period in (period_dists or {}):
            for m in games:
                for tid, banked in ((m.home, m.home_totals), (m.away, m.away_totals)):
                    if banked and tid in index:
                        _add_banked(draws, index[tid], banked, league, week[index[tid]])
        draws = draws * sign
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
        win_pct=pct.mean(0),
        win_pct_sd=pct.std(0),
    )


class SeasonSim:
    """A season simulation that is drawn once and then re-scored for roster changes.

    Screening trades and pickups asks the same question many times: "what are my playoff odds if
    these one or two teams look different?". `simulate_season` answers by drawing every team's
    weeks from scratch each time. This class draws the random part once (the same draws
    `simulate_season` makes with the same seed) and, for a change, re-scores only the matchups
    that involve a changed team. Everyone else's results are reused, so a what-if costs a small
    fraction of a full run and every what-if shares the exact same luck as the baseline.

    The draws are kept in float32 to halve memory on a small server; results agree with
    `simulate_season` to well under a tenth of a percentage point.
    """

    def __init__(
        self,
        league: League,
        dists: list[Dist],
        reverse: np.ndarray,
        n_sims: int = 4000,
        seed: int = 7,
        period_dists: dict[int, list[Dist]] | None = None,
    ):
        rng = np.random.default_rng(seed)
        self.league = league
        self.n_sims = n_sims
        self.ids = [t.id for t in league.teams]
        self.index = {tid: i for i, tid in enumerate(self.ids)}
        n_teams = len(self.ids)
        self.sign = np.where(reverse, -1.0, 1.0)
        self.each_cat = league.scoring_type == EACH_CATEGORY
        self.base_dists = list(dists)
        self.base_periods = period_dists or {}
        chol = np.linalg.cholesky(category_links([c.name for c in league.categories]))
        drift = rng.normal(0.0, TEAM_DRIFT, size=(n_sims, n_teams, len(league.categories)))

        self.periods: list[int] = []
        self.games: dict[int, list[tuple[int, int]]] = {}
        self.noise: dict[int, np.ndarray] = {}
        self.banked: dict[int, dict[int, dict[int, float]]] = {}
        undecided = sorted({m.period for m in league.schedule if m.winner == "UNDECIDED"})
        for period in undecided:
            games = [m for m in league.schedule if m.period == period and m.winner == "UNDECIDED"]
            if not games:
                continue
            z = rng.standard_normal(size=(n_sims, n_teams, len(league.categories))) @ chol.T
            self.noise[period] = (z + drift).astype(np.float32)
            self.periods.append(period)
            self.games[period] = [
                (self.index[m.home], self.index[m.away]) for m in games if m.home in self.index and m.away in self.index
            ]
            if period == league.current_period and period in self.base_periods:
                self.banked[period] = {
                    self.index[tid]: banked
                    for m in games
                    for tid, banked in ((m.home, m.home_totals), (m.away, m.away_totals))
                    if banked and tid in self.index
                }
        self.tiebreak = rng.uniform(0, 1e-6, size=(n_sims, n_teams))

        self.wins = np.tile(np.array([t.wins for t in league.teams], float), (n_sims, 1))
        self.losses = np.tile(np.array([t.losses for t in league.teams], float), (n_sims, 1))
        self.ties = np.tile(np.array([t.ties for t in league.teams], float), (n_sims, 1))
        for period in self.periods:
            for h, a in self.games[period]:
                hd = self._draws(period, h, self._base_dist(period, h))
                ad = self._draws(period, a, self._base_dist(period, a))
                self._score(self.wins, self.losses, self.ties, h, a, hd, ad, 1.0)
        self.base = self._result(self.wins, self.losses, self.ties)

    def _base_dist(self, period: int, i: int) -> Dist:
        week = self.base_periods.get(period)
        return week[i] if week is not None else self.base_dists[i]

    @staticmethod
    def _new_dist(period: int, i: int, dists: list[Dist], period_dists: dict[int, list[Dist]] | None) -> Dist:
        week = (period_dists or {}).get(period)
        return week[i] if week is not None else dists[i]

    def _draws(self, period: int, i: int, d: Dist) -> np.ndarray:
        """Team i's (n_sims, C) signed category totals for one matchup period."""
        draws = d.mu + d.sd * self.noise[period][:, i, :].astype(float)
        banked = self.banked.get(period, {}).get(i)
        if banked:
            wrapped = draws[:, None, :]
            _add_banked(wrapped, 0, banked, self.league, d)
            draws = wrapped[:, 0, :]
        return draws * self.sign

    def _score(self, wins, losses, ties, h: int, a: int, hd: np.ndarray, ad: np.ndarray, weight: float) -> None:
        h_cats = (hd > ad).sum(1)
        a_cats = (ad > hd).sum(1)
        if self.each_cat:
            wins[:, h] += weight * h_cats
            losses[:, h] += weight * a_cats
            wins[:, a] += weight * a_cats
            losses[:, a] += weight * h_cats
        else:
            hw, aw, tie = h_cats > a_cats, a_cats > h_cats, h_cats == a_cats
            wins[:, h] += weight * hw
            losses[:, h] += weight * aw
            wins[:, a] += weight * aw
            losses[:, a] += weight * hw
            ties[:, h] += weight * tie
            ties[:, a] += weight * tie

    def _result(self, wins: np.ndarray, losses: np.ndarray, ties: np.ndarray) -> SeasonResult:
        n_teams = len(self.ids)
        games_played = np.maximum(wins + losses + ties, 1.0)
        pct = (wins + 0.5 * ties) / games_played
        order = np.argsort(-(pct + self.tiebreak), axis=1)
        rank = np.empty_like(order)
        rank[np.arange(self.n_sims)[:, None], order] = np.arange(n_teams)[None, :]
        playoff_teams = min(self.league.playoff_teams, n_teams)
        return SeasonResult(
            team_ids=self.ids,
            wins=wins.mean(0),
            losses=losses.mean(0),
            ties=ties.mean(0),
            playoff_pct=(rank < playoff_teams).mean(0),
            first_seed_pct=(rank == 0).mean(0),
            mean_rank=rank.mean(0) + 1,
            win_pct=pct.mean(0),
            win_pct_sd=pct.std(0),
        )

    def what_if(
        self,
        changed: set[int],
        dists: list[Dist],
        period_dists: dict[int, list[Dist]] | None = None,
    ) -> SeasonResult:
        """The season with teams in ``changed`` (indices) projected by ``dists``/``period_dists``.

        ``period_dists`` only needs entries for the changed teams to be right; the others are
        taken from the baseline.
        """
        if not changed:
            return self.base
        wins, losses, ties = self.wins.copy(), self.losses.copy(), self.ties.copy()
        for period in self.periods:
            for h, a in self.games[period]:
                if h not in changed and a not in changed:
                    continue
                old_h = self._draws(period, h, self._base_dist(period, h))
                old_a = self._draws(period, a, self._base_dist(period, a))
                self._score(wins, losses, ties, h, a, old_h, old_a, -1.0)
                new_h, new_a = old_h, old_a
                if h in changed:
                    new_h = self._draws(period, h, self._new_dist(period, h, dists, period_dists))
                if a in changed:
                    new_a = self._draws(period, a, self._new_dist(period, a, dists, period_dists))
                self._score(wins, losses, ties, h, a, new_h, new_a, 1.0)
        return self._result(wins, losses, ties)

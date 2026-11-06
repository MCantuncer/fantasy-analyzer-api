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
# Teams drift back toward the league average the further ahead the week: bad teams pick up
# free agents, get stars back from injury and trade; good ones lose players. Each week ahead
# keeps this much less of a team's gap to the average (none for the week being played).
# Measured on League A's 2022, 2024, 2025 and 2026 seasons: from week 2 the model's predicted
# gaps between teams' final records were ~2.5x the real ones (slope 0.41), shrinking to about
# right by week 10. At 0.10 the slope is 0.73 at week 2 and 0.95-1.1 from week 4 on.
REVERSION = 0.10


def _pull(period: int, first: int) -> float:
    """Share of a team's gap to the league average still there in ``period``."""
    return (1.0 - REVERSION) ** max(period - first, 0)


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


def _whole_counts(draws: np.ndarray, counts: np.ndarray) -> None:
    """Round the counting categories of ``draws`` (last axis) to whole numbers, in place.

    A week's blocks, steals or threes are whole numbers, so two teams can end level and the
    category is a tie. Continuous draws never tie, which left ties out of every projection.
    With whole numbers the chance of a tie comes from the spread alone: on League A's 2022-2027
    weeks (536 matchups) that predicts 0.15 category ties a matchup against 0.13 real ones
    (BLK 3.6% vs 3.2%, STL 3.7% vs 3.5%, TO 2.0% vs 2.4%, PTS 0.3% vs 0). FG% and FT% stay
    continuous: none of those 536 matchups tied on them.
    """
    if counts.all():
        np.rint(draws, out=draws)
    elif counts.any():
        # Rounding everything and putting the few percentages back is ~2x faster than rounding a masked subset.
        ratios = draws[..., ~counts]
        np.rint(draws, out=draws)
        draws[..., ~counts] = ratios


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
    made_playoffs: np.ndarray | None = None  # (n_sims, T) bool: each simulated season's playoff teams


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
    simulated records spread too little and the odds come out too sure. Each team's
    projection also drifts back toward the league average week by week (REVERSION), since
    rosters, injuries and roles keep changing all season.

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
    n_cats = len(league.categories)
    counts = np.array([c.ratio is None for c in league.categories], bool)
    chol = np.linalg.cholesky(category_links([c.name for c in league.categories]))
    drift = rng.normal(0.0, TEAM_DRIFT, size=(n_sims, n_teams, n_cats))

    periods = sorted({m.period for m in league.schedule if m.winner == "UNDECIDED"})
    for period in periods:
        games = [m for m in league.schedule if m.period == period and m.winner == "UNDECIDED"]
        if not games:
            continue
        week = (period_dists or {}).get(period, dists)
        mu = np.array([d.mu for d in week])
        sd = np.array([d.sd for d in week])
        center = mu.mean(0)
        mu = center + _pull(period, periods[0]) * (mu - center)
        z = rng.standard_normal(size=(n_sims, n_teams, mu.shape[1])) @ chol.T
        draws = mu + sd * (z + drift)
        if period == league.current_period and period in (period_dists or {}):
            for m in games:
                for tid, banked in ((m.home, m.home_totals), (m.away, m.away_totals)):
                    if banked and tid in index:
                        _add_banked(draws, index[tid], banked, league, week[index[tid]])
        _whole_counts(draws, counts)
        draws = draws * sign
        for m in games:
            h, a = index.get(m.home), index.get(m.away)
            if h is None or a is None:
                continue
            h_cats = (draws[:, h, :] > draws[:, a, :]).sum(1)
            a_cats = (draws[:, a, :] > draws[:, h, :]).sum(1)
            if each_cat:
                level = n_cats - h_cats - a_cats
                wins[:, h] += h_cats
                losses[:, h] += a_cats
                wins[:, a] += a_cats
                losses[:, a] += h_cats
                ties[:, h] += level
                ties[:, a] += level
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


# A what-if re-draws its matchups in batches of about this many numbers, so a FINE_SIMS run
# stays a few MB above the stored draws on a 512 MB server (smaller batches also stay in the CPU cache).
BATCH_NUMBERS = 250_000


class SeasonSim:
    """A season simulation that is drawn once and then re-scored for roster changes.

    Screening trades and pickups asks the same question many times: "what are my playoff odds if
    these one or two teams look different?". `simulate_season` answers by drawing every team's
    weeks from scratch each time. This class draws the random part once (the same draws
    `simulate_season` makes with the same seed) and, for a change, re-scores only the matchups
    that involve a changed team. Everyone else's results are reused, so a what-if costs a small
    fraction of a full run and every what-if shares the exact same luck as the baseline.

    Two things keep a what-if cheap:
    - Each team's result in every baseline matchup (categories won and lost, in each simulated
      season) is kept from the first run, so taking a matchup's old result out is a lookup, not a
      second simulation of it.
    - Every re-scored matchup of the season is drawn and compared in one batch of array maths
      instead of one small calculation per week: numpy is fast on big arrays and slow to start
      each operation, so fewer, bigger operations do the same work in less time.

    The random draws are kept in float32 to halve memory on a small server; results agree with
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
        self.n_cats = len(league.categories)
        self.counts = np.array([c.ratio is None for c in league.categories], bool)
        self.base_dists = list(dists)
        self.base_periods = period_dists or {}
        chol = np.linalg.cholesky(category_links([c.name for c in league.categories]))
        drift = rng.normal(0.0, TEAM_DRIFT, size=(n_sims, n_teams, self.n_cats))

        self.periods: list[int] = []
        games_by_period: list[list[tuple[int, int]]] = []
        self.banked: dict[int, dict[int, dict[int, float]]] = {}  # period index -> team -> banked totals
        undecided = sorted({m.period for m in league.schedule if m.winner == "UNDECIDED"})
        # (P, T, S, C), filled in place: one team's seasons for a week are one contiguous block.
        self.noise = np.empty((len(undecided), n_teams, n_sims, self.n_cats), np.float32)
        for k, period in enumerate(undecided):
            games = [m for m in league.schedule if m.period == period and m.winner == "UNDECIDED"]
            z = rng.standard_normal(size=(n_sims, n_teams, self.n_cats)) @ chol.T
            self.noise[k] = (z + drift).transpose(1, 0, 2)
            self.periods.append(period)
            games_by_period.append(
                [
                    (self.index[m.home], self.index[m.away])
                    for m in games
                    if m.home in self.index and m.away in self.index
                ]
            )
            if period == league.current_period and period in self.base_periods:
                self.banked[k] = {
                    self.index[tid]: banked
                    for m in games
                    for tid, banked in ((m.home, m.home_totals), (m.away, m.away_totals))
                    if banked and tid in self.index
                }
        del drift
        n_periods = len(self.periods)
        self.tiebreak = rng.uniform(0, 1e-6, size=(n_sims, n_teams))

        # Who each team plays in each period (-1: no game), and on which side of the matchup.
        self.opponent = np.full((n_periods, n_teams), -1, int)
        self.home = np.zeros((n_periods, n_teams), bool)
        for k, games in enumerate(games_by_period):
            for h, a in games:
                self.opponent[k, h], self.opponent[k, a] = a, h
                self.home[k, h] = True
        # The league average each period, from today's rosters; what-ifs keep it (a trade moves it little).
        self.center = np.array(
            [np.mean([self._base_dist(k, i).mu for i in range(n_teams)], axis=0) for k in range(n_periods)]
        ).reshape(n_periods, self.n_cats)
        self.pull = np.array([_pull(p, self.periods[0]) for p in self.periods])

        # The baseline season, every matchup of a period at once. Each team's categories won and lost
        # in every matchup (or the matchup's win/loss flags) are kept for what-ifs to take back out.
        self.won = np.zeros((n_periods, n_teams, n_sims), np.int8)
        self.lost = np.zeros((n_periods, n_teams, n_sims), np.int8)
        everyone = np.arange(n_teams)
        for k, games in enumerate(games_by_period):
            if not games:
                continue
            draws = self._draws(np.full(n_teams, k), everyone, [self._base_dist(k, i) for i in everyone])
            h = np.array([g[0] for g in games])
            a = np.array([g[1] for g in games])
            hw, aw = self._outcome(draws[h], draws[a])
            self.won[k, h], self.lost[k, h] = hw, aw
            self.won[k, a], self.lost[k, a] = aw, hw
        won, lost = self.won.sum(0, dtype=np.int32).T, self.lost.sum(0, dtype=np.int32).T  # (S, T)
        per_game = self.n_cats if self.each_cat else 1
        left = per_game * (self.opponent >= 0).sum(0)  # (T,) categories (or matchups) left to play
        self.wins = np.array([t.wins for t in league.teams], float) + won
        self.losses = np.array([t.losses for t in league.teams], float) + lost
        self.ties = np.array([t.ties for t in league.teams], float) + (left - won - lost)
        self.base = self._result(self.wins, self.losses, self.ties)

    def _base_dist(self, k: int, i: int) -> Dist:
        week = self.base_periods.get(self.periods[k])
        return week[i] if week is not None else self.base_dists[i]

    def _draws(self, ks: np.ndarray, teams: np.ndarray, ds: list[Dist]) -> np.ndarray:
        """(N, n_sims, C) signed category totals of team ``teams[n]`` in period index ``ks[n]``, projected by
        ``ds[n]``."""
        center, pull = self.center[ks], self.pull[ks][:, None]
        mu = center + pull * (np.array([d.mu for d in ds]) - center)
        sd = np.array([d.sd for d in ds])
        draws = self.noise[ks, teams].astype(float)
        draws *= sd[:, None, :]
        draws += mu[:, None, :]
        for n, (k, i) in enumerate(zip(ks, teams, strict=True)):
            banked = self.banked.get(int(k), {}).get(int(i))
            if banked:
                _add_banked(draws[n][:, None, :], 0, banked, self.league, ds[n])
        _whole_counts(draws, self.counts)
        draws *= self.sign
        return draws

    def _outcome(self, hd: np.ndarray, ad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """What the home and away sides take from matchups (..., n_sims, C): categories won, or win flags."""
        h_cats = np.count_nonzero(hd > ad, axis=-1).astype(np.int8)
        a_cats = np.count_nonzero(ad > hd, axis=-1).astype(np.int8)
        if self.each_cat:
            return h_cats, a_cats
        return (h_cats > a_cats).astype(np.int8), (a_cats > h_cats).astype(np.int8)

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
            made_playoffs=rank < playoff_teams,
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

        def dist(k: int, i: int) -> Dist:
            if i not in changed:
                return self._base_dist(k, i)
            week = (period_dists or {}).get(self.periods[k])
            return week[i] if week is not None else dists[i]

        # Every matchup a changed team plays, once each (a deal's two teams may meet).
        matchups = sorted(
            {
                (k, i, int(self.opponent[k, i])) if self.home[k, i] else (k, int(self.opponent[k, i]), i)
                for i in changed
                for k in np.flatnonzero(self.opponent[:, i] >= 0)
            }
        )
        if not matchups:
            return self.base
        wins, losses, ties = self.wins.copy(), self.losses.copy(), self.ties.copy()
        step = max(1, BATCH_NUMBERS // (self.n_sims * self.n_cats))
        for start in range(0, len(matchups), step):
            batch = np.array(matchups[start : start + step])
            ks, hs, as_ = batch[:, 0], batch[:, 1], batch[:, 2]
            hd = self._draws(ks, hs, [dist(k, i) for k, i in zip(ks, hs, strict=True)])
            ad = self._draws(ks, as_, [dist(k, i) for k, i in zip(ks, as_, strict=True)])
            hw, aw = self._outcome(hd, ad)
            del hd, ad
            for side, won, lost in ((hs, hw, aw), (as_, aw, hw)):
                # New result minus the baseline one, per matchup, added up per team.
                d_won = won.astype(np.int16) - self.won[ks, side]
                d_lost = lost.astype(np.int16) - self.lost[ks, side]
                for i in np.unique(side):
                    rows = side == i
                    dw, dl = d_won[rows].sum(0), d_lost[rows].sum(0)
                    wins[:, i] += dw
                    losses[:, i] += dl
                    ties[:, i] -= dw + dl
        return self._result(wins, losses, ties)

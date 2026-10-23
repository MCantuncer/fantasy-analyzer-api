"""Weekly category model: what a lineup produces in a week and how often it wins each category.

Each team's weekly total in a category is modelled as a normal distribution.
Counting stats use an over-dispersed Poisson variance per game plus schedule
noise. Percentage stats are makes over attempts with binomial variance.

With the NBA calendar, games are counted day by day: each day the best healthy
players who have a game fill the league's lineup slots (PG, SG, SF, PF, C, G, F,
UTIL), so a bench player counts on days a starter sits, a 4-game week counts
more than a 2-game week, and an injured player counts again once he's back.
Without it, every starter plays GAMES_PER_WEEK games.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.espn.parse import IR_SLOT, UTIL_SLOT, Category, League, Player

GAMES_PER_WEEK = 3.4
# Per-game variance as a multiple of the per-game mean, by stat id.
DISPERSION = {0: 7.0, 1: 1.3, 2: 1.0, 3: 2.0, 4: 1.5, 5: 2.0, 6: 2.5, 11: 1.2, 17: 1.3}
DEFAULT_DISPERSION = 1.5
SCHEDULE_CV = 0.08  # weekly games played varies by about 8%
KNOWN_SCHEDULE_CV = 0.04  # with the real calendar, only rest days and minutes vary
# ESPN gives no return dates, so a status means an expected number of days out.
DAYS_OUT = {"DAY_TO_DAY": 2, "SUSPENSION": 5, "OUT": 14, "INJURY_RESERVE": 28}
IR_DAYS_OUT = 28
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
        self.cal = league.calendar
        self.slot_counts = league.lineup_slots or {}
        self._plays: dict[tuple[int, object], np.ndarray] = {}
        self._fills: dict[tuple[int, ...], tuple[int, ...]] = {}
        self._dists: dict[tuple[tuple[int, ...], object], Dist] = {}
        self._masks: dict[int, int] = {}
        self.seats = [slot for slot, n in sorted(self.slot_counts.items()) for _ in range(n)]
        self.horizon: dict[object, tuple[list[int], float]] = {}
        if self.cal:
            last = league.regular_season_periods or max(self.cal.period_days)
            season = [d for p in range(league.current_period, last + 1) for d in self.cal.days_left(p)]
            self.horizon["season"] = (season, max(len(season) / 7, 1.0))
            for p in self.cal.period_days:
                self.horizon[p] = (self.cal.days_left(p), 1.0)

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

    def dist(self, roster: list[Player], horizon: object = "season") -> Dist:
        """A team's weekly totals: an average week of the rest of the regular season, or
        the remaining days of one matchup period when `horizon` is a period number."""
        key = (tuple(sorted(p.id for p in roster)), horizon)
        cached = self._dists.get(key)
        if cached is None:
            cached = self._dists[key] = self._compute_dist(roster, horizon)
        return cached

    def _compute_dist(self, roster: list[Player], horizon: object) -> Dist:
        if not self.cal or horizon not in self.horizon:
            lineup = self.lineup(roster)
            if not lineup:
                return self._empty()
            total, att, var = np.sum([self.row(p) for p in lineup], axis=0)
            return self._dist(total, att, var, SCHEDULE_CV)
        days, weeks = self.horizon[horizon]
        starts = self.starts(roster, horizon)
        if not starts.any():
            return self._empty()
        # row() is one week at self.g games; scale each player to his starts per week.
        rows = np.array([self.row(p) for p in roster]) * (starts / (self.g * weeks))[:, None, None]
        total, att, var = rows.sum(0)
        return self._dist(total, att, var, KNOWN_SCHEDULE_CV)

    def _empty(self) -> Dist:
        return Dist(np.zeros(len(self.cats)), np.ones(len(self.cats)))

    def _dist(self, total: np.ndarray, att: np.ndarray, var: np.ndarray, cv: float) -> Dist:
        with np.errstate(invalid="ignore", divide="ignore"):
            pct = np.where(att > 0, total / np.maximum(att, 1e-9), 0.0)
            ratio_sd = np.sqrt(RATIO_OVERDISPERSION * pct * (1 - pct) / np.maximum(att, 1.0))
        count_sd = np.sqrt(var + (cv * total) ** 2) + 1e-6
        mu = np.where(self.is_ratio, pct, total)
        sd = np.where(self.is_ratio, ratio_sd, count_sd)
        return Dist(mu, sd)

    # ---- the calendar: who plays which day, and who starts ------------------------------

    def return_day(self, p: Player) -> int:
        """First day an injured player is expected back (today for healthy players)."""
        today = self.cal.today if self.cal else 1
        out = DAYS_OUT.get(p.injury_status, 0)
        if p.lineup_slot == IR_SLOT:
            out = max(out, IR_DAYS_OUT)
        return today + out

    def plays(self, p: Player, horizon: object) -> np.ndarray:
        """(D,) bool: days in the horizon his team plays and he's expected to be healthy."""
        key = (p.id, horizon)
        cached = self._plays.get(key)
        if cached is None:
            days, _ = self.horizon[horizon]
            games = self.cal.team_days.get(p.pro_team_id, frozenset())
            back = self.return_day(p)
            cached = self._plays[key] = np.array([d in games and d >= back for d in days], dtype=bool)
        return cached

    def starts(self, roster: list[Player], horizon: object = "season") -> np.ndarray:
        """(P,) games each player starts over the horizon, in roster order."""
        if not roster:
            return np.zeros(0)
        order = sorted(range(len(roster)), key=lambda i: self.value(roster[i]), reverse=True)
        m = np.array([self.plays(roster[i], horizon) for i in order])  # (P, D) in value order
        out = np.zeros(len(roster))
        if not m.size:
            return out
        # Days with the same players available share one lineup: solve each pattern once.
        bits = (m.astype(np.int64) << np.arange(len(order), dtype=np.int64)[:, None]).sum(0)
        patterns, counts = np.unique(bits, return_counts=True)
        masks = [self._seat_mask(roster[i]) for i in order]
        for mask, n in zip(patterns.tolist(), counts.tolist(), strict=True):
            idx = [k for k in range(len(order)) if mask >> k & 1]
            for k in self._fill(tuple([masks[k] for k in idx])):
                out[order[idx[k]]] += n
        return out

    def _seat_mask(self, p: Player) -> int:
        """Bit k set when he can sit in lineup seat k."""
        mask = self._masks.get(p.id)
        if mask is None:
            slots = p.slots or (UTIL_SLOT,)
            mask = self._masks[p.id] = sum(1 << k for k, s in enumerate(self.seats) if s in slots)
        return mask

    def _fill(self, masks: tuple[int, ...]) -> tuple[int, ...]:
        """Which of today's players start, given each one's seat mask, best player first.

        Adding players in value order and keeping each one an augmenting path can
        seat gives the most valuable lineup the slots allow. Only the masks matter,
        so lineups with the same position mix share one answer.
        """
        cached = self._fills.get(masks)
        if cached is not None:
            return cached
        if not self.seats:
            res = tuple(range(min(len(masks), self.n_starters)))
            self._fills[masks] = res
            return res
        holder = [-1] * len(self.seats)  # seat -> index into masks
        seen = 0  # seats already tried in this augmenting search

        def seat(i: int) -> bool:
            """Seat player i, moving others along an augmenting path."""
            nonlocal seen
            free = masks[i] & ~seen
            while free:
                bit = free & -free
                free ^= bit
                seen |= bit
                k = bit.bit_length() - 1
                if holder[k] < 0 or seat(holder[k]):
                    holder[k] = i
                    return True
            return False

        started = []
        for i in range(len(masks)):
            if len(started) == len(self.seats):
                break
            seen = 0
            if seat(i):
                started.append(i)
        res = tuple(started)
        self._fills[masks] = res
        return res

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
        me = dists[idx]
        others = [d for j, d in enumerate(dists) if j != idx]
        return float(np.mean([self.cat_win_prob(me, o).sum() for o in others]))

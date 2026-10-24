"""Trade value: what a player is worth in a trade, in auction dollars ($200 budget scale).

Early in the season owners go by names: ESPN's rankings and what was paid in the
draft. As games pile up, what a player is actually doing takes over. The value
blends both, and the draft-day signals fade as the season produces real numbers:

  * ESPN's preseason rank and auction value (fades to a small weight)
  * average ESPN auction price and this league's draft price (fade out)
  * % owned across ESPN leagues: the market's current verdict
  * rest-of-season value in this league's categories (projection blended with actuals)
  * this season's ESPN rank and the last 15 days, once there are enough games

Every rank is priced on one curve built from ESPN's own rank -> auction dollars.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np

from app.espn.parse import League, Player

if TYPE_CHECKING:
    from app.analysis.model import CategoryModel

# Draft-day signals, at full weight before any games. They fade as games are played.
W_ESPN = 0.25
W_ESPN_FLOOR = 0.10  # ESPN's rank never fades entirely: owners keep looking at it
W_AVG_AUCTION = 0.10
W_PAID = 0.10
DRAFT_FADE_GAMES = 25
# Signals that hold all season.
W_OWNED = 0.15
W_PROJECTION = 0.25
# Production signals, ramping up with games played.
W_SEASON = 0.35
SEASON_FULL_GAMES = 15
W_RECENT = 0.15
RECENT_FULL_GAMES = 8
# A player with no projection or last season needs this many games for his stats to count.
MIN_SAMPLE_GAMES = 5

# Owners value one star over two decent players adding up to the same dollars.
# A package is worth the K-norm of its parts, so two $20 players count as about $32.
STUD_POWER = 1.5


class PerceivedValue:
    def __init__(self, league: League, model: CategoryModel | None = None):
        self.league = league
        self.model = model
        pool = [p for t in league.teams for p in t.roster] + list(league.free_agents)
        # Rank -> dollars curve from ESPN's own rankings, used to price every rank signal.
        ranked = sorted((p.espn_rank, p.espn_auction_value) for p in pool if p.espn_rank)
        self._ranks = np.array([r for r, _ in ranked], float) if ranked else np.array([1.0])
        self._dollars = (
            np.maximum.accumulate(np.array([v for _, v in ranked], float)[::-1])[::-1] if ranked else np.array([0.0])
        )
        by_owned = sorted(pool, key=lambda p: p.percent_owned, reverse=True)
        self._owned_rank = {p.id: r for r, p in enumerate(by_owned, 1) if p.percent_owned > 0}
        self._proj_rank: dict[int, int] = {}
        if model is not None:
            trusted = [p for p in pool if p.has_prior or p.games_played >= MIN_SAMPLE_GAMES]
            by_value = sorted(trusted, key=model.value, reverse=True)
            self._proj_rank = {p.id: r for r, p in enumerate(by_value, 1)}
        self._cache: dict[int, float] = {}

    def dollars_at_rank(self, rank: float) -> float:
        if rank > self._ranks[-1]:
            return 0.0
        return float(np.interp(rank, self._ranks, self._dollars))

    def signals(self, p: Player) -> list[tuple[str, float, float]]:
        """(name, weight, dollars) for every signal this player has."""
        gp = p.games_played
        fade = max(1.0 - gp / DRAFT_FADE_GAMES, 0.0)
        out: list[tuple[str, float, float]] = []
        if p.espn_rank:
            w = W_ESPN_FLOOR + (W_ESPN - W_ESPN_FLOOR) * fade
            out.append(("espn", w, p.espn_auction_value))
        if p.avg_auction_value > 0 and fade > 0:
            out.append(("avg_auction", W_AVG_AUCTION * fade, p.avg_auction_value))
        if fade > 0:
            if p.id in self.league.draft_prices:
                out.append(("paid", W_PAID * fade, self.league.draft_prices[p.id]))
            elif p.id in self.league.draft_picks:
                out.append(("paid", W_PAID * fade, self.dollars_at_rank(self.league.draft_picks[p.id])))
        if p.id in self._owned_rank:
            out.append(("owned", W_OWNED, self.dollars_at_rank(self._owned_rank[p.id])))
        if p.id in self._proj_rank:
            out.append(("projection", W_PROJECTION, self.dollars_at_rank(self._proj_rank[p.id])))
        if p.season_rank and gp > 0:
            out.append(("season", W_SEASON * min(gp / SEASON_FULL_GAMES, 1.0), self.dollars_at_rank(p.season_rank)))
        if p.recent_rank and gp > 0:
            out.append(("recent", W_RECENT * min(gp / RECENT_FULL_GAMES, 1.0), self.dollars_at_rank(p.recent_rank)))
        return out

    def value(self, p: Player) -> float:
        cached = self._cache.get(p.id)
        if cached is not None:
            return cached
        signals = self.signals(p)
        total_w = sum(w for _, w, _ in signals)
        v = sum(w * x for _, w, x in signals) / total_w if total_w else 0.0
        self._cache[p.id] = v
        return v

    def total(self, players: list[Player]) -> float:
        return sum(self.value(p) for p in players)

    def package(self, players: list[Player], price: Callable[[Player], float] | None = None) -> float:
        """What a group of players is worth together in a trade, with the premium for stars.

        ``price`` replaces trade value per player, e.g. with an owner's premium added.
        """
        each = price or self.value
        return float(sum(max(each(p), 0.0) ** STUD_POWER for p in players) ** (1 / STUD_POWER)) if players else 0.0

    @cached_property
    def replacement(self) -> float:
        """Value of the best healthy free agent: what an owner gets back for a roster spot."""
        free = [self.value(p) for p in self.league.free_agents if p.available]
        return max(free, default=0.0)

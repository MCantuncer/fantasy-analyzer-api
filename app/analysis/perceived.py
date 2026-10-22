"""Perceived value: what a player is worth in the eyes of an owner judging a trade.

Owners don't run simulations. They look at ESPN's rankings, what they paid in the
draft, how widely a player is owned, and how he has played this season. This
blends those signals into one auction-dollar figure ($200 budget scale).
"""

from __future__ import annotations

import numpy as np

from app.espn.parse import League, Player

# Weights before re-normalising over the signals a player actually has.
W_ESPN = 0.35
W_AVG_AUCTION = 0.25
W_PAID = 0.25
# This season's rank takes over gradually, reaching this weight at FULL_SEASON_GAMES.
W_SEASON_MAX = 0.45
FULL_SEASON_GAMES = 30


class PerceivedValue:
    def __init__(self, league: League):
        self.league = league
        pool = [p for t in league.teams for p in t.roster] + list(league.free_agents)
        # Rank -> dollars curve from ESPN's own rankings, used to price picks and season ranks.
        ranked = sorted((p.espn_rank, p.espn_auction_value) for p in pool if p.espn_rank)
        self._ranks = np.array([r for r, _ in ranked], float) if ranked else np.array([1.0])
        self._dollars = (
            np.maximum.accumulate(np.array([v for _, v in ranked], float)[::-1])[::-1] if ranked else np.array([0.0])
        )
        self._cache: dict[int, float] = {}

    def dollars_at_rank(self, rank: float) -> float:
        if rank > self._ranks[-1]:
            return 0.0
        return float(np.interp(rank, self._ranks, self._dollars))

    def value(self, p: Player) -> float:
        cached = self._cache.get(p.id)
        if cached is not None:
            return cached
        signals: list[tuple[float, float]] = []
        if p.espn_rank:
            signals.append((W_ESPN, p.espn_auction_value))
        if p.avg_auction_value > 0:
            signals.append((W_AVG_AUCTION, p.avg_auction_value))
        if p.id in self.league.draft_prices:
            signals.append((W_PAID, self.league.draft_prices[p.id]))
        elif p.id in self.league.draft_picks:
            signals.append((W_PAID, self.dollars_at_rank(self.league.draft_picks[p.id])))
        if p.season_rank and p.games_played > 0:
            w = W_SEASON_MAX * min(p.games_played / FULL_SEASON_GAMES, 1.0)
            signals.append((w, self.dollars_at_rank(p.season_rank)))
        total_w = sum(w for w, _ in signals)
        v = sum(w * x for w, x in signals) / total_w if total_w else 0.0
        self._cache[p.id] = v
        return v

    def total(self, players: list[Player]) -> float:
        return sum(self.value(p) for p in players)

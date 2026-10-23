"""Roster tiers: who is a keeper, who is trade bait, and who can actually be cut.

Drops and the players owners cut to make room in a trade come only from the
droppable tier. A player most ESPN leagues roster has trade value: if he doesn't
fit, the move is to trade him, not to release him for nothing.
"""

from __future__ import annotations

from collections.abc import Set as AbstractSet
from typing import Literal

from app.espn.parse import Player

Tier = Literal["core", "tradeable", "droppable", "injured"]

# A player rostered in at least this share of ESPN leagues has a market.
TRADEABLE_MIN_OWNED = 50.0
# ESPN ranks inside this are worth something in a trade even when ownership lags
# (about the top three quarters of what a 10-team league rosters).
TRADEABLE_MAX_RANK = 100
# This season's rank counts once a player has this many games.
SEASON_RANK_GAMES = 8


def has_market(p: Player) -> bool:
    if p.percent_owned >= TRADEABLE_MIN_OWNED:
        return True
    if p.espn_rank and p.espn_rank <= TRADEABLE_MAX_RANK:
        return True
    return bool(p.season_rank and p.games_played >= SEASON_RANK_GAMES and p.season_rank <= TRADEABLE_MAX_RANK)


def tier(p: Player, keep: AbstractSet[int] = frozenset()) -> Tier:
    """core: you keep him. injured: out, valued when he's back. tradeable: has a market.
    droppable: replaceable from free agency."""
    if p.id in keep:
        return "core"
    if not p.available:
        return "injured"
    return "tradeable" if has_market(p) else "droppable"


def droppable(roster: list[Player], keep: AbstractSet[int] = frozenset()) -> list[Player]:
    return [p for p in roster if tier(p, keep) == "droppable"]


TIER_REASON: dict[Tier, str] = {
    "core": "You marked him as a keeper.",
    "injured": "Out right now. Hold him unless the injury is long term.",
    "tradeable": "Has a market (well owned or well ranked). If he doesn't fit, trade him rather than drop him.",
    "droppable": "Lightly owned and replaceable from free agency.",
}

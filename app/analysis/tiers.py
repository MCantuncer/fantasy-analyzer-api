"""Roster tiers: who is a keeper, who is trade bait, and who can actually be cut.

Drops and the players owners cut to make room in a trade come only from the
droppable tier. A player most ESPN leagues roster has trade value: if he doesn't
fit, the move is to trade him, not to release him for nothing.

The reverse holds in a trade: a player off the waiver wire is not currency. The other
owner could have added him for free, so he can ride along in a deal but never pays for it.
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
# A free-agent pickup this recent is still a waiver player to the rest of the league:
# every owner could have had him for nothing and passed.
WAIVER_FRESH_DAYS = 21


def has_market(p: Player) -> bool:
    """Owners pay for him: most leagues roster him, or ESPN ranks him among the players a league rosters.

    A few good weeks of season totals are not a market on their own. ESPN's rank and the share
    of leagues rostering a player both move during the season, so a real breakout shows up there.
    """
    if p.percent_owned >= TRADEABLE_MIN_OWNED:
        return True
    return bool(p.espn_rank and p.espn_rank <= TRADEABLE_MAX_RANK)


def off_the_wire(p: Player, now_ms: int) -> bool:
    """A player the other owners could have for free, or just could: no market, or added from
    free agency in the last few weeks. In a trade he is a throw-in, worth a free agent to them."""
    if not has_market(p):
        return True
    if p.acquisition_type != "ADD" or p.acquired_ms is None:
        return False
    return 0 <= (now_ms - p.acquired_ms) / 86_400_000 < WAIVER_FRESH_DAYS


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

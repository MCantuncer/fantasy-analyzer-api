"""Owner premium: why some players cost more than their trade value.

Owners don't sell every player at what he's worth on paper. Badges mark the ones
they hold on to, and each badge adds a premium on top of trade value that the
owner needs before letting the player go:

  * Jewel: a star name. Trading him has to feel spectacular.
  * Keystone: he carries a category for his team. Take him out and the team
    stops winning it, so the owner holds on even if he isn't a star.
  * Fresh signing: just traded for or picked up. Owners rarely flip a player
    they just went and got.
  * On the block: the owner listed him on ESPN's trade block, so he's cheaper.
  * Not for sale: marked untouchable on ESPN, or by you. Never targeted.

The premium grows for owners who never make moves and shrinks for frequent
traders: research on the endowment effect finds experienced traders barely
overvalue what they own.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, time
from typing import TYPE_CHECKING, Literal

import numpy as np

from app import schemas
from app.analysis.tiers import has_market
from app.espn.parse import League, Player

if TYPE_CHECKING:
    from app.analysis.model import CategoryModel
    from app.analysis.perceived import PerceivedValue

BadgeKind = Literal["jewel", "keystone", "fresh_signing", "not_for_sale", "on_the_block"]

JEWEL_RANK = 12  # top of the league by trade value, about the first round in a 10-team league
JEWEL_TOP_RANK = 5
JEWEL_PREMIUM = 0.20
JEWEL_TOP_PREMIUM = 0.25
# A big name in a slump stays a jewel: top 20 by trade value and top 12 in ESPN's preseason rank.
JEWEL_STICKY_RANK = 20

# Keystone: swapping him for the best free agent costs his team this much weekly win rate
# in one category it competes in, and he supplies this share of the team's production
# above free-agent level there.
KEYSTONE_SWING = 0.15
KEYSTONE_SHARE = 0.30
KEYSTONE_MIN_WIN = 0.5
KEYSTONE_PREMIUM = 0.15
KEYSTONE_MIN_DOLLARS = 3.0  # a cheap keystone still costs something to pry loose

FRESH_DAYS = 21
FRESH_PREMIUM = 0.10  # fades to zero over FRESH_DAYS

ON_THE_BLOCK = -0.10
MAX_PREMIUM = 0.35

# Owner activity: no moves at all makes an owner stickier, frequent trading looser.
ACTIVITY_MIN_WEEKS = 3  # judge activity only after this many weeks
PASSIVE_FACTOR = 1.25
ACTIVE_FACTOR = 0.6
ACTIVE_TRADES = 2
ACTIVE_MOVES_PER_WEEK = 2.0

UNTOUCHABLE = "UNTOUCHABLE"
BLOCK = "ON_THE_BLOCK"


@dataclass(frozen=True)
class Badge:
    kind: BadgeKind
    reason: str
    category: str | None = None  # keystone only


def as_of_ms(league: League) -> int:
    """The league's 'today' in epoch milliseconds: the current scoring day when the calendar is known."""
    if league.calendar is not None:
        day = league.calendar.date_of(league.calendar.today)
        return int(datetime.combine(day, time(12), tzinfo=UTC).timestamp() * 1000)
    return int(datetime.now(UTC).timestamp() * 1000)


class OwnerPremium:
    def __init__(
        self,
        league: League,
        model: CategoryModel,
        perceived: PerceivedValue,
        not_for_sale: set[int] | frozenset[int] = frozenset(),
        now_ms: int | None = None,
    ):
        self.league = league
        self.model = model
        self.perceived = perceived
        self.now_ms = as_of_ms(league) if now_ms is None else now_ms
        self.owner = {p.id: t for t in league.teams for p in t.roster}
        self.manual = set(not_for_sale)
        self.untouchable = {pid for t in league.teams for pid, s in t.trade_block.items() if s == UNTOUCHABLE}
        self.on_block = {pid for t in league.teams for pid, s in t.trade_block.items() if s == BLOCK}
        self._jewels = self._find_jewels()
        self._keystones = self._find_keystones()
        self._badges: dict[int, list[Badge]] = {}

    # ---- detection ---------------------------------------------------------------------

    def _find_jewels(self) -> dict[int, int]:
        """Player id -> rank by trade value, for the league's star names."""
        pool = [p for t in self.league.teams for p in t.roster] + list(self.league.free_agents)
        ranked = sorted(pool, key=self.perceived.value, reverse=True)
        out = {}
        for r, p in enumerate(ranked[:JEWEL_STICKY_RANK], 1):
            if p.id in self.owner and (r <= JEWEL_RANK or (p.espn_rank and p.espn_rank <= JEWEL_RANK)):
                out[p.id] = r
        return out

    def _find_keystones(self) -> dict[int, tuple[str, float, float]]:
        """Player id -> (category, team's weekly win rate with him, without him)."""
        m = self.model
        healthy = [
            p for p in self.league.free_agents if p.available and p.pro_team_id and (p.has_prior or p.games_played >= 5)
        ]
        filler = max(healthy, key=m.value, default=None)
        if filler is None:
            return {}
        z_fill = m.z_scores(filler)
        dists = [m.dist(t.roster) for t in self.league.teams]
        out = {}
        for i, t in enumerate(self.league.teams):
            base = m.avg_cat_win(i, dists)
            surplus = {p.id: np.maximum(m.z_scores(p) - z_fill, 0.0) for p in t.roster}
            total = sum((surplus[p.id] for p in m.lineup(t.roster)), np.zeros(len(m.cats)))
            for p in t.roster:
                if not p.available:
                    continue
                d = list(dists)
                d[i] = m.dist([q for q in t.roster if q is not p] + [filler])
                drop = base - m.avg_cat_win(i, d)
                share = surplus[p.id] / np.where(total > 0, total, 1.0)
                ok = (base >= KEYSTONE_MIN_WIN) & (drop >= KEYSTONE_SWING) & (share >= KEYSTONE_SHARE)
                if ok.any():
                    c = int(np.argmax(np.where(ok, drop, -1.0)))
                    out[p.id] = (m.cats[c].name, float(base[c]), float(base[c] - drop[c]))
        return out

    def _fresh_days(self, p: Player) -> float | None:
        """Days since his owner went and got him, if that counts as a fresh signing."""
        if p.acquired_ms is None or p.acquisition_type not in ("TRADE", "ADD"):
            return None
        # ESPN doesn't send bid amounts with the roster, so a pickup counts when the player has a market.
        if p.acquisition_type == "ADD" and not has_market(p):
            return None
        days = (self.now_ms - p.acquired_ms) / 86_400_000
        return days if 0 <= days < FRESH_DAYS else None

    # ---- results -----------------------------------------------------------------------

    def badges(self, p: Player) -> list[Badge]:
        cached = self._badges.get(p.id)
        if cached is not None:
            return cached
        out: list[Badge] = []
        team = self.owner.get(p.id)
        if p.id in self.untouchable:
            out.append(Badge("not_for_sale", "Marked untouchable on ESPN's trade block."))
        elif p.id in self.manual:
            out.append(Badge("not_for_sale", "You marked him not for sale."))
        if p.id in self._jewels:
            r = self._jewels[p.id]
            why = (
                f"#{r} in the league by trade value. Trading him takes a spectacular offer."
                if r <= JEWEL_RANK
                else "A big name: owners still price him as a star even in a slump."
            )
            out.append(Badge("jewel", why))
        if p.id in self._keystones and team is not None:
            cat, with_him, without = self._keystones[p.id]
            out.append(
                Badge(
                    "keystone",
                    f"Carries {team.abbrev}'s {cat}: they win it {with_him:.0%} of weeks, {without:.0%} without him.",
                    category=cat,
                )
            )
        days = self._fresh_days(p)
        if days is not None:
            how = "traded for" if p.acquisition_type == "TRADE" else "picked up"
            when = "today" if days < 1 else f"{days:.0f} day{'s' if days >= 1.5 else ''} ago"
            out.append(Badge("fresh_signing", f"Just {how} ({when}). Owners rarely flip a player they just got."))
        if p.id in self.on_block:
            out.append(Badge("on_the_block", "On his owner's ESPN trade block: they want to move him."))
        self._badges[p.id] = out
        return out

    def not_for_sale(self, p: Player) -> bool:
        return p.id in self.untouchable or p.id in self.manual

    def keystone(self, p: Player) -> str | None:
        k = self._keystones.get(p.id)
        return k[0] if k else None

    def is_attached(self, p: Player) -> bool:
        """A jewel or keystone: moving him should take a clearly better deal."""
        return p.id in self._jewels or p.id in self._keystones

    def activity_factor(self, team_id: int) -> float:
        weeks = self.league.current_period - 1
        team = self.league.team(team_id)
        if team is None or weeks < ACTIVITY_MIN_WEEKS:
            return 1.0
        if team.trades == 0 and team.acquisitions == 0:
            return PASSIVE_FACTOR
        if team.trades >= ACTIVE_TRADES or (team.acquisitions + 3 * team.trades) / weeks >= ACTIVE_MOVES_PER_WEEK:
            return ACTIVE_FACTOR
        return 1.0

    def rate(self, p: Player) -> float:
        """Premium as a share of trade value, after stacking and owner activity."""
        ups = []
        r = self._jewels.get(p.id)
        if r is not None:
            ups.append(JEWEL_TOP_PREMIUM if r <= JEWEL_TOP_RANK else JEWEL_PREMIUM)
        if p.id in self._keystones:
            ups.append(KEYSTONE_PREMIUM)
        days = self._fresh_days(p)
        if days is not None:
            ups.append(FRESH_PREMIUM * (1 - days / FRESH_DAYS))
        ups.sort(reverse=True)
        rate = min(ups[0] + 0.5 * ups[1], MAX_PREMIUM) if len(ups) > 1 else (ups[0] if ups else 0.0)
        if p.id in self.on_block:
            rate += ON_THE_BLOCK
        team = self.owner.get(p.id)
        return rate * (self.activity_factor(team.id) if team is not None else 1.0)

    def dollars(self, p: Player) -> float:
        """What the owner wants on top of trade value, in auction dollars (negative when he's on the block)."""
        value = self.perceived.value(p)
        extra = value * self.rate(p)
        if p.id in self._keystones and extra >= 0:
            team = self.owner.get(p.id)
            factor = self.activity_factor(team.id) if team is not None else 1.0
            extra = max(extra, KEYSTONE_MIN_DOLLARS * factor)
        return max(extra, -value)

    def price(self, p: Player) -> float:
        """Trade value plus the owner premium: what it takes to get him."""
        return self.perceived.value(p) + self.dollars(p)


def badge_refs(premium: OwnerPremium, p: Player) -> list[schemas.PlayerBadge]:
    return [schemas.PlayerBadge(kind=b.kind, category=b.category, reason=b.reason) for b in premium.badges(p)]

"""Trade finder: deals that help you and that the other owner has a real reason to accept.

A deal is suggested only when
  * it raises your expected categories won,
  * it raises theirs too, counting only the categories they compete in (not their punts), and
  * it doesn't look like a loss to an owner who goes by rankings and draft prices.

"On paper" follows how owners judge deals: one star is worth more than two players
adding up to the same dollars, and a player they must cut to make room counts
against the deal unless waivers have someone as good.

Owners also hold on to some players beyond their trade value: stars, players who
carry a category, fresh signings (see badges.py). Getting one of those costs their
owner premium on top of trade value, and giving up one of yours takes a deal near
the top of what the finder sees.

A 2-for-1 fills the spot it opens with your best free pickup: the free agent who does the
most for your team, as on the pickups list, not just the best by value. That pickup needs
no trade: you could cut your weakest player for him today. So such a deal is measured
against your roster after that free pickup, not against today's roster, and the trade
gets credit only for what it adds on top. The same goes for a deal that cuts or sends
the player you'd drop for that free agent (a 1-for-2 that makes room by cutting him):
replacing him is free, so it isn't the trade's gain. And it goes for the other team:
when a deal opens a spot on their roster, their side is measured against the pickup
they could make today.

A player of mine who came off the waiver wire (no market, or added from free agency in the
last few weeks, see tiers.py) is not currency: the other owner could have had him for free.
He counts for nothing on paper, so he can ride along in a deal but never pays for it.

Trade shapes: 1-for-1, 2-for-2, 2-for-1 and 1-for-2. Real offers show owners treat
them differently at the same value (scripts/trade_shape_backtest.py): a 2-for-2 is a
normal trade, while a 2-for-1 that makes the other owner cut someone to fit two players
rarely gets accepted, however good it looks on paper.

Each surviving deal gets an acceptance estimate and a list of reasons to use when
pitching it. Deals are ranked by playoff-odds gain weighted by that estimate.

The final word is the season simulation (report.py). A fair deal is shown only if it raises
both teams' playoff odds by a point. Early in the season most deals that help me cost the
partner, because I become a tougher rival for the same playoff spots, so few pass. An edge
deal ("worth a try") raises my odds a point and wins on paper for them, as long as it costs
their playoff odds only a little: owners judge on paper, but a deal that sinks their team
isn't one to send. The finder screens loosely and hands over a queue ordered by how likely a
deal is to pass, and the report simulates down the queue until the list is full.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from itertools import combinations
from typing import Literal

import numpy as np

from app import schemas
from app.analysis.badges import OwnerPremium, best_pickup, pickup_pool
from app.analysis.model import TURNOVERS, CategoryModel, Dist
from app.analysis.perceived import PerceivedValue
from app.analysis.tiers import droppable, off_the_wire
from app.espn.parse import League, Player, Team

MIN_MY_GAIN = 0.05  # expected categories won per week
MIN_THEIR_GAIN = 0.02  # a clear gain in the categories they play for: a win-win on the court
# Moves' fair screen: what a deal may cost their build before the playoff-odds check, which has
# the final say there. Their odds can rise while their average week stays flat, e.g. when they
# get healthy players for injured ones, so a stricter screen would hide deals that help them.
SCREEN_MIN_THEIR_GAIN = -0.05
MIN_VALUE_RATIO = 0.85  # they must get back at least this share of perceived value
MAX_VALUE_RATIO = 1.4  # never overpay by more than this on paper
MAX_PER_TARGET = 2
MAX_PER_PARTNER = 2  # a list is ideas across the league, not one owner four times
MAX_PER_GIVE = 3  # each of my players goes in at most this many deals on a list, so it isn't one idea ten times
# Uneven trades (2-for-1, 1-for-2) are only a small part of any list: they look good on paper
# but rarely get done. Even trades (1-for-1, 2-for-2) fill the rest. When there aren't enough
# even ones the list is shorter, not padded with uneven ones. A display rule only: it never
# changes a deal's odds or gains.
MAX_UNEVEN_SHARE = 0.2
# Bold lists also try 3-player deals (3-for-3, and 2-for-3 where I get the extra player), grown
# from the best 2-for-2s with each team: trying every 3-for-3 took 36 s on League A and passed nothing.
# In 4,000 real category-league offers a 3-for-3 was accepted 15% of the time, like a 1-for-1, and was
# 7% of accepted trades; a 3-for-2 that makes them cut a player, 6% (as poor as a 2-for-1), so it's left out.
THREE_PLAYER_CORES = 4

PUNT_WIN = 0.3  # a category an owner wins less often than this is treated as punted
MISPRICED = 5.0  # dollars between owner value and our projection that count as sell-high / buy-low

# Edge mode: they win on paper, you win on the court. Their build may stay flat or
# lose a little, as long as they get at least equal trade value and you gain clearly more.
EDGE_MIN_RATIO = 1.0
EDGE_MIN_THEIR_GAIN = -0.10
EDGE_MIN_MARGIN = 0.10  # my gain minus theirs, in expected categories won per week
WASTED_SHARE = 0.4  # a player whose positive value is at least this much in punted categories
RIVAL_SPOTS = 2  # teams within this many standings places of me are playoff rivals
RIVAL_DISCOUNT = 0.8  # in edge mode, deals that also help a rival rank lower

# Giving up one of your own jewels or keystones takes a deal in the top share of what the
# finder sees that day, and never less than twice the usual bar.
SPECTACULAR_PERCENTILE = 90
SPECTACULAR_FLOOR = 2 * MIN_MY_GAIN
# A deal that takes a team's keystone category away without giving it back.
KEYSTONE_LOSS = 0.10  # weekly win rate lost in that category

# Vision mode: buy a player for what he'll be worth later, while his owner sees what he's
# worth now. Targets are off to a slow start the projection expects to fade: on League A's
# 2025 and 2026 seasons, players flagged this way at weeks 4-12 went on to play about as
# well as projected, nowhere near their season so far (scripts/vision_backtest.py). The
# deal may cost me a little this week, as long as I'm clearly stronger in the playoff weeks.
#
# Injured stars aren't targets yet: the same replay shows good players coming back from an
# absence play about 1.1-2.3 fewer z-points per game than projected (beyond the usual
# regression) and play 20-40% fewer games than expected, so the model would overpay for them.
VISION_MIN_GAMES = 5  # games played before a slow start counts
VISION_SLOW_GAP = 1.0  # projected value above his season so far (z-scores summed over categories)
VISION_MIN_PLAYOFF_GAIN = 0.10  # expected categories won per playoff week
VISION_MIN_SEASON_GAIN = -0.15  # what the deal may cost before the playoffs, per week on average

# Chance the other owner accepts: a logistic regression on 2,826 real offers (accepted vs
# withdrawn) in 13 category leagues, each replayed through this finder on the league as it
# stood that day (scripts/offer_replay.py), every league weighted the same
# (scripts/acceptance_fit.py). Checked league by league on leagues left out of the fit, the
# paper value ratio and the shape predict; the gains in categories won do not (they make the
# odds worse on held-out leagues).
#
# Declined offers are counted: ESPN never shows what was in one, but there are about three for
# every withdrawn offer, so each withdrawn offer in the fit stands for its league's declines
# too. That moves the base rate and nothing else: about 9% of real offers get a yes, a fair
# 1-for-1 lands near 13%, a 2-for-2 near 10% and a 2-for-1 near 6%. Fitted on accepted and
# withdrawn offers alone the same deals showed 31%, 25% and 17%, two to three times the real
# rate in every league left out. The order of deals is sound, the exact percentage is not
# (a league's own rate runs from 2% to 20%), so the label is what to show.
# Coefficients: intercept, log(paper ratio), 2-for-2, 2-for-1, 1-for-2 (1-for-1 = 0).
ACCEPT_COEF = (-1.914, 0.420, -0.248, -0.748, 0.007)
PAPER_CLIP = 1.5  # log paper ratio, about 0.22x to 4.5x
MIN_ODDS = 0.02
# A deal that takes away their keystone category: odds of acceptance multiplied by this.
# Not in the offer data (it has no badges); set so a 1-for-1 drops from 13% to about 8%.
KEYSTONE_ODDS = 0.6

# Labels on the real scale. Likely: one in four, twice a fair 1-for-1, which takes a clear
# overpay. Possible: at least the rate of real offers as a whole, about one in twelve.
LIKELY, POSSIBLE = 0.25, 0.08

Mode = Literal["fair", "edge", "vision"]


@dataclass
class Deal:
    partner_idx: int
    give: list[Player]
    get: list[Player]
    my_drop: Player | None  # I receive more players than I send
    their_drop: Player | None  # they receive more players than they send
    my_gain: float
    their_gain: float  # in the categories they play for
    give_value: float  # perceived value they receive
    get_value: float  # perceived value they send, including a player they must cut
    acceptance: float
    dists: list[Dist]
    reasons: list[str]
    rival: bool = False
    facts: list[schemas.ReasonFact] = field(default_factory=list)  # ``reasons`` as data, same order
    rosters: dict[int, list[Player]] = field(default_factory=dict)  # team index -> roster after the deal
    my_add: Player | None = None  # free agent I pick up for the spot a 2-for-1 opens
    faab: int = 0  # FAAB dollars I send with the players
    premium: float = 0.0  # owner premium included in get_value, in dollars
    warnings: list[str] = field(default_factory=list)  # what I give up that's hard to replace
    # The free pickup (add, drop) the deal is measured against, when the deal spends the spot it needs:
    # its own open spot takes that free agent, or it cuts or sends the player I'd drop for him.
    vs_pickup: tuple[Player, Player] | None = None
    # The same for them: the free pickup (add, drop) they could make today, when the spot the deal
    # opens on their roster takes that free agent.
    their_pickup: tuple[Player, Player] | None = None
    playoff_gain: float = 0.0  # vision mode: change in my expected categories won per playoff week
    vision: list[str] = field(default_factory=list)  # vision mode: why the players I get are worth more later
    # Change in their expected categories won counting every category, as the season simulation does.
    their_strength: float = 0.0

    @property
    def edge(self) -> float:
        """How much more the deal does for me than for them, in categories won per week."""
        return self.my_gain - self.their_gain

    @property
    def score(self) -> float:
        return self.acceptance * self.my_gain


def acceptance_odds(ratio: float, n_give: int = 1, n_get: int = 1) -> float:
    """Chance the other owner says yes, fitted on real offers (see ACCEPT_COEF).

    ``ratio`` is the trade value they get over what they send, owner premium included, so
    stars and keystones need a bigger offer for the same odds. ``n_give``/``n_get`` count the
    players I send and get: a 2-for-1 makes them cut someone. Shapes the data doesn't have
    take the closest even or uneven one.
    """
    c0, c_paper, c22, c21, c12 = ACCEPT_COEF
    if n_give == n_get:
        shape = 0.0 if n_give == 1 else c22
    else:
        shape = c21 if n_give > n_get else c12
    x = c0 + c_paper * float(np.clip(np.log(max(ratio, 1e-6)), -PAPER_CLIP, PAPER_CLIP)) + shape
    return max(float(1 / (1 + np.exp(-x))), MIN_ODDS)


def keystone_odds(p: float) -> float:
    """Acceptance ``p`` for a deal that takes away their keystone category, on the odds scale."""
    return max(p * KEYSTONE_ODDS / (1 - p + p * KEYSTONE_ODDS), MIN_ODDS)


class Picks:
    """Deals picked so far, within the per-target, per-partner, per-player and shape limits of a list of ``n``.

    One deal per core idea: the same main player each way with the same owner is one idea, whatever
    else is added to it ("Allen for Avdija" and "Allen + Mitchell for Avdija + George").

    Used while simulating down the queue: a deal that couldn't make the list isn't simulated.
    """

    def __init__(self, n: int, value: Callable[[Player], float]):
        self.n = n
        self.value = value
        self.uneven_cap = max(1, round(n * MAX_UNEVEN_SHARE))
        self.even = self.uneven = 0
        self.per_target: dict[int, int] = {}
        self.per_partner: dict[int, int] = {}
        self.per_give: dict[int, int] = {}
        self.cores: set[tuple[int, int, int]] = set()

    def _target(self, d: Deal) -> int:
        return max(d.get, key=self.value).id

    def _core(self, d: Deal) -> tuple[int, int, int]:
        """The idea behind a deal: the owner, the main player I send and the main player I get."""
        return d.partner_idx, max(d.give, key=self.value).id, self._target(d)

    def room(self, d: Deal) -> bool:
        if len(d.give) != len(d.get) and self.uneven >= self.uneven_cap:
            return False
        return (
            self._core(d) not in self.cores
            and self.per_target.get(self._target(d), 0) < MAX_PER_TARGET
            and self.per_partner.get(d.partner_idx, 0) < MAX_PER_PARTNER
            and all(self.per_give.get(p.id, 0) < MAX_PER_GIVE for p in d.give)
        )

    def add(self, d: Deal) -> None:
        if len(d.give) != len(d.get):
            self.uneven += 1
        else:
            self.even += 1
        t = self._target(d)
        self.per_target[t] = self.per_target.get(t, 0) + 1
        self.per_partner[d.partner_idx] = self.per_partner.get(d.partner_idx, 0) + 1
        for p in d.give:
            self.per_give[p.id] = self.per_give.get(p.id, 0) + 1
        self.cores.add(self._core(d))

    @property
    def full(self) -> bool:
        return self.even + self.uneven >= self.n


def mix_shapes[T](ranked: list[T], n: int, uneven: Callable[[T], bool]) -> list[T]:
    """The first ``n`` of ``ranked``, in rank order, with at most MAX_UNEVEN_SHARE of ``n`` uneven."""
    cap = max(1, round(n * MAX_UNEVEN_SHARE))
    out: list[T] = []
    used = 0
    for item in ranked:
        if uneven(item):
            if used == cap:
                continue
            used += 1
        out.append(item)
        if len(out) == n:
            break
    return out


def acceptance_label(p: float) -> str:
    return "likely" if p >= LIKELY else "possible" if p >= POSSIBLE else "long_shot"


class TradeFinder:
    def __init__(
        self,
        league: League,
        model: CategoryModel,
        me_idx: int,
        keep: set[int],
        mode: Mode = "fair",
        premium: OwnerPremium | None = None,
        min_their_gain: float = MIN_THEIR_GAIN,
    ):
        """``min_their_gain`` is the fair-mode floor for their gain in the categories they play for."""
        self.mode = mode
        self.min_their_gain = min_their_gain
        self.league = league
        self.model = model
        self.me_idx = me_idx
        self.keep = keep
        self.perceived = premium.perceived if premium is not None else PerceivedValue(league, model)
        self.premium = premium or OwnerPremium(league, model, self.perceived)
        self.teams = league.teams
        self.dists = [model.dist(t.roster) for t in self.teams]
        self.base = [model.strength(i, self.dists) for i in range(len(self.teams))]
        self.base_cats = [self._avg_win(i, self.dists) for i in range(len(self.teams))]
        self.punts = [cats < PUNT_WIN for cats in self.base_cats]
        self.standing: dict[int, int] = {}  # team index -> place in the standings, once games are played
        self.struggling = self._outside_playoffs()
        # My players the other owners could have for free: throw-ins, worth nothing to them on paper.
        self.wire = {p.id for p in self.teams[me_idx].roster if off_the_wire(p, self.premium.now_ms)}
        self.filler = best_pickup(league, model)
        self.free_move, self.free_roster, self.free_base = self._free_move()
        cal = league.calendar
        self.playoff_periods = (
            sorted(p for p in cal.period_days if p > league.regular_season_periods) if cal is not None else []
        )
        self.base_playoff = self.playoff_strength({}) if mode == "vision" else None
        self.free_playoff = (
            self.playoff_strength({self.me_idx: self.free_roster}) if mode == "vision" and self.free_roster else None
        )
        self.rivals = self._rivals()
        # team index -> its category win rates after its best free pickup, and that pickup (add, drop) if it helps
        self._their_free: dict[int, tuple[np.ndarray, tuple[Player, Player] | None]] = {}
        self._gains: dict[tuple[int, int, int], float] = {}  # (partner, mine, theirs) -> my gain from that 1-for-1
        # Our projection in dollars: a player's model-value rank priced on ESPN's rank curve.
        pool = [p for t in self.teams for p in t.roster] + list(league.free_agents)
        by_model = sorted(pool, key=model.value, reverse=True)
        self.model_dollars = {p.id: self.perceived.dollars_at_rank(r) for r, p in enumerate(by_model, 1)}

    def _free_move(self) -> tuple[tuple[Player, Player] | None, list[Player] | None, float | None]:
        """The best pickup I can make without a trade: any free agent I can count on, for one of my
        droppable players, judged on what he does for my team (as the pickups list is).

        Not just the best free agent by value: the one that fits my categories can be someone else,
        and a deal measured against a weaker pickup would get credit for a waiver claim.
        Returns (add, drop), my roster after it, and my strength after it; all None when it doesn't help.
        """
        me = self.teams[self.me_idx]
        drops = droppable(me.roster, self.keep)
        best = None
        for add in pickup_pool(self.league):
            for drop in drops:
                roster = [p for p in me.roster if p.id != drop.id] + [add]
                dists = list(self.dists)
                dists[self.me_idx] = self.model.dist(roster)
                strength = self.model.strength(self.me_idx, dists)
                if strength > self.base[self.me_idx] and (best is None or strength > best[2]):
                    best = ((add, drop), roster, strength)
        return best or (None, None, None)

    def playoff_strength(self, rosters: dict[int, list[Player]]) -> float | None:
        """My expected categories won per playoff week, each week on its own games, with these rosters swapped in."""
        if not self.playoff_periods:
            return None
        teams = [rosters.get(i, t.roster) for i, t in enumerate(self.teams)]
        return float(
            np.mean(
                [self.model.strength(self.me_idx, [self.model.dist(r, p) for r in teams]) for p in self.playoff_periods]
            )
        )

    def vision_signals(self, p: Player) -> list[str]:
        """Why a player is worth more later than his owner sees now. Empty: not a vision target."""
        out = []
        if p.available and p.games_played >= VISION_MIN_GAMES and p.season_line:
            gap = self.model.value(p) - self.model.line_value(p.season_line)
            if gap >= VISION_SLOW_GAP:
                out.append(
                    f"Slow start: his {p.games_played} games so far are well below what he's projected to do the "
                    "rest of the way. Early stats like his usually bounce back."
                )
        return out

    def _outside_playoffs(self) -> set[int]:
        def pct(t: Team) -> float:
            games = t.wins + t.losses + t.ties
            return (t.wins + 0.5 * t.ties) / games if games else 0.5

        if not any(t.wins + t.losses + t.ties for t in self.teams):
            return set()
        order = sorted(range(len(self.teams)), key=lambda i: pct(self.teams[i]), reverse=True)
        self.standing = {i: r for r, i in enumerate(order, 1)}
        return set(order[self.league.playoff_teams :])

    def _rivals(self) -> set[int]:
        """Teams near me in the standings: their gains cost me playoff odds most directly."""

        def pct(t: Team) -> float:
            games = t.wins + t.losses + t.ties
            return (t.wins + 0.5 * t.ties) / games if games else 0.5

        if any(t.wins + t.losses + t.ties for t in self.teams):
            key = [pct(t) for t in self.teams]
        else:
            key = self.base  # no games yet: rank by projected strength
        order = sorted(range(len(self.teams)), key=lambda i: key[i], reverse=True)
        pos = order.index(self.me_idx)
        return {i for k, i in enumerate(order) if i != self.me_idx and abs(k - pos) <= RIVAL_SPOTS}

    def fit_gain(
        self, j: int, dists: list[Dist], baseline: np.ndarray | None = None, every_category: bool = False
    ) -> float:
        """Change in team j's expected categories won, ignoring the categories it punts unless ``every_category``."""
        change = self._avg_win(j, dists) - (self.base_cats[j] if baseline is None else baseline)
        return float(change.sum() if every_category else change[~self.punts[j]].sum())

    def their_free_cats(self, j: int) -> np.ndarray:
        """Team j's category win rates after the best pickup it can make without a trade.

        The same free agent fills the spot a 1-for-2 opens on their roster, so a deal gets
        credit only for what it adds on top of that pickup, as it does on my side.
        """
        return self._their_free_move(j)[0]

    def their_free_move(self, j: int) -> tuple[Player, Player] | None:
        """The pickup (add, drop) behind ``their_free_cats``; None when no free pickup helps team j."""
        return self._their_free_move(j)[1]

    def _their_free_move(self, j: int) -> tuple[np.ndarray, tuple[Player, Player] | None]:
        if j not in self._their_free:
            best, move = self.base_cats[j], None
            if self.filler is not None:
                roster = self.teams[j].roster
                top = self.base[j]
                plays = ~self.punts[j]
                for drop in droppable(roster):
                    dists = list(self.dists)
                    dists[j] = self.model.dist([p for p in roster if p.id != drop.id] + [self.filler])
                    strength = self.model.strength(j, dists)
                    cats = self._avg_win(j, dists)
                    # A pickup they would make: stronger overall, and no worse in the categories they
                    # play for. One that only helps a category they punt is no bar for a deal to clear.
                    if strength > top and cats[plays].sum() >= self.base_cats[j][plays].sum():
                        top, best, move = strength, cats, (self.filler, drop)
            self._their_free[j] = (best, move)
        return self._their_free[j]

    def _worst(self, roster: list[Player], protected: set[int]) -> Player | None:
        """The player an owner would cut to make room: the least valuable droppable one.

        If every candidate has a market, the owner still has to cut someone, and the
        value lost counts against the deal on paper.
        """
        options = [p for p in roster if p.id not in protected]
        cuttable = droppable(options) or options
        return min(cuttable, key=self.model.value) if cuttable else None

    def _my_cut(self, roster: list[Player], protected: set[int]) -> Player | None:
        """Who I cut when I take back more players than I send: never someone with trade value."""
        options = droppable([p for p in roster if p.id not in protected], self.keep)
        return min(options, key=self.model.value) if options else None

    def give_price(self, p: Player) -> float:
        """What one of my players is worth to the other owner: nothing when he came off the waiver wire."""
        return 0.0 if p.id in self.wire else self.perceived.value(p)

    def _evaluate(
        self,
        j: int,
        give: list[Player],
        get: list[Player],
        min_my_gain: float = MIN_MY_GAIN,
        min_their_gain: float | None = None,
        *,
        min_ratio: float | None = None,
        min_margin: float | None = None,
        extra_give_value: float = 0.0,
    ) -> Deal | None:
        """Score one package. The threshold arguments override the mode's defaults."""
        pv = self.perceived
        me, partner = self.teams[self.me_idx], self.teams[j]
        give_ids, get_ids = {p.id for p in give}, {p.id for p in get}
        their_roster = [p for p in partner.roster if p.id not in get_ids] + give
        their_drop = None
        vision = self.mode == "vision"
        if len(give) > len(get):
            their_drop = self._worst(their_roster, give_ids)
            their_roster = [p for p in their_roster if p is not their_drop]

        give_v = pv.package(give, self.give_price) + extra_give_value  # e.g. FAAB dollars sent with the players
        get_v = pv.package(get, self.premium.price)
        premium = get_v - pv.package(get)
        if their_drop is not None:
            get_v += max(pv.value(their_drop) - pv.replacement, 0.0)
        if get_v <= 0:
            return None
        ratio = give_v / get_v
        if min_ratio is None:
            min_ratio = EDGE_MIN_RATIO if self.mode == "edge" else MIN_VALUE_RATIO
        if ratio < min_ratio or ratio > MAX_VALUE_RATIO:
            return None

        my_roster = [p for p in me.roster if p.id not in give_ids] + get
        my_drop = None
        if len(get) > len(give):
            my_drop = self._my_cut(my_roster, self.keep | get_ids)
            if my_drop is None:
                return None
            my_roster = [p for p in my_roster if p is not my_drop]
        # The side that sends more players gets an open spot and fills it from free agency.
        # In a daily league that spot plays real games, so leaving it empty would undersell the deal.
        # It takes the same free agent as my best pickup, so the two are measured alike.
        my_add = (self.free_move[0] if self.free_move else self.filler) if len(give) > len(get) else None
        if my_add is not None:
            my_roster = my_roster + [my_add]
        if len(get) > len(give) and self.filler is not None:
            their_roster = their_roster + [self.filler]

        dists = list(self.dists)
        dists[self.me_idx] = self.model.dist(my_roster)
        dists[j] = self.model.dist(their_roster)
        # A deal that spends the spot my free pickup needs is compared with just making that pickup:
        # its open spot takes the same free agent, or it cuts or sends the player I'd drop for him.
        # Either way the pickup's gain is mine without the trade.
        free = self.free_move
        spends_spot = free is not None and (
            my_add is not None or free[1].id in give_ids or (my_drop is not None and my_drop.id == free[1].id)
        )
        vs_pickup = free if spends_spot else None
        baseline = self.free_base if vs_pickup is not None else self.base[self.me_idx]
        my_gain = self.model.strength(self.me_idx, dists) - baseline
        if vision and min_my_gain == MIN_MY_GAIN:
            min_my_gain = VISION_MIN_SEASON_GAIN
        if my_gain < min_my_gain:
            return None
        # When the deal's open spot on their side takes the free agent, compare with just making that pickup.
        opens_their_spot = len(get) > len(give)
        their_base = self.their_free_cats(j) if opens_their_spot else None
        their_gain = self.fit_gain(j, dists, their_base)
        edge = self.mode == "edge"
        if min_their_gain is None:
            min_their_gain = EDGE_MIN_THEIR_GAIN if edge else self.min_their_gain
        if min_margin is None:
            min_margin = EDGE_MIN_MARGIN if edge else float("-inf")
        if their_gain < min_their_gain or my_gain - their_gain < min_margin:
            return None
        playoff_gain = 0.0
        if vision:
            after = self.playoff_strength({self.me_idx: my_roster, j: their_roster})
            # Measured against the free pickup too, when the deal's open spot takes it.
            before = self.base_playoff if vs_pickup is None else self.free_playoff
            if after is None or before is None:
                return None
            playoff_gain = after - before
            if playoff_gain < VISION_MIN_PLAYOFF_GAIN:
                return None
        if len(give) == len(get) == 1:
            # What a 2-for-2 containing it must beat. Only a 1-for-1 that works counts: one the other
            # owner loses on isn't the simpler deal, and the second pair is what makes it work for them.
            self._gains[(j, give[0].id, get[0].id)] = my_gain
        struggling = j in self.struggling
        facts = self._reasons(j, give, get, dists, struggling)
        acceptance = acceptance_odds(ratio, len(give), len(get))
        if len(give) > len(get):
            facts.append(
                _fact(
                    "roster_squeeze",
                    "against",
                    "They have to cut a player to fit two of yours. Real offers like this get accepted "
                    "less than half as often as a 1-for-1 of the same value.",
                    count=len(give) - len(get),
                )
            )
        lost = self._keystones_lost(j, get, dists)
        if lost:
            acceptance = keystone_odds(acceptance)
            facts.append(
                _fact(
                    "keystone_lost",
                    "against",
                    f"It takes away their {', '.join(lost)} keystone without replacing it: a harder sell.",
                    categories=lost,
                )
            )
        rival = j in self.rivals
        if self.mode == "edge":
            facts = self._edge_reasons(j, give, get, ratio, rival) + facts
        if vs_pickup is not None:
            add, drop = vs_pickup
            facts.append(
                _fact(
                    "vs_pickup",
                    "note",
                    f"Measured against just adding {add.name} for {drop.name}, which needs no trade: "
                    "the odds above count only what the trade adds.",
                    player=add,
                )
            )
        their_pickup = self.their_free_move(j) if opens_their_spot else None
        if their_pickup is not None:
            add, drop = their_pickup
            facts.append(
                _fact(
                    "vs_pickup",
                    "note",
                    f"Their odds are measured against them just adding {add.name} for {drop.name}, which needs "
                    "no trade: the free agent in their open spot isn't counted as the trade's doing.",
                    player=add,
                )
            )
        reasons = [f.text for f in facts]
        return Deal(
            partner_idx=j,
            give=give,
            get=get,
            my_drop=my_drop,
            their_drop=their_drop,
            my_gain=my_gain,
            their_gain=their_gain,
            give_value=give_v,
            get_value=get_v,
            acceptance=acceptance,
            dists=dists,
            reasons=reasons,
            facts=facts,
            rival=rival,
            rosters={self.me_idx: my_roster, j: their_roster},
            my_add=my_add,
            premium=premium,
            warnings=self._warnings(give),
            vs_pickup=vs_pickup,
            their_pickup=their_pickup,
            playoff_gain=playoff_gain,
            vision=[line for p in get for line in self.vision_signals(p)] if vision else [],
            their_strength=self.fit_gain(j, dists, their_base, every_category=True),
        )

    def _keystones_lost(self, j: int, get: list[Player], dists: list[Dist]) -> list[str]:
        """Categories a team loses its keystone in, with nothing in the deal making up for it."""
        cats = {self.premium.keystone(p) for p in get} - {None}
        if not cats:
            return []
        change = self._avg_win(j, dists) - self.base_cats[j]
        names = [c.name for c in self.model.cats]
        return [c for c in sorted(cats) if change[names.index(c)] <= -KEYSTONE_LOSS]

    def _warnings(self, give: list[Player]) -> list[str]:
        out = []
        for p in give:
            for b in self.premium.badges(p):
                if b.kind == "jewel":
                    out.append(f"You give up {p.name}, one of the league's jewels.")
                elif b.kind == "keystone":
                    out.append(f"You give up {p.name}, your {b.category} keystone.")
        return out

    def _wasted(self, p: Player, punts: np.ndarray) -> list[str]:
        """Categories a player's value mostly comes from that this team punts."""
        z = np.maximum(self.model.z_scores(p), 0.0)
        if z.sum() <= 0 or z[punts].sum() < WASTED_SHARE * z.sum():
            return []
        return [self.model.cats[c].name for c in np.flatnonzero(punts & (z > 0.25))]

    def _edge_reasons(
        self, j: int, give: list[Player], get: list[Player], ratio: float, rival: bool
    ) -> list[schemas.ReasonFact]:
        """Why the deal favors me even though it looks like a win for them."""
        out = []
        if ratio >= 1.05:
            out.append(
                _fact(
                    "paper_win",
                    "for",
                    f"On paper they win: they get {ratio:.0%} of the trade value they send.",
                    ratio=round(ratio, 3),
                )
            )
        for p in give:
            cats = self._wasted(p, self.punts[self.me_idx])
            if cats:
                text = f"{p.name}'s value is mostly {', '.join(cats)}, which you punt: it costs you little."
                out.append(_fact("punt_mine", "note", text, player=p, categories=cats))
        for p in get:
            cats = self._wasted(p, self.punts[j])
            if cats:
                text = f"{p.name}'s {', '.join(cats)} is wasted on them: they punt it."
                out.append(_fact("punt_theirs", "note", text, player=p, categories=cats))
        if not rival:
            out.append(_fact("not_rival", "note", "They aren't one of your direct playoff rivals."))
        return out

    def _reasons(
        self, j: int, give: list[Player], get: list[Player], dists: list[Dist], struggling: bool
    ) -> list[schemas.ReasonFact]:
        """Talking points for the pitch, and why the deal can work for both sides."""
        pv, out = self.perceived, []
        at = "at" if self.league.is_auction else "like"  # "at $12" / "like a 4th-round pick"
        for kind, players in (("sell_high", give), ("buy_low", get)):
            for p in players:
                if kind == "sell_high" and p.id in self.wire:
                    continue  # nobody pays for a player who was free last week
                owners, ours = pv.value(p), self.model_dollars.get(p.id, 0.0)
                gap = owners - ours if kind == "sell_high" else ours - owners
                if gap >= MISPRICED and pv.describe(owners) != pv.describe(ours):
                    word = "Sell high" if kind == "sell_high" else "Buy low"
                    out.append(
                        _fact(
                            kind,
                            "for",
                            f"{word}: owners value {p.name} {at} {pv.describe(owners)}, {_ours(pv, ours)}.",
                            player=p,
                            owners_value=round(owners, 1),
                            our_value=round(ours, 1),
                            owners_label=pv.describe(owners),
                            our_label=pv.describe(ours) if ours >= 1 else "replacement level",
                        )
                    )
        change = self._avg_win(j, dists) - self.base_cats[j]
        lost = [self.model.cats[c].name for c in np.flatnonzero(self.punts[j] & (change < -0.02))]
        if lost:
            text = f"They punt {', '.join(lost)}, so what they give up there doesn't cost them."
            out.append(_fact("punt_cost", "for", text, categories=lost))
        partner = self.teams[j]
        out_count = sum(not p.available for p in partner.roster)
        if len(give) > len(get) and out_count >= 2:
            text = f"They have {out_count} players out, so an extra healthy body helps them now."
            out.append(_fact("players_out", "for", text, count=out_count))
        if struggling:
            out.append(
                _fact(
                    "chasing",
                    "for",
                    "They're outside the playoff spots, so they have reason to shake things up.",
                    rank=self.standing.get(j),
                    teams=len(self.teams),
                    playoff_spots=self.league.playoff_teams,
                )
            )
        return out

    def candidates(
        self,
        partners: Iterable[int] | None = None,
        two_for_two: bool = True,
        uneven: bool = True,
        on_partner: Callable[[int, int], None] | None = None,
    ) -> list[Deal]:
        """Every deal that passes the checks. ``uneven=False`` leaves out 2-for-1s and 1-for-2s.

        ``on_partner(done, total)`` is called after each partner team, for a progress bar.
        """
        me = self.teams[self.me_idx]
        mine = [p for p in me.roster if p.id not in self.keep and not self.premium.not_for_sale(p)]
        deals: list[Deal] = []
        if self.mode == "vision":
            return self._vision_candidates(mine, partners, on_partner)
        teams = [j for j in (range(len(self.teams)) if partners is None else partners) if j != self.me_idx]
        for n, j in enumerate(teams):
            if on_partner is not None:
                on_partner(n, len(teams))
            theirs = [p for p in self.teams[j].roster if not self.premium.not_for_sale(p)]
            for give in mine:
                for get in theirs:
                    d = self._evaluate(j, [give], [get])
                    if d:
                        deals.append(d)
            if uneven:
                # 2-for-1: package two of mine for one of theirs.
                for give_pair in combinations(mine, 2):
                    for get in theirs:
                        d = self._evaluate(j, list(give_pair), [get])
                        if d:
                            deals.append(d)
                # 1-for-2: one of mine for two of theirs.
                for give in mine:
                    for get_pair in combinations(theirs, 2):
                        d = self._evaluate(j, [give], list(get_pair))
                        if d:
                            deals.append(d)
            if two_for_two:
                pairs = self._two_for_two(j, mine, theirs)
                deals += pairs
                if self.mode == "edge":
                    deals += self._three_player(j, mine, theirs, pairs)
        return deals

    def _three_player(self, j: int, mine: list[Player], theirs: list[Player], cores: list[Deal]) -> list[Deal]:
        """3-for-3s and 2-for-3s grown from the best 2-for-2s with team ``j``: one more player on their
        side, and maybe one more on mine. A deal is kept only when it does more for me than its core."""
        mine = [p for p in mine if p.id not in self.wire]
        theirs = [p for p in theirs if p not in droppable(theirs)]
        out: list[Deal] = []
        seen: set[tuple[frozenset[int], frozenset[int]]] = set()
        for core in sorted(cores, key=self.sim_key, reverse=True)[:THREE_PLAYER_CORES]:
            for extra_get in theirs:
                if extra_get in core.get:
                    continue
                get = core.get + [extra_get]
                for extra_give in [None, *mine]:
                    if extra_give is not None and extra_give in core.give:
                        continue
                    give = core.give + ([extra_give] if extra_give is not None else [])
                    key = (frozenset(p.id for p in give), frozenset(p.id for p in get))
                    if key in seen:
                        continue
                    seen.add(key)
                    d = self._evaluate(j, give, get)
                    if d is not None and d.my_gain > core.my_gain:
                        out.append(d)
        return out

    def _two_for_two(self, j: int, mine: list[Player], theirs: list[Player]) -> list[Deal]:
        """Two of mine for two of theirs.

        Pairs are matched on trade value first, which is cheap, so only packages that could
        pass the value check get the full evaluation. A 2-for-2 is kept only when it does more
        for me than the best 1-for-1 inside it that is a deal on its own: otherwise the extra
        players are filler.
        """
        pv = self.perceived
        lo = EDGE_MIN_RATIO if self.mode == "edge" else MIN_VALUE_RATIO
        # A player off the waiver wire is a throw-in, not half of a 2-for-2: that deal is a 2-for-1.
        mine = [p for p in mine if p.id not in self.wire]
        theirs = [p for p in theirs if p not in droppable(theirs)]
        gives = [(pair, pv.package(list(pair))) for pair in combinations(mine, 2)]
        gets = [(pair, pv.package(list(pair), self.premium.price)) for pair in combinations(theirs, 2)]
        out = []
        for give_pair, give_v in gives:
            for get_pair, get_v in gets:
                if get_v <= 0 or not lo <= give_v / get_v <= MAX_VALUE_RATIO:
                    continue
                d = self._evaluate(j, list(give_pair), list(get_pair))
                if d is None:
                    continue
                inside = max(
                    (self._gains.get((j, g.id, t.id), float("-inf")) for g in give_pair for t in get_pair),
                    default=float("-inf"),
                )
                if d.my_gain > inside:
                    out.append(d)
        return out

    def _vision_candidates(
        self,
        mine: list[Player],
        partners: Iterable[int] | None,
        on_partner: Callable[[int, int], None] | None = None,
    ) -> list[Deal]:
        """One target with a vision signal for one or two of my healthy players: they get production now."""
        healthy = [p for p in mine if p.available]
        deals: list[Deal] = []
        teams = [j for j in (range(len(self.teams)) if partners is None else partners) if j != self.me_idx]
        for n, j in enumerate(teams):
            if on_partner is not None:
                on_partner(n, len(teams))
            targets = [p for p in self.teams[j].roster if not self.premium.not_for_sale(p) and self.vision_signals(p)]
            for get in targets:
                for give in healthy:
                    d = self._evaluate(j, [give], [get])
                    if d:
                        deals.append(d)
                for pair in combinations(healthy, 2):
                    d = self._evaluate(j, list(pair), [get])
                    if d:
                        deals.append(d)
        return deals

    def rank_key(self, d: Deal) -> float:
        if self.mode == "vision":
            return d.acceptance * d.playoff_gain
        if self.mode == "edge":
            return d.score * (RIVAL_DISCOUNT if d.rival and d.their_gain > 0 else 1.0)
        return d.score

    def spectacular(self, deals: list[Deal]) -> float:
        """The gain a deal needs before it gives up one of my jewels or keystones."""
        if not deals:
            return SPECTACULAR_FLOOR
        return max(SPECTACULAR_FLOOR, float(np.percentile([d.my_gain for d in deals], SPECTACULAR_PERCENTILE)))

    def sim_key(self, d: Deal) -> float:
        """How likely a deal is to raise both teams' playoff odds: the smaller of the two gains.

        On League A (2026-11-05, 2,549 deals simulated) the deals that passed ranked 74th and
        129th by this key, against 1,308th and 2,022nd by ``rank_key``, which looks at my side only.
        """
        if self.mode == "vision":
            return self.rank_key(d)
        return min(d.my_gain, d.their_strength)

    def queue(self, on_partner: Callable[[int, int], None] | None = None) -> list[Deal]:
        """Every deal worth simulating, the likeliest to help both teams' playoff odds first."""
        deals = self.candidates(on_partner=on_partner)
        bar = self.spectacular(deals)
        deals = [d for d in deals if d.my_gain >= bar or not any(self.premium.is_attached(p) for p in d.give)]
        return sorted(deals, key=self.sim_key, reverse=True)

    def picks(self, n: int) -> Picks:
        return Picks(n, self.perceived.value)

    def pitch_needs(self, d: Deal, min_change: float = 0.03, limit: int = 2) -> list[schemas.PitchNeed]:
        """The categories where the deal helps them most, with their rank and what it adds a game.

        Only categories they compete in, and never turnovers first.
        """
        j = d.partner_idx
        before = self._avg_win(j, self.dists)
        after = self._avg_win(j, d.dists)
        change = after - before
        mus = np.array([x.mu for x in self.dists])
        signed = np.where(self.model.reverse, -mus, mus)
        ranks = (signed > signed[j]).sum(0) + 1
        # A category they punt is no selling point, and fewer turnovers never open a pitch:
        # no owner trades for them. Turnovers can still back up a real need.
        helped = [c for c in np.argsort(-change) if change[c] >= min_change and not self.punts[j][c]]
        lead = next((c for c in helped if self.model.cats[c].stat_id != TURNOVERS), None)
        order = [] if lead is None else ([lead] + [c for c in helped if c != lead])[:limit]
        out = []
        for c in order:
            cat = self.model.cats[c]
            per_game = None
            if not cat.ratio:
                delta = sum(p.per_game.get(cat.stat_id, 0.0) for p in d.give) - sum(
                    p.per_game.get(cat.stat_id, 0.0) for p in d.get
                )
                if cat.reverse:
                    delta = -delta
                if delta >= 0.5:
                    per_game = round(delta, 1)
            out.append(schemas.PitchNeed(category=cat.name, league_rank=int(ranks[c]), per_game=per_game))
        return out

    def pitch(self, d: Deal) -> str:
        """One line on why the deal helps them, in the categories where they need it most."""
        needs = self.pitch_needs(d)
        if not needs:
            return "Balances their roster without costing them a category."
        parts = []
        for n in needs:
            line = f"{n.category} ({_ordinal(n.league_rank)} in the league)"
            if n.per_game is not None:
                line += f", +{n.per_game:.1f} a game"
            parts.append(line)
        return "Helps their " + " and ".join(parts) + "."

    def _avg_win(self, i: int, dists: list[Dist]) -> np.ndarray:
        return self.model.avg_cat_win(i, dists)


def _fact(kind: schemas.FactKind, tone: str, text: str, player: Player | None = None, **data) -> schemas.ReasonFact:
    if player is not None:
        data.update(player_id=player.id, player_name=player.name)
    return schemas.ReasonFact(kind=kind, tone=tone, text=text, **data)


def _ours(pv: PerceivedValue, dollars: float) -> str:
    if dollars < 1:
        return "but he's near replacement level in this league's categories"
    return f"but he's worth {pv.describe(dollars)} in this league's categories"


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"

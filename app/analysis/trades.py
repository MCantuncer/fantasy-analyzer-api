"""Trade finder: deals that help you and that the other owner has a real reason to accept.

A deal is suggested only when
  * it raises your expected categories won,
  * it raises theirs too, counting only the categories they compete in (not their punts), and
  * it doesn't look like a loss to an owner who goes by rankings and draft prices.

"On paper" follows how owners judge deals: one star is worth more than two players
adding up to the same dollars, and a player they must cut to make room counts
against the deal unless waivers have someone as good.

Each surviving deal gets an acceptance estimate and a list of reasons to use when
pitching it. Deals are ranked by playoff-odds gain weighted by that estimate.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import combinations
from typing import Literal

import numpy as np

from app.analysis.model import CategoryModel, Dist
from app.analysis.perceived import PerceivedValue
from app.analysis.tiers import droppable
from app.espn.parse import League, Player, Team

MIN_MY_GAIN = 0.05  # expected categories won per week
MIN_THEIR_GAIN = 0.02
MIN_VALUE_RATIO = 0.85  # they must get back at least this share of perceived value
MAX_VALUE_RATIO = 1.4  # never overpay by more than this on paper
TOP_PLAYERS_PROTECTED = 2
SHORTLIST = 14
MAX_PER_TARGET = 2
MAX_PER_PARTNER = 4

PUNT_WIN = 0.3  # a category an owner wins less often than this is treated as punted
MISPRICED = 5.0  # dollars between owner value and our projection that count as sell-high / buy-low
STRUGGLING_BONUS = 0.05  # teams outside the playoff spots are more open to shaking things up

# Edge mode: they win on paper, you win on the court. Their build may stay flat or
# lose a little, as long as they get at least equal trade value and you gain clearly more.
EDGE_MIN_RATIO = 1.0
EDGE_MIN_THEIR_GAIN = -0.10
EDGE_MIN_MARGIN = 0.10  # my gain minus theirs, in expected categories won per week
WASTED_SHARE = 0.4  # a player whose positive value is at least this much in punted categories
RIVAL_SPOTS = 2  # teams within this many standings places of me are playoff rivals
RIVAL_DISCOUNT = 0.8  # in edge mode, deals that also help a rival rank lower

LIKELY, POSSIBLE = 0.6, 0.4

Mode = Literal["fair", "edge"]


@dataclass
class Deal:
    partner_idx: int
    give: list[Player]
    get: list[Player]
    my_drop: Player | None  # I receive more players than I send
    their_drop: Player | None  # they receive more players than they send
    my_gain: float
    their_gain: float
    give_value: float  # perceived value they receive
    get_value: float  # perceived value they send, including a player they must cut
    acceptance: float
    dists: list[Dist]
    reasons: list[str]
    rival: bool = False
    rosters: dict[int, list[Player]] = field(default_factory=dict)  # team index -> roster after the deal
    my_add: Player | None = None  # free agent I pick up for the spot a 2-for-1 opens
    faab: int = 0  # FAAB dollars I send with the players

    @property
    def edge(self) -> float:
        """How much more the deal does for me than for them, in categories won per week."""
        return self.my_gain - self.their_gain

    @property
    def score(self) -> float:
        return self.acceptance * self.my_gain


def acceptance_odds(ratio: float, their_gain: float, gives_star: bool, struggling: bool = False) -> float:
    """Rough chance the other owner says yes. A heuristic, not a fitted model."""
    p = float(np.interp(ratio, [0.85, 0.95, 1.05, 1.2, 1.4], [0.2, 0.4, 0.55, 0.7, 0.8]))
    if their_gain >= 0.15:
        p += 0.1
    elif their_gain < 0.05:
        p -= 0.1
    if gives_star and ratio < 1.15:
        p -= 0.2
    if struggling:
        p += STRUGGLING_BONUS
    return float(np.clip(p, 0.05, 0.9))


def acceptance_label(p: float) -> str:
    return "likely" if p >= LIKELY else "possible" if p >= POSSIBLE else "long_shot"


class TradeFinder:
    def __init__(self, league: League, model: CategoryModel, me_idx: int, keep: set[int], mode: Mode = "fair"):
        self.mode = mode
        self.league = league
        self.model = model
        self.me_idx = me_idx
        self.keep = keep
        self.perceived = PerceivedValue(league, model)
        self.teams = league.teams
        self.dists = [model.dist(t.roster) for t in self.teams]
        self.base = [model.strength(i, self.dists) for i in range(len(self.teams))]
        self.base_cats = [self._avg_win(i, self.dists) for i in range(len(self.teams))]
        self.punts = [cats < PUNT_WIN for cats in self.base_cats]
        self.struggling = self._outside_playoffs()
        healthy = [
            p for p in league.free_agents if p.available and p.pro_team_id and (p.has_prior or p.games_played >= 5)
        ]
        self.filler = max(healthy, key=model.value, default=None)
        self.rivals = self._rivals()
        # Our projection in dollars: a player's model-value rank priced on ESPN's rank curve.
        pool = [p for t in self.teams for p in t.roster] + list(league.free_agents)
        by_model = sorted(pool, key=model.value, reverse=True)
        self.model_dollars = {p.id: self.perceived.dollars_at_rank(r) for r, p in enumerate(by_model, 1)}

    def _outside_playoffs(self) -> set[int]:
        def pct(t: Team) -> float:
            games = t.wins + t.losses + t.ties
            return (t.wins + 0.5 * t.ties) / games if games else 0.5

        if not any(t.wins + t.losses + t.ties for t in self.teams):
            return set()
        order = sorted(range(len(self.teams)), key=lambda i: pct(self.teams[i]), reverse=True)
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

    def fit_gain(self, j: int, dists: list[Dist]) -> float:
        """Change in team j's expected categories won, ignoring the categories it punts."""
        change = self._avg_win(j, dists) - self.base_cats[j]
        return float(change[~self.punts[j]].sum())

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

    def _stars(self, team: Team) -> set[int]:
        top = sorted(team.roster, key=self.perceived.value, reverse=True)[:TOP_PLAYERS_PROTECTED]
        return {p.id for p in top}

    def _evaluate(
        self,
        j: int,
        give: list[Player],
        get: list[Player],
        stars: set[int],
        *,
        min_ratio: float | None = None,
        min_my_gain: float = MIN_MY_GAIN,
        min_their_gain: float | None = None,
        min_margin: float | None = None,
        extra_give_value: float = 0.0,
    ) -> Deal | None:
        """Score one package. The keyword thresholds override the mode's defaults."""
        pv = self.perceived
        me, partner = self.teams[self.me_idx], self.teams[j]
        give_ids, get_ids = {p.id for p in give}, {p.id for p in get}
        their_roster = [p for p in partner.roster if p.id not in get_ids] + give
        their_drop = None
        if len(give) > len(get):
            their_drop = self._worst(their_roster, give_ids)
            their_roster = [p for p in their_roster if p is not their_drop]

        give_v = pv.package(give) + extra_give_value  # e.g. FAAB dollars sent with the players
        get_v = pv.package(get)
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
        my_add = self.filler if len(give) > len(get) else None
        if my_add is not None:
            my_roster = my_roster + [my_add]
        if len(get) > len(give) and self.filler is not None:
            their_roster = their_roster + [self.filler]

        dists = list(self.dists)
        dists[self.me_idx] = self.model.dist(my_roster)
        dists[j] = self.model.dist(their_roster)
        my_gain = self.model.strength(self.me_idx, dists) - self.base[self.me_idx]
        if my_gain < min_my_gain:
            return None
        their_gain = self.fit_gain(j, dists)
        edge = self.mode == "edge"
        if min_their_gain is None:
            min_their_gain = EDGE_MIN_THEIR_GAIN if edge else MIN_THEIR_GAIN
        if min_margin is None:
            min_margin = EDGE_MIN_MARGIN if edge else float("-inf")
        if their_gain < min_their_gain or my_gain - their_gain < min_margin:
            return None
        gives_star = bool(get_ids & stars)
        struggling = j in self.struggling
        reasons = self._reasons(j, give, get, dists, struggling)
        rival = j in self.rivals
        if self.mode == "edge":
            reasons = self._edge_reasons(j, give, get, ratio, rival) + reasons
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
            acceptance=acceptance_odds(ratio, their_gain, gives_star, struggling),
            dists=dists,
            reasons=reasons,
            rival=rival,
            rosters={self.me_idx: my_roster, j: their_roster},
            my_add=my_add,
        )

    def _wasted(self, p: Player, punts: np.ndarray) -> list[str]:
        """Categories a player's value mostly comes from that this team punts."""
        z = np.maximum(self.model.z_scores(p), 0.0)
        if z.sum() <= 0 or z[punts].sum() < WASTED_SHARE * z.sum():
            return []
        return [self.model.cats[c].name for c in np.flatnonzero(punts & (z > 0.25))]

    def _edge_reasons(self, j: int, give: list[Player], get: list[Player], ratio: float, rival: bool) -> list[str]:
        """Why the deal favors me even though it looks like a win for them."""
        out = []
        if ratio >= 1.05:
            out.append(f"On paper they win: they get {ratio:.0%} of the trade value they send.")
        for p in give:
            cats = self._wasted(p, self.punts[self.me_idx])
            if cats:
                out.append(f"{p.name}'s value is mostly {', '.join(cats)}, which you punt: it costs you little.")
        for p in get:
            cats = self._wasted(p, self.punts[j])
            if cats:
                out.append(f"{p.name}'s {', '.join(cats)} is wasted on them: they punt it.")
        if not rival:
            out.append("They aren't one of your direct playoff rivals.")
        return out

    def _reasons(self, j: int, give: list[Player], get: list[Player], dists: list[Dist], struggling: bool) -> list[str]:
        """Talking points for the pitch, and why the deal can work for both sides."""
        pv, out = self.perceived, []
        for p in give:
            owners, ours = pv.value(p), self.model_dollars.get(p.id, 0.0)
            if owners - ours >= MISPRICED:
                out.append(f"Sell high: owners value {p.name} at ${owners:.0f}, {_ours(ours)}.")
        for p in get:
            owners, ours = pv.value(p), self.model_dollars.get(p.id, 0.0)
            if ours - owners >= MISPRICED:
                out.append(f"Buy low: owners value {p.name} at ${owners:.0f}, {_ours(ours)}.")
        change = self._avg_win(j, dists) - self.base_cats[j]
        lost = [self.model.cats[c].name for c in np.flatnonzero(self.punts[j] & (change < -0.02))]
        if lost:
            out.append(f"They punt {', '.join(lost)}, so what they give up there doesn't cost them.")
        partner = self.teams[j]
        out_count = sum(not p.available for p in partner.roster)
        if len(give) > len(get) and out_count >= 2:
            out.append(f"They have {out_count} players out, so an extra healthy body helps them now.")
        if len(get) > len(give):
            out.append("They turn two roster spots into one better player and can grab someone off waivers.")
        if struggling:
            out.append("They're outside the playoff spots, so they have reason to shake things up.")
        return out

    def candidates(self, partners: Iterable[int] | None = None) -> list[Deal]:
        me = self.teams[self.me_idx]
        mine = [p for p in me.roster if p.id not in self.keep]
        deals: list[Deal] = []
        for j in range(len(self.teams)) if partners is None else partners:
            if j == self.me_idx:
                continue
            partner = self.teams[j]
            stars = self._stars(partner)
            theirs = partner.roster
            for give in mine:
                for get in theirs:
                    d = self._evaluate(j, [give], [get], stars)
                    if d:
                        deals.append(d)
            # 2-for-1: package two of mine for one of theirs.
            for give_pair in combinations(mine, 2):
                for get in theirs:
                    d = self._evaluate(j, list(give_pair), [get], stars)
                    if d:
                        deals.append(d)
            # 1-for-2: one of mine for two of theirs.
            for give in mine:
                for get_pair in combinations(theirs, 2):
                    d = self._evaluate(j, [give], list(get_pair), stars)
                    if d:
                        deals.append(d)
        return deals

    def rank_key(self, d: Deal) -> float:
        if self.mode == "edge":
            return d.score * (RIVAL_DISCOUNT if d.rival and d.their_gain > 0 else 1.0)
        return d.score

    def shortlist(self) -> list[Deal]:
        deals = sorted(self.candidates(), key=self.rank_key, reverse=True)
        picked: list[Deal] = []
        per_target: dict[int, int] = {}
        per_partner: dict[int, int] = {}
        for d in deals:
            target = max(d.get, key=self.perceived.value).id
            if per_target.get(target, 0) >= MAX_PER_TARGET or per_partner.get(d.partner_idx, 0) >= MAX_PER_PARTNER:
                continue
            per_target[target] = per_target.get(target, 0) + 1
            per_partner[d.partner_idx] = per_partner.get(d.partner_idx, 0) + 1
            picked.append(d)
            if len(picked) == SHORTLIST:
                break
        return picked

    def pitch(self, d: Deal) -> str:
        """One line on why the deal helps them, in the categories where they need it most."""
        j = d.partner_idx
        before = self._avg_win(j, self.dists)
        after = self._avg_win(j, d.dists)
        change = after - before
        mus = np.array([x.mu for x in self.dists])
        signed = np.where(self.model.reverse, -mus, mus)
        ranks = (signed > signed[j]).sum(0) + 1
        order = [c for c in np.argsort(-change) if change[c] >= 0.03][:2]
        if not order:
            return "Balances their roster without costing them a category."
        parts = []
        for c in order:
            cat = self.model.cats[c]
            line = f"{cat.name} ({_ordinal(int(ranks[c]))} in the league)"
            if not cat.ratio:
                delta = sum(p.per_game.get(cat.stat_id, 0.0) for p in d.give) - sum(
                    p.per_game.get(cat.stat_id, 0.0) for p in d.get
                )
                if cat.reverse:
                    delta = -delta
                if delta >= 0.5:
                    line += f", +{delta:.1f} a game"
            parts.append(line)
        return "Helps their " + " and ".join(parts) + "."

    def _avg_win(self, i: int, dists: list[Dist]) -> np.ndarray:
        return self.model.avg_cat_win(i, dists)


def _ours(dollars: float) -> str:
    if dollars < 1:
        return "but he's near replacement level in this league's categories"
    return f"but he's worth ${dollars:.0f} in this league's categories"


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"

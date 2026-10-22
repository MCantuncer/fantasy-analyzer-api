"""Trade finder: deals that help you and that the other owner has a real reason to accept.

A deal is suggested only when
  * it raises your expected categories won,
  * it raises theirs too, under their own build, and
  * it doesn't look like a loss to an owner who goes by rankings and draft prices.

Each surviving deal gets an acceptance estimate, and deals are ranked by
playoff-odds gain weighted by that estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from app.analysis.model import CategoryModel, Dist
from app.analysis.perceived import PerceivedValue
from app.espn.parse import League, Player, Team

MIN_MY_GAIN = 0.05  # expected categories won per week
MIN_THEIR_GAIN = 0.02
MIN_VALUE_RATIO = 0.85  # they must get back at least this share of perceived value
MAX_VALUE_RATIO = 1.4  # never overpay by more than this on paper
TOP_PLAYERS_PROTECTED = 2
SHORTLIST = 14
MAX_PER_TARGET = 2
MAX_PER_PARTNER = 4

LIKELY, POSSIBLE = 0.6, 0.4


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
    get_value: float  # perceived value they send
    acceptance: float
    dists: list[Dist]

    @property
    def score(self) -> float:
        return self.acceptance * self.my_gain


def acceptance_odds(ratio: float, their_gain: float, gives_star: bool) -> float:
    """Rough chance the other owner says yes. A heuristic, not a fitted model."""
    p = float(np.interp(ratio, [0.85, 0.95, 1.05, 1.2, 1.4], [0.2, 0.4, 0.55, 0.7, 0.8]))
    if their_gain >= 0.15:
        p += 0.1
    elif their_gain < 0.05:
        p -= 0.1
    if gives_star and ratio < 1.15:
        p -= 0.2
    return float(np.clip(p, 0.05, 0.9))


def acceptance_label(p: float) -> str:
    return "likely" if p >= LIKELY else "possible" if p >= POSSIBLE else "long_shot"


class TradeFinder:
    def __init__(self, league: League, model: CategoryModel, me_idx: int, keep: set[int]):
        self.league = league
        self.model = model
        self.me_idx = me_idx
        self.keep = keep
        self.perceived = PerceivedValue(league)
        self.teams = league.teams
        self.dists = [model.dist(t.roster) for t in self.teams]
        self.base = [model.strength(i, self.dists) for i in range(len(self.teams))]

    def _worst(self, roster: list[Player], protected: set[int]) -> Player | None:
        """The player an owner would cut to make room: lowest model value, not protected."""
        options = [p for p in roster if p.id not in protected]
        return min(options, key=self.model.value) if options else None

    def _stars(self, team: Team) -> set[int]:
        top = sorted(team.roster, key=self.perceived.value, reverse=True)[:TOP_PLAYERS_PROTECTED]
        return {p.id for p in top}

    def _evaluate(self, j: int, give: list[Player], get: list[Player], stars: set[int]) -> Deal | None:
        pv = self.perceived
        give_v, get_v = pv.total(give), pv.total(get)
        if get_v <= 0:
            return None
        ratio = give_v / get_v
        if ratio < MIN_VALUE_RATIO or ratio > MAX_VALUE_RATIO:
            return None

        me, partner = self.teams[self.me_idx], self.teams[j]
        give_ids, get_ids = {p.id for p in give}, {p.id for p in get}
        my_roster = [p for p in me.roster if p.id not in give_ids] + get
        their_roster = [p for p in partner.roster if p.id not in get_ids] + give
        my_drop = their_drop = None
        if len(get) > len(give):
            my_drop = self._worst(my_roster, self.keep | get_ids)
            my_roster = [p for p in my_roster if p is not my_drop]
        if len(give) > len(get):
            their_drop = self._worst(their_roster, give_ids)
            their_roster = [p for p in their_roster if p is not their_drop]

        dists = list(self.dists)
        dists[self.me_idx] = self.model.dist(my_roster)
        dists[j] = self.model.dist(their_roster)
        my_gain = self.model.strength(self.me_idx, dists) - self.base[self.me_idx]
        if my_gain < MIN_MY_GAIN:
            return None
        their_gain = self.model.strength(j, dists) - self.base[j]
        if their_gain < MIN_THEIR_GAIN:
            return None
        gives_star = bool(get_ids & stars)
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
            acceptance=acceptance_odds(ratio, their_gain, gives_star),
            dists=dists,
        )

    def candidates(self) -> list[Deal]:
        me = self.teams[self.me_idx]
        mine = [p for p in me.roster if p.id not in self.keep]
        deals: list[Deal] = []
        for j, partner in enumerate(self.teams):
            if j == self.me_idx:
                continue
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

    def shortlist(self) -> list[Deal]:
        deals = sorted(self.candidates(), key=lambda d: d.score, reverse=True)
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
        return np.mean([self.model.cat_win_prob(dists[i], d) for k, d in enumerate(dists) if k != i], axis=0)


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"

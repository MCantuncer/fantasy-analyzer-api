"""Checks on a trade idea: is what a manager says about it true?

A manager who calls an idea bad may say why ("I can just pick that guy up", "Lopez is
hurt all the time") or say nothing. Either way every check runs on the idea as it was
shown, against today's league, and each returns a verdict with its evidence:

  * confirmed: the data shows the problem,
  * not_supported: the data says otherwise,
  * cant_check: the idea can't be rebuilt, or the check isn't built yet.

The checks only read the model; none of them changes it. A confirmed verdict is a lead
for a fix that must hold on past seasons, never a weight to tune.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from app import schemas
from app.analysis.durability import injury_prone
from app.analysis.report import LOCKED, MIN_SAMPLE_GAMES, Analyzer
from app.analysis.tiers import droppable
from app.analysis.trades import TradeFinder
from app.espn.parse import League, Player

CheckId = str

FREE_PICKUP_SHARE = 0.6  # a pickup that alone gets this share of the deal's gain carries the deal
NOISE_SE = 2.0  # a gain within this many standard errors of zero isn't distinguishable from noise
CATEGORY_HIT = 0.05  # weekly win rate lost in a category that counts as a real cost
LATERAL = 0.5  # net model value (z, summed over categories) below half a category's spread is a sideways move
CONTESTED = 0.5  # categories you win at least this often are ones you compete in

TITLES: dict[CheckId, str] = {
    "free_pickup": "Free pickup credit",
    "noise": "Within noise",
    "lateral_swap": "Lateral swap",
    "build_conflict": "Build conflict",
    "small_sample": "Small sample",
    "injury": "Injury and durability",
    "role_change": "Role change",
    "acceptance": "Would they accept",
    "already_offered": "Already offered",
}
ORDER = list(TITLES)
# Which checks a quick reason points at first.
REASON_CHECKS: dict[str, list[CheckId]] = {
    "fake_gain": ["free_pickup", "noise", "lateral_swap"],
    "wont_accept": ["acceptance"],
    "bad_for_build": ["build_conflict"],
    "player_concern": ["injury", "small_sample", "role_change"],
    "already_tried": ["already_offered"],
}


class StaleIdea(ValueError):
    """The idea's players are no longer where it put them."""


@dataclass
class Rebuilt:
    give: list[Player]
    get: list[Player]
    my_adds: list[Player]  # free agents I pick up for the spots the deal opens
    my_drops: list[Player]  # who I cut to fit the extra players I get
    partner_idx: int
    my_roster: list[Player]
    their_roster: list[Player]


def focus_order(reasons: Iterable[str], extra: Iterable[CheckId] = ()) -> list[CheckId]:
    """Every check, the ones the reasons (or a reader) point at first."""
    first = [c for r in reasons for c in REASON_CHECKS.get(r, [])] + [c for c in extra if c in TITLES]
    return list(dict.fromkeys(first + ORDER))


class IdeaChecker(Analyzer):
    def __init__(self, league: League, team_id: int, idea: schemas.TradeIdea, keep_player_ids: list[int] | None = None):
        super().__init__(league, team_id, keep_player_ids)
        self.idea = idea
        self.finder = TradeFinder(league, self.model, self.idx, self.keep, premium=self.premium)
        self.deal = self._rebuild()
        self.before = self.playoff_pct(self.dists)
        self.deal_dists = list(self.dists)
        self.deal_dists[self.idx] = self.model.dist(self.deal.my_roster)
        self.deal_dists[self.deal.partner_idx] = self.model.dist(self.deal.their_roster)
        self.after = self.playoff_pct(
            self.deal_dists, {self.idx: self.deal.my_roster, self.deal.partner_idx: self.deal.their_roster}
        )

    def _rebuild(self) -> Rebuilt:
        idea = self.idea
        partner_idx = next((i for i, t in enumerate(self.teams) if t.id == idea.partner_team_id), None)
        if partner_idx is None or partner_idx == self.idx:
            raise StaleIdea("The other team isn't in this league.")
        mine = {p.id: p for p in self.me.roster}
        theirs = {p.id: p for p in self.teams[partner_idx].roster}
        free = {p.id: p for p in self.league.free_agents}

        def pick(pool: dict[int, Player], ref: schemas.PlayerRef | None, where: str) -> Player | None:
            if ref is None:
                return None
            if ref.id not in pool:
                raise StaleIdea(f"{ref.name} is no longer {where}.")
            return pool[ref.id]

        give = [pick(mine, r, "on your roster") for r in idea.give]
        get = [pick(theirs, r, f"on {self.teams[partner_idx].name}") for r in idea.get]
        my_add = pick(free, idea.my_add, "a free agent")
        my_drop = pick(mine, idea.my_drop, "on your roster")
        their_drop = pick(theirs, idea.their_drop, f"on {self.teams[partner_idx].name}")
        out_ids = {p.id for p in give} | ({my_drop.id} if my_drop else set())
        my_roster = [p for p in self.me.roster if p.id not in out_ids] + get + ([my_add] if my_add else [])
        gone = {p.id for p in get} | ({their_drop.id} if their_drop else set())
        their_roster = [p for p in self.teams[partner_idx].roster if p.id not in gone] + give
        if len(get) > len(give) and self.finder.filler is not None:
            their_roster.append(self.finder.filler)  # as the finder does: they fill the open spot
        adds, drops = [my_add] if my_add else [], [my_drop] if my_drop else []
        return Rebuilt(give, get, adds, drops, partner_idx, my_roster, their_roster)

    # ---- checks --------------------------------------------------------------------------

    def run(self, order: list[CheckId]) -> list[schemas.CheckVerdict]:
        checks: dict[CheckId, Callable[[], tuple[str, str]]] = {
            "free_pickup": self.free_pickup,
            "noise": self.noise,
            "lateral_swap": self.lateral_swap,
            "build_conflict": self.build_conflict,
            "small_sample": self.small_sample,
            "injury": self.injury,
        }
        out = []
        for c in order:
            status, evidence = checks[c]() if c in checks else ("cant_check", "This check isn't built yet.")
            out.append(schemas.CheckVerdict(check=c, title=TITLES[c], status=status, evidence=evidence))
        return out

    @property
    def gain(self) -> float:
        return self.after - self.before

    def free_pickup(self) -> tuple[str, str]:
        if not self.deal.my_adds:
            return "not_supported", "The deal doesn't rely on a free agent pickup."
        add = self.deal.my_adds[0]
        # The pickup alone: the same free agent for my weakest droppable player.
        options = droppable(self.me.roster, self.keep) or list(self.deal.give)
        drop = min(options, key=self.model.value)
        roster = [p for p in self.me.roster if p.id != drop.id] + [add]
        dists = list(self.dists)
        dists[self.idx] = self.model.dist(roster)
        alone = self.playoff_pct(dists, {self.idx: roster}) - self.before
        pts = f"{100 * alone:+.1f} of the deal's {100 * self.gain:+.1f} playoff points"
        if self.gain > 0 and alone >= FREE_PICKUP_SHARE * self.gain:
            return "confirmed", f"Adding {add.name} for {drop.name}, with no trade, gets {pts}."
        return "not_supported", f"Adding {add.name} for {drop.name} alone gets {pts}; the trade adds the rest."

    def noise(self) -> tuple[str, str]:
        base = self._sim.base.made_playoffs if self._sim is not None else None
        moved = {self.idx, self.deal.partner_idx}
        new = self._sim.what_if(moved, self.deal_dists, self.period_dists(self._changed())) if self._sim else None
        if base is None or new is None or new.made_playoffs is None:
            return "cant_check", "The season simulation isn't available for this league."
        d = new.made_playoffs[:, self.idx].astype(float) - base[:, self.idx].astype(float)
        se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
        text = f"{100 * self.gain:+.1f} playoff points, give or take {100 * NOISE_SE * se:.1f} from simulation noise."
        if abs(self.gain) <= NOISE_SE * se:
            return "confirmed", text
        return "not_supported", text

    def _changed(self) -> dict[int, list[Player]]:
        return {self.idx: self.deal.my_roster, self.deal.partner_idx: self.deal.their_roster}

    def lateral_swap(self) -> tuple[str, str]:
        """Is the talent coming in better than the talent going out, apart from the free pickup?"""
        v = self.model.value
        net = sum(v(p) for p in self.deal.get) - sum(v(p) for p in self.deal.give)
        net -= sum(v(p) for p in self.deal.my_drops)
        if self.deal.my_adds:
            # An open spot's pickup needs no trade: it would replace one of my weakest droppable players anyway.
            options = droppable(self.me.roster, self.keep) or list(self.deal.give)
            net += sum(v(p) for p in sorted(options, key=v)[: len(self.deal.my_adds)])
        names = lambda ps: " + ".join(f"{p.name} ({v(p):+.1f})" for p in ps)  # noqa: E731
        text = f"Model value in: {names(self.deal.get)}; out: {names(self.deal.give)}; net {net:+.1f}"
        text += ", counting the free pickup as something you could do anyway." if self.deal.my_adds else "."
        return ("confirmed" if net <= LATERAL else "not_supported"), text

    def build_conflict(self) -> tuple[str, str]:
        before = self.avg_cat_win(self.idx, self.dists)
        change = self.avg_cat_win(self.idx, self.deal_dists) - before
        hits = [
            f"{self.cat_names[c]} {100 * change[c]:+.0f} pts (you win it {100 * before[c]:.0f}% of weeks now"
            + (", a strength" if before[c] >= LOCKED else "")
            + ")"
            for c in np.argsort(change)
            if change[c] <= -CATEGORY_HIT and before[c] >= CONTESTED
        ]
        if hits:
            return "confirmed", "It costs categories you compete in: " + "; ".join(hits) + "."
        return "not_supported", "No category you win at least half the time drops by 5 points or more."

    def _incoming(self) -> list[Player]:
        return self.deal.get + self.deal.my_adds

    def small_sample(self) -> tuple[str, str]:
        thin = [p for p in self._incoming() if not p.has_prior and p.games_played < 2 * MIN_SAMPLE_GAMES]
        if thin:
            text = "; ".join(f"{p.name}: {p.games_played} games and no prior season" for p in thin)
            return "confirmed", f"The projection rests on a few games: {text}."
        return "not_supported", "Everyone you get has a prior season or 10+ games this season."

    def injury(self) -> tuple[str, str]:
        found = []
        for p in self._incoming():
            if not p.available:
                found.append(f"{p.name} is out ({p.injury_status.replace('_', ' ').lower()})")
            elif injury_prone(p):
                found.append(f"{p.name} plays about {100 * (p.durability or 0):.0f}% of games over recent seasons")
        if found:
            return "confirmed", "; ".join(found) + "."
        return "not_supported", "Everyone you get is healthy and has a normal games-played record."

"""Trade for a player: how a chosen player fits you, what his owner needs, and the best offer to send.

Offers come from the trade finder's rules (perceived value plus owner premium, acceptance
odds, punt-aware gain, pitch lines) with two changes for a player you already want:
  * any gain for you counts, since you picked him, and
  * the owner's build only has to stay about even, because owners mostly judge on paper.

Packages are one or two of your players for him, or two or three for him plus one of
the owner's low-value players, so nobody has to cut more than one player.
Players marked not for sale are never offered or asked for. Your own jewels and keystones
can be offered, since you picked the target, and each offer warns when it includes one.
The best offer has the highest playoff-odds gain weighted by acceptance; the fallbacks
are the likeliest to be accepted and the one that costs you least on paper.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from itertools import combinations

import numpy as np

from app import schemas
from app.analysis.report import WHAT_IF_SIMS, Analyzer, record
from app.analysis.sim import simulate_season
from app.analysis.trades import MIN_VALUE_RATIO, PUNT_WIN, Deal, TradeFinder
from app.espn.parse import League, Player

MIN_MY_GAIN = 0.0
MIN_THEIR_GAIN = -0.05  # the owner's build can't get noticeably worse
FILLERS = 3  # the owner's lowest-value players that can be added to balance rosters
SIM_TOP = 10
NEED_BAND = (PUNT_WIN, 0.6)  # categories they compete in but don't control
WOULD_VALUE = 4
FIT_LABELS = [(0.4, "great"), (0.2, "good"), (0.05, "small")]
CAT_MOVE = 0.02  # change in weekly win % worth naming
THROW_IN = 5.0  # players owners value under this many dollars are throw-ins


class TargetError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class TargetAnalyzer(Analyzer):
    def __init__(
        self,
        league: League,
        team_id: int,
        player_id: int,
        keep_player_ids: list[int] | None = None,
        not_for_sale_player_ids: list[int] | None = None,
    ):
        super().__init__(league, team_id, keep_player_ids, not_for_sale_player_ids)
        owner = next((i for i, t in enumerate(self.teams) if any(p.id == player_id for p in t.roster)), None)
        if owner is None:
            raise TargetError("player_not_rostered", "That player isn't on a team in this league.")
        if owner == self.idx:
            raise TargetError("own_player", "That player is already on your team.")
        self.owner_idx = owner
        self.owner = self.teams[owner]
        self.target = next(p for p in self.owner.roster if p.id == player_id)
        self.finder = TradeFinder(league, self.model, self.idx, self.keep, premium=self.premium)
        self.base_cats = self.avg_cat_win(self.idx, self.dists)
        self.before = self.playoff_pct(self.dists)

    # ---- how he fits you ---------------------------------------------------------------

    def fit(self) -> schemas.TargetFit:
        # Never cut someone with trade value; with nobody droppable this is a ceiling with a full roster.
        drop = self.finder._my_cut(self.me.roster, set())
        roster = [p for p in self.me.roster if p is not drop] + [self.target]
        dists = list(self.dists)
        dists[self.idx] = self.model.dist(roster)
        gain = self.model.strength(self.idx, dists) - self.model.strength(self.idx, self.dists)
        cats = self.avg_cat_win(self.idx, dists) - self.base_cats
        return schemas.TargetFit(
            label=next((label for floor, label in FIT_LABELS if gain >= floor), "none"),
            my_cats_gain=round(gain, 3),
            my_drop=self.ref(drop) if drop else None,
            helps=self._named(cats),
            hurts=self._named(-cats),
            playoff_pct_before=round(self.before, 3),
            playoff_pct_after=round(self.playoff_pct(dists, {self.idx: roster}), 3),
            category_changes={n: round(float(cats[c]), 3) for c, n in enumerate(self.cat_names)},
        )

    def _named(self, change: np.ndarray) -> list[str]:
        """Categories that move by at least CAT_MOVE, biggest first."""
        return [self.cat_names[c] for c in np.argsort(-change) if change[c] >= CAT_MOVE]

    # ---- his owner ---------------------------------------------------------------------

    def owner_read(self) -> schemas.OwnerRead:
        j = self.owner_idx
        win = self.finder.base_cats[j]
        punts = self.finder.punts[j]
        needs = [c for c in np.argsort(win) if not punts[c] and NEED_BAND[0] <= win[c] < NEED_BAND[1]]
        build = next(b for b in self.builds() if b.team_id == self.owner.id)
        season_odds = simulate_season(
            self.league, self.dists, self.model.reverse, n_sims=WHAT_IF_SIMS, period_dists=self.period_dists()
        ).playoff_pct
        return schemas.OwnerRead(
            team_id=self.owner.id,
            team_name=self.owner.name,
            record=record(self.owner),
            build=build,
            punts=[self.cat_names[c] for c in np.flatnonzero(punts)],
            needs=[self.cat_names[c] for c in needs[:3]],
            playoff_pct=round(float(season_odds[j]), 3),
            outside_playoffs=j in self.finder.struggling,
            target_is_star=any(b.kind == "jewel" for b in self.premium.badges(self.target)),
            would_value=self._would_value(),
        )

    def _would_value(self) -> list[schemas.OwnerWant]:
        j, wants = self.owner_idx, []
        for p in self._mine():
            roster = self.owner.roster + [p]
            cut = self.finder._worst(roster, {p.id})
            dists = list(self.dists)
            dists[j] = self.model.dist([q for q in roster if q is not cut])
            gain = self.finder.fit_gain(j, dists)
            if gain <= 0:
                continue
            cats = self.avg_cat_win(j, dists) - self.finder.base_cats[j]
            cats[self.finder.punts[j]] = 0.0
            wants.append(
                schemas.OwnerWant(player=self.ref(p), their_cats_gain=round(gain, 3), helps=self._named(cats)[:2])
            )
        wants.sort(key=lambda w: w.their_cats_gain, reverse=True)
        return wants[:WOULD_VALUE]

    # ---- offers ------------------------------------------------------------------------

    def packages(self) -> list[tuple[list[Player], list[Player]]]:
        mine = self._mine()
        others = [p for p in self.owner.roster if p is not self.target and not self.premium.not_for_sale(p)]
        fillers = sorted(others, key=self.model.value)[:FILLERS]
        out: list[tuple[list[Player], list[Player]]] = []
        for n in (1, 2):
            out += [(list(give), [self.target]) for give in combinations(mine, n)]
        for f in fillers:
            for n in (1, 2, 3):
                out += [(list(give), [self.target, f]) for give in combinations(mine, n)]
        return out

    def _mine(self) -> list[Player]:
        """My players that can go in an offer."""
        return [p for p in self.me.roster if p.id not in self.keep and not self.premium.not_for_sale(p)]

    def deals(self, min_their_gain: float) -> list[Deal]:
        found = []
        for give, get in self.packages():
            d = self.finder._evaluate(self.owner_idx, give, get, MIN_MY_GAIN, min_their_gain)
            if d:
                found.append(d)
        return found

    def offers(self, min_their_gain: float = MIN_THEIR_GAIN) -> list[schemas.TargetOffer]:
        deals = sorted(self.deals(min_their_gain), key=lambda d: d.score, reverse=True)
        # Keep the best package for each core (the players that carry value, ignoring throw-ins),
        # so fallbacks are real alternatives, then simulate the top ones.
        seen, unique = set(), []
        for d in deals:
            key = (self._core(d.give), self._core(d.get))
            if key not in seen:
                seen.add(key)
                unique.append(d)
        ideas = []
        for d in unique[:SIM_TOP]:
            after = self.playoff_pct(d.dists, d.rosters)
            if after >= self.before:
                ideas.append(self.trade_idea(self.finder, d, self.before, after, self.base_cats))
        if not ideas:
            return []

        def payoff(t: schemas.TradeIdea) -> float:
            return t.acceptance * (t.playoff_pct_after - t.playoff_pct_before)

        best = max(ideas, key=payoff)
        picked = [self._offer(best, "best", "Best mix of playoff gain and the chance they say yes.")]
        rest = [t for t in ideas if t is not best]
        safer = max(rest, key=lambda t: (t.acceptance, payoff(t)), default=None)
        if safer and safer.acceptance > best.acceptance:
            picked.append(self._offer(safer, "safer", "Most likely to be accepted, for a smaller gain."))
            rest.remove(safer)
        cheaper = min(rest, key=lambda t: t.value_they_get, default=None)
        if cheaper and cheaper.value_they_get < best.value_they_get:
            picked.append(self._offer(cheaper, "cheaper", "Gives up the least on paper; a good opening offer."))
            rest.remove(cheaper)
        for t in sorted(rest, key=payoff, reverse=True)[: 3 - len(picked)]:
            picked.append(self._offer(t, "alternative", "Another way to get there with different players."))
        return picked

    def _core(self, players: list[Player]) -> frozenset[int]:
        return frozenset(p.id for p in players if self.perceived.value(p) >= THROW_IN)

    @staticmethod
    def _offer(t: schemas.TradeIdea, angle: str, note: str) -> schemas.TargetOffer:
        return schemas.TargetOffer(**t.model_dump(), angle=angle, angle_note=note)

    def _no_offer_note(self, fit: schemas.TargetFit) -> str:
        name = self.target.name
        if fit.label == "none":
            return f"{name} doesn't improve your categories, so no package makes you better."
        if self.premium.not_for_sale(self.target):
            return f"{name} is marked not for sale, so there's no offer to send."
        mine = sorted(self._mine(), key=self.perceived.value, reverse=True)
        if self.perceived.package(mine[:3]) < MIN_VALUE_RATIO * self.premium.price(self.target):
            return f"Owners value {name} far above anything you can send without hurting your own team."
        tail = " Try freeing up players you've kept." if self.keep else ""
        return f"No package for {name} both helps you and looks fair to his owner.{tail}"

    def target_report(self) -> schemas.TargetReport:
        started = time.perf_counter()
        fit = self.fit()
        for_sale = not self.premium.not_for_sale(self.target)
        offers = self.offers() if for_sale else []
        note = None
        if not offers and for_sale and fit.label != "none":
            # Nothing keeps their build even: show the best long shots, judged on paper value alone.
            offers = self.offers(min_their_gain=-np.inf)
            if offers:
                note = (
                    f"Every package that's fair on paper weakens {self.owner.name}'s build, "
                    "so these are long shots that rely on the owner going by value."
                )
        report = schemas.TargetReport(
            team_id=self.me.id,
            team_name=self.me.name,
            target=self.ref(self.target),
            fit=fit,
            owner=self.owner_read(),
            offers=offers,
            note=note if offers else self._no_offer_note(fit),
            generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
            elapsed_ms=0,
        )
        report.elapsed_ms = int((time.perf_counter() - started) * 1000)
        return report

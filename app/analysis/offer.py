"""Check any trade: one the manager was offered, or one they plan to send.

The manager names the players on both sides (one to three each, any shape) and the engine
answers for both teams, the same way it scores its own ideas:
  * playoff odds before and after, on the fine season simulation, with the noise on my change,
  * weekly category win chances and fantasy playoff weeks, both teams,
  * trade value on paper (owner premium included) and the chance they say yes,
  * the roster moves the deal forces: who I cut for extra players (the manager can pick),
    who they likely cut, and the free agents that fill an open spot,
  * flags from the idea checks: a category I compete in drops, an injured or thin-sample
    player comes in, a sideways swap, a jewel, keystone or kept player goes out.

No gates: unlike the finder, a check never hides a deal. It reports what the deal does.
When the deal opens spots on my roster, the odds are measured against just picking up the
same free agents for my weakest droppable players, so a trade gets no credit for a pickup
I could make without it, as in the finder.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Literal

import numpy as np

from app import schemas
from app.analysis.checks import IdeaChecker, Rebuilt
from app.analysis.model import Dist
from app.analysis.report import FINE_SIMS, LOCK_ODDS, Analyzer, partner_gains, record
from app.analysis.sim import SeasonResult
from app.analysis.tiers import droppable
from app.analysis.trades import Deal, TradeFinder, _fact, acceptance_odds, keystone_odds
from app.espn.parse import League, Player

MAX_SIDE = 3
MIN_ODDS_MOVE = 0.01  # a point of playoff odds: the smallest change a verdict calls
NOISE_SE = 2.0  # a change inside this many standard errors of zero is a close call
PINNED = 1 - LOCK_ODDS  # odds this close to 0 or 1 can't move: judge on weekly categories instead
CATS_CALL = 0.10  # expected categories won per week that make a call when odds are pinned
CAT_MOVE = 0.02  # change in weekly win chance worth naming
CUT_OPTIONS = 4  # cheapest cuts that get their own odds, for a one-player cut
FLAG_CHECKS = ["build_conflict", "injury", "small_sample", "lateral_swap"]

Call = Literal["take", "close", "pass"]
PartnerCall = Literal["gain", "even", "lose"]
Direction = Literal["received", "sending"]


def _pts(x: float) -> str:
    n = round(100 * x)
    return f"{n} point" if n == 1 else f"{n} points"


def _and(names: list[str]) -> str:
    return " and ".join(names[:2])


def _mix(helps: list[str], hurts: list[str]) -> str:
    """The categories that move, as a sentence: 'Stronger in AST and STL, weaker in TO.'"""
    parts = ([f"stronger in {_and(helps)}"] if helps else []) + ([f"weaker in {_and(hurts)}"] if hurts else [])
    text = ", ".join(parts)
    return f" {text[:1].upper()}{text[1:]}." if parts else ""


class OfferError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class OfferChecker(IdeaChecker):
    """Scores one named trade for both sides. Reuses the idea checks for its flags."""

    def __init__(
        self,
        league: League,
        team_id: int,
        partner_team_id: int,
        give_ids: list[int],
        get_ids: list[int],
        my_drop_ids: list[int] | None = None,
        keep_player_ids: list[int] | None = None,
        not_for_sale_player_ids: list[int] | None = None,
        direction: Direction = "received",
    ):
        Analyzer.__init__(self, league, team_id, keep_player_ids, not_for_sale_player_ids)
        self.direction = direction
        j = next((i for i, t in enumerate(self.teams) if t.id == partner_team_id), None)
        if j is None:
            raise OfferError("team_not_found", "That team isn't in this league.")
        if j == self.idx:
            raise OfferError("own_team", "Pick another team to trade with.")
        self.j = j
        self.partner = self.teams[j]
        self.give = self._pick(self.me.roster, give_ids, "Pick the players you send from your own roster.")
        self.get = self._pick(self.partner.roster, get_ids, f"Pick the players you get from {self.partner.name}.")
        self.finder = TradeFinder(league, self.model, self.idx, self.keep, premium=self.premium)
        self.free = self._free_agents()
        self.my_drops = self._my_drops(my_drop_ids or [])
        self.deal = self._build()

    # ---- the deal ----------------------------------------------------------------------

    @staticmethod
    def _pick(roster: list[Player], ids: list[int], message: str) -> list[Player]:
        if not 1 <= len(ids) <= MAX_SIDE or len(set(ids)) != len(ids):
            raise OfferError("bad_players", f"Each side takes one to {MAX_SIDE} different players.")
        by_id = {p.id: p for p in roster}
        if any(i not in by_id for i in ids):
            raise OfferError("bad_players", message)
        return [by_id[i] for i in ids]

    def _free_agents(self) -> list[Player]:
        """Free agents that can fill an open spot, best first: the finder's filler and the next ones like it."""
        # The same rule as the finder's filler (badges.best_pickup): healthy, with enough games to trust.
        pool = [
            p
            for p in self.league.free_agents
            if p.fully_healthy and p.pro_team_id and (p.has_prior or p.games_played >= 5)
        ]
        return sorted(pool, key=self.model.value, reverse=True)[:MAX_SIDE]

    def _my_drops(self, chosen: list[int]) -> list[Player]:
        """Who I cut for the extra players I get: the manager's pick, else my least valuable droppable players."""
        need = max(len(self.get) - len(self.give), 0)
        give_ids = {p.id for p in self.give}
        options = [p for p in self.me.roster if p.id not in give_ids]
        if chosen:
            by_id = {p.id: p for p in options}
            if len(chosen) != need or len(set(chosen)) != need or any(i not in by_id for i in chosen):
                raise OfferError(
                    "bad_drops", f"Pick {need} of your other players to cut." if need else "This trade needs no cut."
                )
            return [by_id[i] for i in chosen]
        v = self.model.value
        cuttable = sorted(droppable(options, self.keep), key=v)
        rest = sorted((p for p in options if p not in cuttable and p.id not in self.keep), key=v)
        return (cuttable + rest)[:need]

    def _build(self) -> Rebuilt:
        give, get, j = self.give, self.get, self.j
        give_ids, get_ids = {p.id for p in give}, {p.id for p in get}
        drop_ids = {p.id for p in self.my_drops}
        my_adds = self.free[: max(len(give) - len(get), 0)]
        my_roster = [p for p in self.me.roster if p.id not in give_ids | drop_ids] + get + my_adds
        their_roster = [p for p in self.partner.roster if p.id not in get_ids] + give
        self.their_drops: list[Player] = []
        for _ in range(len(give) - len(get)):
            cut = self.finder._worst(their_roster, give_ids | {p.id for p in self.their_drops})
            if cut is None:
                break
            self.their_drops.append(cut)
        their_roster = [p for p in their_roster if p not in self.their_drops]
        self.their_adds = self.free[: max(len(get) - len(give), 0)]
        their_roster += self.their_adds
        return Rebuilt(give, get, my_adds, self.my_drops, j, my_roster, their_roster)

    def _free_baseline(self) -> tuple[list[Player], list[Dist]] | None:
        """My roster after picking up the deal's free agents with no trade, for my weakest droppable players,
        keeping only the pickups that help. None when there are none."""
        adds = self.deal.my_adds
        if not adds:
            return None
        roster = list(self.me.roster)
        dists = list(self.dists)
        strength = self.model.strength(self.idx, dists)
        for add in adds:
            best = None
            for drop in droppable(roster, self.keep):
                trial = [p for p in roster if p.id != drop.id] + [add]
                d = list(dists)
                d[self.idx] = self.model.dist(trial)
                s = self.model.strength(self.idx, d)
                if s > strength and (best is None or s > best[2]):
                    best = (trial, d, s)
            if best is not None:
                roster, dists, strength = best
        return (roster, dists) if roster != self.me.roster else None

    # ---- scoring -----------------------------------------------------------------------

    def _odds(self) -> None:
        """Both teams' odds on FINE_SIMS seasons, and the noise on my change."""
        rosters = {self.idx: self.deal.my_roster, self.j: self.deal.their_roster}
        self.deal_dists = list(self.dists)
        self.deal_dists[self.idx] = self.model.dist(self.deal.my_roster)
        self.deal_dists[self.j] = self.model.dist(self.deal.their_roster)
        today = self.season_result(self.dists, fine=True)
        free = self._free_baseline()
        base = today if free is None else self.season_result(free[1], {self.idx: free[0]}, fine=True)
        self.base_dists = self.dists if free is None else free[1]
        after = self.season_result(self.deal_dists, rosters, fine=True)
        self.today, self.base_result, self.after_result = today, base, after
        self.before = float(base.playoff_pct[self.idx])
        self.after = float(after.playoff_pct[self.idx])
        self.noise_pts = self._noise(base, after)
        self.free_roster = free[0] if free else None

    def _noise(self, base: SeasonResult, after: SeasonResult) -> float:
        """Twice the standard error of my odds change, from the paired simulated seasons."""
        if base.made_playoffs is None or after.made_playoffs is None:
            return 0.0
        d = after.made_playoffs[:, self.idx].astype(float) - base.made_playoffs[:, self.idx].astype(float)
        if len(d) < 2:
            return 0.0
        return float(NOISE_SE * d.std(ddof=1) / np.sqrt(len(d)))

    def _engine_deal(self) -> Deal:
        """The deal as the finder would score it: values, acceptance, gains and talking points."""
        f, j, give, get = self.finder, self.j, self.give, self.get
        pv = self.perceived
        give_v = pv.package(give)
        get_v = pv.package(get, self.premium.price)
        premium = get_v - pv.package(get)
        get_v += sum(max(pv.value(p) - pv.replacement, 0.0) for p in self.their_drops)
        ratio = give_v / get_v if get_v > 0 else float("inf")
        rosters = {self.idx: self.deal.my_roster, j: self.deal.their_roster}
        my_gain = self.model.strength(self.idx, self.deal_dists) - self.model.strength(self.idx, self.base_dists)
        their_base = f.their_free_cats(j) if self.their_adds else None
        their_gain = f.fit_gain(j, self.deal_dists, their_base)
        facts = f._reasons(j, give, get, self.deal_dists, j in f.struggling)
        acceptance = acceptance_odds(min(ratio, 1e6), len(give), len(get))
        lost = f._keystones_lost(j, get, self.deal_dists)
        if lost:
            acceptance = keystone_odds(acceptance)
            text = f"It takes away their {', '.join(lost)} keystone without replacing it: a harder sell."
            facts.append(_fact("keystone_lost", "against", text, categories=lost))
        return Deal(
            partner_idx=j,
            give=give,
            get=get,
            my_drop=self.my_drops[0] if self.my_drops else None,
            their_drop=self.their_drops[0] if self.their_drops else None,
            my_gain=my_gain,
            their_gain=their_gain,
            give_value=give_v,
            get_value=get_v,
            acceptance=acceptance,
            dists=self.deal_dists,
            reasons=[x.text for x in facts],
            facts=facts,
            rival=j in f.rivals,
            rosters=rosters,
            my_add=self.deal.my_adds[0] if self.deal.my_adds else None,
            premium=premium,
            warnings=f._warnings(give),
            their_strength=f.fit_gain(j, self.deal_dists, their_base, every_category=True),
        )

    def _changes(self, i: int, before: list[Dist], after: list[Dist], skip: np.ndarray | None = None):
        change = self.avg_cat_win(i, after) - self.avg_cat_win(i, before)
        if skip is not None:
            change = np.where(skip, 0.0, change)
        helps = [self.cat_names[c] for c in np.argsort(-change) if change[c] >= CAT_MOVE]
        hurts = [self.cat_names[c] for c in np.argsort(change) if change[c] <= -CAT_MOVE]
        return helps, hurts

    def _call(self, my_gain: float, helps: list[str], hurts: list[str]) -> tuple[Call, str]:
        gain = self.after - self.before
        pinned = (self.before >= LOCK_ODDS and self.after >= LOCK_ODDS) or (
            self.before <= PINNED and self.after <= PINNED
        )
        if pinned:
            where = f"You're all but {'in' if self.before >= LOCK_ODDS else 'out'} either way"
            if my_gain >= CATS_CALL:
                return "take", f"{where}, but you win more categories a week."
            if my_gain <= -CATS_CALL:
                return "pass", f"{where}, and you win fewer categories a week."
            return "close", f"{where}, and your weekly categories barely move."
        mix = _mix(helps, hurts)
        if gain >= MIN_ODDS_MOVE and gain > self.noise_pts:
            return "take", f"Your playoff odds go up {_pts(gain)}.{mix}"
        if gain <= -MIN_ODDS_MOVE and -gain > self.noise_pts:
            return "pass", f"Your playoff odds go down {_pts(-gain)}.{mix}"
        return "close", f"Your playoff odds barely move: within the {100 * self.noise_pts:.1f}-point noise.{mix}"

    def _partner_call(self, before: float, after: float, paper: bool) -> tuple[PartnerCall, str]:
        on_paper = "they win on paper" if paper else "they lose on paper"
        move = after - before
        if partner_gains(before, after):
            return "gain", f"Their playoff odds go up {_pts(move)}, and {on_paper}."
        if move <= -MIN_ODDS_MOVE:
            joint = "though" if paper else "and"
            return "lose", f"Their playoff odds go down {_pts(-move)}, {joint} {on_paper}."
        return "even", f"Their playoff odds barely move, and {on_paper}."

    def _flags(self) -> tuple[list[schemas.CheckVerdict], list[schemas.CheckVerdict]]:
        """The concerns the data confirms, and every check that ran (so a page can say which came back clear)."""

        def flag(check: str, title: str, evidence: str) -> schemas.CheckVerdict:
            return schemas.CheckVerdict(check=check, title=title, status="confirmed", evidence=evidence)

        checks = self.run(FLAG_CHECKS)
        flags = [v for v in checks if v.status == "confirmed"]
        kept = [p.name for p in self.give if p.id in self.keep]
        if kept:
            flags.append(flag("keep", "You keep him", f"You marked {', '.join(kept)} as a keeper."))
        flags += [flag("premium", "Hard to replace", text) for text in self.finder._warnings(self.give)]
        cut = [p.name for p in self.my_drops if p not in droppable(self.me.roster, self.keep)]
        if cut:
            text = f"To fit the players you get you'd cut {', '.join(cut)}, who has trade value: trade him instead."
            flags.append(flag("forced_cut", "Forced cut", text))
        return flags, checks

    def _drop_options(self) -> list[schemas.OfferCutOption]:
        """Players I could cut instead, cheapest first, when the deal brings in more than it sends.

        For a one-player cut, each of the cheapest CUT_OPTIONS comes with my playoff odds if I cut him
        instead, on the same simulated seasons, so the page can show what a different cut costs.
        """
        need = len(self.get) - len(self.give)
        if need <= 0:
            return []
        give_ids = {p.id for p in self.give}
        chosen = {p.id for p in self.my_drops}
        options = sorted(
            (p for p in self.me.roster if p.id not in give_ids and p.id not in self.keep), key=self.model.value
        )
        out = []
        for rank, p in enumerate(options):
            odds = None
            if p.id in chosen:
                odds = self.after
            elif need == 1 and rank < CUT_OPTIONS:
                roster = [q for q in self.deal.my_roster if q.id != p.id] + self.my_drops
                dists = list(self.deal_dists)
                dists[self.idx] = self.model.dist(roster)
                result = self.season_result(dists, {self.idx: roster, self.j: self.deal.their_roster}, fine=True)
                odds = float(result.playoff_pct[self.idx])
            out.append(
                schemas.OfferCutOption(
                    player=self.ref(p),
                    chosen=p.id in chosen,
                    playoff_pct_after=None if odds is None else round(odds, 3),
                )
            )
        return out

    def check(self) -> schemas.OfferCheck:
        started = time.perf_counter()
        self._odds()
        d = self._engine_deal()
        j = self.j
        theirs = (float(self.today.playoff_pct[j]), float(self.after_result.playoff_pct[j]))
        finish = (
            (float(self.base_result.mean_rank[self.idx]), float(self.after_result.mean_rank[self.idx])),
            (float(self.today.mean_rank[j]), float(self.after_result.mean_rank[j])),
        )
        base_cats = self.avg_cat_win(self.idx, self.base_dists)
        idea = self.trade_idea(self.finder, d, self.before, self.after, base_cats, theirs, finish)
        helps, hurts = self._changes(self.idx, self.base_dists, self.deal_dists)
        their_helps, their_hurts = self._changes(j, self.dists, self.deal_dists, self.finder.punts[j])
        call, why = self._call(d.my_gain, helps, hurts)
        paper = d.give_value >= d.get_value - 1e-9
        their_call, their_why = self._partner_call(*theirs, paper)
        flags, checks = self._flags()
        return schemas.OfferCheck(
            direction=self.direction,
            team_id=self.me.id,
            team_name=self.me.name,
            partner_team_id=self.partner.id,
            partner_team_name=self.partner.name,
            partner_record=record(self.partner),
            call=call,
            call_reason=why,
            odds_noise=round(self.noise_pts, 3),
            helps=helps,
            hurts=hurts,
            partner_call=their_call,
            partner_reason=their_why,
            partner_helps=their_helps,
            partner_hurts=their_hurts,
            partner_punts=[self.cat_names[c] for c in np.flatnonzero(self.finder.punts[j])],
            my_punts=[self.cat_names[c] for c in np.flatnonzero(self.finder.punts[self.idx])],
            paper_win=paper,
            my_drops=[self.ref(p) for p in self.my_drops],
            my_drop_options=self._drop_options(),
            my_adds=[self.ref(p) for p in self.deal.my_adds],
            their_drops=[self.ref(p) for p in self.their_drops],
            their_adds=[self.ref(p) for p in self.their_adds],
            vs_pickup=self.free_roster is not None,
            flags=flags,
            checks=checks,
            idea=idea,
            sims=FINE_SIMS,
            generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )

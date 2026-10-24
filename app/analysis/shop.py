"""Shop your players: pick one or more of your own players and find the best deals for them league-wide.

Every package sends only players you picked, one to three of them, for one or two of
the other team's players, so nobody has to cut more than one player. A deal must
  * look even or better for the other owner on paper (trade value),
  * leave their build at least about even in the categories they compete in, and
  * help you at least as much as it helps them.
Deals where their build improves too are "win-win"; the rest are "paper wins" for them
that you win on the court. Ranked like the trade finder: playoff-odds gain weighted by
the chance the owner accepts.

In leagues with a FAAB budget, a few FAAB dollars can sweeten a package: they count as
trade value for the other owner at FAAB_RATE per dollar. FAAB spent never comes back,
so it is added only when it gets a deal over the line or makes a yes clearly likelier,
always the smallest amount that does it, capped at a share of your budget, and every
dollar lowers the deal's rank.
"""

from __future__ import annotations

import time
from dataclasses import replace
from datetime import UTC, datetime
from itertools import combinations

from app import schemas
from app.analysis.report import Analyzer, record
from app.analysis.trades import MIN_THEIR_GAIN, Deal, TradeFinder
from app.espn.parse import League, Player

MAX_GIVE = 3
MAX_GET = 2
MIN_RATIO = 0.95  # they get back at least about what they send, as owners price players
MIN_MY_GAIN = 0.03
MIN_THEIR_GAIN_SHOP = -0.05  # their build may slip a little if they win on paper
MIN_MARGIN = 0.0  # the deal must do at least as much for you as for them
SIM_TOP = 12
MAX_PER_PARTNER = 3
MAX_PER_GET = 2
RESULTS = 10
MIN_ODDS_GAIN = 0.01  # a deal must move your playoff odds by at least a point, about twice the sim noise

FAAB_CAP_SHARE = 0.10  # default most FAAB to add to one deal, as a share of what you have left
# FAAB buys waiver-level players, so a FAAB dollar is worth far less than a draft dollar in a trade.
FAAB_RATE = 0.15  # trade value (auction dollars) an owner sees in one FAAB dollar
FAAB_MAX_SHARE = 0.25  # FAAB covers at most this share of the value they send: it sweetens, it doesn't buy
FAAB_MIN_LIFT = 0.10  # FAAB must raise the chance of a yes by at least this much
FAAB_COST = 2.0  # each 10% of your remaining budget lowers a deal's rank by 20%

TARGETS = 3  # categories the search aims at
TARGET_WEIGHT = 1.0  # a win-chance point in a target category counts double
TARGET_MIN_GAIN = 0.0  # deals must not leave you worse off in your targets overall


class ShopError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class ShopAnalyzer(Analyzer):
    def __init__(
        self,
        league: League,
        team_id: int,
        offer_player_ids: list[int],
        keep_player_ids: list[int] | None = None,
        faab_max: int | None = None,
        not_for_sale_player_ids: list[int] | None = None,
    ):
        offered = set(offer_player_ids)
        # A player you're shopping can't also be one you keep.
        keep = [i for i in keep_player_ids or [] if i not in offered]
        super().__init__(league, team_id, keep, not_for_sale_player_ids)
        by_id = {p.id: p for p in self.me.roster}
        missing = offered - by_id.keys()
        if missing:
            raise ShopError("not_your_player", "Pick players from your own roster.")
        self.offered = [by_id[i] for i in offer_player_ids if i in by_id]
        self.finder = TradeFinder(league, self.model, self.idx, self.keep, mode="edge", premium=self.premium)
        self.faab_left = self.me.faab_remaining or 0
        cap = round(FAAB_CAP_SHARE * self.faab_left) if faab_max is None else faab_max
        self.faab_cap = max(0, min(cap, self.faab_left))
        self.base_cats = self.avg_cat_win(self.idx, self.dists)
        self.targets = self._targets()

    def _targets(self) -> list[tuple[int, str]]:
        """Categories to aim at: ones you're close to winning or losing each week, and ones
        the players you're shopping carry, so the return replaces what leaves. Punts are skipped."""
        win = self.base_cats
        punts = self.finder.punts[self.idx]
        dists = list(self.dists)
        offered = {p.id for p in self.offered}
        dists[self.idx] = self.model.dist([p for p in self.me.roster if p.id not in offered])
        lost = win - self.avg_cat_win(self.idx, dists)
        scored = []
        for c in range(len(win)):
            if punts[c]:
                continue
            swing = 0.5 - abs(float(win[c]) - 0.5)  # 0.5 at a coin flip, 0 when locked or lost
            score = swing + float(max(lost[c], 0.0))
            reason = "carried" if lost[c] >= swing and lost[c] >= 0.05 else "swing"
            scored.append((score, c, reason))
        scored.sort(reverse=True)
        return [(c, reason) for _, c, reason in scored[:TARGETS]]

    def target_gain(self, d: Deal) -> float:
        change = self.avg_cat_win(self.idx, d.dists) - self.base_cats
        return float(sum(change[c] for c, _ in self.targets))

    def packages(self, j: int) -> list[tuple[list[Player], list[Player]]]:
        theirs = self.teams[j].roster
        gets = {n: [list(g) for g in combinations(theirs, n)] for n in range(1, MAX_GET + 1)}
        out = []
        for k in range(1, min(MAX_GIVE, len(self.offered)) + 1):
            for give in combinations(self.offered, k):
                for n in range(max(1, k - 1), min(MAX_GET, k + 1) + 1):
                    out += [(list(give), get) for get in gets[n]]
        return out

    def _try(self, j: int, give: list[Player], get: list[Player], faab: int = 0) -> Deal | None:
        d = self.finder._evaluate(
            j,
            give,
            get,
            min_ratio=MIN_RATIO,
            min_my_gain=MIN_MY_GAIN,
            min_their_gain=MIN_THEIR_GAIN_SHOP,
            min_margin=MIN_MARGIN,
            extra_give_value=FAAB_RATE * faab,
        )
        if d is None or not faab:
            return d
        if FAAB_RATE * faab > FAAB_MAX_SHARE * d.get_value:
            return None
        return replace(d, faab=faab)

    def _sweeten(self, j: int, give: list[Player], get: list[Player], plain: Deal | None):
        """The smallest FAAB add that gets the deal done or makes a yes clearly likelier, if any."""
        if not self.faab_cap:
            return None
        floor = plain.acceptance + FAAB_MIN_LIFT if plain else 0.0

        def ok(d: Deal | None) -> bool:
            return d is not None and d.acceptance >= floor

        best = self._try(j, give, get, self.faab_cap)
        if not ok(best):
            return None
        lo, hi = 1, self.faab_cap  # trade value rises with FAAB, so the first amount that works is the one
        while lo < hi:
            mid = (lo + hi) // 2
            d = self._try(j, give, get, mid)
            if ok(d):
                best, hi = d, mid
            else:
                lo = mid + 1
        return best if best.faab == lo else self._try(j, give, get, lo)

    def _faab_cost(self, faab: int) -> float:
        return min(FAAB_COST * faab / self.faab_left, 1.0) if self.faab_left else 0.0

    def rank_key(self, d: Deal) -> float:
        aimed = d.my_gain + TARGET_WEIGHT * self.target_gain(d)
        return d.acceptance * aimed * (1.0 - self._faab_cost(d.faab))

    def deals(self, j: int) -> list[Deal]:
        found = []
        for give, get in self.packages(j):
            plain = self._try(j, give, get)
            sweet = self._sweeten(j, give, get, plain)
            # One version per package: with FAAB only when it ranks higher after its cost.
            options = [d for d in (plain, sweet) if d and self.target_gain(d) >= TARGET_MIN_GAIN]
            if options:
                found.append(max(options, key=self.rank_key))
        return sorted(found, key=self.rank_key, reverse=True)

    def _to_sim(self, by_team: dict[int, list[Deal]]) -> list[Deal]:
        """The best deal with each team, then the best overall, keeping some variety."""
        picked = [ds[0] for ds in by_team.values() if ds]
        per_partner = {d.partner_idx: 1 for d in picked}
        per_get: dict[tuple[int, ...], int] = {}
        for d in picked:
            key = tuple(sorted(p.id for p in d.get))
            per_get[key] = per_get.get(key, 0) + 1
        rest = sorted((d for ds in by_team.values() for d in ds[1:]), key=self.rank_key, reverse=True)
        extra = 0
        for d in rest:
            if extra >= SIM_TOP:
                break
            key = tuple(sorted(p.id for p in d.get))
            if per_partner.get(d.partner_idx, 0) >= MAX_PER_PARTNER or per_get.get(key, 0) >= MAX_PER_GET:
                continue
            per_partner[d.partner_idx] = per_partner.get(d.partner_idx, 0) + 1
            per_get[key] = per_get.get(key, 0) + 1
            picked.append(d)
            extra += 1
        return picked

    def _payoff(self, t: schemas.ShopDeal) -> float:
        """Playoff gain weighted by the chance of a yes, boosted by your targets, minus the FAAB it spends."""
        aimed = 1.0 + TARGET_WEIGHT * max(t.target_gain, 0.0)
        return t.acceptance * (t.playoff_pct_after - t.playoff_pct_before) * aimed * (1.0 - self._faab_cost(t.faab))

    def report(self) -> schemas.ShopReport:
        started = time.perf_counter()
        by_team = {j: self.deals(j) for j in range(len(self.teams)) if j != self.idx}
        base_cats = self.base_cats
        before = self.playoff_pct(self.dists)
        offered_ids = {p.id for p in self.offered}
        shown: list[schemas.ShopDeal] = []
        for d in self._to_sim(by_team):
            after = self.playoff_pct(d.dists, d.rosters)
            # Small category gains can vanish in the full simulation; only keep moves that hold up there.
            if after - before < MIN_ODDS_GAIN:
                continue
            idea = self.trade_idea(self.finder, d, before, after, base_cats)
            if d.faab:
                idea.reasons.append(
                    f"You add ${d.faab} FAAB of your ${self.faab_left}. It never comes back, "
                    "so send it only with the trade itself."
                )
            shown.append(
                schemas.ShopDeal(
                    **idea.model_dump(),
                    kind="win_win" if d.their_gain >= MIN_THEIR_GAIN else "paper_win",
                    uses_all={p.id for p in d.give} == offered_ids,
                    faab=d.faab,
                    target_gain=round(self.target_gain(d), 3),
                )
            )
        shown.sort(key=self._payoff, reverse=True)

        teams = []
        for j, ds in by_team.items():
            t = self.teams[j]
            best = next((s for s in shown if s.partner_team_id == t.id), None)
            teams.append(
                schemas.ShopTeam(team_id=t.id, team_name=t.name, record=record(t), deals_found=len(ds), best=best)
            )

        def market(row: schemas.ShopTeam) -> tuple:
            b = row.best
            return (b is not None, self._payoff(b) if b else -1.0, row.deals_found)

        teams.sort(key=market, reverse=True)
        return schemas.ShopReport(
            team_id=self.me.id,
            team_name=self.me.name,
            offered=[self.ref(p) for p in self.offered],
            playoff_pct_before=round(before, 3),
            faab_remaining=self.me.faab_remaining,
            faab_cap=self.faab_cap,
            targets=[
                schemas.ShopTarget(category=self.cat_names[c], win_pct=round(float(self.base_cats[c]), 3), reason=r)
                for c, r in self.targets
            ],
            deals=shown[:RESULTS],
            teams=teams,
            generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )

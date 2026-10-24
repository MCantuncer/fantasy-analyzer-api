"""Shop your players: pick one or more of your own players and find the best deals for them league-wide.

Every package sends only players you picked, one to three of them, for one or two of
the other team's players, so nobody has to cut more than one player. A deal must
  * look even or better for the other owner on paper (trade value),
  * leave their build at least about even in the categories they compete in, and
  * help you at least as much as it helps them.
Deals where their build improves too are "win-win"; the rest are "paper wins" for them
that you win on the court. Ranked like the trade finder: playoff-odds gain weighted by
the chance the owner accepts.
"""

from __future__ import annotations

import time
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


class ShopError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class ShopAnalyzer(Analyzer):
    def __init__(
        self, league: League, team_id: int, offer_player_ids: list[int], keep_player_ids: list[int] | None = None
    ):
        offered = set(offer_player_ids)
        # A player you're shopping can't also be one you keep.
        super().__init__(league, team_id, [i for i in keep_player_ids or [] if i not in offered])
        by_id = {p.id: p for p in self.me.roster}
        missing = offered - by_id.keys()
        if missing:
            raise ShopError("not_your_player", "Pick players from your own roster.")
        self.offered = [by_id[i] for i in offer_player_ids if i in by_id]
        self.finder = TradeFinder(league, self.model, self.idx, self.keep, mode="edge")

    def packages(self, j: int) -> list[tuple[list[Player], list[Player]]]:
        theirs = self.teams[j].roster
        gets = {n: [list(g) for g in combinations(theirs, n)] for n in range(1, MAX_GET + 1)}
        out = []
        for k in range(1, min(MAX_GIVE, len(self.offered)) + 1):
            for give in combinations(self.offered, k):
                for n in range(max(1, k - 1), min(MAX_GET, k + 1) + 1):
                    out += [(list(give), get) for get in gets[n]]
        return out

    def deals(self, j: int) -> list[Deal]:
        stars = self.finder._stars(self.teams[j])
        found = []
        for give, get in self.packages(j):
            d = self.finder._evaluate(
                j,
                give,
                get,
                stars,
                min_ratio=MIN_RATIO,
                min_my_gain=MIN_MY_GAIN,
                min_their_gain=MIN_THEIR_GAIN_SHOP,
                min_margin=MIN_MARGIN,
            )
            if d:
                found.append(d)
        return sorted(found, key=self.finder.rank_key, reverse=True)

    def _to_sim(self, by_team: dict[int, list[Deal]]) -> list[Deal]:
        """The best deal with each team, then the best overall, keeping some variety."""
        picked = [ds[0] for ds in by_team.values() if ds]
        per_partner = {d.partner_idx: 1 for d in picked}
        per_get: dict[tuple[int, ...], int] = {}
        for d in picked:
            key = tuple(sorted(p.id for p in d.get))
            per_get[key] = per_get.get(key, 0) + 1
        rest = sorted((d for ds in by_team.values() for d in ds[1:]), key=self.finder.rank_key, reverse=True)
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

    def report(self) -> schemas.ShopReport:
        started = time.perf_counter()
        by_team = {j: self.deals(j) for j in range(len(self.teams)) if j != self.idx}
        base_cats = self.avg_cat_win(self.idx, self.dists)
        before = self.playoff_pct(self.dists)
        offered_ids = {p.id for p in self.offered}
        shown: list[schemas.ShopDeal] = []
        for d in self._to_sim(by_team):
            after = self.playoff_pct(d.dists, d.rosters)
            # Small category gains can vanish in the full simulation; only keep moves that hold up there.
            if after < before:
                continue
            idea = self.trade_idea(self.finder, d, before, after, base_cats)
            shown.append(
                schemas.ShopDeal(
                    **idea.model_dump(),
                    kind="win_win" if d.their_gain >= MIN_THEIR_GAIN else "paper_win",
                    uses_all={p.id for p in d.give} == offered_ids,
                )
            )
        shown.sort(key=lambda t: t.acceptance * (t.playoff_pct_after - t.playoff_pct_before), reverse=True)

        teams = []
        for j, ds in by_team.items():
            t = self.teams[j]
            best = next((s for s in shown if s.partner_team_id == t.id), None)
            teams.append(
                schemas.ShopTeam(team_id=t.id, team_name=t.name, record=record(t), deals_found=len(ds), best=best)
            )

        def market(row: schemas.ShopTeam) -> tuple:
            b = row.best
            payoff = b.acceptance * (b.playoff_pct_after - b.playoff_pct_before) if b else -1.0
            return (b is not None, payoff, row.deals_found)

        teams.sort(key=market, reverse=True)
        return schemas.ShopReport(
            team_id=self.me.id,
            team_name=self.me.name,
            offered=[self.ref(p) for p in self.offered],
            playoff_pct_before=round(before, 3),
            deals=shown[:RESULTS],
            teams=teams,
            generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )

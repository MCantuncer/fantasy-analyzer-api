"""Trade of the day: the best win-win deal between any two teams in the league.

Every pair of teams is scanned with the same trade finder the per-team report uses,
from both sides. A deal qualifies only if
  * each owner, judging it the way owners do (rankings, draft prices), sees fair value,
  * it raises each team's expected categories won, and also in the categories that
    team competes in (its punts are ignored), and
  * neither team's playoff odds drop in the season simulation.

Deals are ranked by the smaller of the two gains, weighted by how likely both owners
are to say yes, so the pick is good for both sides rather than great for one.

Teams take turns in the spotlight, one per day in a fixed order per league, and the
spotlight team's best deal wins whenever it is close to the overall best. That keeps
the feature moving around the league instead of naming the same two teams every day.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import UTC, date, datetime

import numpy as np

from app import schemas
from app.analysis.model import CategoryModel, Dist
from app.analysis.perceived import PerceivedValue
from app.analysis.sim import simulate_season
from app.analysis.trades import Deal, TradeFinder, acceptance_label
from app.espn.parse import League, Player

SIM_TOP = 8  # candidates checked in the season simulation
SIM_SPOTLIGHT = 3  # extra spotlight-team candidates checked
SIMS = 2000
SPOTLIGHT_SHARE = 0.6  # the spotlight deal wins if it scores at least this share of the best

CATEGORY_WORDS = {
    "PTS": "points",
    "REB": "rebounds",
    "AST": "assists",
    "STL": "steals",
    "BLK": "blocks",
    "3PM": "threes",
    "FG%": "FG%",
    "FT%": "FT%",
    "TO": "fewer turnovers",
}


@dataclass
class Pair:
    """One deal seen from both sides: ``a`` from team i's finder, ``b`` from team j's."""

    a: Deal
    b: Deal
    i: int
    j: int

    @property
    def gain_i(self) -> float:
        return self.b.their_gain

    @property
    def gain_j(self) -> float:
        return self.a.their_gain

    @property
    def score(self) -> float:
        return min(self.gain_i, self.gain_j) * float(np.sqrt(self.a.acceptance * self.b.acceptance))


def spotlight_index(league: League, day: date) -> int:
    """The team in the spotlight today: every team gets a turn, in a fixed order per league."""
    order = list(range(len(league.teams)))
    random.Random(league.id * 10_000 + league.season).shuffle(order)
    return order[day.toordinal() % len(order)]


class DailyTrade:
    def __init__(self, league: League, day: date):
        self.league = league
        self.day = day
        self.model = CategoryModel(league)
        self.perceived = PerceivedValue(league)
        self.teams = league.teams
        self.cat_names = [c.name for c in self.model.cats]
        self.finders = [TradeFinder(league, self.model, i, set()) for i in range(len(self.teams))]
        self.dists: list[Dist] = self.finders[0].dists
        self.spotlight = spotlight_index(league, day)

    def pairs(self) -> list[Pair]:
        """Every deal that works for both owners, scanning each pair of teams once."""
        out = []
        n = len(self.teams)
        for i in range(n):
            for a in self.finders[i].candidates(partners=range(i + 1, n)):
                j = a.partner_idx
                b = self.finders[j]._evaluate(i, a.get, a.give, self.finders[j]._stars(self.teams[i]))
                if b is not None:
                    out.append(Pair(a=a, b=b, i=i, j=j))
        return sorted(out, key=lambda p: p.score, reverse=True)

    def playoff_odds(self, dists: list[Dist]) -> np.ndarray:
        return simulate_season(self.league, dists, self.model.reverse, n_sims=SIMS).playoff_pct

    def pick(self) -> tuple[Pair | None, np.ndarray, np.ndarray | None, int]:
        pairs = self.pairs()
        involved = [p for p in pairs if self.spotlight in (p.i, p.j)]
        shortlist = pairs[:SIM_TOP] + [p for p in involved[:SIM_SPOTLIGHT] if p not in pairs[:SIM_TOP]]
        before = self.playoff_odds(self.dists)
        held: list[tuple[Pair, np.ndarray]] = []
        for p in shortlist:
            after = self.playoff_odds(p.a.dists)
            # Small category gains can vanish in the full simulation; both sides must hold up there.
            if after[p.i] >= before[p.i] and after[p.j] >= before[p.j]:
                held.append((p, after))
        if not held:
            return None, before, None, len(pairs)
        best = held[0]
        spot = next((h for h in held if self.spotlight in (h[0].i, h[0].j)), None)
        if spot is not None and spot[0].score >= SPOTLIGHT_SHARE * best[0].score:
            best = spot
        return best[0], before, best[1], len(pairs)

    # ---- output ------------------------------------------------------------------------

    def ref(self, p: Player) -> schemas.PlayerRef:
        return schemas.PlayerRef(
            id=p.id,
            name=p.name,
            positions=p.positions,
            injury_status=p.injury_status,
            percent_owned=round(p.percent_owned, 1),
            value=round(self.model.value(p), 2),
            perceived_value=round(self.perceived.value(p), 1),
        )

    def side(
        self,
        idx: int,
        sends: list[Player],
        drop: Player | None,
        gain: float,
        acceptance: float,
        pitch: str,
        reasons: list[str],
        dists: list[Dist],
        before: np.ndarray,
        after: np.ndarray,
    ) -> schemas.TradeSide:
        t = self.teams[idx]
        finder = self.finders[idx]
        change = self.model.avg_cat_win(idx, dists) - finder.base_cats[idx]
        return schemas.TradeSide(
            team_id=t.id,
            team_name=t.name,
            abbrev=t.abbrev,
            sends=[self.ref(p) for p in sends],
            drop=self.ref(drop) if drop else None,
            cats_gain=round(gain, 3),
            category_changes={n: round(float(change[c]), 3) for c, n in enumerate(self.cat_names)},
            punts=[n for c, n in enumerate(self.cat_names) if finder.punts[idx][c]],
            acceptance=round(acceptance, 2),
            acceptance_label=acceptance_label(acceptance),
            pitch=pitch,
            reasons=reasons,
            playoff_pct_before=round(float(before[idx]), 3),
            playoff_pct_after=round(float(after[idx]), 3),
        )

    def headline(self, sides: list[schemas.TradeSide]) -> str:
        parts = []
        for s in sides:
            top = max(s.category_changes, key=lambda c: s.category_changes[c])
            parts.append(f"{CATEGORY_WORDS.get(top, top)} for {s.team_name}")
        return parts[0][0].upper() + parts[0][1:] + ", " + parts[1]

    def report(self) -> schemas.TradeOfTheDay:
        pair, before, after, found = self.pick()
        base = schemas.TradeOfTheDay(
            status="none",
            date=self.day.isoformat(),
            league_id=self.league.id,
            season=self.league.season,
            spotlight_team_id=self.teams[self.spotlight].id,
            team_pairs=len(self.teams) * (len(self.teams) - 1) // 2,
            win_win_deals=found,
            generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )
        if pair is None or after is None:
            return base
        a, b = pair.a, pair.b
        # A deal's pitch and reasons are written for its partner: ``a`` speaks to team j, ``b`` to team i.
        side_i = self.side(
            pair.i, a.give, a.my_drop, pair.gain_i, b.acceptance,
            self.finders[pair.j].pitch(b), _story(b.reasons), a.dists, before, after,
        )  # fmt: skip
        side_j = self.side(
            pair.j, a.get, a.their_drop, pair.gain_j, a.acceptance,
            self.finders[pair.i].pitch(a), _story(a.reasons), a.dists, before, after,
        )  # fmt: skip
        sides = [side_i, side_j]
        # Lead with the spotlight team so its owner sees their name first.
        if self.spotlight == pair.j:
            sides.reverse()
        base.status = "ready"
        base.sides = sides
        base.headline = self.headline(sides)
        return base


def _story(reasons: list[str]) -> list[str]:
    """Reasons fit for the whole league to read: drop sell-high / buy-low notes that call one side a loser."""
    return [r for r in reasons if not r.startswith(("Sell high", "Buy low"))]


def trade_of_the_day(league: League, day: date | None = None) -> schemas.TradeOfTheDay:
    return DailyTrade(league, day or datetime.now(UTC).date()).report()

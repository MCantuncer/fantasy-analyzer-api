"""Trade of the day: the best win-win deal between any two teams in the league.

Every pair of teams is scanned with the same trade finder the per-team report uses,
from both sides. A deal qualifies only if
  * each owner, judging it the way owners do (rankings, draft prices), sees fair value,
  * it raises each team's expected categories won, and also in the categories that
    team competes in (its punts are ignored), and
  * both teams' playoff odds visibly go up in the season simulation (the same one the
    Standings page shows, on each week's real games and injuries): a side whose odds stay
    flat gains nothing its owner can see, so the deal is not suggested.

Deals are ranked by the smaller of the two gains, weighted by how likely both owners
are to say yes, so the pick is good for both sides rather than great for one.

Only even trades qualify, 1-for-1 or 2-for-2. A 2-for-1 or 1-for-2 can look good on paper
for both sides, but real owners rarely make them (scripts/trade_shape_backtest.py), so one
never headlines the league.

Players a manager saved as keepers on their account are never offered: the pick should be a
deal both owners could actually make.

Only fully healthy players change hands: a day-to-day, out or injured-reserve player
makes a poor headline even when the projections allow for the games he'll miss.

Teams take turns in the spotlight, one per day in a fixed order per league, and the
spotlight team's best deal wins whenever it is close to the overall best. That keeps
the feature moving around the league instead of naming the same two teams every day.

The pick is checked again during the day. It stays while it still qualifies and is
close to the best deal on offer, and is replaced when an injury, a roster move or a
clearly better deal comes along.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import UTC, date, datetime

import numpy as np

from app import schemas
from app.analysis.badges import OwnerPremium, badge_refs
from app.analysis.durability import injury_prone
from app.analysis.model import CategoryModel, Dist
from app.analysis.perceived import PerceivedValue
from app.analysis.report import Progress, one_at_a_time, partner_gains
from app.analysis.sim import SeasonSim
from app.analysis.tiers import tier
from app.analysis.trades import Deal, TradeFinder, acceptance_label
from app.espn.parse import PRO_TEAM_ABBREV, League, Player

SIM_TOP = 8  # candidates checked in the season simulation
SIM_SPOTLIGHT = 3  # extra spotlight-team candidates checked
SIMS = 4000
SPOTLIGHT_SHARE = 0.6  # the spotlight deal wins if it scores at least this share of the best
KEEP_SHARE = 0.85  # on a re-check, the current pick stays if it scores at least this share of the best
GAIN_MIN = 0.02  # a category gain worth a tile on the card: the web shows gains from 2 points
GAIN_TILES = 4  # gain tiles per side

CATEGORY_WORDS = {
    "PTS": "points",
    "REB": "rebounds",
    "AST": "assists",
    "STL": "steals",
    "BLK": "blocks",
    "3PM": "3s",
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

    def pick_id(self, teams: list) -> str:
        """Stable id for the deal: both teams and the players each sends."""
        a = teams[self.i].id, sorted(p.id for p in self.a.give)
        b = teams[self.j].id, sorted(p.id for p in self.a.get)
        return "|".join(f"{t}:{','.join(map(str, ids))}" for t, ids in sorted([a, b]))


def fully_healthy(p: Player) -> bool:
    return p.fully_healthy


def spotlight_index(league: League, day: date) -> int:
    """The team in the spotlight today: every team gets a turn, in a fixed order per league."""
    order = list(range(len(league.teams)))
    random.Random(league.id * 10_000 + league.season).shuffle(order)
    return order[day.toordinal() % len(order)]


# Where the scan is, for the card's status line: (stage, share done 0..1). Screening every pair
# of teams is nearly all of the time; the season simulation of the shortlist is the rest.
SCREEN_SPAN = ("Screening deals", 0.04, 0.94)
SIM_SPAN = ("Simulating the season", 0.94, 0.99)


class DailyTrade:
    def __init__(
        self,
        league: League,
        day: date,
        progress: Progress | None = None,
        keepers: dict[int, list[int]] | None = None,
    ):
        """``keepers``: team id -> players its manager marked as keepers. They are never offered."""
        self.league = league
        self.day = day
        self.progress = progress
        self.model = CategoryModel(league)
        self.perceived = PerceivedValue(league, self.model)
        self.teams = league.teams
        self.cat_names = [c.name for c in self.model.cats]
        self.premium = OwnerPremium(league, self.model, self.perceived)
        kept = keepers or {}
        self.finders = [
            TradeFinder(league, self.model, i, set(kept.get(t.id, ())), premium=self.premium)
            for i, t in enumerate(self.teams)
        ]
        self.dists: list[Dist] = self.finders[0].dists
        self.spotlight = spotlight_index(league, day)
        self._sim: SeasonSim | None = None

    def pairs(self) -> list[Pair]:
        """Every deal that works for both owners, scanning each pair of teams once."""
        out = []
        n = len(self.teams)
        total, done = n * (n - 1) // 2, 0
        for i in range(n):

            def screened(k: int, _: int, before: int = done) -> None:
                self._step(SCREEN_SPAN, before + k, total)

            # The headline deal is an even one: uneven trades look good on paper but rarely get done.
            for a in self.finders[i].candidates(partners=range(i + 1, n), uneven=False, on_partner=screened):
                if not all(fully_healthy(p) for p in a.give + a.get):
                    continue
                j = a.partner_idx
                if any(p.id in self.finders[j].keep for p in a.get):
                    continue  # their manager keeps him
                b = self.finders[j]._evaluate(i, a.get, a.give)
                if b is not None:
                    out.append(Pair(a=a, b=b, i=i, j=j))
            done += n - 1 - i
        self._step(SCREEN_SPAN, total, total)
        return sorted(out, key=lambda p: p.score, reverse=True)

    def _step(self, span: tuple[str, float, float], done: int, total: int) -> None:
        if self.progress is not None:
            stage, lo, hi = span
            self.progress(stage, lo + (hi - lo) * (done / total if total else 1.0))

    def period_dists(self, changed: dict[int, list[Player]] | None = None) -> dict[int, list[Dist]] | None:
        """Each team's projection for every remaining matchup period, with changed rosters swapped in."""
        if not self.model.cal:
            return None
        rosters = [(changed or {}).get(i, t.roster) for i, t in enumerate(self.teams)]
        periods = sorted({m.period for m in self.league.schedule if m.winner == "UNDECIDED"})
        return {p: [self.model.dist(r, p) for r in rosters] for p in periods}

    def playoff_odds(self, pair: Pair | None = None) -> np.ndarray:
        """Every team's playoff odds today, or after ``pair``'s deal: paired draws, so the change is the deal's."""
        if self._sim is None:
            self._sim = SeasonSim(
                self.league, self.dists, self.model.reverse, n_sims=SIMS, period_dists=self.period_dists()
            )
        if pair is None:
            return self._sim.base.playoff_pct
        changed = {pair.i, pair.j}
        return self._sim.what_if(changed, pair.a.dists, self.period_dists(pair.a.rosters)).playoff_pct

    @staticmethod
    def both_gain(pair: Pair, before: np.ndarray, after: np.ndarray) -> bool:
        return partner_gains(before[pair.i], after[pair.i]) and partner_gains(before[pair.j], after[pair.j])

    def pick(self, keep_id: str | None = None) -> tuple[Pair | None, np.ndarray, np.ndarray | None, int]:
        """The day's deal. ``keep_id`` is the current pick on a re-check: it stays unless it no longer
        qualifies or a deal scoring clearly better has come along."""
        pairs = self.pairs()
        involved = [p for p in pairs if self.spotlight in (p.i, p.j)]
        shortlist = pairs[:SIM_TOP] + [p for p in involved[:SIM_SPOTLIGHT] if p not in pairs[:SIM_TOP]]
        before = self.playoff_odds()
        held: list[tuple[Pair, np.ndarray]] = []
        for k, p in enumerate(shortlist):
            self._step(SIM_SPAN, k, len(shortlist))
            after = self.playoff_odds(p)
            # Small category gains can vanish in the full simulation; both owners must see their odds go up.
            if self.both_gain(p, before, after):
                held.append((p, after))
        best = held[0] if held else None
        spot = next((h for h in held if self.spotlight in (h[0].i, h[0].j)), None)
        if best and spot and spot[0].score >= SPOTLIGHT_SHARE * best[0].score:
            best = spot
        current = next((p for p in pairs if p.pick_id(self.teams) == keep_id), None) if keep_id else None
        if current is not None and (best is None or current.score >= KEEP_SHARE * best[0].score):
            after = self.playoff_odds(current)
            if self.both_gain(current, before, after):
                best = (current, after)
        if best is None:
            return None, before, None, len(pairs)
        return best[0], before, best[1], len(pairs)

    # ---- output ------------------------------------------------------------------------

    def ref(self, p: Player) -> schemas.PlayerRef:
        return schemas.PlayerRef(
            id=p.id,
            name=p.name,
            pro_team=PRO_TEAM_ABBREV.get(p.pro_team_id, ""),
            positions=p.positions,
            injury_status=p.injury_status,
            percent_owned=round(p.percent_owned, 1),
            value=round(self.model.value(p), 2),
            perceived_value=round(self.perceived.value(p), 1),
            tier=tier(p),
            badges=badge_refs(self.premium, p),
            owner_premium=round(self.premium.dollars(p), 1),
            games_share=round(p.durability, 3) if p.durability is not None else None,
            injury_prone=injury_prone(p),
            games_out=round(p.games_out, 1) if p.games_out is not None else None,
            expected_back=self.model.back_date(p),
            out_report=p.out_report,
        )

    def side(
        self,
        idx: int,
        sends: list[Player],
        drop: Player | None,
        gain: float,
        acceptance: float,
        pitch: str,
        facts: list[schemas.ReasonFact],
        needs: list[schemas.PitchNeed],
        gains: list[schemas.PitchNeed],
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
            reasons=[f.text for f in facts],
            facts=facts,
            pitch_needs=needs,
            gains=gains,
            playoff_pct_before=round(float(before[idx]), 3),
            playoff_pct_after=round(float(after[idx]), 3),
        )

    def headline(self, sides: list[schemas.TradeSide]) -> str:
        parts = []
        for s in sides:
            top = max(s.category_changes, key=lambda c: s.category_changes[c])
            parts.append(f"{CATEGORY_WORDS.get(top, top)} for {s.team_name}")
        return parts[0][0].upper() + parts[0][1:] + ", " + parts[1]

    def report(self, keep_id: str | None = None) -> schemas.TradeOfTheDay:
        pair, before, after, found = self.pick(keep_id)
        base = schemas.TradeOfTheDay(
            status="none",
            date=self.day.isoformat(),
            league_id=self.league.id,
            season=self.league.season,
            spotlight_team_id=self.teams[self.spotlight].id,
            team_pairs=len(self.teams) * (len(self.teams) - 1) // 2,
            win_win_deals=found,
            generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
            checked_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )
        if pair is None or after is None:
            return base
        a, b = pair.a, pair.b
        # A deal's pitch and reasons are written for its partner: ``a`` speaks to team j, ``b`` to team i.
        side_i = self.side(
            pair.i, a.give, a.my_drop, pair.gain_i, b.acceptance,
            self.finders[pair.j].pitch(b), _story(b.facts), self.finders[pair.j].pitch_needs(b),
            self.finders[pair.j].pitch_needs(b, GAIN_MIN, GAIN_TILES), a.dists, before, after,
        )  # fmt: skip
        side_j = self.side(
            pair.j, a.get, a.their_drop, pair.gain_j, a.acceptance,
            self.finders[pair.i].pitch(a), _story(a.facts), self.finders[pair.i].pitch_needs(a),
            self.finders[pair.i].pitch_needs(a, GAIN_MIN, GAIN_TILES), a.dists, before, after,
        )  # fmt: skip
        sides = [side_i, side_j]
        # Lead with the spotlight team so its owner sees their name first.
        if self.spotlight == pair.j:
            sides.reverse()
        base.status = "ready"
        base.pick_id = pair.pick_id(self.teams)
        base.score = round(pair.score, 4)
        base.sides = sides
        base.headline = self.headline(sides)
        return base


def _story(facts: list[schemas.ReasonFact]) -> list[schemas.ReasonFact]:
    """Reasons fit for the whole league to read: drop sell-high / buy-low notes that call one side a loser."""
    return [f for f in facts if f.kind not in ("sell_high", "buy_low")]


def trade_of_the_day(
    league: League,
    day: date | None = None,
    keep_id: str | None = None,
    progress: Progress | None = None,
    keepers: dict[int, list[int]] | None = None,
) -> schemas.TradeOfTheDay:
    # Shares the one big-run slot with team reports: building every team's trade finder is the big part.
    with one_at_a_time(progress, "Waiting its turn", 0.0):
        return DailyTrade(league, day or datetime.now(UTC).date(), progress, keepers).report(keep_id)

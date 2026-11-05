"""When to make a pickup: the day the swap does the most for this week's matchup.

A free agent joins your roster the moment you add him and counts for his game that day
if it hasn't tipped off yet. A player on waivers joins when his claim processes. The
player you drop is gone from then on, so a swap made on a day he plays and the new man
doesn't throws that game away: adding Brook Lopez (no game tonight) for Georges Niang
(playing tonight) costs Niang's game, and making it tomorrow keeps it.

Each day left in the matchup is tried as the day the swap takes effect, plus "after this
week", and scored on the categories you're expected to win against your real opponent,
with what's already banked. The rest of the season is the same whichever day this week
it happens, so this week decides. Waiting only wins when it's clearly better: the
player could be gone by then.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.analysis.model import CategoryModel, Dist
from app.analysis.projection import full_week
from app.espn.parse import League, Player

# Expected categories this matchup that waiting must add before it's worth the risk of
# someone else picking him up. A player starting one more game moves a close category
# by a few hundredths; less than this is noise.
WAIT_MIN_GAIN = 0.02


@dataclass
class Timing:
    earliest_day: int  # the first day the swap can take effect (scoring period)
    best_day: int  # the day it should take effect; past the matchup's last day = next week
    cats_earliest: float  # expected categories won this matchup, change from no move, if made now
    cats_best: float  # the same if made on best_day
    drop_games_kept: int  # games the dropped player still plays for you by waiting
    add_games_missed: int  # games of the new player you miss by waiting


class MatchupTiming:
    """Scores swaps by the day they take effect, for one team's current matchup."""

    def __init__(self, league: League, team_id: int, model: CategoryModel, now_ms: int):
        self.league = league
        self.model = model
        self.now_ms = now_ms
        self.ready = False
        cal = model.cal
        period = league.current_period
        if not cal or period not in model.horizon:
            return
        m = next((x for x in league.schedule if x.period == period and team_id in (x.home, x.away)), None)
        team = league.team(team_id)
        if m is None or team is None:
            return
        home = m.home == team_id
        opponent = league.team(m.away if home else m.home)
        if opponent is None:
            return
        self.period = period
        self.days = np.array(model.horizon[period][0], dtype=int)
        if not len(self.days):
            return
        self.roster = team.roster
        self.banked = m.home_totals if home else m.away_totals
        self.theirs = full_week(model, model.dist(opponent.roster, period), m.away_totals if home else m.home_totals)
        self.base = self._cats(model.dist(self.roster, period))
        self.ready = True

    def _cats(self, mine: Dist) -> float:
        return float(self.model.cat_win_prob(full_week(self.model, mine, self.banked), self.theirs).sum())

    def _tipped(self, p: Player, day: int, ms: int) -> bool:
        """His game that day has tipped off by ``ms`` (no tip time known: not yet)."""
        tip = self.model.cal.tips.get(p.pro_team_id, {}).get(day)
        return tip is not None and tip <= ms

    def best(self, add: Player, drop: Player) -> Timing | None:
        """The best day this week for ``add`` to replace ``drop``. None when nothing this week
        depends on it: no matchup, or the earliest he can join is after the matchup ends."""
        if not self.ready:
            return None
        cal = self.model.cal
        joins_ms = self.now_ms
        if add.status == "WAIVERS" and add.waiver_ms and add.waiver_ms > self.now_ms:
            joins_ms = add.waiver_ms
        earliest = max(cal.day_at(joins_ms), int(self.days[0]))
        last = int(self.days[-1])
        if earliest > last:
            return None
        add_days = self.model.plays(add, self.period)
        drop_days = self.model.plays(drop, self.period)
        # On the day he joins, the new player counts only if his game is still to come,
        # and the dropped one keeps a game that has already started.
        add_first = not self._tipped(add, earliest, joins_ms)
        drop_first = self._tipped(drop, earliest, joins_ms)
        roster = self.roster + [add]
        tried: list[tuple[int, float, np.ndarray, np.ndarray]] = []
        for start in range(earliest, last + 2):  # last + 1: wait for next week
            after = self.days > start
            on = self.days == start
            if start == earliest:
                a = add_days & (after | (on & add_first))
                d = drop_days & (self.days < start) | (drop_days & on & drop_first)
            else:
                a = add_days & (after | on)
                d = drop_days & (self.days < start)
            gain = self._cats(self.model.dist_with(roster, self.period, {add.id: a, drop.id: d})) - self.base
            tried.append((start, gain, a, d))
        first = tried[0]
        top = max(g for _, g, _, _ in tried)
        # The earliest day within reach of the best: a later one must clearly beat it.
        pick = first if top - first[1] < WAIT_MIN_GAIN else max(tried, key=lambda t: t[1])
        return Timing(
            earliest_day=earliest,
            best_day=pick[0],
            cats_earliest=first[1],
            cats_best=pick[1],
            drop_games_kept=int(pick[3].sum() - first[3].sum()),
            add_games_missed=int(first[2].sum() - pick[2].sum()),
        )

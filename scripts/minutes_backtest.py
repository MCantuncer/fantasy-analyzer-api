"""Checks the minutes handling in the per-game projection on past seasons.

    uv run python -m scripts.minutes_backtest

ESPN's preseason projection tends to give players back an old role: players it projected
for 3+ more minutes than they played the season before got 83-88% of those minutes. Two
changes to the projection (app/espn/parse.py) are checked here, against the projection
as it was (ESPN's line blended per stat with the season so far):

  * a: the prior's minutes move this share of the way toward last season's minutes
    (LAST_MINUTES_WEIGHT), counting stats scaled along
  * K: once games are played, minutes follow the season so far with weight gp / (gp + K)
    (MINUTES_WEIGHT_GAMES), counting stats scaled along

For players ESPN projected for 20+ games, after N of their games (0, 5, 10, 20), each
version predicts their per-game line over the rest of the season (20+ more games). The
score is root mean squared error relative to the old projection, averaged over the nine
categories (below 1 is better). The constants are picked on 2023-24 and kept only if they
are no worse on 2024-25 and 2025-26 at any N. Players with a 3+ minute gap between ESPN's
projection and last season are shown on their own.

Uses the caches of scripts/durability_backtest.py (.cache/season_pools, .cache/gamelogs).
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Callable
from dataclasses import dataclass

from app.espn.parse import (
    ACTUAL_WEIGHT_BY_STAT,
    ACTUAL_WEIGHT_GAMES,
    GAMES_PLAYED,
    LAST_MINUTES_WEIGHT,
    MINUTES,
    MINUTES_WEIGHT_GAMES,
)
from scripts.durability_backtest import POOLS
from scripts.projection_backtest import COUNTING, RATIOS, _total, game_lines, projections

FIT, TEST = 2024, (2025, 2026)
CUTS = (0, 5, 10, 20)
MIN_REST = 20
MIN_LAST_GAMES = 20
STATS = [*COUNTING.values(), 13, 14, 15, 16, MINUTES]
CATS = [*COUNTING, *RATIOS]
TRIED_A = (0.0, 0.15, 0.25, 0.35)
TRIED_K = (None, 5.0, 7.0, 10.0, 15.0)
GAP = 3.0  # minutes ESPN projected above last season


@dataclass
class Case:
    cut: int
    proj: dict[int, float]
    last: dict[int, float] | None  # last season per game
    before: list[dict[int, float]]
    after: list[dict[int, float]]

    @property
    def gap(self) -> float:
        return self.proj.get(MINUTES, 0.0) - self.last.get(MINUTES, 0.0) if self.last else 0.0


def season_lines(season: int) -> dict[int, dict[int, float]]:
    """Per-game lines for players with 20+ games in ``season``."""
    out = {}
    for e in json.loads((POOLS / f"{season}.json").read_text()):
        p = e.get("player") or e
        for s in p.get("stats") or []:
            if s.get("seasonId") == season and s.get("statSourceId") == 0 and s.get("statSplitTypeId") == 0:
                total = {int(k): float(v) for k, v in (s.get("stats") or {}).items()}
                gp = total.get(GAMES_PLAYED, 0.0)
                if gp >= MIN_LAST_GAMES:
                    out[int(p["id"])] = {k: v / gp for k, v in total.items()}
    return out


def cases(season: int) -> list[Case]:
    lines, last = game_lines(season), season_lines(season - 1)
    out = []
    for pid, proj in projections(season).items():
        games = lines.get(pid, [])
        out += [Case(n, proj, last.get(pid), games[:n], games[n:]) for n in CUTS if len(games) >= n + MIN_REST]
    return out


def _scaled(line: dict[int, float], minutes: float) -> dict[int, float]:
    scale = minutes / line[MINUTES] if line.get(MINUTES) else 1.0
    return {k: (minutes if k == MINUTES else v * scale) for k, v in line.items()}


def predict(c: Case, a: float, k: float | None) -> dict[int, float]:
    prior = {s: c.proj.get(s, 0.0) for s in STATS}
    if a and c.last and c.last.get(MINUTES):
        prior = _scaled(prior, (1 - a) * prior[MINUTES] + a * c.last[MINUTES])
    if not c.cut:
        return prior
    out = {}
    for s in STATS:
        w = c.cut / (c.cut + ACTUAL_WEIGHT_BY_STAT.get(s, ACTUAL_WEIGHT_GAMES))
        out[s] = (1 - w) * prior[s] + w * _total(c.before, s) / c.cut
    if k is not None:
        w = c.cut / (c.cut + k)
        out = _scaled(out, (1 - w) * prior[MINUTES] + w * _total(c.before, MINUTES) / c.cut)
    return out


def rmse(cs: list[Case], cat: str, f: Callable[[Case], dict[int, float]]) -> float:
    num = den = 0.0
    for c in cs:
        p = f(c)
        if cat in COUNTING:
            s = COUNTING[cat]
            pred, actual, weight = p[s], _total(c.after, s) / len(c.after), len(c.after)
        else:
            makes, attempts = RATIOS[cat]
            pred = p[makes] / p[attempts] if p[attempts] else 0.0
            weight = _total(c.after, attempts)
            actual = _total(c.after, makes) / weight if weight else 0.0
        num += weight * (pred - actual) ** 2
        den += weight
    return (num / den) ** 0.5 if den else 0.0


def relative(cs: list[Case], a: float, k: float | None) -> float:
    def old(c: Case) -> dict[int, float]:
        return predict(c, 0.0, None)

    def new(c: Case) -> dict[int, float]:
        return predict(c, a, k)

    return statistics.mean(rmse(cs, cat, new) / rmse(cs, cat, old) for cat in CATS)


def row(data: dict[int, list[Case]], a: float, k: float | None, keep: Callable[[Case], bool]) -> str:
    cells = []
    for season, cs in data.items():
        rel = [relative([c for c in cs if c.cut == n and keep(c)], a, k) for n in CUTS]
        cells.append(f"{season}: " + " ".join(f"{x:.3f}" for x in rel))
    return "  |  ".join(cells)


def main() -> None:
    data = {s: cases(s) for s in (FIT, *TEST)}
    print(f"error relative to the old projection after {', '.join(map(str, CUTS))} games (below 1 is better)")
    for a in TRIED_A:
        for k in TRIED_K:
            mark = " <- chosen" if a == LAST_MINUTES_WEIGHT and k == MINUTES_WEIGHT_GAMES else ""
            print(f"  a {a:.2f} K {k or '-':>4}  {row(data, a, k, lambda c: True)}{mark}")
    print(f"\nplayers ESPN projected for {GAP:g}+ more minutes than last season, chosen constants")
    print(f"  {row(data, LAST_MINUTES_WEIGHT, MINUTES_WEIGHT_GAMES, lambda c: c.gap >= GAP)}")


if __name__ == "__main__":
    main()

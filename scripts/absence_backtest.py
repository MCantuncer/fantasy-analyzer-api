"""Fits and checks the games-still-to-miss table (app/analysis/absence.py) on past seasons.

    uv run python -m scripts.absence_backtest              # fit on each season, test on the next
    uv run python -m scripts.absence_backtest --table 2026 # print the table fitted on one season

Every run of team games a regular missed in a row is an absence. For each game of it
(after k missed so far), the outcome is how many more he missed, cut off at the
season's end. Absences that start in a team's last 20 games are left out: late-season
rest and shutdowns aren't injuries a trade can plan around. Regulars are players ESPN
projected for 50+ games with 60+ team games logged.

The table predicts the mean of that outcome by k. It's compared with the flat guess the
model used before (two weeks, about 7 team games) by root mean squared error, the
measure that gets expected games right. Durability (games-played record) was tried as a
second input: spells of injury-prone players weren't consistently longer, so it's out.

Uses the caches of scripts/durability_backtest.py (.cache/season_pools, .cache/gamelogs).
"""

from __future__ import annotations

import argparse
import bisect
import statistics
from dataclasses import dataclass

from app.analysis.absence import REMAINING
from scripts.durability_backtest import MIN_PROJECTED, game_logs, season_pool

SEASONS = (2024, 2025, 2026)
MIN_LOGGED = 60
LATE = 20  # absences starting in a team's last this many games are left out
FLAT = 7.0  # the old guess: 14 days out, about 7 team games
BOUNDS = [k for k, _ in REMAINING]
CUTS = (1, 5, 10)


@dataclass
class Case:
    season: int
    missed: int  # games missed in a row so far
    rest: int  # games he went on to miss


def cases(season: int) -> list[Case]:
    _, projected = season_pool(season)
    out = []
    for pid, games in game_logs(season).items():
        if projected.get(pid, 0) < MIN_PROJECTED or len(games) < MIN_LOGGED:
            continue
        n, i = len(games), 0
        while i < n:
            if games[i]:
                i += 1
                continue
            j = i
            while j < n and not games[j]:
                j += 1
            if i < n - LATE:
                out += [Case(season, k, j - i - k) for k in range(1, j - i + 1) if i + k < n]
            i = j
    return out


def bucket(k: int) -> int:
    return bisect.bisect_right(BOUNDS, k) - 1


def fit(cs: list[Case]) -> list[float]:
    means = [statistics.fmean([c.rest for c in cs if bucket(c.missed) == b] or [0.0]) for b in range(len(BOUNDS))]
    for b in range(1, len(means)):  # non-decreasing: a longer absence so far never means a shorter rest
        means[b] = max(means[b], means[b - 1])
    return [round(m, 1) for m in means]


def rmse(cs: list[Case], predict) -> float:
    return statistics.fmean((predict(c) - c.rest) ** 2 for c in cs) ** 0.5


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--table", type=int, help="print the table fitted on this season")
    args = ap.parse_args()
    by_season = {s: cases(s) for s in SEASONS}
    if args.table:
        print(list(zip(BOUNDS, fit(by_season[args.table]), strict=True)))
        return
    print("root mean squared error in games still to miss (lower is better); n = absence-games")
    for fit_season, test_season in zip(SEASONS, SEASONS[1:], strict=False):
        table = fit(by_season[fit_season])
        test = by_season[test_season]
        print(f"\nfit {fit_season}, test {test_season}: table {dict(zip(BOUNDS, table, strict=True))}")
        for k in CUTS:
            sub = [c for c in test if c.missed >= k]
            actual = statistics.fmean(c.rest for c in sub)
            pred = statistics.fmean(table[bucket(c.missed)] for c in sub)
            print(
                f"  out {k:2}+ games  n={len(sub):5}  flat {rmse(sub, lambda c: FLAT):5.1f}"
                f"  table {rmse(sub, lambda c, t=table: t[bucket(c.missed)]):5.1f}"
                f"  mean missed {actual:4.1f}, table said {pred:4.1f}"
            )


if __name__ == "__main__":
    main()

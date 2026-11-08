"""Checks the age trend in the per-game projection (app/espn/parse.py, PEAK_AGE) on past seasons.

    uv run python -m scripts.age_backtest            # chosen constants, fit 2023-24, test 2024-25 + 2025-26
    uv run python -m scripts.age_backtest --fit      # also search the constants on the fit season

For players ESPN projected for 20+ games, after N of their games (0, 5, 10, 20, 30, 40),
the projection as the app builds it (parse._blend_stats: ESPN's preseason line, minutes
pulled toward last season's, blended per stat with the season so far) predicts their
per-game line over the rest of the season (15+ more games), which game logs then tell.
It's run with and without the player's age.

Scored by the value of the line in standard 9-cat terms (z-scores against the 200 players
ESPN projected to score most, summed; percentages weighted by attempts): root mean squared
error of predicted against actual value, weighted by games left. Also printed: how far
actual value lands from predicted by age group (positive: the projection was too low),
which is what the trend corrects.

Ages come from app/espn/birth_dates.csv. Uses the caches of scripts/durability_backtest.py
(.cache/season_pools, .cache/gamelogs).
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
from dataclasses import dataclass

from app.espn import parse
from app.espn.ages import season_age
from app.espn.parse import GAMES_PLAYED, MINUTES, _blend_stats
from scripts.durability_backtest import POOLS
from scripts.minutes_backtest import season_lines
from scripts.projection_backtest import COUNTING, RATIOS, _total, game_lines, projections

FIT, TEST = 2024, (2025, 2026)
CUTS = (0, 5, 10, 20, 30, 40)
FIT_CUTS = (5, 10, 20, 30)
MIN_REST = 15
SCALE_POOL = 200
STATS = [*COUNTING.values(), 13, 14, 15, 16, MINUTES]
GROUPS = ((0, 23, "under 23"), (23, 26, "23-25"), (26, 30, "26-29"), (30, 33, "30-32"), (33, 99, "33+"))
TRIED_YOUNG = (0.0, 0.02, 0.03, 0.04, 0.05, 0.06)
TRIED_OLD = (0.0, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02)


@dataclass
class Case:
    season: int
    cut: int
    age: float
    player: dict  # ESPN-shaped stats as of the cut, what parse._blend_stats reads
    after: list[dict[int, float]]


def _names(season: int) -> dict[int, str]:
    return {int((e.get("player") or e)["id"]): (e.get("player") or e)["fullName"] for e in _pool(season)}


def _pool(season: int) -> list[dict]:
    return json.loads((POOLS / f"{season}.json").read_text())


def _entry(season: int, source: int, line: dict[int, float]) -> dict:
    return {
        "seasonId": season,
        "statSourceId": source,
        "statSplitTypeId": 0,
        "averageStats": {str(k): v for k, v in line.items()},
    }


def cases(season: int) -> list[Case]:
    names, lines, last = _names(season), game_lines(season), season_lines(season - 1)
    out = []
    for pid, proj in projections(season).items():
        age = season_age(names.get(pid, ""), season)
        games = lines.get(pid, [])
        if age is None:
            continue
        for n in CUTS:
            if len(games) < n + MIN_REST:
                continue
            stats = [_entry(season, 1, proj)]
            if pid in last:
                stats.append(_entry(season - 1, 0, last[pid]))
            if n:
                so_far = {s: _total(games[:n], s) / n for s in STATS} | {GAMES_PLAYED: float(n)}
                stats.append(_entry(season, 0, so_far))
            out.append(Case(season, n, age, {"stats": stats}, games[n:]))
    return out


def scales(season: int) -> dict[str, tuple[float, float]]:
    """Mean and spread of each category among the players ESPN projected to score most."""
    top = sorted(projections(season).values(), key=lambda p: -p.get(0, 0.0))[:SCALE_POOL]
    out = {}
    for cat, s in COUNTING.items():
        vals = [p.get(s, 0.0) for p in top]
        out[cat] = (statistics.fmean(vals), statistics.pstdev(vals))
    for cat, (m, a) in RATIOS.items():
        pct = sum(p.get(m, 0.0) for p in top) / sum(p.get(a, 0.0) for p in top)
        out[cat] = (pct, statistics.pstdev([p.get(m, 0.0) - pct * p.get(a, 0.0) for p in top]))
    return out


def value(line: dict[int, float], sc: dict[str, tuple[float, float]]) -> float:
    total = 0.0
    for cat, s in COUNTING.items():
        mean, sd = sc[cat]
        z = (line.get(s, 0.0) - mean) / sd
        total += -z if cat == "TO" else z
    for cat, (m, a) in RATIOS.items():
        pct, sd = sc[cat]
        total += (line.get(m, 0.0) - pct * line.get(a, 0.0)) / sd
    return total


def scored(cs: list[Case], sc: dict, with_age: bool) -> list[tuple[Case, float, float]]:
    """(case, predicted value, actual value) for each case."""
    out = []
    for c in cs:
        pred, _, _ = _blend_stats(c.player, c.season, c.age if with_age else None)
        actual = {s: _total(c.after, s) / len(c.after) for s in STATS}
        out.append((c, value(pred, sc), value(actual, sc)))
    return out


def rmse(rows: list[tuple[Case, float, float]]) -> float:
    weight = sum(len(c.after) for c, _, _ in rows)
    return (sum(len(c.after) * (p - a) ** 2 for c, p, a in rows) / weight) ** 0.5 if weight else 0.0


def bias(rows: list[tuple[Case, float, float]]) -> float:
    weight = sum(len(c.after) for c, _, _ in rows)
    return sum(len(c.after) * (a - p) for c, p, a in rows) / weight if weight else 0.0


def with_trend(young: float, old: float):
    parse.YOUNG_TREND, parse.OLD_TREND = young, old


def fit(cs: list[Case], sc: dict) -> tuple[float, float]:
    chosen = (parse.YOUNG_TREND, parse.OLD_TREND)
    fit_cases = [c for c in cs if c.cut in FIT_CUTS]

    def err(g: tuple[float, float]) -> float:
        with_trend(*g)
        return rmse(scored(fit_cases, sc, True))

    best = min(itertools.product(TRIED_YOUNG, TRIED_OLD), key=err)
    with_trend(*chosen)
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fit", action="store_true", help="search the constants on the fit season first")
    args = ap.parse_args()
    data = {s: cases(s) for s in (FIT, *TEST)}
    sc = {s: scales(s) for s in data}
    if args.fit:
        young, old = fit(data[FIT], sc[FIT])
        print(f"fitted on {FIT}: YOUNG_TREND {young}  OLD_TREND {old}")
        with_trend(young, old)
    print(f"YOUNG_TREND {parse.YOUNG_TREND}  OLD_TREND {parse.OLD_TREND}  PEAK_AGE {parse.PEAK_AGE}")
    print("\nvalue error per game, without age -> with age (lower is better)")
    for season, cs in data.items():
        cells = []
        for n in CUTS:
            here = [c for c in cs if c.cut == n]
            before, after = rmse(scored(here, sc[season], False)), rmse(scored(here, sc[season], True))
            cells.append(f"{n:2}: {before:.3f}->{after:.3f}")
        print(f"  {season} after N games  " + "  ".join(cells))
    print("\nactual minus predicted value by age, after 5-40 games, without age -> with age (0 is unbiased)")
    for season, cs in data.items():
        played = [c for c in cs if c.cut]
        cells = []
        for lo, hi, label in GROUPS:
            group = [c for c in played if lo <= c.age < hi]
            before, after = bias(scored(group, sc[season], False)), bias(scored(group, sc[season], True))
            cells.append(f"{label} {before:+.2f}->{after:+.2f}")
        print(f"  {season}  " + "  ".join(cells))


if __name__ == "__main__":
    main()

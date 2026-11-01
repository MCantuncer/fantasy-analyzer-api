"""Checks how fast this season's stats should take over from the preseason projection, per category.

    uv run python -m scripts.projection_backtest

A player's per-game projection is ESPN's preseason projection blended with his season
so far: weight gp / (gp + K) on the season (app/espn/parse.py). For players ESPN
projected for 20+ games, after N of their games (5, 10, 20, 30), the blend predicts
their per-game line over the rest of the season (20+ more games), which game logs then
tell. Percentages are blended as makes and attempts and scored weighted by attempts.

Reported as error relative to K = 20 (below 1 is better), for each K tried, on every
season: a K is kept only if it wins on the fit season (2023-24) and on both seasons
after. Also printed: how far off the season so far is on its own (what owners see) and
the projection on its own, relative to the blend.

Uses the caches of scripts/durability_backtest.py (.cache/season_pools, .cache/gamelogs).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from app.espn.parse import ACTUAL_WEIGHT_BY_STAT, ACTUAL_WEIGHT_GAMES, GAMES_PLAYED
from scripts.durability_backtest import LOGS, POOLS, _get

SEASONS = (2024, 2025, 2026)
COUNTING = {"PTS": 0, "REB": 6, "AST": 3, "STL": 2, "BLK": 1, "3PM": 17, "TO": 11}
RATIOS = {"FG%": (13, 14), "FT%": (15, 16)}
CUTS = (5, 10, 20, 30)
MIN_REST = 20
TRIED = (10.0, 20.0, 30.0, 45.0, 90.0)


@dataclass
class Case:
    cut: int
    proj: dict[int, float]  # preseason projection, per game
    before: list[dict[int, float]]  # game lines up to the cut
    after: list[dict[int, float]]  # game lines after it


def projections(season: int) -> dict[int, dict[int, float]]:
    flt = {"limit": 500, "sortPercOwned": {"sortPriority": 1, "sortAsc": False}}
    out = {}
    for e in _get(season, flt, POOLS / f"{season}.json"):
        p = e.get("player") or e
        for s in p.get("stats") or []:
            if s.get("seasonId") == season and s.get("statSourceId") == 1 and s.get("statSplitTypeId") == 0:
                total = {int(k): float(v) for k, v in (s.get("stats") or {}).items()}
                gp = total.get(GAMES_PLAYED, 0.0)
                if gp >= 20:
                    out[int(p["id"])] = {k: v / gp for k, v in total.items()}
    return out


def game_lines(season: int) -> dict[int, list[dict[int, float]]]:
    """Each player's lines in the games he played, in order."""
    days: dict[int, dict[int, dict[int, float]]] = {}
    for team in range(1, 31):
        for e in json.loads((LOGS / f"{season}-{team}.json").read_text()):
            p = e.get("player") or e
            mine = days.setdefault(int(p["id"]), {})
            for s in p.get("stats") or []:
                if s.get("seasonId") != season or s.get("statSplitTypeId") != 5 or s.get("statSourceId"):
                    continue
                if (s.get("stats") or {}).get(str(GAMES_PLAYED)):
                    mine[int(s["scoringPeriodId"])] = {int(k): float(v) for k, v in s["stats"].items()}
    return {pid: [g[d] for d in sorted(g)] for pid, g in days.items()}


def cases(season: int) -> list[Case]:
    lines = game_lines(season)
    out = []
    for pid, proj in projections(season).items():
        games = lines.get(pid, [])
        out += [Case(n, proj, games[:n], games[n:]) for n in CUTS if len(games) >= n + MIN_REST]
    return out


def _total(games: list[dict[int, float]], stat: int) -> float:
    return sum(g.get(stat, 0.0) for g in games)


def blend(c: Case, stat: int, k: float | None) -> float:
    """Per-game blend; ``k=None`` is the season so far on its own."""
    w = 1.0 if k is None else c.cut / (c.cut + k)
    return (1 - w) * c.proj.get(stat, 0.0) + w * _total(c.before, stat) / c.cut


def rmse(cs: list[Case], cat: str, k: float | None) -> float:
    num = den = 0.0
    for c in cs:
        if cat in COUNTING:
            stat = COUNTING[cat]
            pred, actual, weight = blend(c, stat, k), _total(c.after, stat) / len(c.after), len(c.after)
        else:
            makes, attempts = RATIOS[cat]
            pa = blend(c, attempts, k)
            pred = blend(c, makes, k) / pa if pa else 0.0
            weight = _total(c.after, attempts)
            actual = _total(c.after, makes) / weight if weight else 0.0
        num += weight * (pred - actual) ** 2
        den += weight
    return (num / den) ** 0.5


def _stat(cat: str) -> int:
    return COUNTING[cat] if cat in COUNTING else RATIOS[cat][0]


def main() -> None:
    data = {s: cases(s) for s in SEASONS}
    print("error relative to K = 20 (below 1 is better)")
    for cat in [*COUNTING, *RATIOS]:
        stat = _stat(cat)
        chosen = ACTUAL_WEIGHT_BY_STAT.get(stat, ACTUAL_WEIGHT_GAMES)
        cells = []
        for s in SEASONS:
            base = rmse(data[s], cat, 20.0)
            cells.append(f"{s}: " + " ".join(f"K{k:g} {rmse(data[s], cat, k) / base:.3f}" for k in TRIED))
        print(f"  {cat:4} chosen K{chosen:g}  " + "  |  ".join(cells))
    print("\nseason so far alone / projection alone, relative to the chosen blend (above 1: the blend is better)")
    for s in SEASONS[1:]:
        for n in CUTS:
            cs = [c for c in data[s] if c.cut == n]
            cells = []
            for cat in [*COUNTING, *RATIOS]:
                stat = _stat(cat)
                fit = rmse(cs, cat, ACTUAL_WEIGHT_BY_STAT.get(stat, ACTUAL_WEIGHT_GAMES))
                cells.append(f"{cat} {rmse(cs, cat, None) / fit:.2f}/{rmse(cs, cat, 1e9) / fit:.2f}")
            print(f"  {s} after {n:2} games (n={len(cs)}): " + "  ".join(cells))


if __name__ == "__main__":
    main()

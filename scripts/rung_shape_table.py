"""Builds app/analysis/rung_shape.json: how often a player reaches n of a stat, given our line.

    uv run --with pyarrow python -m scripts.rung_shape_table

A sportsbook prices player props as "n+" rungs ("Flagg 30+"). To set our own chance beside each
rung, this replays the shipped next-game line (gamebox.apply on the backtest cases of
scripts/next_game_matchup_backtest.py, 2021-22 to 2025-26, players with a minutes line of
MIN_MINUTES+) and counts, for lines near each grid value, how often the player got to n or more.

Per stat: a grid of line values; at each, the share of games with the stat at n or more for
n = 0..N_MAX, from the games whose line is within a bandwidth of the grid value (widened until
MIN_GAMES games). Chances rise with the line (made monotone).

Blind check printed at the end: each season's games scored with a table built from the other four
(chance buckets: what the table said vs how often it happened).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.analysis import gamebox
from app.analysis.rungs import STATS, TABLE
from scripts.durability_backtest import season_pool
from scripts.next_game_matchup_backtest import SEASONS, cases

MIN_MINUTES = 12.0
MIN_GAMES = 400
GRID = {
    "pts": (2.0, 40.0, 0.5, 1.0, 60),
    "reb": (0.5, 16.0, 0.1, 0.3, 30),
    "ast": (0.5, 13.0, 0.1, 0.3, 25),
    "stl": (0.1, 3.0, 0.05, 0.08, 10),
    "blk": (0.1, 4.0, 0.05, 0.08, 12),
    "three_pm": (0.1, 6.0, 0.05, 0.12, 14),
}  # stat: (first line, last line, step, bandwidth, N_MAX)


def collect() -> dict[int, dict[str, np.ndarray]]:
    out = {}
    for season in SEASONS:
        for s in range(season - 4, season + 1):
            season_pool(s)
        rows = []
        for _, c in cases(season):
            new, _, _ = gamebox.apply(c.line, c.context, c.mates, c.gained, c.back)
            if new.get(gamebox.MINUTES, 0.0) < MIN_MINUTES:
                continue
            rows.append([new.get(k, 0.0) for k in STATS.values()] + [c.actual.get(k, 0.0) for k in STATS.values()])
        a = np.array(rows)
        out[season] = {name: a[:, [i, i + len(STATS)]] for i, name in enumerate(STATS)}
        print(season, len(a), flush=True)
    return out


def build(pairs: np.ndarray, stat: str) -> dict:
    lo, hi, step, bw0, n_max = GRID[stat]
    lines = np.round(np.arange(lo, hi + step / 2, step), 3)
    line, got = pairs[:, 0], pairs[:, 1]
    rows = []
    for g in lines:
        bw = bw0
        near = np.abs(line - g) < bw
        while near.sum() < MIN_GAMES:
            bw *= 1.5
            near = np.abs(line - g) < bw
        x = got[near]
        rows.append([float(np.mean(x >= n - 0.5)) for n in range(n_max + 1)])
    chance = np.maximum.accumulate(np.array(rows), axis=0)  # more expected -> never less likely
    return {"lines": lines.tolist(), "chance": np.round(chance, 4).tolist()}


def blind(data: dict[int, dict[str, np.ndarray]]) -> None:
    from app.analysis.rungs import chance_from

    for stat in STATS:
        print(f"\n{stat}: chance bucket, said vs happened per left-out season")
        for season in SEASONS:
            train = np.vstack([data[s][stat] for s in SEASONS if s != season])
            table = build(train, stat)
            test = data[season][stat]
            said, hit = [], []
            for line, got in test[:: max(1, len(test) // 4000)]:
                for n in range(1, GRID[stat][4] + 1):
                    p = chance_from(table, line, n)
                    if 0.02 <= p <= 0.98:
                        said.append(p)
                        hit.append(got >= n - 0.5)
            said, hit = np.array(said), np.array(hit)
            cells = []
            for a, b in ((0.02, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 0.98)):
                k = (said >= a) & (said < b)
                if k.any():
                    cells.append(f"{said[k].mean() * 100:.0f}->{hit[k].mean() * 100:.0f}")
            print(f"  {season}: " + "  ".join(cells))


def main() -> None:
    data = collect()
    table = {stat: build(np.vstack([data[s][stat] for s in SEASONS]), stat) for stat in STATS}
    Path(TABLE).write_text(json.dumps({"seasons": list(SEASONS), "min_minutes": MIN_MINUTES, "stats": table}))
    print("saved", TABLE)
    blind(data)


if __name__ == "__main__":
    main()

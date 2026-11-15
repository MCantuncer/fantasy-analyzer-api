"""Checks how much of a hot or cold run lasts, by what it was made of (analysis/form.py).

    uv run python -m scripts.hot_run_backtest

For every player and every second game of five seasons (2021-22 to 2025-26), his games in the 14
days up to that game are the run (3 or more), and the games before it plus last season (counted as
LAST_SEASON_GAMES games, when he played MIN_LAST_SEASON_GAMES) are his usual level, as form.hot_run
builds them from ESPN's splits. form.split_run splits the gap into minutes, production and shooting.
The value of his next 10 games over his usual level is then fitted on the three parts by least
squares (with an intercept). Each part's coefficient is the share of it that lasted.

Printed: the shares in each season and over all five, for every run and for runs at least HOT_GAP
above his usual, and how the verdicts' runs did.

Uses the caches of scripts/durability_backtest.py (.cache/season_pools, .cache/gamelogs).
"""

from __future__ import annotations

import numpy as np

from app.analysis.form import (
    FIRE_SHARE,
    HOT_GAP,
    KEEP,
    LAST_SEASON_GAMES,
    LINE,
    MIN_RUN_GAMES,
    MINUTES,
    ROLE_SHARE,
    split_run,
    value,
)
from scripts.minutes_backtest import season_lines
from scripts.next_games_backtest import dated_lines

SEASONS = (2022, 2023, 2024, 2025, 2026)
RUN_DAYS = 14
AFTER = 10
MIN_BEFORE = 8  # games before the run


def cases(season: int) -> np.ndarray:
    """Rows: minutes, production and shooting parts, then the next AFTER games' value over his usual."""
    last, lines = season_lines(season - 1), dated_lines(season)
    out = []
    for pid, games in lines.items():
        days = np.array([d for d, _ in games])
        rows = np.array([[g.get(k, 0.0) for k in LINE] for _, g in games])
        prior = np.array([last[pid].get(k, 0.0) for k in LINE]) if pid in last else None
        for end in range(MIN_BEFORE + MIN_RUN_GAMES, len(games) - AFTER + 1, 2):
            start = int(np.searchsorted(days, days[end - 1] - RUN_DAYS))
            if end - start < MIN_RUN_GAMES or start < MIN_BEFORE:
                continue
            before = rows[:start].sum(0)
            usual_arr = (
                (before + LAST_SEASON_GAMES * prior) / (start + LAST_SEASON_GAMES)
                if prior is not None
                else before / start
            )
            usual = dict(zip(LINE, usual_arr, strict=True))
            if usual[MINUTES] <= 0:
                continue
            recent = dict(zip(LINE, rows[start:end].mean(0), strict=True))
            nxt = dict(zip(LINE, rows[end : end + AFTER].mean(0), strict=True))
            out.append([*split_run(usual, recent), value(nxt) - value(usual)])
    return np.array(out)


def fit(c: np.ndarray) -> np.ndarray:
    x = np.column_stack([np.ones(len(c)), c[:, :3]])
    return np.linalg.lstsq(x, c[:, 3], rcond=None)[0][1:]


def main() -> None:
    by_season = {s: cases(s) for s in SEASONS}
    every = np.concatenate(list(by_season.values()))
    print("share that lasted over the next 10 games: minutes, production, shooting")
    for s, c in by_season.items():
        hot = c[c[:, :3].sum(1) >= HOT_GAP]
        print(f"  {s}: all runs {np.round(fit(c), 2)} ({len(c)})   hot runs {np.round(fit(hot), 2)} ({len(hot)})")
    hot = every[every[:, :3].sum(1) >= HOT_GAP]
    print(f"  five seasons: all runs {np.round(fit(every), 2)}   hot runs {np.round(fit(hot), 2)}   in use {KEEP}")
    gap = hot[:, :3].sum(1)
    share = hot[:, 0] / gap
    print("\nhot runs by verdict: count, gap, next 10 games over his usual (share of gap)")
    for name, pick in (
        ("bigger role", share >= ROLE_SHARE),
        ("mixed", (share > FIRE_SHARE) & (share < ROLE_SHARE)),
        ("straw fire", share <= FIRE_SHARE),
    ):
        g, after = gap[pick].mean(), hot[pick, 3].mean()
        print(f"  {name:12s} {pick.sum():6d}  {g:5.2f}  {after:5.2f} ({after / g:.0%})")


if __name__ == "__main__":
    main()

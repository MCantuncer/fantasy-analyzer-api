"""Checks how far this week's minutes should lean on a player's last 7 and 15 days.

    uv run python -m scripts.next_games_backtest

Before each of a player's games (from his 4th on), the projection as the app builds it
(parse._blend_stats, from ESPN's preseason projection, last season and this season so far) gives
his minutes. His average minutes over his games in the 7 and 15 days before that game are what
ESPN's "last 7 days" and "last 15 days" splits hold on game day. The weights on the gaps between
those and the projection (model.RECENT_MINUTES_WEIGHTS) are fitted by least squares on the fit
seasons and scored on the test season, for the next game and for the sum of the next four games
(about a week), in minutes and in points (counting stats follow the minutes, as in the model).

Printed: the weights fitted on each season alone; then, for every season, its errors with weights
fitted on the other four (never on itself), against the plain projection, each window alone, and
the weights in use. ESPN's stored projections for 2022-23 are rest-of-season ones, so that season's
projection is better than a real one was and leaves less for recent minutes to add.

Uses the caches of scripts/durability_backtest.py (.cache/season_pools, .cache/gamelogs).
"""

from __future__ import annotations

import json

import numpy as np

from app.analysis.model import RECENT_MINUTES_WEIGHTS
from app.espn.ages import season_age
from app.espn.parse import GAMES_PLAYED, MINUTES, PROVEN_GAMES, PROVEN_SEASONS, _blend_stats
from scripts.age_backtest import _entry, _names
from scripts.durability_backtest import LOGS, game_logs, season_pool
from scripts.minutes_backtest import season_lines
from scripts.projection_backtest import projections

SEASONS = (2022, 2023, 2024, 2025, 2026)
POINTS = 0
WINDOWS = (7, 15)  # days, as ESPN's splits
FIRST = 3  # games played before the first case
AHEAD = 4  # games summed for the week-sized check


def dated_lines(season: int) -> dict[int, list[tuple[int, dict[int, float]]]]:
    """Each player's (day, line) for the games he played, in order."""
    game_logs(season)  # fills the cache when it's missing
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
    return {pid: sorted(g.items()) for pid, g in days.items()}


def cases(season: int) -> np.ndarray:
    """Rows: projected minutes, 7-day and 15-day minutes (NaN: no game), then for the next game
    and the next AHEAD games: minutes, points, projected points."""
    proj, last, names, lines = projections(season), season_lines(season - 1), _names(season), dated_lines(season)
    past = [season_pool(season - k)[0] for k in range(1, PROVEN_SEASONS + 1)]
    out = []
    for pid, pr in proj.items():
        games = lines.get(pid, [])
        age = season_age(names.get(pid, ""), season)
        proven = all(p.get(pid, 0) >= PROVEN_GAMES for p in past)
        keys = sorted({k for _, g in games for k in g})
        total = dict.fromkeys(keys, 0.0)
        for i, (day, g) in enumerate(games):
            if FIRST <= i <= len(games) - AHEAD:
                so_far = {k: v / i for k, v in total.items()}
                so_far[GAMES_PLAYED] = i
                stats = [_entry(season, 1, pr)] + ([_entry(season - 1, 0, last[pid])] if pid in last else [])
                line = _blend_stats({"stats": [*stats, _entry(season, 0, so_far)]}, season, age, proven=proven)[0]
                minutes = line.get(MINUTES, 0.0)
                if minutes > 0:
                    recent = []
                    for w in WINDOWS:
                        inside = [x[MINUTES] for d, x in games[:i] if d >= day - w]
                        recent.append(float(np.mean(inside)) if inside else np.nan)
                    nxt = games[i : i + AHEAD]
                    out.append(
                        [minutes, *recent, g.get(MINUTES, 0.0), g.get(POINTS, 0.0), line.get(POINTS, 0.0)]
                        + [sum(x.get(MINUTES, 0.0) for _, x in nxt), sum(x.get(POINTS, 0.0) for _, x in nxt)]
                    )
            for k in keys:
                total[k] += g.get(k, 0.0)
    return np.array(out)


def gaps(c: np.ndarray) -> np.ndarray:
    return np.nan_to_num(c[:, 1:3] - c[:, [0]])


def fit(c: np.ndarray, use: tuple[int, ...] = (0, 1)) -> np.ndarray:
    """Weights on the 7-day and 15-day gaps (a window left out of ``use`` gets 0)."""
    w = np.zeros(2)
    w[list(use)] = np.linalg.lstsq(gaps(c)[:, list(use)], c[:, 3] - c[:, 0], rcond=None)[0]
    return w


def scores(c: np.ndarray, w: np.ndarray) -> tuple[float, float, float, float]:
    """Mean misses: next game minutes and points, next AHEAD games minutes and points."""
    minutes = np.clip(c[:, 0] + gaps(c) @ w, 0, 48)
    per_minute = c[:, 5] / c[:, 0]
    return (
        float(np.abs(c[:, 3] - minutes).mean()),
        float(np.abs(c[:, 4] - per_minute * minutes).mean()),
        float(np.abs(c[:, 6] - AHEAD * minutes).mean()),
        float(np.abs(c[:, 7] - AHEAD * per_minute * minutes).mean()),
    )


def main() -> None:
    by_season = {s: cases(s) for s in SEASONS}
    for s, c in by_season.items():
        print(f"{s}: {len(c)} games, weights fitted on it alone (7 days, 15 days): {np.round(fit(c), 3)}")
    every = np.concatenate(list(by_season.values()))
    print(f"fitted on all: {np.round(fit(every), 3)}  in use: {RECENT_MINUTES_WEIGHTS}")
    print(f"\nmean miss, each season with weights from the others: next game minutes, points | next {AHEAD} games")
    for s, c in by_season.items():
        others = np.concatenate([by_season[o] for o in SEASONS if o != s])
        print(f"{s}  weights {np.round(fit(others), 3)}")
        for name, ww in (
            ("projection alone", np.zeros(2)),
            ("last 7 days only", fit(others, (0,))),
            ("last 15 days only", fit(others, (1,))),
            ("both", fit(others)),
            ("in use", np.array(RECENT_MINUTES_WEIGHTS)),
        ):
            m, p, m4, p4 = scores(c, ww)
            print(f"  {name:18s} {m:5.2f} {p:5.2f} | {m4:5.2f} {p4:6.2f}")


if __name__ == "__main__":
    main()

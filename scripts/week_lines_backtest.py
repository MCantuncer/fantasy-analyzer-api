"""Replays the week's lines built game by game (gamebox.week_line) on five past seasons.

    uv run python -m scripts.week_lines_backtest

On the first day of each week (every 7th day of a season, 2021-22 to 2025-26) and for each of the
~500 most-owned players with 4+ games behind him, two calls for his games of that week are compared
with what he did in them:

  * today's week line: his projection (parse._blend_stats) at his last 7 and 15 days of minutes
    (CategoryModel.horizon_line), the same for every game,
  * the week line built game by game (CategoryModel.week_line, gamebox.week_line): that line through
    gamebox.apply for each game he played that week, with the league box built from the games before
    the week's first day (opponent, home court, his form, his minutes lately), averaged.

Neither side knows which teammates sit that week, and both count the games he went on to play: this
checks the lines, not who plays. Benchings by coach's decision are known only up to the week's first day.

Printed per category: per season the % less squared miss of the week total game by game against
today's week line, as it ships, and with both lines set to level first (a + b x, fitted on the other
seasons), so that only what moves players apart counts. Then the same for the week's value (the nine
categories, each over its spread across player-weeks, turnovers against) and how well each line
ranks players by it (rank correlation with what happened, mean over weeks).

Uses the caches of scripts/next_game_matchup_backtest.py.
"""

from __future__ import annotations

import bisect
import json

import numpy as np

from app.analysis import gamebox, gamescore
from app.analysis.model import RECENT_MINUTES_WEIGHTS
from app.espn.ages import season_age
from app.espn.parse import GAMES_PLAYED, MINUTES, PROVEN_GAMES, PROVEN_SEASONS, _blend_stats, with_minutes
from scripts.age_backtest import _entry, _names
from scripts.durability_backtest import LOGS, POOLS, game_logs, season_pool
from scripts.minutes_backtest import season_lines
from scripts.next_game_backtest import FIRST, season_games
from scripts.next_game_matchup_backtest import CATS, SEASONS, cats, sat_days, schedule
from scripts.projection_backtest import projections

WEEK = 7
VALUE = slice(1, None)  # the nine categories: everything but minutes
TURNOVERS = CATS.index("TO")


def season_weeks(season: int) -> tuple[list, list, list, list]:
    """Per player-week: (week's first day), today's week line x his games, the game-by-game line x his
    games, and what he did (cats(), summed over his games that week)."""
    game_logs(season)
    g = season_games(season)
    rows = []
    for team in range(1, 31):
        rows += gamebox.parse_rows(json.loads((LOGS / f"{season}-{team}.json").read_text()), season)
    prior = gamebox.prior_rates(json.loads((POOLS / f"{season - 1}.json").read_text()), season - 1)
    game_logs(season - 1)
    past = []
    for team in range(1, 31):
        past += gamebox.parse_rows(json.loads((LOGS / f"{season - 1}-{team}.json").read_text()), season - 1)
    last_season = gamebox.carry(past)
    score_start = gamescore.season_end(gamescore.team_games(past))
    sched = gamebox.parse_schedule(schedule(season))
    sat = sat_days(season, g)
    proj, last, names = projections(season), season_lines(season - 1), _names(season)
    pools = [season_pool(season - k)[0] for k in range(1, PROVEN_SEASONS + 1)]
    first = min(r.day for r in rows)
    days, now, new, act = [], [], [], []
    for d in range(first, max(r.day for r in rows) + 1, WEEK):
        known = {pid: [x for x in xs if x < d] for pid, xs in sat.items()}
        box = gamebox.build(season, [r for r in rows if r.day < d], prior, sched, last_season, score_start, known)
        for pid, pr in proj.items():
            gl = sorted(((day, team, x) for day, team, x in g.get(pid, {}).values()), key=lambda t: t[0])
            played = [day for day, *_ in gl]
            i, j = bisect.bisect_left(played, d), bisect.bisect_left(played, d + WEEK)
            if i < FIRST or j <= i:
                continue
            keys = sorted({k for *_, x in gl[:i] for k in x})
            so_far = {k: sum(x.get(k, 0.0) for *_, x in gl[:i]) / i for k in keys} | {GAMES_PLAYED: i}
            stats = [_entry(season, 1, pr)] + ([_entry(season - 1, 0, last[pid])] if pid in last else [])
            proven = all(p.get(pid, 0) >= PROVEN_GAMES for p in pools)
            age = season_age(names.get(pid, ""), season)
            line = _blend_stats({"stats": [*stats, _entry(season, 0, so_far)]}, season, age, proven=proven)[0]
            m = line.get(MINUTES, 0.0)
            if m <= 0:
                continue
            shift = 0.0
            for w, span in zip(RECENT_MINUTES_WEIGHTS, (7, 15), strict=True):
                inside = [x[MINUTES] for day, _, x in gl[:i] if day >= d - span]
                if inside:
                    shift += w * (float(np.mean(inside)) - m)
            line = with_minutes(line, min(max(m + shift, 0.0), 40.0))
            games = [gamebox.context(box, pid, team, day) for day, team, _ in gl[i:j]]
            if any(c is None or c.day != day for c, (day, *_) in zip(games, gl[i:j], strict=True)):
                continue
            n = j - i
            days.append(d)
            now.append([n * v for v in cats(line)])
            new.append([n * v for v in cats(gamebox.week_line(line, games))])
            act.append(np.sum([cats(x) for *_, x in gl[i:j]], axis=0).tolist())
    return days, now, new, act


def gains(now: dict, new: dict, act: dict, level: bool) -> list[float]:
    """% less squared miss of ``new`` than ``now`` per season; ``level``: both fitted a + b x on the others."""
    out = []
    for s in SEASONS:
        errs = []
        for x in (now, new):
            pred = x[s]
            if level:
                xs = np.concatenate([x[t] for t in SEASONS if t != s])
                ys = np.concatenate([act[t] for t in SEASONS if t != s])
                a, b = np.linalg.lstsq(np.c_[np.ones(len(xs)), xs], ys, rcond=None)[0]
                pred = a + b * pred
            errs.append(np.mean((act[s] - pred) ** 2))
        out.append((errs[0] - errs[1]) / errs[0] * 100)
    return out


def ranks(x: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(x)).astype(float)


def main() -> None:
    per = {}
    for season in SEASONS:
        days, now, new, act = season_weeks(season)
        per[season] = (np.array(days), np.array(now), np.array(new), np.array(act))
        print(f"{season}: {len(days)} player-weeks", flush=True)
    spread = np.concatenate([per[s][3] for s in SEASONS]).std(0)
    sign = np.ones(len(CATS))
    sign[TURNOVERS] = -1

    def value(x: np.ndarray) -> np.ndarray:
        return (x / spread * sign)[:, VALUE].sum(1)

    print("\n% less squared miss of the week total, game by game against today's week line, by season:")
    print("        as it ships                      | both set to level first")

    def show(name: str, pick) -> None:
        now, new, act = ({s: pick(per[s][i]) for s in SEASONS} for i in (1, 2, 3))
        raw, lev = gains(now, new, act, False), gains(now, new, act, True)
        mark = "  better in all five" if min(lev) > 0 else ""
        print(f"  {name:5s} {' '.join(f'{v:+5.1f}%' for v in raw)} | {' '.join(f'{v:+5.1f}%' for v in lev)}{mark}")

    for c, name in enumerate(CATS):
        show(name, lambda x, c=c: x[:, c])
    show("VALUE", value)
    print("\nranking players by the week's value (rank correlation with what happened, mean over weeks):")
    for i, name in ((1, "today's week line"), (2, "game by game")):
        row = []
        for s in SEASONS:
            days, *_, act = per[s]
            pred = per[s][i]
            cors = [
                np.corrcoef(ranks(value(pred[days == d])), ranks(value(act[days == d])))[0, 1] for d in np.unique(days)
            ]
            row.append(float(np.mean(cors)))
        print(f"  {name:18s} " + " ".join(f"{v:.4f}" for v in row))


if __name__ == "__main__":
    main()

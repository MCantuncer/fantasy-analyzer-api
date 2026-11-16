"""Replays the next-game line with the game in it (analysis/gamebox.py) on five past seasons.

    uv run python -m scripts.next_game_matchup_backtest

Before each of a player's games (from his 4th on, the ~500 most-owned players, 2021-22 to 2025-26),
three lines are compared with what he did:

  * his season average so far,
  * today's next-game line (report.next_game_line without a game context): his projection
    (parse._blend_stats) at his last 7 and 15 days of minutes, plus the minutes teammates ruled out
    leave him (tonight.py), at NEXT_GAME_LEVEL,
  * the same line through gamebox.apply, with the league box built by gamebox.build from every
    game played before that day (all 30 teams' game logs), last season's per-minute lines as the
    start, and the NBA schedule (opponent, home, back-to-back).

This runs the app's own code day by day, so it checks what ships: the weights in gamebox.py were
fitted on these seasons (signals kept only when each lowered the squared miss in all five seasons,
fitted with that season left out and fitted on earlier seasons only), and here should beat today's
line in every season, for every category but FT%.

Printed per category: the mean miss of each line, and per season the % less squared miss of the
new line against today's line.

Uses the caches of scripts/durability_backtest.py (.cache/season_pools, .cache/gamelogs) and keeps
the NBA schedules in .cache/schedules.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import httpx
import numpy as np

from app.analysis import gamebox
from app.analysis.model import RECENT_MINUTES_WEIGHTS
from app.analysis.report import NEXT_GAME_LEVEL
from app.analysis.tonight import OTHER_ROLE, SAME_ROLE
from app.espn.ages import season_age
from app.espn.parse import GAMES_PLAYED, MINUTES, PROVEN_GAMES, PROVEN_SEASONS, _blend_stats, with_minutes
from scripts.age_backtest import _entry, _names
from scripts.durability_backtest import LOGS, POOLS, game_logs, season_pool
from scripts.minutes_backtest import season_lines
from scripts.next_game_backtest import FIRST, out_for, regulars_out, season_games, season_roles
from scripts.projection_backtest import projections

SEASONS = (2022, 2023, 2024, 2025, 2026)
SCHEDULES = Path(".cache/schedules")
URL = "https://lm-api-reads.fantasy.espn.com/apis/v3/games/fba/seasons/{season}"
PTS, BLK, STL, AST, REB, TO, FGM, FGA, FTM, FTA, TPM = 0, 1, 2, 3, 6, 11, 13, 14, 15, 16, 17
STATS = (MINUTES, PTS, REB, AST, STL, BLK, TPM, TO, FGM, FGA, FTM, FTA)
CATS = ("MIN", "PTS", "REB", "AST", "STL", "BLK", "3PM", "TO", "FG", "FT")
FG0, FT0 = 0.4756, 0.7973


def schedule(season: int) -> dict:
    path = SCHEDULES / f"{season}.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        resp = httpx.get(URL.format(season=season), params={"view": "proTeamSchedules_wl"}, timeout=90)
        resp.raise_for_status()
        path.write_text(resp.text)
    return json.loads(path.read_text())


def cats(line: dict[int, float]) -> list[float]:
    """Minutes, the seven counting categories, then FG and FT as makes over the league's rate."""
    return [line.get(s, 0.0) for s in (MINUTES, PTS, REB, AST, STL, BLK, TPM, TO)] + [
        line.get(FGM, 0.0) - FG0 * line.get(FGA, 0.0),
        line.get(FTM, 0.0) - FT0 * line.get(FTA, 0.0),
    ]


def season_cases(season: int) -> tuple[list, list, list, list]:
    """Per case: today's line, the new line, his season average so far and what he did (cats())."""
    game_logs(season)
    g = season_games(season)
    rows = []
    for team in range(1, 31):
        rows += gamebox.parse_rows(json.loads((LOGS / f"{season}-{team}.json").read_text()), season)
    prior = gamebox.prior_rates(json.loads((POOLS / f"{season - 1}.json").read_text()), season - 1)
    sched = gamebox.parse_schedule(schedule(season))
    sides: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for pid, gl in g.items():
        for gid, (_d, team, line) in gl.items():
            sides[gid][team].append((pid, line))
    out = regulars_out(g, sides)
    roles = season_roles(g, season)
    proj, last, names = projections(season), season_lines(season - 1), _names(season)
    past = [season_pool(season - k)[0] for k in range(1, PROVEN_SEASONS + 1)]
    todo: dict[int, list] = defaultdict(list)
    for pid, pr in proj.items():
        gl = sorted(((d, gid, team, x) for gid, (d, team, x) in g.get(pid, {}).items()), key=lambda t: t[0])
        age = season_age(names.get(pid, ""), season)
        proven = all(p.get(pid, 0) >= PROVEN_GAMES for p in past)
        keys = sorted({k for *_, x in gl for k in x})
        total = dict.fromkeys(keys, 0.0)
        for i, (day, gid, team, x) in enumerate(gl):
            if i >= FIRST:
                so_far = {k: v / i for k, v in total.items()} | {GAMES_PLAYED: i}
                stats = [_entry(season, 1, pr)] + ([_entry(season - 1, 0, last[pid])] if pid in last else [])
                line = _blend_stats({"stats": [*stats, _entry(season, 0, so_far)]}, season, age, proven=proven)[0]
                m = line.get(MINUTES, 0.0)
                if m > 0:
                    shift = 0.0
                    for w, days in zip(RECENT_MINUTES_WEIGHTS, (7, 15), strict=True):
                        inside = [y[MINUTES] for d, *_, y in gl[:i] if d >= day - days]
                        if inside:
                            shift += w * (float(np.mean(inside)) - m)
                    line = with_minutes(line, min(max(m + shift, 0.0), 40.0))
                    todo[day].append((pid, gid, team, line, x, so_far))
            for k in keys:
                total[k] += x.get(k, 0.0)
    now, new, avg, act = [], [], [], []
    for day in sorted(todo):
        box = gamebox.build(season, [r for r in rows if r.day < day], prior, sched)
        for pid, gid, team, line, x, so_far in todo[day]:
            opp = next((t for t in sides[gid] if t != team), None)
            ctx = gamebox.context(box, pid, team, day)
            if opp is None or ctx is None or ctx.day != day:
                continue
            same, other, _ = out_for(out, roles, pid, gid, team, opp)
            minutes = line.get(MINUTES, 0.0)
            gained = min(SAME_ROLE * same + OTHER_ROLE * other, 40.0 - minutes) if minutes < 40.0 else 0.0
            today = with_minutes(line, minutes + gained) if gained else line
            today = {k: v * NEXT_GAME_LEVEL.get(k, 1.0) for k, v in today.items()}
            mates = []
            for q, q_min in out.get((gid, team), []):
                f = box.form.get(q)
                usage = f.usage if f is not None else 0.0
                mates.append(gamebox.Teammate("", f.minutes_long if f is not None else q_min, usage))
            line_new, _, _ = gamebox.apply(today, ctx, mates, gained)
            now.append(cats(today))
            new.append(cats(line_new))
            avg.append(cats(so_far))
            act.append(cats(x))
    return now, new, avg, act


def main() -> None:
    per: dict[int, tuple[np.ndarray, ...]] = {}
    for season in SEASONS:
        per[season] = tuple(np.array(a) for a in season_cases(season))
        print(f"{season}: {len(per[season][0])} games", flush=True)
    print("\nmean miss: season average | today's line | with the game; then % less squared miss than today, by season")
    for c, name in enumerate(CATS):
        miss = {k: [] for k in ("avg", "now", "new")}
        gains = []
        for season in SEASONS:
            now, new, avg, act = per[season]
            for k, v in (("avg", avg), ("now", now), ("new", new)):
                miss[k].append(np.abs(act[:, c] - v[:, c]))
            e0, e1 = np.mean((act[:, c] - now[:, c]) ** 2), np.mean((act[:, c] - new[:, c]) ** 2)
            gains.append((e0 - e1) / e0 * 100)
        m = {k: float(np.mean(np.concatenate(v))) for k, v in miss.items()}
        mark = "  better in all five" if min(gains) > 0 else ""
        print(
            f"  {name:4s} {m['avg']:.3f} | {m['now']:.3f} | {m['new']:.3f}   "
            + " ".join(f"{g:+.1f}%" for g in gains)
            + mark
        )


if __name__ == "__main__":
    main()

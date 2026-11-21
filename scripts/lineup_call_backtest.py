"""Replays Our call with who plays tonight (analysis/lineupcall.py) on five seasons and this one.

    uv run python -m scripts.lineup_call_backtest [2027]

For every complete game, the call as shipped before (gamescore parts only) and with the two lineup
pieces, from the games before that day. Who is out tonight stands in for ESPN's OUT tags: players in
the team's rotation (lineupcall.Lineups.rotation) who didn't play. Printed per season: squared miss
of a team's points, the margin and the total, the winner picked; then the two weights fitted blind
(season left out, and earlier seasons only) against the shipped ones.

Uses the caches of scripts/durability_backtest.py and the schedules of next_game_matchup_backtest.py.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace

import numpy as np

from app.analysis import gamebox, gamescore, lineupcall
from scripts.durability_backtest import LOGS, POOLS, game_logs, season_pool
from scripts.next_game_matchup_backtest import schedule

SEASONS = (2022, 2023, 2024, 2025, 2026)


def season_rows(season: int) -> list[gamebox.GameRow]:
    game_logs(season)
    rows: list[gamebox.GameRow] = []
    for team in range(1, 31):
        rows += gamebox.parse_rows(json.loads((LOGS / f"{season}-{team}.json").read_text()), season)
    return rows


def season_cases(season: int) -> dict[str, np.ndarray]:
    season_pool(season - 1)
    rows = season_rows(season)
    prior = gamebox.prior_rates(json.loads((POOLS / f"{season - 1}.json").read_text()), season - 1)
    start = gamescore.season_end(gamescore.team_games(season_rows(season - 1)))
    sched = gamebox.parse_schedule(schedule(season))
    box = gamebox.build(season, rows, prior, sched, None, start)
    plain = replace(box, lineups=None, _day_scores=box._day_scores)
    lu = box.lineups
    assert lu is not None
    by_game: dict[int, list[gamescore.TeamGame]] = {}
    for g in box.team_games:
        by_game.setdefault(g.game, []).append(g)
    out: dict[str, list[float]] = {k: [] for k in ("old", "new", "talent", "absent", "pts", "game")}
    for gid, sides in by_game.items():
        if len(sides) != 2:
            continue
        day = sides[0].day
        for g in sides:
            game = next((x for x in sched.get(g.team, []) if x.day == day), None)
            if game is None:
                continue
            outs: set[int] = set()
            for t in (g.team, g.opponent):
                went = set(lu.played.get((gid, t), (0, []))[1])
                outs |= {p for p in lu.rotation(t, day) if p not in went}
            away, home = (g.opponent, g.team) if game.home else (g.team, g.opponent)
            pa, ph = box.call_parts(day, away, home, outs)
            oa, oh = plain.call_parts(day, away, home)
            mine, old = (ph, oh) if game.home else (pa, oa)
            out["old"].append(sum(old.values()))
            out["new"].append(sum(mine.values()))
            out["talent"].append(mine["lineup"] / lineupcall.TALENT)
            out["absent"].append(mine["absences"] / lineupcall.ABSENT)
            out["pts"].append(g.stats["pf"])
            out["game"].append(gid)
    return {k: np.array(v) for k, v in out.items()}


def report(per: dict[int, dict[str, np.ndarray]]) -> None:
    print("squared miss old -> new (% less): team points | margin | total; winner picked old -> new")
    for s, d in per.items():
        order = np.argsort(d["game"], kind="stable")
        a, b = order[0::2], order[1::2]
        ok = d["game"][a] == d["game"][b]
        a, b = a[ok], b[ok]
        pts = d["pts"]

        def pct(x0: np.ndarray, x1: np.ndarray) -> str:
            e0, e1 = np.mean(x0**2), np.mean(x1**2)
            return f"{e0:.1f} -> {e1:.1f} ({(e0 - e1) / e0 * 100:+.2f}%)"

        team = pct(pts - d["old"], pts - d["new"])
        mar = pct((pts[a] - pts[b]) - (d["old"][a] - d["old"][b]), (pts[a] - pts[b]) - (d["new"][a] - d["new"][b]))
        tot = pct((pts[a] + pts[b]) - (d["old"][a] + d["old"][b]), (pts[a] + pts[b]) - (d["new"][a] + d["new"][b]))
        w0 = np.mean(np.sign(pts[a] - pts[b]) == np.sign(d["old"][a] - d["old"][b])) * 100
        w1 = np.mean(np.sign(pts[a] - pts[b]) == np.sign(d["new"][a] - d["new"][b])) * 100
        print(f"  {s - 1}-{s % 100:02d}: {team} | {mar} | {tot}; winner {w0:.1f}% -> {w1:.1f}%")
    past = [s for s in per if s in SEASONS]
    x = {s: np.column_stack([per[s]["talent"], per[s]["absent"]]) for s in per}
    y = {s: per[s]["pts"] - per[s]["old"] for s in per}
    print(f"weights shipped: talent {lineupcall.TALENT}, absent {lineupcall.ABSENT}; refit blind:")
    for s in per:
        tr = [t for t in past if t != s]
        if not tr:
            continue
        w = np.linalg.lstsq(np.vstack([x[t] for t in tr]), np.concatenate([y[t] for t in tr]), rcond=None)[0]
        fw = [t for t in past if t < s]
        f = (
            np.linalg.lstsq(np.vstack([x[t] for t in fw]), np.concatenate([y[t] for t in fw]), rcond=None)[0]
            if fw
            else None
        )
        e0 = np.mean(y[s] ** 2)
        blind = [f"left out {np.round(w, 3)} {(e0 - np.mean((y[s] - x[s] @ w) ** 2)) / e0 * 100:+.2f}%"]
        if f is not None:
            blind.append(f"earlier only {np.round(f, 3)} {(e0 - np.mean((y[s] - x[s] @ f) ** 2)) / e0 * 100:+.2f}%")
        print(f"  {s}: team points, " + "; ".join(blind))
    xa, ya = np.vstack([x[t] for t in past]), np.concatenate([y[t] for t in past])
    print(f"  all five seasons: {np.round(np.linalg.lstsq(xa, ya, rcond=None)[0], 4)}")


def main() -> None:
    seasons = [int(a) for a in sys.argv[1:]] or [*SEASONS, 2027]
    per = {}
    for s in seasons:
        per[s] = season_cases(s)
        print(f"{s}: {len(per[s]['pts'])} team games", flush=True)
    report(per)


if __name__ == "__main__":
    main()

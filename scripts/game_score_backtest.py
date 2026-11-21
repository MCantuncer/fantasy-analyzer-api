"""Replays a game's expected score (analysis/gamescore.py) and its check on next-game lines on five seasons.

    uv run python -m scripts.game_score_backtest            # both parts
    uv run python -m scripts.game_score_backtest --scores   # part 1 only (a minute)

Can's rule (2026-11-15): a weight is kept only if it lowers the squared miss in EVERY season, tested
blind, both ways: fitted with that season left out, and fitted on earlier seasons only.

Part 1, the score. Before each game of 2021-22 to 2025-26, every team's score from the games before
that day (gamescore.build, last season's end as the start). The weights on top of the ratings (level,
home, the two three-point luck terms, back-to-back) are fitted again blind; printed per season: the
squared miss with the shipped weights and the refit, against a team's own average so far and the
league's, the winner picked, and how big the calls' moves from the team average run against what
happened (1.0: right size; below 1: overfit).

Part 2, the check on player lines. scripts/next_game_matchup_backtest.py's cases (the app's code day
by day), each team's lines summed per game (report.team_sum_gap): the share of the gap each stat
moves by (report.TEAM_SUM_SHARE) fitted again blind, and the % less squared miss against the line
without the check, per category and season.

Uses the caches of scripts/durability_backtest.py and the schedules of next_game_matchup_backtest.py.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from app.analysis import gamebox, gamescore
from app.analysis.gamebox import MINUTES, PTS
from app.analysis.report import TEAM_SUM_SHARE, team_adjust, team_sum_gap
from scripts.durability_backtest import LOGS, game_logs
from scripts.next_game_matchup_backtest import CATS, SEASONS, cases, cats, schedule

WEIGHTS = ("level", "home", "opp_three_luck", "own_three_luck", "back_to_back")
SHIPPED = np.array(
    [gamescore.LEVEL, gamescore.HOME, gamescore.OPP_THREE_LUCK, gamescore.OWN_THREE_LUCK, gamescore.BACK_TO_BACK]
)


def season_rows(season: int) -> list[gamebox.GameRow]:
    game_logs(season)
    rows: list[gamebox.GameRow] = []
    for team in range(1, 31):
        rows += gamebox.parse_rows(json.loads((LOGS / f"{season}-{team}.json").read_text()), season)
    return rows


def score_cases(season: int) -> dict[str, list[float]]:
    """Every complete game of a season from both sides: the signals before tip-off and the score."""
    games = gamescore.team_games(season_rows(season))
    start = gamescore.season_end(gamescore.team_games(season_rows(season - 1)))
    sched = gamebox.parse_schedule(schedule(season))
    box = gamebox.Box(season, {}, {}, {}, schedule=sched)
    out: dict[str, list[float]] = defaultdict(list)
    by_day: dict[int, list[gamescore.TeamGame]] = defaultdict(list)
    for g in games:
        by_day[g.day].append(g)
    before: list[gamescore.TeamGame] = []
    for day in sorted(by_day):
        s = gamescore.build(before, start)
        for g in by_day[day]:
            game = next((x for x in sched.get(g.team, []) if x.day == day), None)
            x = gamescore.signals(s, g.team, g.opponent)
            out["ratings"].append(x["ratings"])
            out["home"].append(0.0 if game is None else (1.0 if game.home else -1.0))
            out["opp_three_luck"].append(x["opp_three_luck"])
            out["own_three_luck"].append(x["own_three_luck"])
            out["back_to_back"].append(1.0 if box.back_to_back(g.team, day) else 0.0)
            out["average"].append(x["average"])
            out["league"].append(s.rate("pf"))
            out["pts"].append(g.stats["pf"])
            out["opp_pts"].append(g.stats["pa"])
            out["game"].append(g.game)
        before += by_day[day]
    return out


def part_scores() -> list[str]:
    per = {s: {k: np.array(v) for k, v in score_cases(s).items()} for s in SEASONS}
    d = {k: np.concatenate([per[s][k] for s in SEASONS]) for k in per[SEASONS[0]]}
    season = np.concatenate([np.full(len(per[s]["pts"]), s) for s in SEASONS])
    x = np.column_stack([np.ones(len(season)), *(d[k] for k in WEIGHTS[1:])])
    y = d["pts"] - d["ratings"]
    lines = [
        "## 1. The score",
        "",
        f"{len(season):,} team games. Mean miss (points) per season: league average | team average | shipped |"
        " refit left out | refit forward; winner picked; move size (1.0 right).",
        "",
    ]
    ok = []
    refit = {}
    for i, s in enumerate(SEASONS):
        te = season == s
        ship = d["ratings"][te] + x[te] @ SHIPPED
        w_l = np.linalg.lstsq(x[season != s], y[season != s], rcond=None)[0]
        loso = d["ratings"][te] + x[te] @ w_l
        fwd = None
        if i:
            w_f = np.linalg.lstsq(x[season < s], y[season < s], rcond=None)[0]
            fwd = d["ratings"][te] + x[te] @ w_f
        act = d["pts"][te]
        lvl = np.mean((d["pts"] - d["league"])[season != s])
        avg_lvl = np.mean((d["pts"] - d["average"])[season != s])
        miss = [np.abs(act - (d["league"][te] + lvl)), np.abs(act - (d["average"][te] + avg_lvl)), np.abs(act - ship)]
        miss += [np.abs(act - loso)] + ([np.abs(act - fwd)] if fwd is not None else [])
        refit[s] = w_l
        # the pick: the two sides of a game, in order
        games = d["game"][te]
        order = np.argsort(games, kind="stable")
        a, b = order[0::2], order[1::2]
        same = games[a] == games[b]
        a, b = a[same], b[same]
        win = np.mean(np.sign(act[a] - act[b]) == np.sign(ship[a] - ship[b])) * 100
        dx, dy = ship - (d["average"][te] + avg_lvl), act - (d["average"][te] + avg_lvl)
        size = float(np.sum(dx * dy) / np.sum(dx * dx))
        beats = np.mean((act - ship) ** 2) < np.mean((act - (d["average"][te] + avg_lvl)) ** 2)
        ok.append(beats)
        lines.append(
            f"- {s - 1}-{s % 100:02d}: "
            + " | ".join(f"{float(np.mean(m)):.2f}" for m in miss)
            + f"; winner {win:.1f}%; move size {size:.2f}"
        )
    w_all = np.linalg.lstsq(x, y, rcond=None)[0]
    # Each weight earns its place: the squared miss without it, refit blind, must be higher every season.
    drops = []
    for j, name in enumerate(WEIGHTS[1:], start=1):
        keep = [k for k in range(x.shape[1]) if k != j]
        worse = []
        for i, s in enumerate(SEASONS):
            te = season == s
            for tr in (season != s, season < s) if i else (season != s,):
                full = np.linalg.lstsq(x[tr], y[tr], rcond=None)[0]
                less = np.linalg.lstsq(x[tr][:, keep], y[tr], rcond=None)[0]
                e_full = np.mean((y[te] - x[te] @ full) ** 2)
                e_less = np.mean((y[te] - x[te][:, keep] @ less) ** 2)
                worse.append((e_less - e_full) / e_less * 100)
        drops.append(f"{name} {min(worse):+.2f}..{max(worse):+.2f}%{' every season' if min(worse) > 0 else ''}")
    lines += ["", "Each weight's % less squared miss (left out and forward): " + "; ".join(drops)]
    lines += [
        "",
        "Weights (" + ", ".join(WEIGHTS) + "): shipped " + ", ".join(f"{v:+.3f}" for v in SHIPPED),
        "refit on all " + ", ".join(f"{v:+.3f}" for v in w_all),
        "refit per season left out: "
        + "; ".join(f"{s}: " + ", ".join(f"{v:+.2f}" for v in w) for s, w in refit.items()),
        f"Beats the team's own average in every season: {'yes' if all(ok) else 'no'}",
        "",
    ]
    return lines


def part_lines() -> list[str]:
    cols: dict[str, list] = defaultdict(list)
    for season in SEASONS:
        day_cases: dict[tuple[int, int], list] = defaultdict(list)
        for _box, c in cases(season):
            new, _, _ = gamebox.apply(c.line, c.context, c.mates, c.gained, c.back, c.absent)
            day_cases[(c.day, c.team)].append((c, new))
        for group in day_cases.values():
            lines_pts = sum(new.get(PTS, 0.0) for _, new in group)
            season_pts = sum(c.so_far.get(PTS, 0.0) for c, _ in group)
            for c, new in group:
                m = c.context
                gap = 0.0 if m.team_score is None else team_sum_gap(m.team_score, lines_pts, season_pts, m.team_points)
                cols["season"].append(season)
                cols["gap"].append(gap)
                cols["new"].append(cats(new))
                cols["shipped"].append(cats(team_adjust(new, gap)))
                cols["act"].append(cats(c.actual))
        print(f"{season}: {len(cols['season'])} games so far", flush=True)
    season = np.array(cols["season"])
    gap = np.array(cols["gap"])
    new, shipped, act = (np.array(cols[k]) for k in ("new", "shipped", "act"))
    lines = [
        "## 2. Lines held to the team's score",
        "",
        f"{len(season):,} player games. Per category: shipped share | refit share; % less squared miss than the"
        " line without the check, per season: shipped | left out | forward.",
        "",
    ]
    # Each category's value moves by share x gap x its value (FG and FT impacts with the shot counts).
    for c, cat in enumerate(CATS):
        z = new[:, c] * gap
        ship_g, loso_g, fwd_g = [], [], []
        for s in SEASONS:
            te = season == s
            e0 = np.mean((act[te, c] - new[te, c]) ** 2)
            ship_g.append((e0 - np.mean((act[te, c] - shipped[te, c]) ** 2)) / e0 * 100)
            for tr, out in ((season != s, loso_g), (season < s, fwd_g)):
                if not tr.any():
                    continue
                b = np.sum(z[tr] * (act[tr, c] - new[tr, c])) / max(np.sum(z[tr] ** 2), 1e-9)
                out.append((e0 - np.mean((act[te, c] - new[te, c] - b * z[te]) ** 2)) / e0 * 100)
        b_all = np.sum(z * (act[:, c] - new[:, c])) / max(np.sum(z**2), 1e-9)
        key = {"MIN": MINUTES, "PTS": 0, "REB": 6, "AST": 3, "STL": 2, "FG": 13}.get(cat)
        share = TEAM_SUM_SHARE.get(key, 0.0) if key is not None else 0.0
        every = min(loso_g + fwd_g) > 0
        lines.append(
            f"- {cat}: {share:.3f} | {b_all:+.3f}; "
            + " ".join(f"{g:+.2f}%" for g in ship_g)
            + " | "
            + " ".join(f"{g:+.2f}%" for g in loso_g)
            + " | "
            + " ".join(f"{g:+.2f}%" for g in fwd_g)
            + ("  every season" if every else "")
        )
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", action="store_true", help="part 1 only")
    args = ap.parse_args()
    lines = ["# Game score check", ""] + part_scores()
    print("\n".join(lines), flush=True)
    if not args.scores:
        print("\n".join(part_lines()))


if __name__ == "__main__":
    main()

"""Blind test of fouls in the live estimates (app/liveline.py, app/quarters.py) on five seasons of NBA
play-by-play (cdn.nba.com, github.com/shufinskiy/nba_data, one "cdnnba_<year>.csv" a season):

    uv run python -m scripts.foul_backtest <folder>

Players (the live line's checkpoints, every 6 minutes): his personal and offensive fouls so far. Foul
trouble is quarter + 1 fouls under 5 (2 in the 1st quarter, 3 in the 2nd, 4 in the 3rd; at a quarter's
end, the next quarter's allowance), then 5 fouls, then fouled out (6, no minutes left). On top of the
shipped live line (its weights as they are), FOUL_TROUBLE and FIVE_FOULS scale his minutes left; they are
fitted with each season left out and on earlier seasons only, and each must lower the squared miss of the
rest of his points in every season (and dropping it must hurt every season).

Quarters (3, 6 and 9 minutes into each quarter): the rest of the quarter in progress against the live
quarters number (pre-game quarter, fitted with the season left out as in quarter_backtest, moved by
GIVE_BACK). Team fouls are personal fouls (not offensive, not technical); a side is in the bonus when
the opponent has 4+ this quarter, or 1+ in its last two minutes. Pieces: BONUS (a minute left while in
the bonus) and IN_QUARTER (a minute, x share played x share left), least squares, same rules.

Also printed: whether each team's own bonus scoring, and each team's minutes for players in foul
trouble, hold from one season to the next (correlation), and a blind try of last season's team effect.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from app import liveline, quarters
from scripts import live_line_backtest as llb
from scripts import quarter_backtest as qb

SEASONS = (2021, 2022, 2023, 2024, 2025)
IN_QUARTER_CHECKS = (180, 360, 540)  # seconds into a quarter


def read_games(path: Path):
    """Per game: players {pid: (team, made [(t, pts)], stints [(on, off)], fouls [t])} in regulation
    seconds, and each team's team fouls [(period, seconds into it)]."""
    with path.open(newline="") as fh:
        game, players, on, seen, period, tf = None, {}, {}, set(), 0, defaultdict(list)
        for r in csv.DictReader(fh):
            if r["gameId"] != game or int(r["period"] or 0) != period:
                for pid, start in on.items():
                    players[pid][2].append((start, period * 720.0))
                on, seen = {}, set()
                if r["gameId"] != game:
                    if game is not None:
                        yield int(game), players, tf
                    game, players, tf = r["gameId"], {}, defaultdict(list)
                period = int(r["period"] or 0)
            if period > 4 or not r["teamTricode"] or not r["personId"] or r["personId"] == "0":
                continue
            pid = int(float(r["personId"]))
            s = 720.0 - llb.clock(r["clock"])
            t = (period - 1) * 720.0 + s
            me = players.setdefault(pid, (r["teamTricode"], [], [], []))
            kind, sub = r["actionType"], r["subType"]
            if pid not in seen:
                seen.add(pid)
                if not (kind == "substitution" and sub == "in"):
                    on[pid] = (period - 1) * 720.0
            if kind == "substitution":
                if sub == "in":
                    on[pid] = t
                elif sub == "out" and pid in on:
                    me[2].append((on.pop(pid), t))
            elif r["shotResult"] == "Made" and kind in llb.POINTS:
                me[1].append((t, llb.POINTS[kind]))
            elif kind == "foul" and sub in ("personal", "offensive"):
                me[3].append(t)
                if sub == "personal":
                    tf[r["teamTricode"]].append((period, s))
        if game is not None:
            for pid, start in on.items():
                players[pid][2].append((start, period * 720.0))
            yield int(game), players, tf


def collect(folder: Path) -> tuple[np.ndarray, np.ndarray]:
    """Player checkpoints (live line columns + fouls, rest minutes, team, player) and team-quarter
    checkpoints (season, game, team, opponent, period, seconds in, rest, opponent fouls so far)."""
    decay = 0.5 ** (1 / llb.HALF_LIFE)
    usual: dict[int, list[float]] = {}
    prow, trow = [], []
    for year in (SEASONS[0] - 1, *SEASONS):
        for gid, players, tf in sorted(read_games(folder / f"cdnnba_{year}.csv"), key=lambda x: x[0]):
            pts = defaultdict(list)
            for team, ev, _, _ in players.values():
                pts[team] += ev
            if len(pts) != 2:
                continue
            teams = list(pts)
            for team in teams if year in SEASONS else ():
                other = teams[1] if teams[0] == team else teams[0]
                for per in range(1, 5):
                    for s in IN_QUARTER_CHECKS:
                        t0, t = (per - 1) * 720.0, (per - 1) * 720.0 + s
                        rest = sum(p for x, p in pts[team] if t < x <= t0 + 720)
                        opp = sorted(x for p, x in tf[other] if p == per and x <= s)
                        bonus = len(opp) >= 4 or (s >= 600 and any(x >= 600 for x in opp))
                        trow.append((year, gid, qb.ABBREV[team], qb.ABBREV[other], per, s, rest, bonus))
            for pid, (team, ev, stints, fl) in players.items():
                total = sum(b - a for a, b in stints) / 60
                if total <= 0:
                    continue
                u = usual.get(pid)
                if year in SEASONS and u and u[3] >= llb.MIN_GAMES:
                    other = next(k for k in pts if k != team)
                    for t in llb.CHECKS:
                        m = sum(max(0.0, min(b, t) - a) for a, b in stints) / 60
                        so = sum(p for x, p in ev if x <= t)
                        rest = sum(p for x, p in ev if x > t)
                        mar = sum(p for x, p in pts[team] if x <= t) - sum(p for x, p in pts[other] if x <= t)
                        f = sum(1 for x in fl if x <= t)
                        prow.append(
                            (year, u[0] / u[1], u[1] / u[2], t / 60, m, so, rest, mar, f, total - m, qb.ABBREV[team])
                        )
                p = sum(x for _, x in ev)
                u = u or [0.0, 0.0, 0.0, 0]
                usual[pid] = [u[0] * decay + p, u[1] * decay + total, u[2] * decay + 1, u[3] + 1]
        print(year, len(prow), len(trow), flush=True)
    return np.array(prow), np.array(trow)


def players(x: np.ndarray) -> None:
    year, el, rest, f, restmin, team = x[:, 0], x[:, 3], x[:, 6], x[:, 8], x[:, 9], x[:, 10]
    rate, mins = parts(x)
    quarter = np.where(el % 12 == 0, el / 12 + 1, np.ceil(el / 12))
    trouble = (f >= quarter + 1) & (f < 5)
    five, out = f == 5, f >= 6

    def pred(i, a, b):
        return np.where(out[i], 0, rate[i] * mins[i] * (1 + a * trouble[i] + b * five[i]))

    def fit(i, keep=(True, True)):
        grid = [
            (((rest[i] - pred(i, a, b)) ** 2).mean(), a, b)
            for a in (np.arange(-0.6, 0.31, 0.05) if keep[0] else [0.0])
            for b in (np.arange(-0.9, 0.31, 0.05) if keep[1] else [0.0])
        ]
        return min(grid)[1:]

    def mse(i, a, b):
        return float(((rest[i] - pred(i, a, b)) ** 2).mean())

    print("\nplayers: squared miss of the rest of his points; fouled out alone, then trouble + 5 fouls on top")
    print(f"  minutes left, real vs ours: in trouble {restmin[trouble].mean():.2f} vs {mins[trouble].mean():.2f};")
    print(f"  at 5 fouls {restmin[five].mean():.2f} vs {mins[five].mean():.2f}; fouled out 0 vs {mins[out].mean():.2f}")
    every = np.arange(len(x))
    for s in SEASONS:
        te = every[year == s]
        plain = float(((rest[te] - rate[te] * mins[te]) ** 2).mean())
        lo = fit(every[year != s])
        fw = fit(every[year < s]) if s > SEASONS[0] else lo
        print(
            f"  {s}: {plain:.3f} out {mse(te, 0, 0) - plain:+.4f} fouls {mse(te, *lo) - mse(te, 0, 0):+.4f} "
            f"(left out) {mse(te, *fw) - mse(te, 0, 0):+.4f} (earlier only)"
        )
    for name, keep in (("foul trouble", (False, True)), ("5 fouls", (True, False))):
        hurt = []
        for s in SEASONS:
            te, tr = every[year == s], every[year != s]
            hurt.append(round(mse(te, *fit(tr, keep)) - mse(te, *fit(tr)), 4))
        print(f"  without {name}: worse by {hurt}")
    a, b = fit(every)
    print(f"  FOUL_TROUBLE {a:+.2f} FIVE_FOULS {b:+.2f}")
    res = restmin - np.where(out, 0, mins * (1 + a * trouble + b * five))
    spots = trouble | five
    for s in SEASONS[:-1]:
        now = [res[(year == s) & (team == t) & spots].mean() for t in range(1, 31)]
        nxt = [res[(year == s + 1) & (team == t) & spots].mean() for t in range(1, 31)]
        print(f"  each team's minutes in foul trouble, {s} to {s + 1}: r = {np.corrcoef(now, nxt)[0, 1]:+.2f}")


def parts(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The shipped live line's points a minute and minutes left (as in live_line_backtest.predict)."""
    _, ppm, umin, el, m, so, _, mar = x[:, :8].T
    left = (48 - el) / 48
    gap = (so - ppm * m) / (m + liveline.FORM_MINUTES)
    rate = np.maximum(ppm + liveline.FORM * gap, 0)
    blow = np.clip((np.abs(mar) - liveline.MARGIN_FROM) / 10, 0, 2)
    by_now = umin * el / 48
    pace = (m - by_now) / np.maximum(by_now, 3)
    mins = umin * left * (1 + liveline.MARGIN * blow * (umin - 24) / 10 + liveline.MINUTES_PACE * pace)
    mins += liveline.HOT_ROLE * gap * (umin < liveline.ROLE_MINUTES)
    return rate, np.clip(mins, 0, 48 - el)


def teams(trow: np.ndarray, pre: dict) -> None:
    key = {(int(r[1]), int(r[2]), int(r[4]), int(r[5])): i for i, r in enumerate(trow)}
    keep, y = [], []
    for i, (_, g, tm, o, per, s, rest, _) in enumerate(trow):
        a, b, j = pre.get((int(g), int(tm))), pre.get((int(g), int(o))), key.get((int(g), int(o), int(per), int(s)))
        if a is None or b is None or j is None:
            continue
        k, frac = int(per) - 1, s / 720
        (pa, qa), (pb, qo) = a, b
        real = sum(qa[: k + 1]) - rest
        real_o = sum(qo[: k + 1]) - trow[j, 6]
        past = (real - real_o) - ((sum(pa[:k]) + pa[k] * frac) - (sum(pb[:k]) + pb[k] * frac))
        ours = (pa[k] + quarters.GIVE_BACK * past / (k + frac)) * (1 - frac)
        keep.append(i)
        y.append(rest - ours)
    t, y = trow[keep], np.array(y)
    year, team, frac, bonus = t[:, 0], t[:, 2], t[:, 5] / 720, t[:, 7]
    x = np.column_stack([bonus * (1 - frac) * 12, frac * (1 - frac) * 12])
    names = ("bonus", "in-quarter shape")

    def fit(m, cols=(0, 1)):
        w = np.zeros(2)
        w[list(cols)] = np.linalg.lstsq(x[np.ix_(m, list(cols))], y[m], rcond=None)[0]
        return w

    def mse(m, w):
        return float(((y[m] - x[m] @ w) ** 2).mean())

    print("\nquarters: squared miss of the rest of the quarter in progress (live quarters) and with fouls minus it")
    print(f"  share of checkpoints in the bonus {bonus.mean():.3f}")
    for s in SEASONS:
        te = year == s
        lo, fw = fit(year != s), fit(year < s) if s > SEASONS[0] else fit(year != s)
        base = float((y[te] ** 2).mean())
        print(f"  {s}: {base:.3f} {mse(te, lo) - base:+.4f} (left out) {mse(te, fw) - base:+.4f} (earlier only)")
    for c, n in enumerate(names):
        hurt = [round(mse(year == s, fit(year != s, (1 - c,))) - mse(year == s, fit(year != s)), 4) for s in SEASONS]
        print(f"  without {n}: worse by {hurt}")
    w = fit(np.ones(len(y), bool))
    print(f"  BONUS {w[0]:.3f} IN_QUARTER {w[1]:.3f}")
    r = y - x @ w
    on = bonus == 1
    for s in SEASONS[:-1]:
        now = [r[(year == s) & (team == k) & on].mean() for k in range(1, 31)]
        nxt = [r[(year == s + 1) & (team == k) & on].mean() for k in range(1, 31)]
        print(f"  each team's own scoring in the bonus, {s} to {s + 1}: r = {np.corrcoef(now, nxt)[0, 1]:+.2f}")


def main(folder: Path) -> None:
    prow, trow = collect(folder)
    players(prow)
    seasons = {y: qb.read_season(folder / f"cdnnba_{y}.csv") for y in (SEASONS[0] - 1, *SEASONS)}
    rows = qb.replay(seasons)
    ids = [t for y in sorted(seasons) for g in seasons[y] for t in (g["home"], g["away"])]
    tested = [(r, t) for r, t in zip(rows, ids, strict=True) if r["season"] in SEASONS]
    pre = {}
    for year in SEASONS:
        test = [x for x in tested if x[0]["season"] == year]
        share, b = qb.fit([x[0] for x in tested if x[0]["season"] != year], True)
        for (r, t), p in zip(test, qb.predict([x[0] for x in test], share, b, True), strict=True):
            pre[(r["game"], t)] = (p.tolist(), r["q"])
    teams(trow, pre)


if __name__ == "__main__":
    main(Path(sys.argv[1]))

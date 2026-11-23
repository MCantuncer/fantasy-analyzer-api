"""Blind test of the live line (app/liveline.py) on five seasons of NBA play-by-play.

Data: cdn.nba.com play-by-play as collected in github.com/shufinskiy/nba_data (one CSV a season,
"cdnnba_<year>.csv", 2020 = 2020-21). Download and untar them into a folder, then

    uv run python -m scripts.live_line_backtest <folder>

From the rows: each player's made baskets with their time, his time on the floor (a player is on at
a period's start unless his first row of the period is a sub in; stints end at a sub out or the
period's end) and his team's running margin. His usual, the stand-in for the tip-off line: points a
minute and minutes a game from his earlier games, recent ones counting most (half-life 20 games,
carried across seasons), from his 16th game on. Checkpoints every 6 minutes of regulation (6-42).

The plain line (usual points a minute x usual minutes x the share left) is the bar. The live line's
weights are fitted with the tested season left out, and again on earlier seasons only; it ships only
if it lowers the squared miss of the rest of his points in every season both ways, and so must each
piece (dropping it must hurt every season). Last printed: the weights on all five seasons and the
final-points miss by minutes played (liveline.FINAL_WITHIN).
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from app import liveline

SEASONS = (2021, 2022, 2023, 2024, 2025)  # 2021-22 to 2025-26; 2020-21 warms the usuals up
HALF_LIFE = 20.0
MIN_GAMES = 15
CHECKS = tuple(range(360, 2880, 360))  # seconds of regulation played
POINTS = {"2pt": 2, "3pt": 3, "freethrow": 1}


def clock(cl: str) -> float:
    """Seconds left in the period from "PT11M43.00S"."""
    try:
        m, s = cl[2:-1].split("M")
        return int(m) * 60 + float(s)
    except ValueError:
        return 0.0


def read_games(path: Path):
    """Per game: {player: (team, [(t, pts)], [(on, off)])} in regulation seconds, one game at a time."""
    with path.open(newline="") as fh:
        rows = csv.DictReader(fh)
        game, players, on, seen, period = None, {}, {}, set(), 0
        for r in rows:
            if r["gameId"] != game or int(r["period"] or 0) != period:
                p0 = (period - 1) * 720.0
                for pid, start in on.items():
                    players[pid][2].append((start, p0 + 720.0))
                on, seen = {}, set()
                if r["gameId"] != game:
                    if game is not None:
                        yield game, players
                    game, players = r["gameId"], {}
                period = int(r["period"] or 0)
            if period > 4 or not r["teamTricode"] or not r["personId"] or r["personId"] == "0":
                continue
            pid = int(float(r["personId"]))
            t = (period - 1) * 720.0 + 720.0 - clock(r["clock"])
            me = players.setdefault(pid, (r["teamTricode"], [], []))
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
            elif r["shotResult"] == "Made" and kind in POINTS:
                me[1].append((t, POINTS[kind]))
        if game is not None:
            p0 = (period - 1) * 720.0
            for pid, start in on.items():
                players[pid][2].append((start, p0 + 720.0))
            yield game, players


def checkpoints(folder: Path) -> np.ndarray:
    """season, usual pts/min, usual min, t, minutes so far, points so far, rest points, margin."""
    decay = 0.5 ** (1 / HALF_LIFE)
    usual: dict[int, list[float]] = {}
    out = []
    for year in (2020, *SEASONS):
        for _, players in sorted(read_games(folder / f"cdnnba_{year}.csv"), key=lambda x: x[0]):
            team_pts: dict[str, list[tuple[float, int]]] = defaultdict(list)
            for team, ev, _ in players.values():
                team_pts[team] += ev
            if len(team_pts) != 2:
                continue
            for pid, (team, ev, stints) in players.items():
                total = sum(b - a for a, b in stints) / 60
                if total <= 0:
                    continue
                u = usual.get(pid)
                if year in SEASONS and u and u[3] >= MIN_GAMES:
                    ppm, umin = u[0] / u[1], u[1] / u[2]
                    other = next(k for k in team_pts if k != team)
                    for t in CHECKS:
                        m = sum(max(0.0, min(b, t) - a) for a, b in stints) / 60
                        so = sum(p for x, p in ev if x <= t)
                        rest = sum(p for x, p in ev if x > t)
                        margin = sum(p for x, p in team_pts[team] if x <= t) - sum(
                            p for x, p in team_pts[other] if x <= t
                        )
                        out.append((year, ppm, umin, t / 60, m, so, rest, margin))
                pts = sum(p for _, p in ev)
                u = u or [0.0, 0.0, 0.0, 0]
                usual[pid] = [u[0] * decay + pts, u[1] * decay + total, u[2] * decay + 1, u[3] + 1]
        print(year, len(out), flush=True)
    return np.array(out)


def predict(p: np.ndarray, x: np.ndarray) -> np.ndarray:
    form, form_min, margin, pace_w, hot = p
    _, ppm, umin, el, m, so, _, mar = x.T
    left = (48 - el) / 48
    gap = (so - ppm * m) / (m + form_min)
    rate = np.maximum(ppm + form * gap, 0)
    blow = np.clip((np.abs(mar) - liveline.MARGIN_FROM) / 10, 0, 2)
    by_now = umin * el / 48
    pace = (m - by_now) / np.maximum(by_now, 3)
    mins = umin * left * (1 + margin * blow * (umin - 24) / 10 + pace_w * pace) + hot * gap * (umin < 28)
    return rate * np.clip(mins, 0, 48 - el)


def fit(x: np.ndarray, start: np.ndarray, keep: np.ndarray | None = None) -> np.ndarray:
    """Least squares on the rest of his points (Levenberg-Marquardt, numeric slopes); ``keep`` marks
    the weights fitted (the rest stay at ``start``)."""
    p = start.astype(float).copy()
    keep = np.ones(len(p), bool) if keep is None else keep
    lam = 1e-3
    y = x[:, 6]
    for _ in range(60):
        r = predict(p, x) - y
        cost = r @ r
        jac = np.zeros((len(x), len(p)))
        for i in np.flatnonzero(keep):
            dp = p.copy()
            dp[i] += 1e-4 * max(1.0, abs(p[i]))
            jac[:, i] = (predict(dp, x) - predict(p, x)) / (dp[i] - p[i])
        a, g = jac.T @ jac, jac.T @ r
        while True:
            step = np.zeros(len(p))
            step[keep] = -np.linalg.solve((a + lam * np.diag(np.diag(a)))[np.ix_(keep, keep)], g[keep])
            new = p + step
            new[1] = max(new[1], 0.5)
            rn = predict(new, x) - y
            if rn @ rn < cost:
                p, lam = new, lam / 3
                break
            lam *= 10
            if lam > 1e8:
                return p
        if abs(cost - rn @ rn) < 1e-7 * cost:
            break
    return p


def main(folder: Path) -> None:
    x = checkpoints(folder)
    shipped = np.array(
        [liveline.FORM, liveline.FORM_MINUTES, liveline.MARGIN, liveline.MINUTES_PACE, liveline.HOT_ROLE]
    )
    base = x[:, 1] * x[:, 2] * (48 - x[:, 3]) / 48

    def mse(e):
        return float(np.mean(e**2))

    print("season  plain  left-out  earlier-only   (squared miss of the rest of his points)")
    for s in SEASONS:
        te = x[x[:, 0] == s]
        lo = fit(x[(x[:, 0] != s) & (x[:, 0] >= SEASONS[0])], shipped)
        fw = fit(x[x[:, 0] < s], shipped) if s > SEASONS[0] else lo
        b = base[x[:, 0] == s]
        print(
            s,
            round(mse(te[:, 6] - b), 3),
            round(mse(te[:, 6] - predict(lo, te)), 3),
            round(mse(te[:, 6] - predict(fw, te)), 3),
        )
    names = ("form", "margin minutes", "minutes pace", "hot role minutes")
    for name, i in zip(names, (0, 2, 3, 4), strict=True):
        hurt = []
        for s in SEASONS:
            te, tr = x[x[:, 0] == s], x[(x[:, 0] != s) & (x[:, 0] >= SEASONS[0])]
            full = fit(tr, shipped)
            keep = np.ones(5, bool)
            keep[i] = False
            z = shipped.copy()
            z[i] = 0.0
            without = fit(tr, z, keep)
            hurt.append(round(mse(te[:, 6] - predict(without, te)) - mse(te[:, 6] - predict(full, te)), 3))
        print(f"without {name}: worse by {hurt}")
    allp = fit(x, shipped)
    print("weights, all seasons (FORM, FORM_MINUTES, MARGIN, MINUTES_PACE, HOT_ROLE):", np.round(allp, 3))
    final_miss = np.abs(x[:, 6] - predict(allp, x))
    for t in CHECKS:
        m = (x[:, 3] == t / 60) & (x[:, 2] >= 20)
        q = final_miss[m]
        print(f"  {t // 60} min: half within {np.quantile(q, 0.5):.1f}, 8 in 10 within {np.quantile(q, 0.8):.1f}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))

"""Blind test of points by quarter (app/quarters.py) on five seasons of NBA play-by-play.

Data: cdn.nba.com play-by-play as collected in github.com/shufinskiy/nba_data (one CSV a season,
"cdnnba_<year>.csv", 2020 = 2020-21). Download and untar them into a folder, then

    uv run python -m scripts.quarter_backtest <folder>

Each game's quarters (regulation, both sides) come from the running score; each team's possessions
and threes come from the same rows, so Our call (analysis/gamescore.py, without tonight's lineup) is
replayed day by day exactly as the Scores page makes it. Then every team-quarter is predicted:

  * league shape: the call x the league's share of a game's points in that quarter;
  * shipped: league shape + the favourite term (called margin, by quarter) + the team's own shape and
    the opponent's allowed shape (quarters.team_shape, this season so far plus last season's).

Weights are fitted with the tested season left out, and again on earlier seasons only. A piece ships
only if it lowers the squared miss in every season both ways. Last printed: the LAST_SEASON table
for the season after the newest one read.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from itertools import groupby
from pathlib import Path

import numpy as np

from app import quarters
from app.analysis import gamescore as gs
from app.quarters import QUARTERS, QuarterGame, Shapes

SEASONS = (2021, 2022, 2023, 2024, 2025)  # tested; 2020 gives the first one its start
# NBA tricodes whose ESPN pro team id isn't found by abbreviation.
PRO_ID = {"NOP": 3, "GSW": 9, "NYK": 18, "SAS": 24, "UTA": 26, "WAS": 27}
ABBREV = {"ATL": 1, "BOS": 2, "CHI": 4, "CLE": 5, "DAL": 6, "DEN": 7, "DET": 8, "HOU": 10, "IND": 11}
ABBREV |= {"LAC": 12, "LAL": 13, "MIA": 14, "MIL": 15, "MIN": 16, "BKN": 17, "ORL": 19, "PHI": 20, "PHX": 21}
ABBREV |= {"POR": 22, "SAC": 23, "OKC": 25, "TOR": 28, "MEM": 29, "CHA": 30} | PRO_ID


def read_season(path: Path) -> list[dict]:
    """One row per finished game: day, home/away pro ids, quarter points and box sums each side."""
    out = []
    with path.open(newline="") as fh:
        for gid, rows in groupby(csv.DictReader(fh), key=lambda r: r["gameId"]):
            game = _game(int(gid), list(rows))
            if game is not None:
                out.append(game)
    return sorted(out, key=lambda g: (g["day"], g["id"]))


def _game(gid: int, rows: list[dict]) -> dict | None:
    home = away = None
    last_h = last_a = 0
    end = defaultdict(lambda: [0, 0])  # period -> score at its end
    box = {s: defaultdict(float) for s in ("h", "a")}
    start = None
    for r in rows:
        try:
            period = int(r["period"])
            h, a = int(float(r["scoreHome"] or last_h)), int(float(r["scoreAway"] or last_a))
        except ValueError:
            continue
        start = start or r["timeActual"]
        team = r["teamTricode"]
        if team and h > last_h and home is None:
            home = team
        if team and a > last_a and away is None:
            away = team
        last_h, last_a = h, a
        end[period] = [h, a]
        side = "h" if team == home else "a" if team == away else None
        if side is None:
            continue
        kind = r["actionType"]
        b = box[side]
        if kind in ("2pt", "3pt"):
            b["fga"] += 1
            if kind == "3pt":
                b["tpa"] += 1
                b["tpm"] += r["shotResult"] == "Made"
        elif kind == "freethrow":
            b["fta"] += 1
        elif kind == "turnover":
            b["to"] += 1
        elif kind == "rebound" and r["subType"] == "offensive":
            b["oreb"] += 1
    if home is None or away is None or home == away or len(end) < QUARTERS or start is None:
        return None
    cum = [end[p] for p in range(1, QUARTERS + 1)]
    hq = [cum[0][0]] + [cum[k][0] - cum[k - 1][0] for k in range(1, QUARTERS)]
    aq = [cum[0][1]] + [cum[k][1] - cum[k - 1][1] for k in range(1, QUARTERS)]
    day = (datetime.fromisoformat(start.replace("Z", "+00:00")) - timedelta(hours=10)).date().toordinal()
    return {"id": gid, "day": day, "home": ABBREV[home], "away": ABBREV[away], "hq": hq, "aq": aq} | {
        "hpts": last_h,
        "apts": last_a,
        "hbox": box["h"],
        "abox": box["a"],
    }


def team_games(games: list[dict]) -> list[gs.TeamGame]:
    out = []
    for g in games:
        poss = sum(b["fga"] + 0.44 * b["fta"] - b["oreb"] + b["to"] for b in (g["hbox"], g["abox"])) / 2
        for t, o, me, them, pf, pa in (
            (g["home"], g["away"], g["hbox"], g["abox"], g["hpts"], g["apts"]),
            (g["away"], g["home"], g["abox"], g["hbox"], g["apts"], g["hpts"]),
        ):
            st = {"pf": pf, "pa": pa, "poss": poss, "tpm": me["tpm"], "tpa": me["tpa"], "fta": me["fta"]}
            st |= {"ftm": 0.78 * me["fta"], "tpm_a": them["tpm"], "tpa_a": them["tpa"]}
            st |= {"fta_a": them["fta"], "ftm_a": 0.78 * them["fta"]}
            out.append(gs.TeamGame(g["id"], g["day"], t, o, st))
    return sorted(out, key=lambda x: (x.day, x.game, x.team))


def replay(seasons: dict[int, list[dict]]) -> list[dict]:
    """Every team-game with its call, the opponent's call, its quarters and its shape features."""
    rows = []
    start: gs.SeasonEnd | None = None
    decay = 0.5 ** (1 / gs.HALF_LIFE)
    for year in sorted(seasons):
        games = seasons[year]
        prev = seasons.get(year - 1)
        if prev:
            table: dict[int, tuple] = {}
            for g in prev:
                for t, mine, theirs in ((g["home"], g["hq"], g["aq"]), (g["away"], g["aq"], g["hq"])):
                    s, a = table.setdefault(t, ([0.0] * QUARTERS, [0.0] * QUARTERS))
                    quarters._add(s, mine)
                    quarters._add(a, theirs)
            quarters.LAST_SEASON[year + 1] = table  # ESPN's season key: 2021-22 (pbp 2021) is 2022 there
        tg = team_games(games)
        s = gs.Scores(start=start)
        if start is not None:
            s.league = {k: gs.LEAGUE_PRIOR_GAMES * v for k, v in start.league.items()}
            s.games = gs.LEAGUE_PRIOR_GAMES
        shapes = Shapes()
        last_day: dict[int, int] = {}
        by_day: dict[int, list[dict]] = defaultdict(list)
        for g in games:
            by_day[g["day"]].append(g)
        tg_day: dict[int, list[gs.TeamGame]] = defaultdict(list)
        for t in tg:
            tg_day[t.day].append(t)
        for day in sorted(by_day):
            for g in by_day[day]:
                calls = {}
                for t, o, home in ((g["home"], g["away"], True), (g["away"], g["home"], False)):
                    calls[t] = gs.expected(s, t, o, home, last_day.get(t) == day - 1)
                for t, o, q in ((g["home"], g["away"], g["hq"]), (g["away"], g["home"], g["aq"])):
                    rows.append(
                        {"season": year, "game": g["id"], "call": calls[t], "margin": calls[t] - calls[o], "q": q}
                        | {"own": quarters.team_shape(year + 1, shapes, t, allowed=False)}
                        | {"opp": quarters.team_shape(year + 1, shapes, o, allowed=True)}
                    )
            for g in by_day[day]:
                shapes.add(QuarterGame(g["id"], g["away"], g["home"], tuple(g["aq"]), tuple(g["hq"])))
            for t in tg_day[day]:
                st = s.teams.get(t.team) or gs._start_of(t.team, start)
                s.teams[t.team] = st
                st["w"] = st["w"] * decay + 1
                s.played[t.team] = s.played.get(t.team, 0) + 1
                for k in gs.FIELDS:
                    st[k] = st[k] * decay + t.stats[k]
                    s.league[k] = s.league.get(k, 0.0) + t.stats[k]
                s.games += 1
                last_day[t.team] = day
        start = gs.season_end(tg)
    return rows


def fit(rows: list[dict], shape: bool) -> tuple[np.ndarray, np.ndarray]:
    """League shares, then per quarter [margin, level] and shared [own, opp] weights on what's left."""
    c = np.array([r["call"] for r in rows])
    q = np.array([r["q"] for r in rows], dtype=float)
    share = q.sum(0) / c.sum()
    n = len(rows)
    blocks, targets = [], []
    for k in range(QUARTERS):
        z = np.zeros((n, 2 * QUARTERS + (2 if shape else 0)))
        z[:, 2 * k] = [r["margin"] for r in rows]
        z[:, 2 * k + 1] = 1.0
        if shape:
            z[:, -2] = c * np.array([r["own"][k] for r in rows])
            z[:, -1] = c * np.array([r["opp"][k] for r in rows])
        blocks.append(z)
        targets.append(q[:, k] - c * share[k])
    b = np.linalg.lstsq(np.vstack(blocks), np.concatenate(targets), rcond=None)[0]
    return share, b


def predict(rows: list[dict], share: np.ndarray, b: np.ndarray | None, shape: bool) -> np.ndarray:
    c = np.array([r["call"] for r in rows])
    p = c[:, None] * share[None, :]
    if b is None:
        return p
    for k in range(QUARTERS):
        p[:, k] += b[2 * k] * np.array([r["margin"] for r in rows]) + b[2 * k + 1]
        if shape:
            p[:, k] += c * (
                b[-2] * np.array([r["own"][k] for r in rows]) + b[-1] * np.array([r["opp"][k] for r in rows])
            )
    return p


def live_test(tested: list[dict]) -> None:
    """quarters.live_quarters: at the end of each quarter, every quarter to come is the pre-game number
    plus a weight x the margin so far past the called one, per quarter played. Fitted with the season
    left out and on earlier seasons only, against the pre-game number alone."""
    pre: dict[int, np.ndarray] = {}
    for year in SEASONS:
        test = [r for r in tested if r["season"] == year]
        train = [r for r in tested if r["season"] != year]
        share, b = fit(train, True)
        for i, p in zip(range(len(test)), predict(test, share, b, True), strict=True):
            pre[id(test[i])] = p
    opp = {}
    for r in tested:
        opp.setdefault(r["game"], []).append(r)
    x, y, season = [], [], []
    for r in tested:
        pair = opp[r["game"]]
        if len(pair) != 2:
            continue
        o = pair[1] if pair[0] is r else pair[0]
        p, po = pre[id(r)], pre[id(o)]
        for j in (1, 2, 3):
            past = (sum(r["q"][:j]) - sum(o["q"][:j])) - (p[:j].sum() - po[:j].sum())
            for k in range(j, QUARTERS):
                x.append(past / j)
                y.append(r["q"][k] - p[k])
                season.append(r["season"])
    x, y, season = np.array(x), np.array(y), np.array(season)
    print("\nlive quarters: squared miss of a quarter to come (pre-game number) and with GIVE_BACK minus it")
    for mode in ("season left out", "earlier seasons only"):
        out = []
        for year in SEASONS:
            m = season == year
            t = (season != year) if mode == "season left out" else (season < year)
            if not t.any():
                t = season != year
            w = float((x[t] * y[t]).sum() / (x[t] ** 2).sum())
            base = float((y[m] ** 2).mean())
            out.append(f"{year}: {base:.3f} {float(((y[m] - w * x[m]) ** 2).mean()) - base:+.3f} (w {w:.4f})")
        print(f"  {mode}: " + "; ".join(out))
    print("  GIVE_BACK, all seasons:", round(float((x * y).sum() / (x**2).sum()), 4))


def main(folder: str) -> None:
    seasons = {}
    for year in (SEASONS[0] - 1, *SEASONS):
        seasons[year] = read_season(Path(folder) / f"cdnnba_{year}.csv")
        print(year, len(seasons[year]), "games", flush=True)
    rows = replay(seasons)
    tested = [r for r in rows if r["season"] in SEASONS]
    for mode in ("season left out", "earlier seasons only"):
        print(f"\n{mode}: squared miss a team-quarter (league shape) and shipped minus it")
        for year in SEASONS:
            test = [r for r in tested if r["season"] == year]
            train = [r for r in tested if (r["season"] != year if mode == "season left out" else r["season"] < year)]
            if not train:
                train = [r for r in rows if r["season"] < year]
            q = np.array([r["q"] for r in test], dtype=float)
            share, _ = fit(train, False)
            base = ((q - predict(test, share, None, False)) ** 2).mean()
            share, b = fit(train, True)
            shipped = ((q - predict(test, share, b, True)) ** 2).mean()
            print(f"  {year}: {base:.3f} {shipped - base:+.3f}")
    live_test(tested)
    share, b = fit(tested, True)
    print("\nall seasons: LEAGUE_SHARE", np.round(share, 4).tolist())
    print(
        "FAVOURITE",
        [round(float(b[2 * k]), 4) for k in range(QUARTERS)],
        "LEVEL",
        [round(float(b[2 * k + 1]), 3) for k in range(QUARTERS)],
    )
    print("OWN_SHAPE", round(float(b[-2]), 3), "OPP_SHAPE", round(float(b[-1]), 3))
    newest = max(seasons)
    print(f"\nLAST_SEASON[{newest + 2}]:")
    table: dict[int, tuple] = {}
    for g in seasons[newest]:
        for t, mine, theirs in ((g["home"], g["hq"], g["aq"]), (g["away"], g["aq"], g["hq"])):
            s, a = table.setdefault(t, ([0] * QUARTERS, [0] * QUARTERS))
            quarters._add(s, mine)
            quarters._add(a, theirs)
    for t in sorted(table):
        print(f"    {t}: ({tuple(table[t][0])}, {tuple(table[t][1])}),")


if __name__ == "__main__":
    main(sys.argv[1])

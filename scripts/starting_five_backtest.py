"""Tonight's starting five in next-game lines (gamebox.starting_five), replayed with the app's code.

    uv run python -m scripts.starting_five_backtest            # eight seasons + the season so far
    uv run python -m scripts.starting_five_backtest 2024       # one season's rows only (cached in .cache)

Each case is the line as it ships (gamebox.apply, the team's sum, then the trees), what he did, and
whether he started (the box score's starters, so this is the term with the confirmed five). The one
weight is fitted on minutes, blind: leave one season out (fit on the other four of 2021-22..2025-26)
and forward (fit on earlier seasons only), then once on all five for 2026-27 so far and, frozen, for
2018-19..2020-21. Each stat then moves with his minutes, as gamebox.starting_five moves it. Can's
rule: a stat is moved only if every one of those is better. Prints the % less squared miss.

Two checks on the result itself: the same test with who started shuffled within each season (should
be about zero), and with a five that is wrong the way nba.com's first listing was on the logged nights
(one starter swapped for the bench player with the highest line, in LISTED_WRONG of team games):
fitted for itself, and at the weight of the real five, which is what feeding a first listing to
gamebox.starting_five would do.
"""

from __future__ import annotations

import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from app.analysis import blackbox, gamebox
from app.analysis.gamebox import GS, MAX_MINUTES, PTS
from app.analysis.report import team_adjust, team_sum_gap
from scripts.durability_backtest import season_pool
from scripts.next_game_matchup_backtest import SEASONS, cases

CURRENT = 2027
OLDER = (2019, 2020, 2021)
CACHE = Path(".cache/starting_five")
CATS = ("MIN", "PTS", "REB", "AST", "STL", "BLK", "3PM", "TO", "FG", "FT")  # blackbox.cats
LISTED_WRONG = 0.3  # 12 of 40 team games on the two logged nights had one starter change after the first listing


def season_rows(season: int) -> dict[str, np.ndarray]:
    path = CACHE / f"{season}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    for s in range(season - 4, season + 1):
        season_pool(s)
    groups = defaultdict(list)
    for _box, c in cases(season):
        f = c.context.form
        if f is None or f.starts_lately is None:
            continue
        new, _, _ = gamebox.apply(c.line, c.context, c.mates, c.gained, c.back, c.absent)
        groups[(c.day, c.team)].append((c, new))
    cols: dict[str, list] = {k: [] for k in ("line", "act", "starts", "lately", "team_game")}
    for (day, team), group in groups.items():
        lines_pts = sum(new.get(PTS, 0.0) for _, new in group)
        season_pts = sum(c.so_far.get(PTS, 0.0) for c, _ in group)
        for c, new in group:
            m = c.context
            gap = 0.0 if m.team_score is None else team_sum_gap(m.team_score, lines_pts, season_pts, m.team_points)
            new = team_adjust(new, gap)
            new = blackbox.adjust(new, blackbox.calls(blackbox.inputs(m, c.line, new, c.mates, c.back, c.gained, gap)))
            cols["line"].append(blackbox.cats(new))
            cols["act"].append(blackbox.cats(c.actual))
            cols["starts"].append(float(c.actual.get(GS, 0.0) > 0))
            cols["lately"].append(m.form.starts_lately)
            cols["team_game"].append(day * 100 + team)
    out = {k: np.array(v) for k, v in cols.items()}
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(out))
    return out


def moved(line: np.ndarray, x: np.ndarray, weight: float) -> np.ndarray:
    """Every category of ``line`` (rows of blackbox.cats) with the term in it: gamebox.starting_five's
    minutes, and each stat with his minutes (FG and FT as their makes and attempts would)."""
    minutes = line[:, 0]
    new = np.clip(minutes + weight * x, 0.0, MAX_MINUTES)
    ratio = np.divide(new, minutes, out=np.ones_like(new), where=minutes > 0)
    out = line * ratio[:, None]
    out[:, 0] = np.where(minutes > 0, new, minutes)
    return out


def fit(line: np.ndarray, act: np.ndarray, x: np.ndarray) -> float:
    return float(x @ (act[:, 0] - line[:, 0])) / float(x @ x)


def gains(line: np.ndarray, act: np.ndarray, x: np.ndarray, weight: float) -> np.ndarray:
    """% less squared miss in each category with the term at ``weight``."""
    before = np.mean((act - line) ** 2, axis=0)
    after = np.mean((act - moved(line, x, weight)) ** 2, axis=0)
    return (before - after) / before * 100


def listed(rows: dict[str, np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """Who starts as a first listing has it: in LISTED_WRONG of team games one real starter is
    swapped for the bench player with the highest minutes line."""
    starts = rows["starts"].copy()
    by_game = defaultdict(list)
    for i, key in enumerate(rows["team_game"]):
        by_game[int(key)].append(i)
    for idx in by_game.values():
        if rng.random() >= LISTED_WRONG:
            continue
        five = [i for i in idx if rows["starts"][i] > 0]
        bench = [i for i in idx if rows["starts"][i] == 0]
        if five and bench:
            starts[rng.choice(five)] = 0.0
            starts[max(bench, key=lambda i: rows["line"][i, 0])] = 1.0
    return starts


def judge(label: str, data: dict[int, dict[str, np.ndarray]], x: dict[int, np.ndarray], cats: tuple[str, ...]) -> None:
    def stack(seasons: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.concatenate([data[s]["line"] for s in seasons]),
            np.concatenate([data[s]["act"] for s in seasons]),
            np.concatenate([x[s] for s in seasons]),
        )

    def test(train: list[int], season: int) -> np.ndarray:
        d = data[season]
        return gains(d["line"], d["act"], x[season], fit(*stack(train)))

    five = [s for s in SEASONS if s in data]
    left_out = [test([t for t in five if t != s], s) for s in five]
    forward = [test([t for t in five if t < s], s) for s in five[1:]]
    frozen = {s: test(five, s) for s in (CURRENT, *OLDER) if s in data}
    print(f"\n{label}: weight on the five seasons {fit(*stack(five)):.2f}")
    for c in cats:
        i = CATS.index(c)
        every = [g[i] for g in (*left_out, *forward, *frozen.values())]
        print(
            f"  {c:3s} left out {' '.join(f'{g[i]:+.2f}' for g in left_out)}"
            f" | forward {' '.join(f'{g[i]:+.2f}' for g in forward)}"
            f" | {' '.join(f'{s} {g[i]:+.2f}' for s, g in frozen.items())}"
            f" | {'every one better' if min(every) > 0 else 'not every one'}"
        )


def main() -> None:
    if len(sys.argv) > 1:
        rows = season_rows(int(sys.argv[1]))
        print(sys.argv[1], len(rows["starts"]), "cases")
        return
    data = {s: season_rows(s) for s in (*OLDER, *SEASONS, CURRENT)}
    n = sum(len(d["starts"]) for d in data.values())
    real = {s: d["starts"] - d["lately"] for s, d in data.items()}
    into = sum(int((v >= 0.5).sum()) for v in real.values())
    out_of = sum(int((v <= -0.5).sum()) for v in real.values())
    print(f"{n} cases; into the five {into / n:.1%}, out of it {out_of / n:.1%}")
    judge("The box score's starters", data, real, CATS)
    rng = np.random.default_rng(7)
    judge("Shuffled within each season", data, {s: rng.permutation(v) for s, v in real.items()}, ("MIN", "PTS"))
    first = {s: listed(d, rng) - d["lately"] for s, d in data.items()}
    judge(f"A first listing, one starter wrong in {LISTED_WRONG:.0%} of team games", data, first, CATS[:5])
    print(f"\nThe same first listing at the real five's weight ({gamebox.STARTING_MINUTES}):")
    for s, d in data.items():
        g = gains(d["line"], d["act"], first[s], gamebox.STARTING_MINUTES)
        print(f"  {s} {' '.join(f'{c} {v:+.2f}' for c, v in zip(CATS[:5], g[:5], strict=True))}")


if __name__ == "__main__":
    main()

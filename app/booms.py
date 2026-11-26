"""Crazy One's boom calls: the players most likely to go way over their line tonight.

Can 2026-11-26: "My expectation is to catch 'much more' than we expected situations." So Crazy One
hunts booms: nights far above a player's line, which Our chance (app/analysis/rungs.py) already prices
right on average. Crazy One believes three fan beliefs on top of it, the ones that caught more booms than
Our chance expected in a five-season blind replay (2021-22..2025-26, each season left out of its own
fit, then 2026-27 so far; files /mnt/project-files/crazy-one/boom-replay/):

    hot       hot hand: his last 3 games' points over his line
    swings    wild swings: how much his points moved over his last 10 games, over his line
    minutes   minutes rollercoaster: how much his minutes moved over his last 10 games
    done_it   done it lately (rebounds): his best rebound game of his last 10, over his line

Points booms (hot + swings + minutes): +0.01..+0.16% log loss every season; his top 5 a night caught
0.63-0.80 booms where Our chance said 0.63-0.65, and beat Our chance's own top 5 in 5 of 6 seasons.
Rebound booms (swings + minutes + done_it): + in 5 of 6 seasons, top 5 caught 0.36-0.53 where Our chance
said 0.39-0.40. Real but small: about 1 boom in 7 picks where Our chance expects 1 in 8. Assists and
threes found nothing, and "soft defense tonight" did worse than nothing (one of the myths he refuses).

RECORDED ONLY, as the rest of Crazy One (app/tunings.py): never Our Estimates, never the Kupons.

A game's boom calls, kept in its tunings record under "booms", top PICKS per stat:
    [player, team, name, stat, line, bar, our chance, Crazy One's chance, [[belief, value, push]]]
value is what he did (hot: points a game over his last 3; swings: his points' spread over his last 10;
minutes: his minutes' spread; done_it: his best rebound game) and push the chance it adds.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable

from app.analysis import rungs
from app.analysis.gamebox import Form

STATS = ("pts", "reb")
MIN_MINUTES = 12.0  # a line under this many minutes isn't called
MIN_GAMES = 5  # games this season before his swings count
PICKS = 5  # a night's boom calls per stat (and the most a game keeps per stat)
HOT_GAMES = 3

HOT, SWINGS, MINUTES, DONE_IT = "hot", "swings", "minutes", "done_it"

# Fitted on 2021-22..2025-26 (logistic, Our chance's logit as the offset, ridge 2): per belief
# (low, high) where its measure is clipped, the five seasons' average, and the logit it adds a unit.
WEIGHTS: dict[str, tuple[tuple[str, float, float, float, float], ...]] = {
    "pts": (
        (HOT, -0.6006, 0.6849, 0.0114, 0.0966),
        (SWINGS, 0.1562, 0.8351, 0.4175, 0.2235),
        (MINUTES, 1.7139, 9.8813, 4.7540, 0.0280),
    ),
    "reb": (
        (SWINGS, 0.1582, 0.9898, 0.4408, 0.6880),
        (DONE_IT, 1.0728, 3.1969, 1.8391, 0.1191),
        (MINUTES, 1.7205, 9.9749, 4.8093, 0.0416),
    ),
}


def bar(stat: str, line: float) -> int | None:
    """What counts as a boom on this line: points 1.5x his line and 8 over it at least (lines of 8+),
    rebounds 1.6x and 5 over (lines of 3+). None for a line too small to boom from."""
    if stat == "pts" and line >= 8:
        return math.ceil(max(1.5 * line, line + 8) - 1e-9)
    if stat == "reb" and line >= 3:
        return math.ceil(max(1.6 * line, line + 5) - 1e-9)
    return None


def measures(stat: str, line: dict[str, float], last: tuple[tuple[float, float, float], ...]) -> dict:
    """Each belief's measure for the model and the plain value shown, {belief: (measure, value)}, from
    his line and his last games ((minutes, points, rebounds), oldest first). Swings are his points'
    spread over his points line for both stats, as the replay found them."""
    minutes = [g[0] for g in last]
    points = [g[1] for g in last]
    spread = statistics.pstdev(points)
    out = {
        MINUTES: (statistics.pstdev(minutes), statistics.pstdev(minutes)),
        SWINGS: (spread / max(line.get("pts") or 0.0, 1.0), spread),
    }
    mine = max(line.get(stat) or 0.0, 1.0)
    if stat == "pts":
        hot = sum(points[-HOT_GAMES:]) / len(points[-HOT_GAMES:])
        out[HOT] = ((hot - (line.get("pts") or 0.0)) / mine, hot)
    else:
        best = max(g[2] for g in last)
        out[DONE_IT] = (best / mine, best)
    return out


def _sig(z: float) -> float:
    return 1 / (1 + math.exp(-z))


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def crazy_chance(stat: str, ours: float, seen: dict[str, tuple[float, float]]) -> tuple[float, list[list]]:
    """Crazy One's chance of the boom and the beliefs that raised it, [belief, value, push], biggest
    push first. A belief at or under its five-season average pushes nothing and isn't listed."""
    parts = {}
    for name, low, high, mean, weight in WEIGHTS[stat]:
        x = min(max(seen[name][0], low), high)
        parts[name] = weight * (x - mean)
    base = _logit(ours)
    p = _sig(base + sum(parts.values()))
    pushed = []
    for name, part in parts.items():
        if part <= 0:
            continue
        without = _sig(base + sum(v for k, v in parts.items() if k != name))
        pushed.append([name, round(seen[name][1], 1), round(p - without, 3)])
    pushed.sort(key=lambda b: -b[2])
    return round(p, 3), pushed


def game_calls(
    lines: dict[int, tuple[int, dict[str, float]]],
    form: dict[int, Form],
    names: dict[int, str],
    out: Iterable[int] = (),
) -> list[list]:
    """A game's boom calls from Live's player lines, the top PICKS per stat by Crazy One's chance.
    Players ruled out (``out``), on a short line or with under MIN_GAMES games aren't called."""
    skip = set(out)
    rows: dict[str, list[list]] = {s: [] for s in STATS}
    for pid, (team, line) in lines.items():
        f = form.get(pid)
        if pid in skip or f is None or len(f.last) < MIN_GAMES or (line.get("minutes") or 0.0) < MIN_MINUTES:
            continue
        for stat in STATS:
            ln = line.get(stat) or 0.0
            n = bar(stat, ln)
            ours = None if n is None else rungs.chance(stat, ln, n)
            if not ours:
                continue
            crazy, why = crazy_chance(stat, ours, measures(stat, line, f.last))
            rows[stat].append([pid, team, names.get(pid, ""), stat, round(ln, 1), n, ours, crazy, why])
    return [r for s in STATS for r in sorted(rows[s], key=lambda r: (-r[7], r[0]))[:PICKS]]


def night_picks(games: Iterable[tuple[int, dict]]) -> list[tuple[int, list]]:
    """A night's boom calls: its top PICKS per stat by Crazy One's chance, over all its games
    ((game id, tunings record game)), as (game id, call)."""
    every = [(gid, b) for gid, g in games for b in g.get("booms") or []]
    out = []
    for stat in STATS:
        mine = sorted((x for x in every if x[1][3] == stat), key=lambda x: (-x[1][7], x[1][0]))
        out += mine[:PICKS]
    return out


def outcome(call: list, results: dict[str, list], index: dict[str, int]) -> tuple[float | None, bool | None]:
    """What he got and whether it was a boom; (None, None) until the line log has his night or when
    he didn't play (results as app/linelog.py keeps them, index: its stat -> column)."""
    got = results.get(str(call[0]))
    if not got:
        return None, None
    value = got[index[call[3]]]
    return value, value >= call[5]


def record(nights: list[list[tuple[float, bool]]]) -> dict:
    """Booms caught against what Our chance expected of the same calls, over the settled nights
    (each a list of (our chance, boomed)): a per-night edge with a 95% range, ``leading`` only when
    the whole range is above zero."""
    from app.tunings import edge_range

    done = [n for n in nights if n]
    picks = sum(len(n) for n in done)
    caught = sum(b for n in done for _, b in n)
    expected = sum(p for n in done for p, _ in n)
    got = edge_range([sum(b for _, b in n) - sum(p for p, _ in n) for n in done])
    return {
        "nights": len(done),
        "picks": picks,
        "caught": caught,
        "expected": round(expected, 2),
        "edge": None if got is None else round(got[0], 3),
        "edge_low": None if got is None else round(got[1], 3),
        "edge_high": None if got is None else round(got[2], 3),
        "leading": got is not None and got[1] > 0,
    }

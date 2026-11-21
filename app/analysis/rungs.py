"""Our chance that a player reaches a sportsbook rung ("n+"), from our line for his next game.

The chances come from rung_shape.json (scripts/rung_shape_table.py): over five replayed seasons,
how often players with a line near ours got to n or more. Between two grid lines the chance is
read on a straight line; past the grid it holds the end value.

``value`` is what one unit on a rung returns on average if our chance is right: chance x odds - 1.
Over 0 means the price pays more than the chance says it should. COMPARISON ONLY: nothing here
feeds the model, and the book's numbers never come back into our lines.
"""

from __future__ import annotations

import json
from bisect import bisect_right
from functools import lru_cache
from pathlib import Path

from app.analysis import gamebox

TABLE = Path(__file__).with_name("rung_shape.json")
STATS = {
    "pts": gamebox.PTS,
    "reb": gamebox.REB,
    "ast": gamebox.AST,
    "stl": gamebox.STL,
    "blk": gamebox.BLK,
    "three_pm": gamebox.TPM,
}


@lru_cache(maxsize=1)
def _tables() -> dict:
    return json.loads(TABLE.read_text())["stats"]


def chance_from(table: dict, line: float, n: int) -> float:
    """P(stat >= n) for a player whose line is ``line``, read off one stat's table."""
    chances = table["chance"]
    if n <= 0:
        return 1.0
    if n >= len(chances[0]):
        return 0.0
    lines = table["lines"]
    i = bisect_right(lines, line)
    if i == 0:
        return chances[0][n]
    if i == len(lines):
        return chances[-1][n]
    a, b = lines[i - 1], lines[i]
    t = (line - a) / (b - a)
    return chances[i - 1][n] * (1 - t) + chances[i][n] * t


def chance(stat: str, line: float | None, n: int) -> float | None:
    """Our chance he gets ``n`` or more of ``stat`` (pts, reb, ast, stl, blk, three_pm); None
    without a line."""
    if line is None or stat not in STATS:
        return None
    return round(chance_from(_tables()[stat], line, n), 3)


def value(p: float | None, odd: float) -> float | None:
    """Average return of one unit at these odds if the chance is right: p x odd - 1."""
    return None if p is None else round(p * odd - 1, 3)

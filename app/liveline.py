"""A player's live line: where his night ends, from his line at tip-off and what has happened since.

Can 2026-11-23: per player, during the game, for live users. Tested on every player's game of five
seasons (2021-22 to 2025-26, cdn.nba.com play-by-play, checkpoints every 6 minutes of game clock),
fitted with each season left out and again on earlier seasons only (scripts/live_line_backtest.py).
The plain line, his usual points a minute x his usual minutes x the share of the game left, is the
bar to beat. Each piece below beat it in every season, both ways of fitting:

  * his night so far: points above or below his usual for the minutes he has played carry over a
    little (FORM, shrunk by FORM_MINUTES so a hot two minutes counts for little). A quiet start
    is not made up later, and a burst (6+ points in a minute) says nothing more than the points;
  * a lopsided score: past a 10-point margin, starters lose minutes and the bench gains them
    (MARGIN, per 10 points over 10, up to 30, x how far his usual minutes are from 24);
  * minutes so far: a player ahead of his usual share keeps playing a little more (MINUTES_PACE);
  * a hot role player (usual minutes under ROLE_MINUTES) gets a little more time (HOT_ROLE).

Together they lower the squared miss of the rest of his points by 1.5-2.3% with each season left
out and 1.5-2.3% fitted on earlier seasons only (2022-23 on), every season better. The
miss shrinks as the game goes (FINAL_WITHIN). Tested and dropped (2026-11-23): team ratings split
into quarters, halves or starter/bench quarters (same as the whole game), each factor head to head
(turnovers, boards, free throws, threes), and bursts of 4+ points in 15 seconds or 6+ in a minute in
any setting (half, close game, trailing, leading, with a three). Foul trouble is not in yet.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

GAME_MINUTES = 48.0
FORM = 0.248  # x (points so far - usual a minute x minutes so far) / (minutes so far + FORM_MINUTES)
FORM_MINUTES = 48.9
MARGIN = -0.232  # minutes share, x margin past 10 (per 10 points, capped at 2) x (usual minutes - 24) / 10
MARGIN_FROM = 10.0
MINUTES_PACE = 0.083  # minutes share, x minutes so far vs his usual share by now (relative)
HOT_ROLE = 7.04  # minutes, x the form gap, for a player whose usual minutes are under ROLE_MINUTES
ROLE_MINUTES = 28.0
# How far his final points land from the live line (players with 20+ usual minutes), by regulation
# minutes played: half land within the first number, 8 in 10 within the second.
FINAL_WITHIN: tuple[tuple[float, float, float], ...] = (
    (0.0, 4.4, 8.3),
    (6.0, 4.1, 7.7),
    (12.0, 3.8, 7.0),
    (18.0, 3.5, 6.4),
    (24.0, 3.1, 5.5),
    (30.0, 2.7, 4.7),
    (36.0, 2.2, 3.6),
    (42.0, 1.4, 2.2),
    (48.0, 0.0, 0.0),
)


@dataclass(frozen=True)
class LiveLine:
    final: float  # points at the end of regulation
    rest: float  # points still to come
    minutes_left: float  # his minutes still to come
    half_within: float
    most_within: float


def within(elapsed: float) -> tuple[float, float]:
    """FINAL_WITHIN at ``elapsed`` minutes, read between its rows."""
    rows = FINAL_WITHIN
    if elapsed <= rows[0][0]:
        return rows[0][1], rows[0][2]
    for (t0, h0, m0), (t1, h1, m1) in zip(rows, rows[1:], strict=False):
        if elapsed <= t1:
            f = (elapsed - t0) / (t1 - t0)
            return h0 + f * (h1 - h0), m0 + f * (m1 - m0)
    return 0.0, 0.0


def live_line(
    pts: float, minutes: float, line_pts: float, line_minutes: float, elapsed: float, margin: float
) -> LiveLine | None:
    """His live line, from his points and minutes so far, his tip-off line (points and minutes), the
    regulation minutes played and his team's margin (positive when ahead). None without a usable line."""
    if line_minutes <= 0 or line_pts < 0 or not math.isfinite(elapsed):
        return None
    elapsed = min(GAME_MINUTES, max(0.0, elapsed))
    left = (GAME_MINUTES - elapsed) / GAME_MINUTES
    ppm = line_pts / line_minutes
    gap = (pts - ppm * minutes) / (minutes + FORM_MINUTES)
    rate = max(0.0, ppm + FORM * gap)
    blow = min(2.0, max(0.0, (abs(margin) - MARGIN_FROM) / 10))
    starter = (line_minutes - 24) / 10
    by_now = line_minutes * elapsed / GAME_MINUTES
    pace = (minutes - by_now) / max(by_now, 3.0)
    mins = line_minutes * left * (1 + MARGIN * blow * starter + MINUTES_PACE * pace)
    if line_minutes < ROLE_MINUTES:
        mins += HOT_ROLE * gap
    mins = max(0.0, min(mins, GAME_MINUTES - elapsed))
    rest = rate * mins
    half, most = within(elapsed)
    return LiveLine(final=pts + rest, rest=rest, minutes_left=mins, half_within=half, most_within=most)

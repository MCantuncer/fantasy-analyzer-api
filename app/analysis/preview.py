"""A game before tip-off, for the pre-game box on Scores: why Our call is what it is, who plays, and
how the players' lines stand against the team's call.

Our call (gamescore.py) adds up from a few pieces a side: the ratings (the pace of this matchup, the
team's offense and the opponent's defense), a level, home court, three-point luck on both ends and
a back-to-back. The margin between the two sides is the same pieces, home side minus away side (the
level cancels), shown from the picked side and rounded so the whole numbers add up to the margin the
card shows.

Player lines are report.next_game_line, the same line the player card shows. They add up to the
team's call only in part: the add-up check moves each line by a share of the gap (TEAM_SUM_SHARE,
what tested best for each player), so the box says what the lines and the rest of the team add up
to against the call, rather than forcing them to agree.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, replace

from app.analysis.absence import is_out
from app.analysis.gamebox import PTS, Box, context
from app.analysis.model import CategoryModel
from app.analysis.report import MIN_TEAM_GAMES, next_game_line
from app.espn.parse import Player
from app.schemas import NextGameLine

# The margin's pieces, in the order the box lists them: gamescore.parts' keys each one sums.
MARGIN_TERMS: dict[str, tuple[str, ...]] = {
    "matchup": ("ratings", "lineup", "absences"),
    "home_court": ("home",),
    "three_point_luck": ("opp_three_luck", "own_three_luck"),
    "back_to_back": ("back_to_back",),
}
ROTATION_GAMES = 5  # a player is in the rotation if he played in one of his team's last this many games
ROTATION_MINUTES = 8.0  # ... and plays this many minutes a game lately (Form.minutes)


@dataclass(frozen=True)
class Term:
    kind: str
    points: int  # rounded so the terms add up to the called margin
    raw: float


@dataclass(frozen=True)
class Margin:
    pick: str  # "away" or "home"
    points: int  # the called margin, whole points, from the picked side: never 0
    raw: float
    terms: tuple[Term, ...]


def margin(away: dict[str, float], home: dict[str, float], away_call: int, home_call: int) -> Margin:
    """The called margin from the picked side, piece by piece: each piece is the home side's part less
    the away side's (gamescore.parts), turned to the picked side, rounded to add up to the card's
    margin (``home_call`` - ``away_call``, whole points, never tied)."""
    pick = "home" if home_call > away_call else "away"
    sign = 1.0 if pick == "home" else -1.0
    raw = {k: sign * sum(home.get(p, 0.0) - away.get(p, 0.0) for p in keys) for k, keys in MARGIN_TERMS.items()}
    target = abs(home_call - away_call)
    rounded = largest_remainder(list(raw.values()), target)
    total = sign * (sum(home.values()) - sum(away.values()))
    return Margin(
        pick,
        target,
        round(total, 2),
        tuple(Term(k, n, round(v, 2)) for (k, v), n in zip(raw.items(), rounded, strict=True)),
    )


def largest_remainder(values: list[float], target: int) -> list[int]:
    """Whole numbers close to ``values`` that add up to ``target``: each floored, then the ones that lost
    the most get one more (or, when the floors already pass the target, the ones that lost least one
    fewer)."""
    floors = [math.floor(v) for v in values]
    left = target - sum(floors)
    order = sorted(range(len(values)), key=lambda i: values[i] - floors[i], reverse=left > 0)
    out = list(floors)
    for i in range(abs(left)):
        out[order[i % len(order)]] += 1 if left > 0 else -1
    return out


def rotation(box: Box, team: int, day: int, known: Iterable[int]) -> list[int]:
    """Players of ``team`` the box has who aren't ``known``: in his team's rotation lately (one of its
    last ROTATION_GAMES games before ``day``, ROTATION_MINUTES+ a game), most minutes first."""
    played = [d for d in box.played.get(team, []) if d < day][-ROTATION_GAMES:]
    if not played:
        return []
    skip = set(known)
    found = [
        (f.minutes, pid)
        for pid, f in box.form.items()
        if f.team == team and pid not in skip and f.last_day >= played[0] and f.minutes >= ROTATION_MINUTES
    ]
    return [pid for _, pid in sorted(found, reverse=True)]


@dataclass(frozen=True)
class TeamLine:
    shown: float  # the shown players' points lines, added up
    rest: float  # the rest of the team: its points a game less the shown players' season points
    total: float
    call: float  # the team's expected points, before rounding
    gap: float  # total - call: + the lines run over the call, - under it


def team_line(shown: float, season: float, team_points: float, call: float) -> TeamLine:
    """How the lines stand against the call, as report.team_sum_gap sees it: the shown players' lines
    plus what the rest of the team scores a game (its points a game less these players' season
    points a game)."""
    rest = max(team_points - season, 0.0)
    total = shown + rest
    return TeamLine(round(shown, 1), round(rest, 1), round(total, 1), round(call, 1), round(total - call, 1))


@dataclass(frozen=True)
class SideLines:
    playing: list[tuple[Player, NextGameLine]]  # most minutes first
    out: list[Player]  # ruled out tonight
    team_line: TeamLine | None  # None before the box has scores


def side_lines(
    model: CategoryModel, box: Box, day: int, team: int, pool: list[Player], deep: list[Player]
) -> SideLines:
    """Every line of ``team`` in its game on ``day``. ``pool``: its players in the league's own list, whose
    lines are the player card's (their add-up check counts the pool, as it does there); ``deep``: rotation
    players read for this box only, checked against the pool and each other. Players are copied with
    this game as their next one, so a later game's box works the same as tonight's."""
    pool = [_at(box, p, team, day) for p in pool]
    deep = [_at(box, p, team, day) for p in deep]
    pool_sums: dict = {}
    deep_sums: dict = {}
    playing, out = [], []
    for group, mates, sums in ((pool, pool, pool_sums), (deep, pool + deep, deep_sums)):
        for p in group:
            if is_out(p):
                out.append(p)
                continue
            if p.game_context is None or p.game_context.day != day:
                continue
            line = next_game_line(model, p, mates, sums)
            if line is not None:
                playing.append((p, line))
    playing.sort(key=lambda x: -x[1].minutes)
    m = next((p.game_context for p, _ in playing), None)
    tl = None
    if m is not None and m.team_score is not None:
        season = sum(p.season_line.get(PTS, 0.0) for p, _ in playing if p.games_played >= MIN_TEAM_GAMES)
        tl = team_line(sum(ln.pts for _, ln in playing), season, m.team_points, m.team_score)
    return SideLines(playing, out, tl)


def _at(box: Box, p: Player, team: int, day: int) -> Player:
    return replace(p, game_context=context(box, p.id, team, day))

"""Who plays tonight, in Our call (gamescore.py): the lineup's talent and the regulars missing.

Can 2026-11-21, after a night when calls missed: is a team's talent (not only its results so far)
worth a weight, and who is out tonight? Team ratings (gamescore.py) are built from results, so they
trail a lineup that is better or worse than its record, and they only slowly learn that a regular is
missing. Two pieces, tested blind on every game of five seasons (2021-22 to 2025-26) and on 2026-27
so far, fitted with the season left out and again on earlier seasons only:

  * lineup talent: what tonight's players scored a minute last season, times the minutes each has
    played lately (his last LINEUP_FORM games), scaled to a full game's 240 minutes. Against the
    call, after the season's usual gap between the two (the talent scale is last season's league,
    the call is this one's), about a tenth of the gap shows up in the score. Players without a last
    season (rookies, deep bench) count at the REPLACEMENT_PCT percentile of last season's rates;
  * regulars missing: for each regular (REGULAR_MINUTES+ over his last LINEUP_FORM games, played
    one of his team's last REGULAR_RECENT games), whether he plays tonight against how often he
    played in the games the team's ratings stand on (decayed like them), times his points a game.
    A 25-point scorer out who usually plays: about 1 point fewer.

Both passed every season for a team's points and the margin; together the squared miss of the
margin falls 1.6-3.7% a season. Who is out tonight is ESPN's OUT tag as it stood before tip-off
(app/pregame.py); without tags every regular counts as playing.

The talent piece needs the season's usual gap, so it counts once the season has TALENT_MIN_GAMES
team games (about the first week without it).
"""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import numpy as np

TALENT = 0.107  # points tonight per point of lineup talent over the call (after the season's usual gap)
ABSENT = 0.036  # points tonight per point a game of regulars playing (+) or missing (-) against usual
TALENT_MIN_GAMES = 60  # team games this season before the usual gap is known
LINEUP_FORM = 10  # his last this many games: minutes and points a game
LINEUP_RECENT = 10  # in the lineup if he played one of his team's last this many games
REGULAR_MINUTES = 15.0
REGULAR_RECENT = 15  # a regular played one of his team's last this many games
REPLACEMENT_PCT = 25  # percentile of last season's points a minute for players without one
FULL_GAME = 240.0
MIN_LINEUP_MINUTES = 150.0  # lineups whose recent minutes add up to less say nothing


@dataclass
class Lineups:
    """Each team's games this season player by player, and what each player scored a minute last season."""

    games: dict[int, dict[int, list[tuple[int, float, float]]]] = field(
        default_factory=dict
    )  # team -> player -> (day, min, pts)
    team_days: dict[int, list[int]] = field(default_factory=dict)  # team -> days it played, in order
    rate: dict[int, float] = field(default_factory=dict)  # player -> points a minute last season
    floor: float = 0.0  # points a minute for a player without a last season
    played: dict[tuple[int, int], tuple[int, list[int]]] = field(default_factory=dict)  # (game, team) -> day, players

    def _form(self, team: int, player: int, day: int) -> tuple[float, float, int] | None:
        """His minutes and points a game over his last LINEUP_FORM games for ``team`` before ``day``,
        and the day of the last one."""
        g = self.games.get(team, {}).get(player, [])
        i = bisect_left(g, (day,))
        last = g[max(0, i - LINEUP_FORM) : i]
        if not last:
            return None
        return sum(x[1] for x in last) / len(last), sum(x[2] for x in last) / len(last), last[-1][0]

    def _before(self, team: int, day: int) -> list[int]:
        d = self.team_days.get(team, [])
        return d[: bisect_left(d, day)]

    def rotation(self, team: int, day: int) -> list[int]:
        """Players who played one of the team's last LINEUP_RECENT games before ``day``."""
        recent = self._before(team, day)[-LINEUP_RECENT:]
        if not recent:
            return []
        out = []
        for pid in self.games.get(team, {}):
            f = self._form(team, pid, day)
            if f is not None and f[2] >= recent[0]:
                out.append(pid)
        return out

    def talent(self, team: int, day: int, players: Iterable[int]) -> float | None:
        """Points a full game from ``players``' last-season rates at their recent minutes."""
        total = minutes = 0.0
        for pid in players:
            f = self._form(team, pid, day)
            if f is None:
                continue
            total += self.rate.get(pid, self.floor) * f[0]
            minutes += f[0]
        if minutes < MIN_LINEUP_MINUTES:
            return None
        return total * FULL_GAME / minutes

    def absent(self, team: int, day: int, out: set[int], half_life: float) -> float:
        """Points a game of regulars playing tonight against how often they played lately (- when missing)."""
        days = self._before(team, day)
        if len(days) < 3:
            return 0.0
        recent = days[-REGULAR_RECENT:]
        w = 0.5 ** ((len(days) - 1 - np.arange(len(days))) / half_life)
        total = 0.0
        for pid, g in self.games.get(team, {}).items():
            f = self._form(team, pid, day)
            if f is None or f[0] < REGULAR_MINUTES or f[2] < recent[0]:
                continue
            mine = {x[0] for x in g}
            share = float(np.sum(w * np.array([d in mine for d in days])) / w.sum())
            total += ((0.0 if pid in out else 1.0) - share) * f[1]
        return total


def build(rows: Iterable, prior: dict[int, dict[int, float]], pts: int, minutes: int) -> Lineups:
    """Lineups from this season's game rows (gamebox.GameRow) and last season's per-minute lines."""
    games: dict[int, dict[int, list[tuple[int, float, float]]]] = defaultdict(lambda: defaultdict(list))
    days: dict[int, set[int]] = defaultdict(set)
    played: dict[tuple[int, int], tuple[int, list[int]]] = {}
    for r in rows:
        games[r.team][r.player].append((r.day, r.line[minutes], r.line[pts]))
        days[r.team].add(r.day)
        played.setdefault((r.game, r.team), (r.day, []))[1].append(r.player)
    rates = {pid: line.get(pts, 0.0) for pid, line in prior.items()}
    return Lineups(
        games={t: {p: sorted(g) for p, g in ps.items()} for t, ps in games.items()},
        team_days={t: sorted(d) for t, d in days.items()},
        rate=rates,
        floor=float(np.percentile(list(rates.values()), REPLACEMENT_PCT)) if rates else 0.0,
        played=played,
    )


def usual_gap(lineups: Lineups, team_games: Iterable, day: int, raw: Callable[[int, int, int], float]) -> float | None:
    """The season's usual gap between lineup talent (who played) and the call, over its games before
    ``day``; None before TALENT_MIN_GAMES. ``raw(day, team, opponent)``: the call without these pieces."""
    gaps = []
    for g in team_games:
        if g.day >= day:
            continue
        got = lineups.played.get((g.game, g.team))
        if got is None:
            continue
        t = lineups.talent(g.team, g.day, got[1])
        if t is not None:
            gaps.append(t - raw(g.day, g.team, g.opponent))
    return float(np.mean(gaps)) if len(gaps) >= TALENT_MIN_GAMES else None

"""How much of the rest of the season a player is expected to play, from his games-played record.

Some players miss a lot of games year after year. A flat availability for everyone
values them as if they were as durable as anyone, which flatters trades for them.

A player's expected share of his team's games is a weighted average of:

- the league's typical share (``LEAGUE_SHARE``), worth ``PRIOR_GAMES`` games of evidence,
  so a short record is pulled toward it;
- each of his last ``HISTORY_SEASONS`` seasons, recent seasons counting more (``DECAY``);
- this season so far: each team game counts ``CURRENT_GAME_WEIGHT`` times a game of an
  old season. A healthy start after injury-hit seasons eases the risk, but only partway:
  in past seasons, injury-prone players who played 9 of their first 10 games still
  played less of the rest than durable players did.

The model plays each player in fewer of his team's games when his expected share is
below a typical regular's (``keep_share``). His missed games are left to the bench,
the same way a known injury is, so the cost is what he adds over who fills in.

The share leaves out an absence that is still running (the injury status covers it), so
the two don't count the same games twice. Every constant is fitted by
``scripts/durability_backtest.py`` on the 2023-24 season and checked on 2024-25 and
2025-26. ESPN's projected games were tried as an input and did worse than the typical
share on their own, so they're left out. News stories ("he put in the work this
summer") can't be measured that way and don't move the number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from app.espn.parse import GAMES_PLAYED, League, Player

HISTORY_SEASONS = 3
FULL_SEASON = 82

# Fitted on 2023-24, checked on 2024-25 and 2025-26 (scripts/durability_backtest.py).
LEAGUE_SHARE = 0.728  # share of his team's games a regular plays once back from any current injury
PRIOR_GAMES = 160.0  # about two seasons: one bad season moves the estimate only partway
DECAY = 0.5  # the season before last counts half as much as last season
CURRENT_GAME_WEIGHT = 5.0
PRONE_SHARE = 0.65  # below this, labelled injury-prone: about the least durable fifth of regulars


@dataclass(frozen=True)
class PastSeason:
    played: int
    games: int  # games in that NBA season (82, or fewer in a shortened one)


def expected_share(
    past: list[PastSeason | None],
    played: int = 0,
    team_games: int = 0,
    *,
    league_share: float = LEAGUE_SHARE,
    prior_games: float = PRIOR_GAMES,
    decay: float = DECAY,
    current_weight: float = CURRENT_GAME_WEIGHT,
) -> float:
    """Expected share (0-1) of his team's remaining games a player plays.

    ``past`` holds his last seasons, most recent first (None: no record that season);
    ``played`` and ``team_games`` are his games and his team's games so far this season.
    """
    num, den = league_share * prior_games, prior_games
    for k, s in enumerate(past[:HISTORY_SEASONS]):
        if s is None or s.games <= 0:
            continue
        w = decay**k * FULL_SEASON
        num += w * min(s.played / s.games, 1.0)
        den += w
    base = num / den
    team_games = max(team_games, played)
    return (base * den + current_weight * played) / (den + current_weight * team_games)


def games_played(players: list[dict], season: int) -> dict[int, int]:
    """Games each player played in one season, from ESPN's season totals."""
    out = {}
    for e in players:
        p = e.get("player") or e
        for st in p.get("stats") or []:
            if st.get("seasonId") != season or st.get("statSourceId") != 0 or st.get("statSplitTypeId") != 0:
                continue
            gp = (st.get("stats") or {}).get(str(GAMES_PLAYED))
            if gp is not None and "id" in p:
                out[int(p["id"])] = int(gp)
    return out


def season_length(played: dict[int, int]) -> int:
    """Games in that NBA season: the most anyone played, capped at 82 (traded players can top it)."""
    return min(max(played.values(), default=FULL_SEASON), FULL_SEASON)


def attach(league: League, past: list[dict[int, int]]) -> None:
    """Sets every player's expected share of his team's remaining games.

    ``past`` holds games played in the last seasons, most recent first.
    """
    lengths = [season_length(gp) for gp in past]
    cal = league.calendar
    players = [p for t in league.teams for p in t.roster] + list(league.free_agents)
    for p in players:
        record = [PastSeason(gp[p.id], n) if p.id in gp else None for gp, n in zip(past, lengths, strict=True)]
        team_games = sum(d < cal.today for d in cal.team_days.get(p.pro_team_id, ())) if cal else 0
        # Without the calendar, this season's games can't be put against his team's.
        played = min(p.games_played, team_games)
        p.durability = expected_share(record, played, team_games)


def keep_share(p: Player) -> float:
    """Share of his team's games the model plays him in, on top of known injuries.

    A typical regular (``LEAGUE_SHARE``) or better plays every game: the flat
    availability in the model already covers the absences everyone has.
    """
    if p.durability is None:
        return 1.0
    return min(p.durability / LEAGUE_SHARE, 1.0)


def kept(index: int, share: float, phase: float) -> bool:
    """Whether a player plays his team's game number ``index`` when he plays ``share`` of them.

    Missed games are spread evenly through the season, starting at ``phase`` (0-1), so
    every stretch of games gets its share of them and teammates don't miss the same nights.
    """
    return math.floor((index + 1) * share + phase) > math.floor(index * share + phase)


def injury_prone(p: Player) -> bool:
    return p.durability is not None and p.durability < PRONE_SHARE

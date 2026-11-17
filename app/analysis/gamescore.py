"""A game's expected score: how many points each NBA team should score in its next game.

Built from the 30 team game logs the league box reads (gamebox.py). Each team's points per
possession, the points per possession it allows, and its pace (possessions a game), with recent
games counting most (half as much every HALF_LIFE games back) and last season's end, regressed
halfway to the league, worth PRIOR_GAMES games at the start. Then

    possessions = pace x opponent's pace / league pace
    points      = possessions x offense x opponent's defense / league points per possession

plus what passed on top of that, tested on every game of five seasons (2021-22 to 2025-26) and kept
only when it lowered the squared miss of a team's score in every season, fitted with that season
left out and again on earlier seasons only (scripts/game_score_backtest.py):

  * home court: about 0.9 points more at home, 0.9 fewer away;
  * three-point luck: about 75% of the opponent's 3P% allowed above the league's doesn't last (a
    defense has little say over open threes going in), nor 40% of the team's own 3P% above it;
  * a back-to-back: about 1.1 points fewer.

Replayed day by day, its mean miss is 8.94-9.43 points a team (a team's own average so far:
9.24-9.99, the league average: 9.34-10.30) and it picks the winner in 62-68% of games. Its calls
move 3+ points from the team's average in 39-51% of games, and those moves are the right size in
every season (actual over called change 0.93-1.04: not an average, and not overfitted). Rest beyond
a back-to-back, the opponent's back-to-back, players ruled out (failed 2021-22 only), recent form over
a shorter run and free-throw luck didn't pass.

report.next_game_line uses it as a check on player lines: teammates' lines should add up to it.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.analysis.gamebox import GameRow

# ESPN stat ids (as gamebox's): points, offensive rebounds, turnovers, shots, free throws, threes, minutes
PTS, OREB, TO, FGA, FTM, FTA, TPM, TPA, MINUTES = 0, 4, 11, 14, 15, 16, 17, 18, 40
TEAM_STATS = (OREB, TPA)  # what gamebox.parse_rows keeps for these sums beyond its own stats
HALF_LIFE = 20.0  # team games
PRIOR_GAMES = 8.0  # team games of last season's end (regressed) a team starts the season with
PRIOR_KEEP = 0.5  # share of a team's gap to the league last season it keeps into the new one
LEAGUE_PRIOR_GAMES = 20.0  # team games of last season's league rates the season's league starts with
FULL_MINUTES = 235.0  # player minutes a team's game needs in the logs to count (fewer: rows missing)
FLOOR = 60.0  # a team's points below this mean a broken sum, not a call

# Points on top of the ratings (scripts/game_score_backtest.py, fitted on all five seasons).
LEVEL = 0.540
HOME = 0.925  # at home; minus this away
OPP_THREE_LUCK = -0.757  # x the opponent's 3P% allowed above the league's, in points tonight
OWN_THREE_LUCK = -0.395  # x the team's own 3P% above the league's, in points tonight
BACK_TO_BACK = -1.094

# The sums a team keeps, per game: its points, the points it allows, possessions (both teams'
# average), its threes and free throws made and tried, and the same allowed.
FIELDS = ("pf", "pa", "poss", "tpm", "tpa", "ftm", "fta", "tpm_a", "tpa_a", "ftm_a", "fta_a")
_START = {"pf": 113.0, "pa": 113.0, "poss": 99.0, "tpm": 13.0, "tpa": 36.0, "ftm": 17.0, "fta": 22.0}
_START |= {"tpm_a": 13.0, "tpa_a": 36.0, "ftm_a": 17.0, "fta_a": 22.0}


@dataclass(frozen=True)
class TeamGame:
    """One team's side of a finished game, summed from its players' rows."""

    game: int
    day: int
    team: int
    opponent: int
    stats: dict[str, float]  # FIELDS


def team_games(rows: list[GameRow]) -> list[TeamGame]:
    """Each complete game twice, once from each side, in day order. Games where either side's
    player minutes come up short (rows missing from the logs) are left out."""
    sums: dict[tuple[int, int], dict[int, float]] = defaultdict(lambda: defaultdict(float))
    days: dict[int, int] = {}
    teams: dict[int, set[int]] = defaultdict(set)
    seen: set[tuple[int, int]] = set()
    for r in rows:
        if (r.game, r.player) in seen:  # a player can be in two teams' logs after a trade
            continue
        seen.add((r.game, r.player))
        s = sums[(r.game, r.team)]
        for k in (PTS, FGA, FTA, FTM, TPM, TPA, OREB, TO, MINUTES):
            s[k] += r.line.get(k, 0.0)
        days[r.game] = r.day
        teams[r.game].add(r.team)
    out = []
    for game, ts in teams.items():
        if len(ts) != 2:
            continue
        a, b = sorted(ts)
        x, y = sums[(game, a)], sums[(game, b)]
        if x[MINUTES] < FULL_MINUTES or y[MINUTES] < FULL_MINUTES:
            continue
        poss = (_possessions(x) + _possessions(y)) / 2
        for t, o, me, them in ((a, b, x, y), (b, a, y, x)):
            stats = {"pf": me[PTS], "pa": them[PTS], "poss": poss, "tpm": me[TPM], "tpa": me[TPA]}
            stats |= {"ftm": me[FTM], "fta": me[FTA], "tpm_a": them[TPM], "tpa_a": them[TPA]}
            stats |= {"ftm_a": them[FTM], "fta_a": them[FTA]}
            out.append(TeamGame(game, days[game], t, o, stats))
    out.sort(key=lambda g: (g.day, g.game, g.team))
    return out


def _possessions(s: dict[int, float]) -> float:
    return s[FGA] + 0.44 * s[FTA] - s[OREB] + s[TO]


@dataclass
class SeasonEnd:
    """Where a season ended: each team's rates a game and the league's, the next season's start."""

    teams: dict[int, dict[str, float]]
    league: dict[str, float]


@dataclass
class Scores:
    """Every team's decayed sums so far (weight first), and the league's sums and team games."""

    teams: dict[int, dict[str, float]] = field(default_factory=dict)
    league: dict[str, float] = field(default_factory=dict)
    games: float = 0.0
    start: SeasonEnd | None = None

    def rate(self, key: str) -> float:
        """The league's rate a team game; the start's while there is none."""
        v = self.league.get(key, 0.0) / self.games if self.games else 0.0
        return v if v > 0 else _START[key]

    def team(self, team: int) -> dict[str, float]:
        return self.teams.get(team) or _start_of(team, self.start)


def build(games: list[TeamGame], start: SeasonEnd | None = None) -> Scores:
    """The season so far from ``games`` (team_games, in day order), starting from last season's end."""
    decay = 0.5 ** (1 / HALF_LIFE)
    s = Scores(start=start)
    if start is not None:
        s.league = {k: LEAGUE_PRIOR_GAMES * v for k, v in start.league.items()}
        s.games = LEAGUE_PRIOR_GAMES
    for g in games:
        t = s.teams.get(g.team)
        if t is None:
            t = s.teams[g.team] = _start_of(g.team, start)
        t["w"] = t["w"] * decay + 1
        for k in FIELDS:
            t[k] = t[k] * decay + g.stats[k]
            s.league[k] = s.league.get(k, 0.0) + g.stats[k]
        s.games += 1
    return s


def _start_of(team: int, start: SeasonEnd | None) -> dict[str, float]:
    """A team's sums before its first game: last season's end regressed to the league, PRIOR_GAMES of it."""
    if start is None:
        base = {k: _START[k] for k in FIELDS}
    else:
        last, lg = start.teams.get(team), start.league
        base = {k: lg[k] + PRIOR_KEEP * (last[k] - lg[k]) if last else lg[k] for k in FIELDS}
    return {"w": PRIOR_GAMES, **{k: PRIOR_GAMES * v for k, v in base.items()}}


def season_end(games: list[TeamGame]) -> SeasonEnd:
    """A finished season's end (build without a start), each team's decayed rates a game."""
    s = build(games)
    return SeasonEnd(
        teams={t: {k: v[k] / v["w"] for k in FIELDS} for t, v in s.teams.items()},
        league={k: s.rate(k) for k in FIELDS},
    )


def signals(s: Scores, team: int, opponent: int) -> dict[str, float]:
    """The ratings' points for ``team`` against ``opponent`` and the luck terms, before the weights."""
    a, b = s.team(team), s.team(opponent)
    lg_poss, lg_pf = s.rate("poss"), s.rate("pf")
    lg_ppp = lg_pf / lg_poss
    lg3 = s.rate("tpm") / s.rate("tpa")
    possessions = (a["poss"] / a["w"]) * (b["poss"] / b["w"]) / lg_poss
    offense, defense = a["pf"] / a["poss"], b["pa"] / b["poss"]
    return {
        "ratings": possessions * offense * defense / lg_ppp,
        "opp_three_luck": 3 * (b["tpm_a"] - lg3 * b["tpa_a"]) / b["poss"] * possessions,
        "own_three_luck": 3 * (a["tpm"] - lg3 * a["tpa"]) / a["poss"] * possessions,
        "average": a["pf"] / a["w"],
    }


def expected(s: Scores, team: int, opponent: int, home: bool | None, back_to_back: bool) -> float:
    """Points ``team`` should score against ``opponent``. ``home`` None: unknown (neutral)."""
    x = signals(s, team, opponent)
    side = 0.0 if home is None else (1.0 if home else -1.0)
    return (
        x["ratings"]
        + LEVEL
        + HOME * side
        + OPP_THREE_LUCK * x["opp_three_luck"]
        + OWN_THREE_LUCK * x["own_three_luck"]
        + (BACK_TO_BACK if back_to_back else 0.0)
    )

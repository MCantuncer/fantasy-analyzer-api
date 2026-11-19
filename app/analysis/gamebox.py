"""The game itself in a player's next-game line: who he plays, where, and how he has been going.

Today's line (report.next_game_line) is his projection at his recent minutes. Tested on every game
of the ~500 most-owned players in five seasons (2021-22 to 2025-26, 90,410 games), a few things
known before tip-off move it closer to what he then does. Each was kept only if it lowered the
squared miss in every season, fitted with that season left out and again on earlier seasons only.
Replaying this code day by day (scripts/next_game_matchup_backtest.py), the squared miss against
today's line falls in every season by 3.0-4.5% for points, 1.8-4.8% for rebounds, 2.9-4.1% for
assists, 1.0-1.7% for steals, 1.0-1.6% for blocks, 1.6-2.6% for threes, 1.4-2.3% for turnovers,
0.4-0.8% for FG impact, 0.1-0.6% for FT impact and 6.0-7.4% for minutes (mean points miss 4.91 ->
4.81; season average 4.96):

  * his form: every game he played, last season's too, counting half as much each FORM_DECAY
    half-life further back, per minute against his line, trusted by how many of that stat his line
    expects over those minutes (a few expected steals mean little). Most of the gap lasts (weights
    0.6-1.0): the line's rates trail what he does lately. Blocks and shots keep his last FORM_GAMES;
  * his minutes: lately (decayed), in his last game and over a longer run; whether he started his
    last game against how often he has started lately (a new starter plays up to about 2 more minutes,
    a benched one up to about 2 fewer); games he just missed, and his first three games back;
  * the opponent's defense in each stat: what the players it faced made against what they usually
    make per minute (DEFENSE_PRIOR_GAMES of league average to start). Raw "points allowed" is half
    as good, and splitting it by guards, wings and bigs, or by his own mix of shots, added nothing;
    free throws follow the opponent's free throws allowed;
  * teammates ruled out: their shots and points a game (more points, assists and turnovers a minute),
    on top of the minutes they leave (tonight.py), and more minutes for bench players;
  * home court (assists, FG%), how much his team is favored (blocks), a back-to-back (steals).

His history against the team, pace, travel, altitude, tip time, rest beyond a back-to-back, the
opponent's injuries, playing a former team, blowout risk for minutes, matchups of his shot or
rebound mix against the opponent's, and a single game's FG% or FT% form didn't pass.

Each change is kept as a reason (kind, the minutes and per-stat change, the words), so the player
card can say why the line moved.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass, field, replace

import numpy as np

from app.analysis import gamescore

PTS, BLK, STL, AST, REB, TO, FGM, FGA, FTM, FTA, TPM = 0, 1, 2, 3, 6, 11, 13, 14, 15, 16, 17
MINUTES, GS, GP = 40, 41, 42  # GS: started the game
COUNTING = (PTS, REB, AST, STL, BLK, TPM, TO, FGM, FGA, FTM, FTA)
GAME_SPLIT = 5

PRIOR_MINUTES = 200.0  # minutes of his last season (or the league's rate) in what he usually makes a minute
MIN_PRIOR_MINUTES = 200.0  # last season counts only with this many minutes
DEFENSE_PRIOR_GAMES = 8.0  # team games of league-average defense a team's own record is blended with
MARGIN_PRIOR_GAMES = 8.0  # team games of an even record in a team's margin a game
HOME_EDGE = 2.5  # points of home court in the expected margin
FORM_GAMES = 10  # blocks and shots: his last this many games
FREED = (REB, AST, BLK)  # what a teammate ruled out leaves behind beyond his shots (RATE_WEIGHTS "freed")
TEAM_MINUTES = 240.0  # player minutes of a team in a game
MISSED_CAP = 10
BENCH_MINUTES = 36.0  # starters' minutes; below it a player gains minutes when teammates sit
HEAVY_MINUTES = 36.0  # lines above this come in lower
MAX_MINUTES = 44.0
FORM_RANGE = (-1.0, 2.0)

# Form, decayed: every game he played counts half as much each HALF-LIFE games further back, last
# season's included (carry()), and the gap to his line's rate is trusted by E / (E + SHRINK), E being
# what his line's rate expects over those (decayed) minutes: a few expected steals mean little.
#   stat: (half-life in games, SHRINK in expected events)
FORM_DECAY: dict[int, tuple[float, float]] = {
    PTS: (12.0, 1.0),
    REB: (20.0, 8.0),
    AST: (35.0, 1.0),
    STL: (35.0, 16.0),
    TPM: (35.0, 1.0),
    TO: (20.0, 8.0),
}
# The same, over this season only with a short half-life: a quick turn on top of the slow one.
FAST_DECAY: dict[int, tuple[float, float]] = {AST: (5.0, 8.0), TPM: (5.0, 2.0)}
RECENT_HALF_LIFE = 5.0  # games: his minutes lately (this season)
LONG_HALF_LIFE = 12.0  # games: his minutes over a longer run, last season's included
START_HALF_LIFE = 8.0  # games: how often he has started lately (this season)
RAMP = {1: 1.0, 2: 0.66, 3: 0.33}  # 1st, 2nd and 3rd game after missing 2+ team games
RAMP_GAP = 2  # team games missed in a row that start a ramp

# Minutes: the line + MINUTES_LEVEL + weight x signal (MINUTE_WEIGHTS).
MINUTES_LEVEL = -0.7527  # lines run this many minutes over what players play
MISSED_MINUTES = -0.4358  # per team game he just missed, up to MISSED_CAP
RAMP_MINUTES = -0.8149  # x RAMP of his game back
BENCH_SHARE = 0.1457  # per minute of teammates ruled out, x (1 - his minutes / BENCH_MINUTES)
RECENT_MINUTES = 0.298  # per minute his recent games (RECENT_HALF_LIFE) ran over the line
LAST_GAME_MINUTES = 0.0676  # per minute his last game ran over the line
LONG_MINUTES = 0.1336  # per minute his longer run (LONG_HALF_LIFE) ran over the line
ROLE_MINUTES = 0.7156  # x (share of games he started lately - 0.5): starters play more than lines say
SWITCH_MINUTES = 2.126  # x (started his last game - share he started lately): a new starter, or benched
HEAVY_LEVEL = -0.5084  # per minute of line over HEAVY_MINUTES
# Coaches' calls a line can't see in his games so far (scripts/next_game_tuning.py, every season of five):
NEW_TEAM_MINUTES = -4.5  # his first game for a new team this season: eased in
SEASON_END_MINUTES = -2.3  # a core player (CORE_MINUTES+) in his team's last SEASON_END_GAMES games: rested
CORE_MINUTES = 28.0
SEASON_END_GAMES = 3
COACH_SAT_MINUTES = -2.2  # he sat a game by coach's decision in the last COACH_SAT_DAYS days: still out of favor
COACH_SAT_DAYS = 4
COACH_SAT_LINE = 10.0  # minutes of line it takes: a deep-bench line has little left to lose
MINUTE_WEIGHTS = {
    "missed": MISSED_MINUTES,
    "ramp": RAMP_MINUTES,
    "bench_out": BENCH_SHARE,
    "recent": RECENT_MINUTES,
    "last_game": LAST_GAME_MINUTES,
    "long": LONG_MINUTES,
    "role": ROLE_MINUTES,
    "switch": SWITCH_MINUTES,
    "heavy": HEAVY_LEVEL,
    "new_team": NEW_TEAM_MINUTES,
    "season_end": SEASON_END_MINUTES,
    "coach_sat": COACH_SAT_MINUTES,
}
MINUTE_KINDS = {
    "missed": "back_from_absence",
    "ramp": "back_from_absence",
    "bench_out": "teammates_out",
    "recent": "recent_minutes",
    "last_game": "recent_minutes",
    "long": "recent_minutes",
    "switch": "starter_switch",
    "heavy": "heavy_minutes",
    "new_team": "new_team",
    "season_end": "season_end",
    "coach_sat": "coach_decision",
}  # role: part of his usual line, not a reason

# Per minute: his line's rate x (1 + sum of weight x signal) + level, per stat.
#   form: decayed (FORM_DECAY) per minute over his line's rate, -1..2, shrunk; for blocks and shots
#     his last FORM_GAMES games, unshrunk. form_fast: the same over FAST_DECAY
#   opp: the opponent's defense in that stat (+0.05 = players make 5% more against it than usual)
#   usage_out: shots a game (FGA + 0.44 FTA + TO) of teammates ruled out, / 100
#   vac: points a game of teammates ruled out, over his team's points a game
#   freed: the share of his team's rebounds (assists, blocks) a game that teammates ruled out make,
#     less their share of its minutes: a rebounder ruled out leaves more of them than his minutes hold
#   home: +0.5 at home, -0.5 away; margin: expected margin / 10; b2b: 1 on a back-to-back
RATE_WEIGHTS: dict[int, dict[str, float]] = {
    PTS: {"form": 0.7683, "opp": 1.0751, "usage_out": 0.1212, "vac": 0.2043},
    REB: {"form": 0.8233, "opp": 0.8943, "freed": 0.5},
    AST: {"form": 0.4122, "form_fast": 0.5042, "opp": 0.8339, "usage_out": 0.2245, "home": 0.0325, "freed": 0.85},
    STL: {"form": 0.9928, "opp": 0.7837, "b2b": -0.0343},
    BLK: {"opp": 0.893, "margin": 0.0576, "form": 0.1027, "freed": 0.5},
    TPM: {"form": 0.6247, "form_fast": 0.2113, "opp": 0.6939},
    TO: {"form": 0.7926, "opp": 0.8042, "usage_out": 0.2392},
    FGA: {"form": 0.5567, "opp": 1.0615, "usage_out": 0.2991},
    FTA: {"opp": 0.9368},
}
RATE_LEVEL = {
    PTS: -0.1758,
    REB: 0.0019,
    AST: -0.0145,
    STL: 0.0,
    BLK: 0.0113,
    TPM: 0.0127,
    TO: -0.0241,
    FGA: -0.2284,
    FTA: -0.0644,
}
# FG%: his line's FG% + weight x signal (opp: the opponent's FG% defense as a share, home as above).
# FT%: his line's (made free throws follow the attempts).
FG_PCT_WEIGHTS = {"opp": 0.3729, "home": 0.0071}

USAGE_FTA = 0.44
_DECAY_ROW = (MINUTES, GS, *COUNTING)  # what a decayed sum holds, after its weight


@dataclass(frozen=True)
class GameRow:
    player: int
    game: int
    day: int
    team: int
    line: dict[int, float]


def parse_rows(players: list[dict], season: int) -> list[GameRow]:
    """Every game a player played in one season, from ESPN's per-game splits (team game logs)."""
    out = []
    for entry in players:
        p = entry.get("player") or entry
        if "id" not in p:
            continue
        for st in p.get("stats") or []:
            if st.get("statSplitTypeId") != GAME_SPLIT or st.get("statSourceId") or not st.get("externalId"):
                continue
            if st.get("seasonId") not in (None, season):
                continue
            s = st.get("stats") or {}
            line = {k: float(s.get(str(k), 0) or 0) for k in (*COUNTING, MINUTES, GS, *gamescore.TEAM_STATS)}
            if line[MINUTES] <= 0:
                continue
            out.append(
                GameRow(
                    int(p["id"]),
                    int(st["externalId"]),
                    int(st.get("scoringPeriodId") or 0),
                    int(st.get("proTeamId") or p.get("proTeamId") or 0),
                    line,
                )
            )
    return out


def prior_rates(players: list[dict], season: int) -> dict[int, dict[int, float]]:
    """Per-minute line of each player with MIN_PRIOR_MINUTES or more in a season (ESPN season totals)."""
    out = {}
    for entry in players:
        p = entry.get("player") or entry
        for st in p.get("stats") or []:
            if st.get("seasonId") != season or st.get("statSourceId") != 0 or st.get("statSplitTypeId") != 0:
                continue
            s = st.get("stats") or {}
            minutes = float(s.get(str(MINUTES), 0) or 0)
            if minutes >= MIN_PRIOR_MINUTES and "id" in p:
                out[int(p["id"])] = {k: float(s.get(str(k), 0) or 0) / minutes for k in COUNTING}
    return out


@dataclass(frozen=True)
class Form:
    rate: dict[int, float]  # per minute over his last FORM_GAMES games
    minutes: float  # a game, decayed (RECENT_HALF_LIFE), this season
    minutes_long: float  # a game over his last FORM_GAMES games
    usage: float  # FGA + 0.44 FTA + TO a game over his last FORM_GAMES games
    last_day: int  # the day of his last game
    team: int  # his team in that game
    decayed: dict[int, tuple[float, float]] = field(default_factory=dict)  # stat -> (stat, minutes), FORM_DECAY
    fast: dict[int, tuple[float, float]] = field(default_factory=dict)  # stat -> (stat, minutes), FAST_DECAY
    minutes_season: float | None = None  # a game, decayed (LONG_HALF_LIFE), last season's included
    last_minutes: float | None = None  # in his last game
    started: bool | None = None  # his last game
    start_share: float | None = None  # of his games lately (START_HALF_LIFE)
    since_return: int = 0  # games played since missing RAMP_GAP+ team games in a row, while in RAMP


Carry = dict[float, list[float]]  # half-life -> [weight, then _DECAY_ROW], decayed to the end of a season


@dataclass(frozen=True)
class Game:
    day: int
    opponent: int
    home: bool
    tip: int | None = None  # tip-off, epoch ms; None when ESPN hasn't set it


@dataclass
class Box:
    """What every team's games so far say, for next-game lines: built once from all 30 game logs."""

    season: int
    defense: dict[int, dict[int, float]]  # team -> stat -> share more (+) or less its opponents make than usual
    margin: dict[int, float]  # team -> points a game over its opponents, shrunk to even
    form: dict[int, Form]  # player -> his recent games
    team_points: dict[int, float] = field(default_factory=dict)  # team -> points a game
    team_made: dict[int, dict[int, float]] = field(default_factory=dict)  # team -> FREED stat -> a game
    played: dict[int, list[int]] = field(default_factory=dict)  # team -> days it played, in order
    schedule: dict[int, list[Game]] = field(default_factory=dict)  # team -> its games this season, in order
    sat: dict[int, list[int]] = field(default_factory=dict)  # player -> days he sat by coach's decision
    scores: gamescore.Scores | None = None  # every team's scoring so far, for its expected score
    team_games: list[gamescore.TeamGame] = field(default_factory=list, repr=False)  # for calls on past days
    score_start: gamescore.SeasonEnd | None = field(default=None, repr=False)
    _day_scores: dict[int, gamescore.Scores] = field(default_factory=dict, repr=False, compare=False)

    def call(self, day: int, away: int, home: int) -> gamescore.Call:
        """Our call for a game on ``day`` from the games before it: the same before tip-off and after."""
        s = self._day_scores.get(day)
        if s is None:
            s = gamescore.build([g for g in self.team_games if g.day < day], self.score_start)
            self._day_scores[day] = s
        a = gamescore.expected(s, away, home, False, self.back_to_back(away, day))
        h = gamescore.expected(s, home, away, True, self.back_to_back(home, day))
        return gamescore.call(a, h)

    def latest_day(self) -> int:
        """The last day with a game in the box: 0 for a box with no games yet."""
        return max((d[-1] for d in self.played.values() if d), default=0)

    def games_missed(self, player: int, team: int, day: int) -> int:
        """Team games since his last one, before ``day``, up to MISSED_CAP. 0 without games of his."""
        f = self.form.get(player)
        if f is None or f.team != team:
            return 0
        return min(sum(1 for d in self.played.get(team, []) if f.last_day < d < day), MISSED_CAP)

    def next_game(self, team: int, today: int) -> Game | None:
        return next((g for g in self.schedule.get(team, []) if g.day >= today), None)

    def back_to_back(self, team: int, day: int) -> bool:
        return any(g.day == day - 1 for g in self.schedule.get(team, []))


def parse_schedule(pro_schedule: dict) -> dict[int, list[Game]]:
    """Each NBA team's games from ESPN's pro schedule: day, opponent, home or away, tip-off."""
    out: dict[int, dict[int, Game]] = defaultdict(dict)
    for team in (pro_schedule.get("settings") or {}).get("proTeams", []):
        for day, games in (team.get("proGamesByScoringPeriod") or {}).items():
            for g in games or []:
                home, away = int(g.get("homeProTeamId") or 0), int(g.get("awayProTeamId") or 0)
                if home and away:
                    tip = None if g.get("startTimeTBD") or not g.get("date") else int(g["date"])
                    out[home][int(day)] = Game(int(day), away, True, tip)
                    out[away][int(day)] = Game(int(day), home, False, tip)
    return {t: [games[d] for d in sorted(games)] for t, games in out.items()}


def _decay_weights(n: int, half_life: float) -> np.ndarray:
    """Weights of n games in order, the last one 1."""
    return 0.5 ** (np.arange(n - 1, -1, -1) / half_life)


def _decay_rows(games: list[GameRow]) -> np.ndarray:
    return np.array([[r.line.get(k, 0.0) for k in _DECAY_ROW] for r in games], dtype=float).reshape(-1, len(_DECAY_ROW))


def _half_lives() -> set[float]:
    return {h for h, _ in FORM_DECAY.values()} | {LONG_HALF_LIFE}


def _this_season_half_lives() -> set[float]:
    return {RECENT_HALF_LIFE, START_HALF_LIFE} | {h for h, _ in FAST_DECAY.values()}


def carry(rows: list[GameRow]) -> dict[int, Carry]:
    """Each player's decayed sums at the end of a season (last season's rows), for the form and the
    longer minutes run of the next one. Small: a few numbers a player."""
    by_player: dict[int, list[GameRow]] = defaultdict(list)
    for r in rows:
        by_player[r.player].append(r)
    out = {}
    for pid, games in by_player.items():
        games.sort(key=lambda r: r.day)
        x = _decay_rows(games)
        out[pid] = {}
        for h in _half_lives():
            w = _decay_weights(len(games), h)
            out[pid][h] = [float(w.sum()), *(w @ x).tolist()]
    return out


def _decayed(x: np.ndarray, half_life: float, before: Carry | None) -> np.ndarray:
    """[weight, then _DECAY_ROW] over this season's games ``x``, plus last season's carried sums."""
    w = _decay_weights(len(x), half_life)
    out = np.concatenate([[w.sum()], w @ x])
    if before and half_life in before:
        out += np.array(before[half_life]) * 0.5 ** (len(x) / half_life)
    return out


def build(
    season: int,
    rows: list[GameRow],
    prior: dict[int, dict[int, float]],
    schedule: dict[int, list[Game]],
    last_season: dict[int, Carry] | None = None,
    score_start: gamescore.SeasonEnd | None = None,
    sat: dict[int, list[int]] | None = None,
) -> Box:
    """The box from every game row so far: defenses, margins and each player's recent games.
    ``last_season``: carry() of last season's rows, for form that runs across the summer;
    ``score_start``: gamescore.season_end of last season's rows, where teams' scoring starts;
    ``sat``: the days each player sat a game by coach's decision (ESPN's box scores, app/benched.py)."""
    last_season = last_season or {}
    # League rate a minute, for players without a last season to start from.
    total_min = sum(r.line[MINUTES] for r in rows) or 1.0
    league = {k: sum(r.line[k] for r in rows) / total_min for k in COUNTING}
    teams_in: dict[int, set[int]] = defaultdict(set)
    for r in rows:
        teams_in[r.game].add(r.team)
    # What each player usually makes a minute before each game: his games before it this season,
    # plus PRIOR_MINUTES of last season (or the league's rate).
    by_player: dict[int, list[GameRow]] = defaultdict(list)
    for r in rows:
        by_player[r.player].append(r)
    faced: dict[tuple[int, int], list[list[float]]] = {}  # (game, defense) -> [actual, expected] per stat
    form: dict[int, Form] = {}
    col = {k: 1 + i for i, k in enumerate(_DECAY_ROW)}  # column in a decayed sum (0: weight)
    for pid, games in by_player.items():
        games.sort(key=lambda r: r.day)
        start = prior.get(pid, league)
        cum = dict.fromkeys(COUNTING, 0.0)
        cum_min = 0.0
        for r in games:
            opp = next((t for t in teams_in[r.game] if t != r.team), None)
            if opp is not None:
                acc = faced.setdefault((r.game, opp), [[0.0] * len(COUNTING), [0.0] * len(COUNTING)])
                for i, k in enumerate(COUNTING):
                    usual = (cum[k] + PRIOR_MINUTES * start[k]) / (cum_min + PRIOR_MINUTES)
                    acc[0][i] += r.line[k]
                    acc[1][i] += usual * r.line[MINUTES]
            for k in COUNTING:
                cum[k] += r.line[k]
            cum_min += r.line[MINUTES]
        last = games[-FORM_GAMES:]
        minutes = sum(r.line[MINUTES] for r in last)
        x = _decay_rows(games)
        before = last_season.get(pid)
        sums = {h: _decayed(x, h, before) for h in _half_lives()}
        this = {h: _decayed(x, h, None) for h in _this_season_half_lives()}
        long = sums[LONG_HALF_LIFE]
        form[pid] = Form(
            rate={k: sum(r.line[k] for r in last) / minutes for k in COUNTING},
            minutes=float(this[RECENT_HALF_LIFE][col[MINUTES]] / this[RECENT_HALF_LIFE][0]),
            minutes_long=minutes / len(last),
            usage=sum(r.line[FGA] + USAGE_FTA * r.line[FTA] + r.line[TO] for r in last) / len(last),
            last_day=games[-1].day,
            team=games[-1].team,
            decayed={k: (float(sums[h][col[k]]), float(sums[h][col[MINUTES]])) for k, (h, _) in FORM_DECAY.items()},
            fast={k: (float(this[h][col[k]]), float(this[h][col[MINUTES]])) for k, (h, _) in FAST_DECAY.items()},
            minutes_season=float(long[col[MINUTES]] / long[0]),
            last_minutes=games[-1].line[MINUTES],
            started=games[-1].line.get(GS, 0.0) > 0,
            start_share=float(this[START_HALF_LIFE][col[GS]] / this[START_HALF_LIFE][0]),
        )
    # Defense: what all its opponents made over what they usually make, with DEFENSE_PRIOR_GAMES
    # average team games blended in.
    per_team: dict[int, list[list[float]]] = {}
    for (_, team), (act, exp) in faced.items():
        acc = per_team.setdefault(team, [[0.0] * len(COUNTING), [0.0] * len(COUNTING)])
        for i in range(len(COUNTING)):
            acc[0][i] += act[i]
            acc[1][i] += exp[i]
    n_games = max(len(faced), 1)
    mean_exp = [sum(exp[i] for _, exp in faced.values()) / n_games for i in range(len(COUNTING))]
    defense = {}
    for team, (act, exp) in per_team.items():
        defense[team] = {
            k: (act[i] + DEFENSE_PRIOR_GAMES * mean_exp[i]) / (exp[i] + DEFENSE_PRIOR_GAMES * mean_exp[i]) - 1
            for i, k in enumerate(COUNTING)
            if mean_exp[i] > 0
        }
    # Margin and points a game, and the days each team played.
    points: dict[tuple[int, int], float] = defaultdict(float)
    made: dict[tuple[int, int], dict[int, float]] = defaultdict(lambda: dict.fromkeys(FREED, 0.0))
    days: dict[int, int] = {}
    for r in rows:
        points[(r.game, r.team)] += r.line[PTS]
        for k in FREED:
            made[(r.game, r.team)][k] += r.line[k]
        days[r.game] = r.day
    margin_sum: dict[int, float] = defaultdict(float)
    points_sum: dict[int, float] = defaultdict(float)
    made_sum: dict[int, dict[int, float]] = defaultdict(lambda: dict.fromkeys(FREED, 0.0))
    played: dict[int, set[int]] = defaultdict(set)
    for game, teams in teams_in.items():
        if len(teams) != 2:
            continue
        a, b = sorted(teams)
        diff = points[(game, a)] - points[(game, b)]
        margin_sum[a] += diff
        margin_sum[b] -= diff
        points_sum[a] += points[(game, a)]
        points_sum[b] += points[(game, b)]
        for t in (a, b):
            for k in FREED:
                made_sum[t][k] += made[(game, t)][k]
        played[a].add(days[game])
        played[b].add(days[game])
    margin = {t: margin_sum[t] / (len(played[t]) + MARGIN_PRIOR_GAMES) for t in played}
    team_days = {t: sorted(d) for t, d in played.items()}
    # Games back from an absence: walk his games, counting his team's games he missed in between.
    for pid, games in by_player.items():
        k, prev = 0, None
        for r in games:
            if prev is None or prev.team != r.team:
                k = 0  # his first game, or for a new team: no games of his there to have missed
            else:
                tdays = team_days.get(r.team, [])
                missed = bisect_left(tdays, r.day) - bisect_right(tdays, prev.day)
                k = 1 if missed >= RAMP_GAP else (k + 1 if k else 0)
            prev = r
        if k in RAMP:
            form[pid] = replace(form[pid], since_return=k)
    return Box(
        season=season,
        defense=defense,
        margin=margin,
        form=form,
        team_points={t: points_sum[t] / len(played[t]) for t in played},
        team_made={t: {k: v / len(played[t]) for k, v in made_sum[t].items()} for t in played},
        played=team_days,
        schedule=schedule,
        scores=gamescore.build(games := gamescore.team_games(rows), score_start),
        team_games=games,
        score_start=score_start,
        sat=sat or {},
    )


@dataclass(frozen=True)
class Teammate:
    """A teammate ruled out of the game (tonight.ruled_out)."""

    name: str
    minutes: float  # a game, lately
    usage: float  # FGA + 0.44 FTA + TO a game, lately
    points: float = 0.0  # a game, lately
    made: dict[int, float] = field(default_factory=dict)  # his rebounds, assists and blocks (FREED) a game, lately


@dataclass(frozen=True)
class GameContext:
    """His next game as the box sees it."""

    day: int
    opponent: int
    home: bool
    back_to_back: bool
    margin: float  # his team's expected margin, points
    defense: dict[int, float]  # the opponent's, per stat
    form: Form | None
    games_missed: int
    tip: int | None = None  # tip-off, epoch ms
    team_points: float = 0.0  # his team's points a game so far
    team_made: dict[int, float] = field(default_factory=dict)  # its rebounds, assists and blocks (FREED) a game
    game_back: int = 0  # 1, 2, 3: his first, second, third game after missing RAMP_GAP+ team games
    team_score: float | None = None  # points his team should score (gamescore.expected)
    new_team: bool = False  # his first game for this team after playing for another one this season
    games_left: int | None = None  # his team's games left this season, this one included
    sat_ago: int | None = None  # days since he last sat a game by coach's decision (none: never, or not known)


def context(box: Box, player: int, team: int, today: int) -> GameContext | None:
    game = box.next_game(team, today)
    if game is None:
        return None
    margin = box.margin.get(team, 0.0) - box.margin.get(game.opponent, 0.0) + (HOME_EDGE if game.home else -HOME_EDGE)
    f = box.form.get(player)
    new_team = f is not None and f.team != team
    f = f if f is not None and f.team == team else None
    missed = box.games_missed(player, team, game.day)
    sat = [d for d in box.sat.get(player, ()) if d < game.day]
    back = 1 if missed >= RAMP_GAP else (f.since_return + 1 if f is not None and f.since_return else 0)
    return GameContext(
        day=game.day,
        opponent=game.opponent,
        home=game.home,
        back_to_back=box.back_to_back(team, game.day),
        margin=margin,
        defense=box.defense.get(game.opponent, {}),
        form=f,
        games_missed=missed,
        tip=game.tip,
        team_points=box.team_points.get(team, 0.0),
        team_made=box.team_made.get(team, {}),
        game_back=back if back in RAMP else 0,
        team_score=expected_score(box, team, game),
        new_team=new_team,
        games_left=sum(1 for g in box.schedule.get(team, []) if g.day >= game.day),
        sat_ago=game.day - max(sat) if sat else None,
    )


def week_contexts(box: Box, player: int, team: int, days: list[int]) -> tuple[GameContext, ...]:
    """His team's games on ``days`` (the days left of a matchup week), each as ``context`` sees it today."""
    playing = {g.day for g in box.schedule.get(team, [])}
    out = []
    for day in days:
        if day in playing:
            m = context(box, player, team, day)
            if m is not None and m.day == day:
                out.append(m)
    return tuple(out)


def week_line(line: dict[int, float], games: tuple[GameContext, ...] | list[GameContext]) -> dict[int, float]:
    """A week's per-game line built game by game: ``line`` (his line this week, the same for every
    game) plus what ``apply`` moves it by in each of ``games``, averaged: the opponent, home court, his
    form, his minutes lately. Only what moves players apart: ``apply``'s levels stay out
    (``level_line``), so a week's totals keep the level they had. What belongs to one game only (a
    game he just missed, his first games back, a coach who just sat him, a first game for a new team)
    counts in the first; teammates ruled out stay with ``line``, which holds the minutes they leave
    him. Free throws keep ``line``'s: game by game didn't beat it there
    (scripts/week_lines_backtest.py). ``line`` itself without games or minutes."""
    if not games or line.get(MINUTES, 0.0) <= 0:
        return line
    level = level_line(line)
    moved: dict[int, float] = defaultdict(float)
    for i, m in enumerate(games):
        if i:
            m = replace(m, games_missed=0, game_back=0, sat_ago=None, new_team=False)
        new = apply(line, m, [])[0]
        for k in (MINUTES, *COUNTING):
            moved[k] += new.get(k, 0.0) - level.get(k, 0.0)
    out = dict(line)
    for k, v in moved.items():
        if k not in (FTM, FTA):
            out[k] = max(line.get(k, 0.0) + v / len(games), 0.0)
    return out


def level_line(line: dict[int, float]) -> dict[int, float]:
    """``line`` through ``apply`` with nothing in it but the levels (MINUTES_LEVEL, RATE_LEVEL): what
    a game's line is measured against to see what the game, his form and his role moved."""
    minutes = line.get(MINUTES, 0.0)
    if minutes <= 0:
        return line
    plain = min(max(minutes + MINUTES_LEVEL, 0.0), MAX_MINUTES)
    out = dict(line)
    out[MINUTES] = plain
    for k in COUNTING:
        out[k] = max(line.get(k, 0.0) / minutes * plain + RATE_LEVEL.get(k, 0.0), 0.0)
    fga = line.get(FGA, 0.0)
    if fga > 0:
        out[FGM] = out[FGA] * line.get(FGM, 0.0) / fga
    out[FTM] = out[FTA] * ft_line_pct(line)
    return out


def expected_score(box: Box, team: int, game: Game) -> float | None:
    """Points ``team`` should score in ``game`` (gamescore.expected), None before the box has scores."""
    if box.scores is None:
        return None
    return gamescore.expected(box.scores, team, game.opponent, game.home, box.back_to_back(team, game.day))


@dataclass
class Reason:
    """One thing that moved his line: the minutes it moved and each stat (stat id -> change)."""

    kind: str  # opponent, form, teammates_out, back_from_absence, heavy_minutes, recent_minutes,
    # starter_switch, new_team, season_end, coach_decision, home, away, favored, underdog, back_to_back
    minutes: float = 0.0
    stats: dict[int, float] = field(default_factory=dict)
    detail: dict[str, float | str] = field(default_factory=dict)

    def add(self, stat: int, change: float) -> None:
        self.stats[stat] = self.stats.get(stat, 0.0) + change


def minute_signals(minutes: float, m: GameContext, out: list[Teammate]) -> dict[str, float]:
    """What moves his minutes (MINUTE_WEIGHTS), for a line of ``minutes``."""
    f = m.form
    sig = dict.fromkeys(MINUTE_WEIGHTS, 0.0)
    sig["missed"] = float(m.games_missed)
    sig["ramp"] = RAMP.get(m.game_back, 0.0)
    sig["bench_out"] = sum(t.minutes for t in out) * max(0.0, 1 - minutes / BENCH_MINUTES)
    sig["heavy"] = max(0.0, minutes - HEAVY_MINUTES)
    sig["new_team"] = float(m.new_team)
    last_games = m.games_left is not None and m.games_left <= SEASON_END_GAMES
    sig["season_end"] = float(last_games and minutes >= CORE_MINUTES)
    sat_lately = m.sat_ago is not None and m.sat_ago <= COACH_SAT_DAYS
    sig["coach_sat"] = float(sat_lately and minutes >= COACH_SAT_LINE)
    if f is not None:
        sig["recent"] = f.minutes - minutes
        if f.last_minutes is not None:
            sig["last_game"] = f.last_minutes - minutes
        if f.minutes_season is not None:
            sig["long"] = f.minutes_season - minutes
        if f.started is not None and f.start_share is not None:
            sig["role"] = f.start_share - 0.5
            sig["switch"] = float(f.started) - f.start_share
    return sig


def rate_signal(name: str, stat: int, rate: float, m: GameContext, out: list[Teammate]) -> float:
    """One per-minute signal of RATE_WEIGHTS for a stat whose line makes ``rate`` a minute."""
    if name == "form":
        return _form(m.form, stat, rate)
    if name == "form_fast":
        f = m.form
        return _shrunk(f.fast.get(stat), rate, FAST_DECAY[stat][1]) if f is not None and stat in FAST_DECAY else 0.0
    if name == "opp":
        return m.defense.get(stat, 0.0)
    if name == "usage_out":
        return sum(t.usage for t in out) / 100
    if name == "vac":
        return min(sum(t.points for t in out) / m.team_points, 1.0) if m.team_points > 0 else 0.0
    if name == "freed":
        team = m.team_made.get(stat, 0.0)
        known = [t for t in out if t.made]
        if not known or team <= 0:
            return 0.0
        share = min(sum(t.made.get(stat, 0.0) for t in known) / team, 1.0)
        return share - min(sum(t.minutes for t in known) / TEAM_MINUTES, 1.0)
    if name == "home":
        return 0.5 if m.home else -0.5
    if name == "margin":
        return m.margin / 10
    if name == "b2b":
        return 1.0 if m.back_to_back else 0.0
    raise KeyError(name)


def apply(
    line: dict[int, float], m: GameContext, out: list[Teammate], gained: float = 0.0
) -> tuple[dict[int, float], dict[int, float], list[Reason]]:
    """His line with the game in it, his usual line (the same model on a plain night: no opponent,
    form, teammates out or absence, at his usual role), and the reasons one differs from the other.
    ``line`` is today's next-game line (report.next_game_line), ``gained`` the minutes in it from
    teammates ruled out (tonight.minutes_gained)."""
    minutes = line.get(MINUTES, 0.0)
    if minutes <= 0:
        return line, line, []
    reasons: dict[str, Reason] = {}

    def reason(kind: str) -> Reason:
        return reasons.setdefault(kind, Reason(kind))

    # Minutes
    sig = minute_signals(minutes, m, out)
    moves = {k: MINUTE_WEIGHTS[k] * x for k, x in sig.items()}
    new_minutes = min(max(minutes + MINUTES_LEVEL + sum(moves.values()), 0.0), MAX_MINUTES)
    rate = {k: line.get(k, 0.0) / minutes for k in COUNTING}
    shifts: dict[str, float] = defaultdict(float)
    for k, change in moves.items():
        if k in MINUTE_KINDS:
            shifts[MINUTE_KINDS[k]] += change
    shifts["teammates_out"] += gained
    for kind, change in shifts.items():
        if abs(change) >= 0.05:
            reason(kind).minutes += change
            for k in COUNTING:
                reason(kind).add(k, rate[k] * change)
    if "back_from_absence" in reasons:
        reasons["back_from_absence"].detail.update(games=m.games_missed, game_back=m.game_back)
    if "starter_switch" in reasons and m.form is not None:
        reasons["starter_switch"].detail["started"] = int(bool(m.form.started))
    if "coach_decision" in reasons:
        reasons["coach_decision"].detail["days"] = m.sat_ago
    if out:
        reason("teammates_out").detail["names"] = ", ".join(t.name for t in out)
        reason("teammates_out").detail["points"] = round(sum(t.points for t in out), 1)
    # Per minute
    kinds = {
        "form": "form",
        "form_fast": "form",
        "opp": "opponent",
        "usage_out": "teammates_out",
        "vac": "teammates_out",
        "freed": "teammates_out",
        "home": "home" if m.home else "away",
        "margin": "favored" if m.margin > 0 else "underdog",
        "b2b": "back_to_back",
    }
    plain = min(max(minutes - gained + MINUTES_LEVEL + moves["role"], 0.0), MAX_MINUTES)
    usual = dict(line)
    usual[MINUTES] = plain
    for k in COUNTING:
        usual[k] = max(rate[k] * plain + RATE_LEVEL.get(k, 0.0), 0.0)
    new = dict(line)
    new[MINUTES] = new_minutes
    for k in COUNTING:
        new[k] = rate[k] * new_minutes
    for k, weights in RATE_WEIGHTS.items():
        base = rate[k] * new_minutes
        total = RATE_LEVEL[k]
        for name, w in weights.items():
            change = base * w * rate_signal(name, k, rate[k], m, out)
            total += change
            if abs(change) > 1e-6:
                reason(kinds[name]).add(k, change)
        new[k] = max(base + total, 0.0)
    # FG%: his line's percentage, moved by the opponent's FG% defense and home court.
    fga = line.get(FGA, 0.0)
    if fga > 0:
        pct = line.get(FGM, 0.0) / fga
        opp_pct = (1 + m.defense.get(FGM, 0.0)) / (1 + m.defense.get(FGA, 0.0)) - 1
        home = rate_signal("home", FGA, 0.0, m, out)
        fg_moves = {"opponent": FG_PCT_WEIGHTS["opp"] * opp_pct, kinds["home"]: FG_PCT_WEIGHTS["home"] * home}
        new_pct = min(max(pct + sum(fg_moves.values()), 0.0), 1.0)
        new[FGM] = new[FGA] * new_pct
        usual[FGM] = usual[FGA] * pct
        for kind, change in fg_moves.items():
            if abs(change) > 1e-6:
                reason(kind).detail["fg_pct"] = round(float(reason(kind).detail.get("fg_pct", 0.0)) + change, 4)
    # FT: his line's percentage on the attempts (a single game's FT% is noise).
    ft_pct = ft_line_pct(line)
    new[FTM] = new[FTA] * ft_pct
    usual[FTM] = usual[FTA] * ft_pct
    if "opponent" in reasons:
        reasons["opponent"].detail["team"] = m.opponent
    return new, usual, list(reasons.values())


def ft_line_pct(line: dict[int, float]) -> float:
    fta = line.get(FTA, 0.0)
    return line.get(FTM, 0.0) / fta if fta > 0 else 0.0


def _form(f: Form | None, stat: int, rate: float) -> float:
    """His form in a stat against his line's rate: decayed and shrunk (FORM_DECAY), else his last
    FORM_GAMES games."""
    if f is None or rate <= 0:
        return 0.0
    if stat in FORM_DECAY:
        return _shrunk(f.decayed.get(stat), rate, FORM_DECAY[stat][1])
    lo, hi = FORM_RANGE
    return min(max(f.rate.get(stat, 0.0) / rate - 1, lo), hi)


def _shrunk(sums: tuple[float, float] | None, rate: float, shrink: float) -> float:
    """(decayed stat, decayed minutes) against his line's ``rate``: the gap, -1..2, times E / (E + shrink)
    with E the count his rate expects over those minutes."""
    if sums is None or rate <= 0 or sums[1] <= 0:
        return 0.0
    lo, hi = FORM_RANGE
    gap = min(max(sums[0] / sums[1] / rate - 1, lo), hi)
    expected = rate * sums[1]
    return gap * expected / (expected + shrink)

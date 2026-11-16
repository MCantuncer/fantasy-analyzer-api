"""The game itself in a player's next-game line: who he plays, where, and how he has been going.

Today's line (report.next_game_line) is his projection at his recent minutes. Tested on every game
of the ~500 most-owned players in five seasons (2021-22 to 2025-26, 90,410 games), a few things
known before tip-off move it closer to what he then does. Each was kept only if it lowered the
squared miss in every season, fitted with that season left out and again on earlier seasons only.
Replaying this code day by day (scripts/next_game_matchup_backtest.py), the squared miss against
today's line falls in every season by 1.9-2.5% for points, 1.5-2.8% for rebounds, 2.0-3.1% for
assists, 0.6-1.2% for steals, 1.0-1.6% for blocks, 1.1-2.0% for threes, 1.0-1.8% for turnovers,
0.4-0.8% for FG impact and 4.1-6.1% for minutes (mean points miss 4.91 -> 4.84; season average 4.96):

  * the opponent's defense in each stat: what the players it faced made against what they usually
    make per minute (DEFENSE_PRIOR_GAMES of league average to start). Raw "points allowed" is half
    as good, and splitting it by guards, wings and bigs added nothing;
  * his last FORM_GAMES games per minute against his line: about a third of the gap lasts for
    points, rebounds and assists, less for the rest, nothing for steals;
  * shots a game of teammates ruled out: more points, assists and turnovers a minute, on top of the
    minutes they leave (tonight.py), and more minutes for bench players;
  * games he just missed: fewer minutes in his first game back;
  * home court (assists, FG%), how much his team is favored (blocks), a back-to-back (steals).

His history against the team, pace, travel, altitude, tip time, rest beyond a back-to-back, the
opponent's injuries and playing a former team didn't pass, or added nothing on top of these.
FT% for one game is noise: no signal passed, so free throws only follow the minutes.

Each change is kept as a reason (kind, the minutes and per-stat change, the words), so the player
card can say why the line moved.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

PTS, BLK, STL, AST, REB, TO, FGM, FGA, FTM, FTA, TPM, MINUTES, GP = 0, 1, 2, 3, 6, 11, 13, 14, 15, 16, 17, 40, 42
COUNTING = (PTS, REB, AST, STL, BLK, TPM, TO, FGM, FGA, FTM, FTA)
GAME_SPLIT = 5

PRIOR_MINUTES = 200.0  # minutes of his last season (or the league's rate) in what he usually makes a minute
MIN_PRIOR_MINUTES = 200.0  # last season counts only with this many minutes
DEFENSE_PRIOR_GAMES = 8.0  # team games of league-average defense a team's own record is blended with
MARGIN_PRIOR_GAMES = 8.0  # team games of an even record in a team's margin a game
HOME_EDGE = 2.5  # points of home court in the expected margin
FORM_GAMES = 10
MINUTE_GAMES = 5
MISSED_CAP = 10
BENCH_MINUTES = 36.0  # starters' minutes; below it a player gains minutes when teammates sit
HEAVY_MINUTES = 36.0  # lines above this come in lower
MAX_MINUTES = 44.0
FORM_RANGE = (-1.0, 2.0)

# Minutes: the line + these (fitted on all five seasons; each sign held in every season).
MINUTES_LEVEL = -0.839  # lines run this many minutes over what players play
MISSED_MINUTES = -0.611  # per team game he just missed, up to MISSED_CAP
BENCH_SHARE = 0.105  # per minute of teammates ruled out, x (1 - his minutes / BENCH_MINUTES)
RECENT_MINUTES = 0.199  # per minute his last MINUTE_GAMES games ran over the line
HEAVY_LEVEL = -0.613  # per minute of line over HEAVY_MINUTES

# Per minute: his line's rate x (1 + sum of weight x signal) + level, per stat.
#   form: his last FORM_GAMES games per minute over his line's rate, -1..2
#   opp: the opponent's defense in that stat (+0.05 = players make 5% more against it than usual)
#   usage_out: shots a game (FGA + 0.44 FTA + TO) of teammates ruled out, / 100
#   home: +0.5 at home, -0.5 away; margin: expected margin / 10; b2b: 1 on a back-to-back
RATE_WEIGHTS: dict[int, dict[str, float]] = {
    PTS: {"form": 0.3745, "opp": 1.0831, "usage_out": 0.318},
    REB: {"form": 0.3248, "opp": 0.8744},
    AST: {"form": 0.3697, "opp": 0.8206, "usage_out": 0.1917, "home": 0.032},
    STL: {"opp": 0.7881, "b2b": -0.0493},
    BLK: {"opp": 0.893, "margin": 0.0576, "form": 0.1027},
    TPM: {"form": 0.2253, "opp": 0.7053},
    TO: {"opp": 0.8134, "form": 0.1746, "usage_out": 0.2274},
    FGA: {"form": 0.5567, "opp": 1.0615, "usage_out": 0.2991},
}
RATE_LEVEL = {
    PTS: -0.0797,
    REB: 0.0167,
    AST: -0.0372,
    STL: -0.0189,
    BLK: 0.0113,
    TPM: -0.0112,
    TO: -0.0164,
    FGA: -0.2284,
}
# FG%: his line's FG% + weight x signal (opp: the opponent's FG% defense as a share, home as above).
FG_PCT_WEIGHTS = {"opp": 0.3729, "home": 0.0071}

USAGE_FTA = 0.44


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
            line = {k: float(s.get(str(k), 0) or 0) for k in (*COUNTING, MINUTES)}
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
    minutes: float  # a game over his last MINUTE_GAMES games
    minutes_long: float  # a game over his last FORM_GAMES games
    usage: float  # FGA + 0.44 FTA + TO a game over his last FORM_GAMES games
    last_day: int  # the day of his last game
    team: int  # his team in that game


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
    played: dict[int, list[int]] = field(default_factory=dict)  # team -> days it played, in order
    schedule: dict[int, list[Game]] = field(default_factory=dict)  # team -> its games this season, in order

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


def build(season: int, rows: list[GameRow], prior: dict[int, dict[int, float]], schedule: dict[int, list[Game]]) -> Box:
    """The box from every game row so far: defenses, margins and each player's recent games."""
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
        form[pid] = Form(
            rate={k: sum(r.line[k] for r in last) / minutes for k in COUNTING},
            minutes=sum(r.line[MINUTES] for r in games[-MINUTE_GAMES:]) / len(games[-MINUTE_GAMES:]),
            minutes_long=minutes / len(last),
            usage=sum(r.line[FGA] + USAGE_FTA * r.line[FTA] + r.line[TO] for r in last) / len(last),
            last_day=games[-1].day,
            team=games[-1].team,
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
    # Margin a game and the days each team played.
    points: dict[tuple[int, int], float] = defaultdict(float)
    days: dict[int, int] = {}
    for r in rows:
        points[(r.game, r.team)] += r.line[PTS]
        days[r.game] = r.day
    margin_sum: dict[int, float] = defaultdict(float)
    played: dict[int, set[int]] = defaultdict(set)
    for game, teams in teams_in.items():
        if len(teams) != 2:
            continue
        a, b = sorted(teams)
        diff = points[(game, a)] - points[(game, b)]
        margin_sum[a] += diff
        margin_sum[b] -= diff
        played[a].add(days[game])
        played[b].add(days[game])
    margin = {t: margin_sum[t] / (len(played[t]) + MARGIN_PRIOR_GAMES) for t in played}
    return Box(
        season=season,
        defense=defense,
        margin=margin,
        form=form,
        played={t: sorted(d) for t, d in played.items()},
        schedule=schedule,
    )


@dataclass(frozen=True)
class Teammate:
    """A teammate ruled out of the game (tonight.ruled_out)."""

    name: str
    minutes: float  # a game, lately
    usage: float  # FGA + 0.44 FTA + TO a game, lately


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


def context(box: Box, player: int, team: int, today: int) -> GameContext | None:
    game = box.next_game(team, today)
    if game is None:
        return None
    margin = box.margin.get(team, 0.0) - box.margin.get(game.opponent, 0.0) + (HOME_EDGE if game.home else -HOME_EDGE)
    f = box.form.get(player)
    return GameContext(
        day=game.day,
        opponent=game.opponent,
        home=game.home,
        back_to_back=box.back_to_back(team, game.day),
        margin=margin,
        defense=box.defense.get(game.opponent, {}),
        form=f if f is not None and f.team == team else None,
        games_missed=box.games_missed(player, team, game.day),
        tip=game.tip,
    )


@dataclass
class Reason:
    """One thing that moved his line: the minutes it moved and each stat (stat id -> change)."""

    kind: str  # opponent, form, teammates_out, back_from_absence, heavy_minutes, recent_minutes, home, away,
    # favored, underdog, back_to_back
    minutes: float = 0.0
    stats: dict[int, float] = field(default_factory=dict)
    detail: dict[str, float | str] = field(default_factory=dict)

    def add(self, stat: int, change: float) -> None:
        self.stats[stat] = self.stats.get(stat, 0.0) + change


def apply(
    line: dict[int, float], m: GameContext, out: list[Teammate], gained: float = 0.0
) -> tuple[dict[int, float], dict[int, float], list[Reason]]:
    """His line with the game in it, his usual line (the same model on a plain night: no opponent,
    form, teammates out or absence), and the reasons one differs from the other. ``line`` is
    today's next-game line (report.next_game_line), ``gained`` the minutes in it from teammates
    ruled out (tonight.minutes_gained)."""
    minutes = line.get(MINUTES, 0.0)
    if minutes <= 0:
        return line, line, []
    reasons: dict[str, Reason] = {}

    def reason(kind: str) -> Reason:
        return reasons.setdefault(kind, Reason(kind))

    # Minutes
    out_minutes = sum(t.minutes for t in out)
    shifts = {
        "back_from_absence": MISSED_MINUTES * m.games_missed,
        "teammates_out": BENCH_SHARE * out_minutes * max(0.0, 1 - minutes / BENCH_MINUTES),
        "recent_minutes": RECENT_MINUTES * (m.form.minutes - minutes) if m.form is not None else 0.0,
        "heavy_minutes": HEAVY_LEVEL * max(0.0, minutes - HEAVY_MINUTES),
    }
    new_minutes = min(max(minutes + MINUTES_LEVEL + sum(shifts.values()), 0.0), MAX_MINUTES)
    rate = {k: line.get(k, 0.0) / minutes for k in COUNTING}
    shifts["teammates_out"] += gained
    for kind, change in shifts.items():
        if abs(change) >= 0.05:
            reason(kind).minutes += change
            for k in COUNTING:
                reason(kind).add(k, rate[k] * change)
    if m.games_missed:
        reason("back_from_absence").detail["games"] = m.games_missed
    if out:
        reason("teammates_out").detail["names"] = ", ".join(t.name for t in out)
    # Per minute
    signals = {
        "form": None,
        "opp": None,
        "usage_out": sum(t.usage for t in out) / 100,
        "home": 0.5 if m.home else -0.5,
        "margin": m.margin / 10,
        "b2b": 1.0 if m.back_to_back else 0.0,
    }
    kinds = {
        "form": "form",
        "opp": "opponent",
        "usage_out": "teammates_out",
        "home": "home" if m.home else "away",
        "margin": "favored" if m.margin > 0 else "underdog",
        "b2b": "back_to_back",
    }
    plain = min(max(minutes - gained + MINUTES_LEVEL, 0.0), MAX_MINUTES)
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
            if name == "form":
                x = _form(m.form, k, rate[k])
            elif name == "opp":
                x = m.defense.get(k, 0.0)
            else:
                x = signals[name]
            change = base * w * x
            total += change
            if abs(change) > 1e-6:
                reason(kinds[name]).add(k, change)
        new[k] = max(base + total, 0.0)
    # FG%: his line's percentage, moved by the opponent's FG% defense and home court.
    fga = line.get(FGA, 0.0)
    if fga > 0:
        pct = line.get(FGM, 0.0) / fga
        opp_pct = (1 + m.defense.get(FGM, 0.0)) / (1 + m.defense.get(FGA, 0.0)) - 1
        moves = {"opponent": FG_PCT_WEIGHTS["opp"] * opp_pct, kinds["home"]: FG_PCT_WEIGHTS["home"] * signals["home"]}
        new_pct = min(max(pct + sum(moves.values()), 0.0), 1.0)
        new[FGM] = new[FGA] * new_pct
        usual[FGM] = usual[FGA] * pct
        for kind, change in moves.items():
            if abs(change) > 1e-6:
                reason(kind).detail["fg_pct"] = round(float(reason(kind).detail.get("fg_pct", 0.0)) + change, 4)
    # FT: the line's percentage at the new minutes (no signal passed for a single game's FT%).
    if "opponent" in reasons:
        reasons["opponent"].detail["team"] = m.opponent
    return new, usual, list(reasons.values())


def _form(f: Form | None, stat: int, rate: float) -> float:
    if f is None or rate <= 0:
        return 0.0
    lo, hi = FORM_RANGE
    return min(max(f.rate.get(stat, 0.0) / rate - 1, lo), hi)

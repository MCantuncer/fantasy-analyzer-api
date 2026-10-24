"""NBA scores and box scores, built from ESPN's fantasy data.

The fantasy API is the one ESPN host this service can count on. It has no
scoreboard, but for any scoring period (one day) it returns every player's line
for each game, tagged with the game id. Adding up a team's points gives the
score, and the NBA calendar (``proTeamSchedules_wl``) gives the start times.

It has no quarter or game clock, so a game is "live" from its start until ESPN
marks its stats official, or until ``GAME_LENGTH`` has passed.

A day is league-independent: every league sees the same games. A day whose
games are all over is saved once and never fetched again.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta

# ESPN stat ids in a game line.
PTS, BLK, STL, AST, REB, TO, FGM, FGA, FTM, FTA, TPM, MIN = 0, 1, 2, 3, 6, 11, 13, 14, 15, 16, 17, 40
GAME_SPLIT = 5  # statSplitTypeId of a single game
# A regulation game takes about 2h15m; this leaves room for overtime and delays.
GAME_LENGTH = timedelta(hours=3)
# A finished day is saved this long after its last game started, once late stat fixes are in.
SETTLE = timedelta(hours=5)


@dataclass
class ScheduledGame:
    id: int
    day: int
    start: datetime
    away: int  # pro team id
    home: int
    official: bool  # ESPN has marked the stats final


@dataclass
class Line:
    player_id: int
    name: str
    pro_team_id: int
    minutes: int
    pts: int
    reb: int
    ast: int
    stl: int
    blk: int
    tpm: int
    to: int
    fgm: int
    fga: int
    ftm: int
    fta: int


@dataclass
class Schedule:
    teams: dict[int, tuple[str, str]]  # pro team id -> (abbrev, "Location Name")
    games: dict[int, list[ScheduledGame]] = field(default_factory=dict)  # day -> games
    opening: date | None = None

    def date_of(self, day: int) -> date | None:
        return self.opening + timedelta(days=day - 1) if self.opening else None

    def latest_started(self, now: datetime) -> int:
        """The last day with a game that has started: last night in the morning, tonight once it tips off."""
        started = [d for d, games in self.games.items() if any(g.start <= now for g in games)]
        return max(started) if started else min(self.games, default=1)


def parse_schedule(raw: dict) -> Schedule:
    teams: dict[int, tuple[str, str]] = {}
    games: dict[int, dict[int, ScheduledGame]] = {}
    for team in (raw.get("settings") or {}).get("proTeams", []):
        tid = int(team["id"])
        if tid == 0:  # "FA", the free agents' pseudo team
            continue
        teams[tid] = (team.get("abbrev", "").upper(), f"{team.get('location', '')} {team.get('name', '')}".strip())
        for day, items in (team.get("proGamesByScoringPeriod") or {}).items():
            for g in items or []:
                games.setdefault(int(day), {})[int(g["id"])] = ScheduledGame(
                    id=int(g["id"]),
                    day=int(day),
                    start=datetime.fromtimestamp(g["date"] / 1000, UTC),
                    away=int(g["awayProTeamId"]),
                    home=int(g["homeProTeamId"]),
                    official=bool(g.get("statsOfficial")),
                )
    sched = Schedule(
        teams=teams, games={d: sorted(gs.values(), key=lambda g: (g.start, g.id)) for d, gs in games.items()}
    )
    if 1 in sched.games:
        # Start times are UTC; US evening games land on the next UTC day, so step back 8 hours.
        sched.opening = (sched.games[1][0].start - timedelta(hours=8)).date()
    return sched


def parse_lines(players: list[dict], day: int) -> dict[int, list[Line]]:
    """Game id -> the lines of everyone who played in it that day, most minutes first."""
    out: dict[int, list[Line]] = {}
    for entry in players:
        p = entry.get("player") or entry
        for st in p.get("stats") or []:
            if st.get("scoringPeriodId") != day or st.get("statSplitTypeId") != GAME_SPLIT or st.get("statSourceId"):
                continue
            s = st.get("stats") or {}
            if not s or not st.get("externalId"):
                continue

            def v(stat: int, s: dict = s) -> int:
                return round(float(s.get(str(stat), 0)))

            if v(MIN) <= 0:
                continue  # did not play
            out.setdefault(int(st["externalId"]), []).append(
                Line(
                    player_id=int(p["id"]),
                    name=p.get("fullName", ""),
                    pro_team_id=int(st.get("proTeamId") or p.get("proTeamId") or 0),
                    minutes=v(MIN),
                    pts=v(PTS),
                    reb=v(REB),
                    ast=v(AST),
                    stl=v(STL),
                    blk=v(BLK),
                    tpm=v(TPM),
                    to=v(TO),
                    fgm=v(FGM),
                    fga=v(FGA),
                    ftm=v(FTM),
                    fta=v(FTA),
                )
            )
    for lines in out.values():
        lines.sort(key=lambda line: (-line.minutes, -line.pts, line.name))
    return out


def game_status(game: ScheduledGame, now: datetime) -> str:
    if game.official or now >= game.start + GAME_LENGTH:
        return "final"
    return "live" if now >= game.start else "scheduled"


def _side(schedule: Schedule, team_id: int, box: list[Line], status: str) -> dict:
    abbrev, name = schedule.teams.get(team_id, ("", ""))
    score = sum(line.pts for line in box if line.pro_team_id == team_id) if status != "scheduled" else None
    return {"pro_team_id": team_id, "abbrev": abbrev, "name": name, "score": score}


def build_day(schedule: Schedule, day: int, players: list[dict], now: datetime) -> dict:
    """One day's games with scores and box scores, in the shape of ``schemas.ScoreDay`` minus league fields."""
    lines = parse_lines(players, day)
    games = []
    for g in schedule.games.get(day, []):
        status = game_status(g, now)
        box = lines.get(g.id, []) if status != "scheduled" else []
        games.append(
            {
                "id": g.id,
                "start": g.start.isoformat().replace("+00:00", "Z"),
                "status": status,
                "away": _side(schedule, g.away, box, status),
                "home": _side(schedule, g.home, box, status),
                "lines": [line.__dict__ for line in box],
            }
        )
    d = schedule.date_of(day)
    return {"day": day, "date": d.isoformat() if d else None, "games": games}


def settled(schedule: Schedule, day: int, now: datetime) -> bool:
    """True once every game that day is over and late stat fixes have had time to land."""
    games = schedule.games.get(day, [])
    return bool(games) and all(game_status(g, now) == "final" for g in games) and now >= games[-1].start + SETTLE


def needs_stats(schedule: Schedule, day: int, now: datetime) -> bool:
    return any(g.start <= now for g in schedule.games.get(day, []))


def week_fixtures(schedule: Schedule, now: datetime) -> list[tuple[int, list[ScheduledGame]]]:
    """Games still to tip off this week (Monday to Sunday, US dates), the way fantasy matchup weeks run."""
    if schedule.opening is None:
        return []
    today = (now - timedelta(hours=8)).date()  # the US date, as for opening night
    sunday = today + timedelta(days=6 - today.weekday())
    out = []
    for day in sorted(schedule.games):
        d = schedule.date_of(day)
        if d is None or not today <= d <= sunday:
            continue
        upcoming = [g for g in schedule.games[day] if g.start > now]
        if upcoming:
            out.append((day, upcoming))
    return out

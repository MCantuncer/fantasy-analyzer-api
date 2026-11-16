"""An NBA team at a glance, for the team card opened from the header search.

Its record and latest results come from the same day boxes as the Scores page (``scores.py``):
the fantasy API's player lines added up per game. A finished day is stored once, so after the
first look a team's season costs no ESPN call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from app.scores import Schedule, ScheduledGame, game_status

LATEST = 5


@dataclass(frozen=True)
class Scorer:
    player_id: int
    name: str
    pts: int
    reb: int
    ast: int


@dataclass(frozen=True)
class Result:
    game_id: int
    day: int
    date: str | None
    start: str
    home: bool
    opponent: str  # abbreviation
    opponent_name: str
    points: int
    opponent_points: int
    top: Scorer | None  # his team's top scorer that night

    @property
    def won(self) -> bool:
        return self.points > self.opponent_points


@dataclass
class TeamSeason:
    abbrev: str
    name: str
    pro_team_id: int
    results: list[Result] = field(default_factory=list)  # oldest first
    next_game: ScheduledGame | None = None

    @property
    def wins(self) -> int:
        return sum(r.won for r in self.results)

    @property
    def losses(self) -> int:
        return len(self.results) - self.wins

    @property
    def streak(self) -> tuple[str, int] | None:
        """("W", 3) for three wins in a row; None before the first game."""
        if not self.results:
            return None
        last = self.results[-1].won
        n = 0
        for r in reversed(self.results):
            if r.won != last:
                break
            n += 1
        return ("W" if last else "L", n)


def team_id_of(schedule: Schedule, abbrev: str) -> int | None:
    want = abbrev.strip().upper()
    return next((tid for tid, (ab, _) in schedule.teams.items() if ab == want), None)


def played_days(schedule: Schedule, team_id: int, now: datetime) -> list[int]:
    """Days on which the team played a game that is over, oldest first."""
    return sorted(
        day
        for day, games in schedule.games.items()
        if any(team_id in (g.away, g.home) and game_status(g, now) == "final" for g in games)
    )


def next_game(schedule: Schedule, team_id: int, now: datetime) -> ScheduledGame | None:
    games = [g for gs in schedule.games.values() for g in gs if team_id in (g.away, g.home) and g.start > now]
    return min(games, key=lambda g: g.start, default=None)


def result_of(game: dict, team_id: int, day: int, date: str | None) -> Result | None:
    """The team's result in one game of a day box, or None when it wasn't in it or it isn't final."""
    if game.get("status") != "final":
        return None
    sides = {"away": game.get("away") or {}, "home": game.get("home") or {}}
    mine = next((k for k, s in sides.items() if s.get("pro_team_id") == team_id), None)
    if mine is None:
        return None
    them = sides["home" if mine == "away" else "away"]
    pts, opp = sides[mine].get("score"), them.get("score")
    if pts is None or opp is None or (pts == 0 and opp == 0):
        return None  # no box yet for a game that should be over
    lines = [ln for ln in game.get("lines", []) if ln.get("pro_team_id") == team_id]
    best = max(lines, key=lambda ln: (ln.get("pts", 0), ln.get("reb", 0) + ln.get("ast", 0)), default=None)
    return Result(
        game_id=int(game["id"]),
        day=day,
        date=date,
        start=game.get("start", ""),
        home=mine == "home",
        opponent=them.get("abbrev", ""),
        opponent_name=them.get("name", ""),
        points=int(pts),
        opponent_points=int(opp),
        top=Scorer(
            player_id=int(best["player_id"]),
            name=best.get("name", ""),
            pts=int(best.get("pts", 0)),
            reb=int(best.get("reb", 0)),
            ast=int(best.get("ast", 0)),
        )
        if best
        else None,
    )


def team_season(schedule: Schedule, team_id: int, day_boxes: dict[int, dict], now: datetime) -> TeamSeason:
    abbrev, name = schedule.teams[team_id]
    results = []
    for day in sorted(day_boxes):
        body = day_boxes[day]
        for g in body.get("games", []):
            r = result_of(g, team_id, day, body.get("date"))
            if r is not None:
                results.append(r)
    results.sort(key=lambda r: (r.start, r.game_id))
    return TeamSeason(abbrev, name, team_id, results, next_game(schedule, team_id, now))

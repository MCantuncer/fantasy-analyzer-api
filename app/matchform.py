"""Two teams' form before they meet, for the match card on Our Estimates.

Every game comes from the Scores page's day boxes (``scores.py``): ESPN's fantasy data, whose game
days start on opening night, so preseason games never get in. A game's kind (Regular Season, NBA Cup,
Play-In, Playoff) is not in that data: ESPN's public scoreboard has it, read once per finished day.
When the scoreboard can't be read, a game simply has no kind and the card shows no tag.

The MVP of a game is the player on the winning side with the best Game Score (John Hollinger's
one-number box score summary). Fantasy lines carry no offensive/defensive rebound split and no
fouls, so rebounds get their blended weight and fouls are left out.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import httpx

from app.gameclock import SCOREBOARD, TIMEOUT, USER_AGENT

# ESPN season types: 1 preseason, 2 regular season, 3 playoffs, 5 play-in.
KINDS = {2: "regular", 3: "playoff", 5: "play_in"}


@dataclass(frozen=True)
class GameKind:
    kind: str  # "regular" | "cup" | "play_in" | "playoff" | "preseason"
    note: str | None  # ESPN's own note: "NBA Cup - Group Play", "NBA Mexico City Game"
    neutral: bool  # played at a neutral site


def kind_of(season_type: int, notes: list[str]) -> str:
    if season_type == 1:
        return "preseason"
    if season_type == 2 and any("nba cup" in n.lower() for n in notes):
        return "cup"
    return KINDS.get(season_type, "regular")


def parse_kinds(raw: dict) -> dict[int, GameKind]:
    """Each game's kind on one scoreboard day, by ESPN game id."""
    out: dict[int, GameKind] = {}
    for event in raw.get("events") or []:
        try:
            season_type = int((event.get("season") or {}).get("type") or 2)
            comp = (event.get("competitions") or [{}])[0]
            notes = [str(n.get("headline") or "") for n in comp.get("notes") or [] if n.get("headline")]
            out[int(event["id"])] = GameKind(
                kind=kind_of(season_type, notes),
                note=notes[0] if notes else None,
                neutral=bool(comp.get("neutralSite")),
            )
        except (KeyError, TypeError, ValueError):
            continue
    return out


async def fetch_kinds(day: date, transport: httpx.AsyncBaseTransport | None = None) -> dict[int, GameKind] | None:
    """That day's game kinds, or None when the scoreboard can't be read."""
    async with httpx.AsyncClient(transport=transport, timeout=TIMEOUT, headers={"User-Agent": USER_AGENT}) as http:
        try:
            resp = await http.get(SCOREBOARD, params={"dates": day.strftime("%Y%m%d")})
            if resp.status_code != 200:
                return None
            return parse_kinds(resp.json())
        except (httpx.HTTPError, ValueError):
            return None


def game_score(line: dict) -> float:
    """Hollinger's Game Score from a fantasy line: rebounds at their blended weight, no fouls."""
    g = line.get
    return (
        g("pts", 0)
        + 0.4 * g("fgm", 0)
        - 0.7 * g("fga", 0)
        - 0.4 * (g("fta", 0) - g("ftm", 0))
        + 0.4 * g("reb", 0)  # about a quarter of rebounds are offensive (0.7), the rest 0.3
        + g("stl", 0)
        + 0.7 * g("ast", 0)
        + 0.7 * g("blk", 0)
        - g("to", 0)
    )


STAT_KEYS = ("minutes", "pts", "reb", "ast", "stl", "blk", "tpm", "to", "fgm", "fga", "ftm", "fta")


@dataclass(frozen=True)
class StarLine:
    player_id: int
    name: str
    team: str  # abbreviation
    game_score: float
    stats: dict[str, int]


@dataclass(frozen=True)
class FormGame:
    game_id: int
    date: str | None
    start: str
    home: bool
    opponent: str
    opponent_name: str
    points: int
    opponent_points: int
    mvp: StarLine | None  # the winning side's best Game Score
    best: StarLine | None  # this team's best Game Score (the MVP itself after a win)
    opp_best: StarLine | None  # the opponent's best Game Score (the MVP itself after a loss)

    @property
    def won(self) -> bool:
        return self.points > self.opponent_points

    @property
    def margin(self) -> int:
        return self.points - self.opponent_points


def _star(lines: list[dict], abbrev: str) -> StarLine | None:
    played = [ln for ln in lines if ln.get("minutes", 0) > 0] or lines
    if not played:
        return None
    top = max(played, key=lambda ln: (game_score(ln), ln.get("pts", 0)))
    return StarLine(
        player_id=int(top["player_id"]),
        name=top.get("name", ""),
        team=abbrev,
        game_score=round(game_score(top), 1),
        stats={k: int(top.get(k, 0) or 0) for k in STAT_KEYS},
    )


def form_game(game: dict, team_id: int, date_: str | None) -> FormGame | None:
    """The team's side of one final game of a day box, or None when it wasn't in it or isn't final."""
    if game.get("status") != "final":
        return None
    sides = {"away": game.get("away") or {}, "home": game.get("home") or {}}
    mine = next((k for k, s in sides.items() if s.get("pro_team_id") == team_id), None)
    if mine is None:
        return None
    me, them = sides[mine], sides["home" if mine == "away" else "away"]
    pts, opp = me.get("score"), them.get("score")
    if pts is None or opp is None or pts == opp:
        return None
    lines = game.get("lines") or []
    winner = me if pts > opp else them
    win_lines = [ln for ln in lines if ln.get("pro_team_id") == winner.get("pro_team_id")]
    my_lines = [ln for ln in lines if ln.get("pro_team_id") == team_id]
    their_lines = [ln for ln in lines if ln.get("pro_team_id") == them.get("pro_team_id")]
    return FormGame(
        game_id=int(game["id"]),
        date=date_,
        start=game.get("start", ""),
        home=mine == "home",
        opponent=them.get("abbrev", ""),
        opponent_name=them.get("name", ""),
        points=int(pts),
        opponent_points=int(opp),
        mvp=_star(win_lines, winner.get("abbrev", "")),
        best=_star(my_lines, me.get("abbrev", "")),
        opp_best=_star(their_lines, them.get("abbrev", "")),
    )


def team_form(team_id: int, day_boxes: dict[int, dict]) -> list[FormGame]:
    """Every final game of the team in these day boxes, most recent first."""
    games = [
        fg
        for day in day_boxes
        for g in day_boxes[day].get("games", [])
        if (fg := form_game(g, team_id, day_boxes[day].get("date"))) is not None
    ]
    games.sort(key=lambda fg: (fg.start, fg.game_id), reverse=True)
    return games

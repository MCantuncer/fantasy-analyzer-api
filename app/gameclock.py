"""Quarter and game clock for live NBA games, from ESPN's public scoreboard.

The fantasy API gives scores but no clock, so on its own a game looks "live" from
tip-off until ESPN marks its stats official. ESPN's site scoreboard knows the
period, the clock and the moment a game ends. It is best effort: when it can't be
reached, games keep the fantasy-only status and show no clock.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date

import httpx

SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
# One game's box score, which marks the ten players on the floor as "active" while it is played.
SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"
# ESPN's site API refuses (403) a user agent that carries a link.
USER_AGENT = "Mozilla/5.0 (compatible; fantasy-analyzer/1.0)"
TIMEOUT = 5.0


@dataclass(frozen=True)
class Clock:
    state: str  # "pre", "in" or "post"
    label: str | None  # "Q4 2:31", "Half", "End Q3", "OT 0:48"; None unless in progress
    teams: frozenset[str]  # the two team abbreviations, a fallback when ids don't match


def period_name(period: int) -> str:
    if period <= 4:
        return f"Q{period}"
    return "OT" if period == 5 else f"{period - 4}OT"


def clock_label(period: int, display_clock: str, detail: str) -> str | None:
    """A short live label. ESPN's own detail decides halftime and the end of a period."""
    if period <= 0:
        return None
    d = detail.lower()
    if "half" in d:
        return "Half"
    if d.startswith("end") or display_clock in ("0.0", "0:00", ""):
        return f"End {period_name(period)}"
    return f"{period_name(period)} {display_clock}"


def parse_scoreboard(raw: dict) -> dict[int, Clock]:
    out: dict[int, Clock] = {}
    for event in raw.get("events") or []:
        try:
            status = event["status"]
            kind = status.get("type") or {}
            state = kind.get("state") or ""
            label = (
                clock_label(
                    int(status.get("period") or 0), str(status.get("displayClock") or ""), kind.get("shortDetail") or ""
                )
                if state == "in"
                else None
            )
            comp = (event.get("competitions") or [{}])[0]
            teams = frozenset(c["team"]["abbreviation"].upper() for c in comp.get("competitors") or [])
            out[int(event["id"])] = Clock(state=state, label=label, teams=teams)
        except (KeyError, TypeError, ValueError):
            continue
    return out


async def fetch_clocks(day: date, transport: httpx.AsyncBaseTransport | None = None) -> dict[int, Clock] | None:
    """That day's clocks by ESPN game id, or None when the scoreboard can't be read."""
    async with httpx.AsyncClient(transport=transport, timeout=TIMEOUT, headers={"User-Agent": USER_AGENT}) as http:
        try:
            resp = await http.get(SCOREBOARD, params={"dates": day.strftime("%Y%m%d")})
            if resp.status_code != 200:
                return None
            return parse_scoreboard(resp.json())
        except (httpx.HTTPError, ValueError):
            return None


def apply_clocks(body: dict, clocks: dict[int, Clock]) -> dict:
    """A copy of a score day with each game's status corrected and its clock added."""
    by_teams = {c.teams: c for c in clocks.values()}
    games = []
    for g in body["games"]:
        clock = clocks.get(g["id"]) or by_teams.get(
            frozenset((g["away"]["abbrev"].upper(), g["home"]["abbrev"].upper()))
        )
        g = {**g, "clock": None}
        if clock is not None and g["status"] != "scheduled":
            if clock.state == "post":
                g["status"] = "final"
            elif clock.state == "in":
                g["status"] = "live"
                g["clock"] = clock.label
        games.append(g)
    return {**body, "games": games}


def parse_on_court(raw: dict) -> frozenset[int] | None:
    """The ESPN ids of the players on the floor, or None when the box score doesn't say clearly.

    ESPN flags the players on the court as ``active``. It is trusted only when each team has
    exactly five, so a box score that flags nobody (or everybody) never dims a whole team.
    """
    teams = (raw.get("boxscore") or {}).get("players") or []
    if len(teams) != 2:
        return None
    out: set[int] = set()
    for team in teams:
        try:
            athletes = (team.get("statistics") or [{}])[0].get("athletes") or []
            on = {int(a["athlete"]["id"]) for a in athletes if a.get("active") is True}
        except (KeyError, TypeError, ValueError, IndexError):
            return None
        if len(on) != 5:
            return None
        out |= on
    return frozenset(out)


async def fetch_on_court(
    game_ids: list[int], transport: httpx.AsyncBaseTransport | None = None
) -> dict[int, frozenset[int]]:
    """Who is on the floor in each game in progress, by ESPN game id; games ESPN can't tell are left out."""
    if not game_ids:
        return {}
    async with httpx.AsyncClient(transport=transport, timeout=TIMEOUT, headers={"User-Agent": USER_AGENT}) as http:

        async def one(gid: int) -> tuple[int, frozenset[int] | None]:
            try:
                resp = await http.get(SUMMARY, params={"event": gid})
                if resp.status_code != 200:
                    return gid, None
                return gid, parse_on_court(resp.json())
            except (httpx.HTTPError, ValueError):
                return gid, None

        pairs = await asyncio.gather(*(one(g) for g in game_ids))
    return {gid: on for gid, on in pairs if on is not None}


def apply_on_court(body: dict, on_court: dict[int, frozenset[int]]) -> dict:
    """A copy of a score day where each line of a game in progress says whether he is on the floor now."""
    games = []
    for g in body["games"]:
        on = on_court.get(g["id"]) if g["status"] == "live" else None
        if on is not None:
            g = {**g, "lines": [{**ln, "on_court": ln["player_id"] in on} for ln in g["lines"]]}
        games.append(g)
    return {**body, "games": games}

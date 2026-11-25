"""Tonight's starting fives as nba.com lists them before tip-off, saved night by night.

Can 2026-11-25 (the area map of our misses, dig A): a player who starts tonight but usually comes
off the bench plays 3.6-4.4 minutes more than our line, every season, and a starting-five term cut
the minutes miss by 3.4-5.5% in every one of five seasons, blind. The replay knew the starters from
the box score; live, they come from nba.com's daily lineups file, where each team's five is
"Expected" during the day and "Confirmed" once the team says so (with its inactive list).

Before the term moves any number we need to know how early the five firms up, how often "Expected"
turns out right, and whether our server can read nba.com at all. So this watch only records: every
few minutes in the last LEAD before a game, it reads the file and, for each team still to tip off,
keeps every change it sees. Nothing here feeds the model yet.

A day's record (JSON keys as strings):

    {"teams": {espn team: {
        "game": nba game id, "tip": iso, "checked": iso,   checked: the last look before tip-off
        "confirmed_at": iso | None,                         the first look that saw "Confirmed"
        "looks": [{"at": iso, "status": "Expected" | "Confirmed" | ...,
                   "starters": [[nba id, name, position], ...],
                   "inactive": [[nba id, name], ...]}, ...]}}}      a look is added only on a change
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import httpx

from app.scores import Schedule

URL = "https://stats.nba.com/js/data/leaders/00_daily_lineups_{:%Y%m%d}.json"
# stats.nba.com answers browsers; it turns away requests that don't look like a page on nba.com.
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0 Safari/537.36",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "Accept": "application/json",
}
TIMEOUT = 15.0
LEAD = timedelta(minutes=90)  # games starting this soon are watched
# nba.com team codes that ESPN writes differently.
ESPN_CODE = {"GSW": "GS", "NOP": "NO", "NYK": "NY", "SAS": "SA", "UTA": "UTAH", "WAS": "WSH"}


def days_to_watch(schedule: Schedule, now: datetime) -> list[int]:
    """Days with a game that hasn't started yet and starts within LEAD."""
    return sorted(d for d, games in schedule.games.items() if any(now < g.start <= now + LEAD for g in games))


async def fetch(day: date, transport: httpx.AsyncBaseTransport | None = None) -> dict:
    """nba.com's lineups for one (New York) date. Raises when the file can't be read, so the watch
    can say why: refused, missing or changed."""
    async with httpx.AsyncClient(transport=transport, timeout=TIMEOUT, headers=HEADERS) as http:
        resp = await http.get(URL.format(day))
        resp.raise_for_status()
        raw = resp.json()
    if not isinstance(raw, dict) or not isinstance(raw.get("games"), list):
        raise ValueError("no games list in the lineups file")
    return raw


def parse(raw: dict) -> dict[str, dict]:
    """Team code (ESPN's spelling) -> {"game", "status", "starters", "inactive"} for every team listed."""
    out: dict[str, dict] = {}
    for g in raw.get("games") or []:
        for side in ("awayTeam", "homeTeam"):
            team = g.get(side) or {}
            code = str(team.get("teamAbbreviation") or "").upper()
            if not code:
                continue
            players = team.get("players") or []
            starters = [
                [p.get("personId"), p.get("playerName") or "", p.get("position")] for p in players if p.get("position")
            ]
            inactive = [
                [p.get("personId"), p.get("playerName") or ""]
                for p in players
                if str(p.get("rosterStatus") or "").lower() == "inactive"
            ]
            statuses = {p.get("lineupStatus") for p in players if p.get("position")}
            status = statuses.pop() if len(statuses) == 1 else ("Mixed" if statuses else None)
            out[ESPN_CODE.get(code, code)] = {
                "game": str(g.get("gameId") or ""),
                "status": status,
                "starters": starters,
                "inactive": inactive,
            }
    return out


def update(record: dict | None, schedule: Schedule, day: int, listed: dict[str, dict], now: datetime) -> dict:
    """The day's record with this look written in for every team whose game is within LEAD and
    hasn't started. A look is appended only when the status, the five or the inactive list changed."""
    out = {"teams": {k: dict(v) for k, v in ((record or {}).get("teams") or {}).items()}}
    stamp = now.isoformat(timespec="seconds")
    for g in schedule.games.get(day, []):
        if not now < g.start <= now + LEAD:
            continue
        for team in (g.away, g.home):
            code = schedule.teams.get(team, ("", ""))[0].upper()
            seen = listed.get(code)
            entry = out["teams"].get(str(team)) or {
                "game": None,
                "tip": g.start.isoformat(timespec="seconds"),
                "confirmed_at": None,
                "looks": [],
            }
            entry["checked"] = stamp
            if seen is not None:
                look = {"status": seen["status"], "starters": seen["starters"], "inactive": seen["inactive"]}
                last = entry["looks"][-1] if entry["looks"] else None
                if last is None or {k: last[k] for k in look} != look:
                    entry["looks"] = [*entry["looks"], {"at": stamp, **look}]
                entry["game"] = seen["game"] or entry.get("game")
                if seen["status"] == "Confirmed" and not entry.get("confirmed_at"):
                    entry["confirmed_at"] = stamp
            out["teams"][str(team)] = entry
    return out

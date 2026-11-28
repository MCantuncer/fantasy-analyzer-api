"""Tonight's starting fives as nba.com lists them before tip-off, saved night by night.

Can 2026-11-25 (the area map of our misses, dig A): a player who starts tonight but usually comes
off the bench plays 3.6-4.4 minutes more than our line, every season, and a starting-five term cut
the minutes miss by 3.4-5.5% in every one of five seasons, blind. The replay knew the starters from
the box score; live, they come from nba.com's daily lineups file, where each team's five is
"Expected" during the day and "Confirmed" once the team says so (with its inactive list).

Every few minutes in the last LEAD before a game the watch reads the file and, for each team still
to tip off, keeps every change it sees. The first two logged nights (40 team games): the first look
came 84-89 minutes before tip-off and was "Expected"; "Confirmed" came 10-27 minutes before; for 12
teams one starter changed in between.

Since 2026-11-28 a confirmed five feeds tonight's next-game lines (confirmed_fives, then
gamebox.starting_five). Only a confirmed one: replayed over nine seasons, the term at its weight
with one starter wrong in 30% of team games made minutes worse (scripts/starting_five_backtest.py).
An "Expected" five is still only recorded.

The evening look (2026-11-28, Can: a confirmed five comes when Turkey is asleep, and after the day's
first tip no pickup can be made): on 2026-11-28 at 15:30 in New York nba.com already had an
"Expected" five up for all ten teams of the night. In the EVENING before the line log's evening cut
(linelog.early_cut: 23:00 in Istanbul, or the day's first tip-off) the watch also reads every team of
the day, so the five a reader there could have seen is kept as the team's first look, to be set
against its confirmed five later. Recorded only.

A day's record (JSON keys as strings):

    {"teams": {espn team: {
        "game": nba game id, "tip": iso, "checked": iso,   checked: the last look before tip-off
        "confirmed_at": iso | None,                         the first look that saw "Confirmed"
        "looks": [{"at": iso, "status": "Expected" | "Confirmed" | ...,
                   "starters": [[nba id, name, position], ...],
                   "inactive": [[nba id, name], ...]}, ...]}}}      a look is added only on a change
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime, timedelta

import httpx

from app.linelog import early_cut
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
EVENING = timedelta(minutes=20)  # the evening look is taken this long before the line log's evening cut
# nba.com team codes that ESPN writes differently.
ESPN_CODE = {"GSW": "GS", "NOP": "NO", "NYK": "NY", "SAS": "SA", "UTA": "UTAH", "WAS": "WSH"}


def days_to_watch(schedule: Schedule, now: datetime) -> list[int]:
    """Days with a game that hasn't started yet and starts within LEAD."""
    return sorted(d for d, games in schedule.games.items() if any(now < g.start <= now + LEAD for g in games))


def evening_days(schedule: Schedule, now: datetime) -> list[int]:
    """Days whose evening cut (linelog.early_cut) is within EVENING: every team gets a look now."""
    out = []
    for day in schedule.games:
        cut = early_cut(schedule, day)
        if cut is not None and cut - EVENING <= now <= cut:
            out.append(day)
    return sorted(out)


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


def update(
    record: dict | None,
    schedule: Schedule,
    day: int,
    listed: dict[str, dict],
    now: datetime,
    evening: bool = False,
) -> dict:
    """The day's record with this look written in for every team whose game is within LEAD and
    hasn't started; with ``evening``, for every team still to tip off (the evening look). A look is
    appended only when the status, the five or the inactive list changed."""
    out = {"teams": {k: dict(v) for k, v in ((record or {}).get("teams") or {}).items()}}
    stamp = now.isoformat(timespec="seconds")
    for g in schedule.games.get(day, []):
        if not now < g.start or not (evening or g.start <= now + LEAD):
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


def confirmed_fives(record: dict | None, match: Callable[[str, int], int | None]) -> dict[int, frozenset[int]]:
    """Team -> the players of its confirmed five, from a day's record: each team whose last look
    reads "Confirmed" with five starters, every one found by ``match(nba.com name, team)`` (our
    player id, or None). A five with a starter we can't name is left out whole: half a five would
    call a starter a bench player."""
    out: dict[int, frozenset[int]] = {}
    for team, entry in ((record or {}).get("teams") or {}).items():
        looks = entry.get("looks") or []
        if not looks or looks[-1].get("status") != "Confirmed":
            continue
        found = {match(name, int(team)) for _nba, name, *_ in looks[-1].get("starters") or []}
        if len(found) == 5 and None not in found:
            out[int(team)] = frozenset(found)
    return out

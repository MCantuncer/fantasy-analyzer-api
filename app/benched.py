"""Who sat a game without being hurt, night by night, from ESPN's box scores.

A game log shows the games a player played. It can't say why he missed one: an injury, rest, or his
coach leaving him on the bench. ESPN's box score can: every player in uniform who didn't play is
listed with a reason ("COACH'S DECISION", "RIGHT KNEE CONTUSION", "PERSONAL", "NOT WITH TEAM"). A
player his coach just sat is out of favor, and plays less than his line says when he is back on the
floor (app/analysis/gamebox.py, ``COACH_SAT_MINUTES``).

A finished game never changes, so each night is read once (one summary a game) and saved for good:
{player: reason} for every player who did not play, JSON keys as strings. All reasons are kept, not
only the coach's, so other absences can be tested later.
"""

from __future__ import annotations

import asyncio

import httpx

from app.garbagetime import SUMMARY, TIMEOUT, USER_AGENT

COACH_DECISION = "COACH'S DECISION"
LOOKBACK_NIGHTS = 6  # game nights read on a first run: a little more than the days a benching counts
KEEP_DAYS = 10  # saved nights the league box reads back


def parse_sat(raw: dict) -> dict[int, str] | None:
    """player -> ESPN's reason, for each player in one game's box score who did not play.
    None when the summary has no box score for both teams (a game not played yet, or ESPN's hiccup)."""
    teams = ((raw or {}).get("boxscore") or {}).get("players") or []
    if len(teams) != 2:
        return None
    out: dict[int, str] = {}
    for team in teams:
        groups = team.get("statistics") or []
        athletes = groups[0].get("athletes") if groups else None
        if not athletes:
            return None
        for a in athletes:
            pid = (a.get("athlete") or {}).get("id")
            if a.get("didNotPlay") and pid:
                out[int(pid)] = str(a.get("reason") or "").strip().upper()
    return out


async def fetch_night(game_ids: list[int], transport: httpx.AsyncBaseTransport | None = None) -> dict[int, str] | None:
    """Everyone who did not play in these finished games, with ESPN's reason. None unless every game
    answered: a night half read would look like nobody sat in the other games."""
    async with httpx.AsyncClient(transport=transport, timeout=TIMEOUT, headers={"User-Agent": USER_AGENT}) as http:

        async def one(gid: int) -> dict[int, str] | None:
            try:
                resp = await http.get(SUMMARY, params={"event": gid})
                return parse_sat(resp.json()) if resp.status_code == 200 else None
            except (httpx.HTTPError, ValueError):
                return None

        games = await asyncio.gather(*(one(g) for g in game_ids))
    if any(g is None for g in games):
        return None
    return {pid: reason for g in games for pid, reason in g.items()}


def coach_days(nights: dict[int, dict]) -> dict[int, list[int]]:
    """player -> the days he sat by coach's decision, in order, from saved nights (day -> record)."""
    out: dict[int, list[int]] = {}
    for day in sorted(nights):
        for pid, reason in nights[day].items():
            if COACH_DECISION in reason:
                out.setdefault(int(pid), []).append(day)
    return out

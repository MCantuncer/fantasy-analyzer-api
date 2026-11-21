"""Each player's injury status as it stood before his game tipped off, saved day by day.

Can 2026-11-17: whether tonight's lineup news (who is out, doubtful, back) sharpens our call for a
game can only be tested fairly with the news as it stood before tip-off, and ESPN keeps only today's
tags. So a watch reads ESPN's tags every few minutes and, for each team playing in the next day,
keeps the last reading before its game starts. Once a team's game tips off its entries stop moving:
what is saved is what anyone could have known at tip-off (to within one look).

A day's record: {"teams": {team: {"tip": iso, "checked": iso}}, "players": {player: {"team", "status"}}},
JSON keys as strings. Players are ESPN's most-owned (news.facts.parse_states of player_states), which
covers every team's rotation.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from app.news.facts import PlayerState
from app.scores import Schedule

AHEAD = timedelta(hours=24)  # days with a game starting this soon are watched


def days_to_watch(schedule: Schedule, now: datetime) -> list[int]:
    """Days with a game that hasn't started yet and starts within AHEAD."""
    return sorted(d for d, games in schedule.games.items() if any(now < g.start <= now + AHEAD for g in games))


def update(record: dict | None, schedule: Schedule, day: int, states: dict[int, PlayerState], now: datetime) -> dict:
    """The day's record with this look written in for every team whose game hasn't started."""
    out = {"teams": dict((record or {}).get("teams", {})), "players": dict((record or {}).get("players", {}))}
    tips = {team: g.start for g in schedule.games.get(day, []) for team in (g.away, g.home)}
    waiting = {team for team, start in tips.items() if now < start}
    if not waiting:
        return out
    stamp = now.isoformat(timespec="seconds")
    for team in waiting:
        out["teams"][str(team)] = {"tip": tips[team].isoformat(timespec="seconds"), "checked": stamp}
    # A team still waiting gets a fresh list: a player traded away or dropped from the pool leaves it.
    out["players"] = {pid: v for pid, v in out["players"].items() if v["team"] not in waiting}
    for s in states.values():
        if s.pro_team_id in waiting:
            out["players"][str(s.id)] = {"team": s.pro_team_id, "status": s.status}
    return out


def ruled_out(record: dict | None) -> set[int]:
    """Players whose last saved tag before their game is OUT (Our call's who plays tonight)."""
    players = (record or {}).get("players", {})
    return {int(pid) for pid, v in players.items() if v.get("status") == "OUT"}

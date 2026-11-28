"""Each player's next-game line as it stood at tip-off, saved day by day, with what he then did.

Can 2026-11-19: save every player's estimate for the day and his result, so over time we see who is
"consistent" (lands near his line) and who "jumps" (far above or below it, both ways). The line must
be the one anyone could see before the game, so a watch looks at the lines in the last hours before
tip-off and, for each team, keeps its last look before its game starts (as app/pregame.py does for
injury tags). Once the night has settled, the box build fills in what each player did.

Built to stay small: one row per NBA day, numbers in fixed-order arrays instead of keyed objects, ids
as JSON keys. Since 2026-11-27 the league is read with main.LINE_LOG_FREE_AGENTS free agents, so with
the rosters every NBA player in tonight's games has a line (the black box's team and opponent misses
need them all); a full night is about 450 players, 60-80 KB before Postgres compresses it. A day's record:

    {"teams": {team: [tip, checked]},             ISO times
     "lines": {player: [team, *STATS]},           his line, last look before his team tipped off
     "engine": {player: [*STATS]},                the same look before the trees (analysis/blackbox.py)
     "build": {player: [[*STATS], [[kind, text, minutes, *REASON_STATS], ...]]}
                                                  his plain-night line (usual) and the reasons that moved it
     "results": {player: [*STATS] | []}}          once settled; [] = his team played, he didn't

"engine" (since 2026-11-27) is what the engine's own misses are counted against, so the trees can learn
from past misses without learning from their own corrections.

"build" (since 2026-11-28, Can: "we need some whys for each estimation, especially for the Kupons") keeps
how each line was built at that same look: his usual line and the reasons the game moved it (gamebox.py,
biggest first, as the player card shows them), so a bet's why stays the one anyone could see before tip.

A player whose team has no game rows that night (postponed) gets no result.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime, timedelta

from app.analysis.form import AST, BLK, FGA, FGM, FTA, FTM, MINUTES, PTS, REB, STL, TO, TPM
from app.analysis.gamebox import GameRow
from app.analysis.model import CategoryModel
from app.analysis.report import next_game_parts
from app.espn.parse import League
from app.schemas import NextGameLine
from app.scores import Schedule

# The order of every saved array: NextGameLine's field and the game row's stat id.
STATS: tuple[tuple[str, int], ...] = (
    ("minutes", MINUTES),
    ("pts", PTS),
    ("reb", REB),
    ("ast", AST),
    ("stl", STL),
    ("blk", BLK),
    ("three_pm", TPM),
    ("to", TO),
    ("fgm", FGM),
    ("fga", FGA),
    ("ftm", FTM),
    ("fta", FTA),
)
# Builds are kept for players this many minutes and up: the book prices rotation players only, and the
# rest of the 450 would add most of the size for nothing.
BUILD_MIN_MINUTES = 10.0
# A saved reason's stat changes, in order (NextGameReason's fields).
REASON_STATS = ("pts", "reb", "ast", "stl", "blk", "three_pm", "to")
# The stats the read-back summarises (makes and attempts stay saved for FG% and FT% later).
SUMMARY = ("minutes", "pts", "reb", "ast", "stl", "blk", "three_pm", "to")
AHEAD = timedelta(hours=18)  # days with a game starting this soon are looked at: the whole night, like the iddaa log


def days_to_watch(schedule: Schedule, now: datetime) -> list[int]:
    """Days with a game that hasn't started yet and starts within AHEAD."""
    return sorted(d for d, games in schedule.games.items() if any(now < g.start <= now + AHEAD for g in games))


Look = tuple[int, NextGameLine] | tuple[int, NextGameLine, dict[int, float] | None]


def tonight_lines(league: League) -> dict[int, dict[int, Look]]:
    """Every player's next-game line in the league's pool (rosters and the free agents it reads: for the
    line log, every NBA player), by
    the day of that game: day -> player -> (NBA team, line, the engine's line before the trees). Players
    out or without a game are left out."""
    pool = [*(p for t in league.teams for p in t.roster), *league.free_agents]
    by_team: dict[int, list] = defaultdict(list)
    for p in pool:
        by_team[p.pro_team_id].append(p)
    model, sums = CategoryModel(league), {}
    out: dict[int, dict[int, Look]] = defaultdict(dict)
    for p in pool:
        m = p.game_context
        if m is None:
            continue
        line, engine = next_game_parts(model, p, by_team[p.pro_team_id], sums)
        if line is not None:
            out[m.day][p.id] = (p.pro_team_id, line, engine)
    return dict(out)


def pack(line: NextGameLine) -> list[float]:
    return [round(getattr(line, name), 1) for name, _ in STATS]


def pack_build(line: NextGameLine) -> list | None:
    """His usual line and the reasons that moved it, as saved under "build"; None without a usual line."""
    if line.usual is None:
        return None
    usual = [round(getattr(line.usual, name), 1) for name, _ in STATS]
    reasons = [
        [r.kind, r.text, round(r.minutes, 1), *(round(getattr(r, name), 2) for name in REASON_STATS)]
        for r in line.reasons
    ]
    return [usual, reasons]


def update(record: dict | None, schedule: Schedule, day: int, lines: dict[int, Look], now: datetime) -> dict:
    """The day's record with this look written in for every team whose game hasn't started: its
    players' lines replace the last look's (a player ruled out since then leaves it)."""
    out = {
        "teams": dict((record or {}).get("teams", {})),
        "lines": dict((record or {}).get("lines", {})),
        "engine": dict((record or {}).get("engine", {})),
        "build": dict((record or {}).get("build", {})),
        "results": dict((record or {}).get("results", {})),
    }
    tips = {team: g.start for g in schedule.games.get(day, []) for team in (g.away, g.home)}
    waiting = {team for team, start in tips.items() if now < start}
    if not waiting:
        return out
    stamp = now.isoformat(timespec="seconds")
    for team in waiting:
        out["teams"][str(team)] = [tips[team].isoformat(timespec="seconds"), stamp]
    gone = {pid for pid, v in out["lines"].items() if v[0] in waiting}
    out["lines"] = {pid: v for pid, v in out["lines"].items() if pid not in gone}
    out["engine"] = {pid: v for pid, v in out["engine"].items() if pid not in gone}
    out["build"] = {pid: v for pid, v in out["build"].items() if pid not in gone}
    for pid, (team, line, *rest) in lines.items():
        if team in waiting:
            out["lines"][str(pid)] = [team, *pack(line)]
            build = pack_build(line) if line.minutes >= BUILD_MIN_MINUTES else None
            if build is not None:
                out["build"][str(pid)] = build
            engine = rest[0] if rest else None
            if engine is not None:
                out["engine"][str(pid)] = [round(engine.get(stat, 0.0), 1) for _, stat in STATS]
    return out


def settle(record: dict, rows: Iterable[GameRow]) -> dict:
    """The record with each lined player's result from the night's game rows."""
    played = {r.player: r for r in rows}
    teams = {r.team for r in played.values()}
    results = {}
    for pid, (team, *_) in record.get("lines", {}).items():
        r = played.get(int(pid))
        if r is not None:
            results[pid] = [round(r.line.get(stat, 0.0), 1) for _, stat in STATS]
        elif team in teams:
            results[pid] = []
    return {**record, "results": results}


def summary(records: Iterable[dict]) -> list[dict]:
    """Per player over these settled days: games, nights he had a line but didn't play, and for each
    SUMMARY stat his average line and result, the average miss (result less line: + he beat it) and
    the swing, the typical size of a miss either way (root mean square). A small swing is a player who
    lands near his line; a big one, a player who jumps above and below it."""
    index = {name: i for i, (name, _) in enumerate(STATS)}
    games: dict[str, list[tuple[list[float], list[float]]]] = defaultdict(list)
    team: dict[str, int] = {}
    dnp: dict[str, int] = defaultdict(int)
    for rec in records:
        results = rec.get("results", {})
        for pid, (t, *line) in rec.get("lines", {}).items():
            got = results.get(pid)
            if got is None:
                continue
            team[pid] = t
            if got:
                games[pid].append((line, got))
            else:
                dnp[pid] += 1
    out = []
    for pid in team:
        played = games.get(pid, [])
        stats = {}
        for name in SUMMARY:
            i = index[name]
            misses = [got[i] - line[i] for line, got in played]
            n = len(misses)
            stats[name] = {
                "line": round(sum(line[i] for line, _ in played) / n, 1) if n else 0.0,
                "actual": round(sum(got[i] for _, got in played) / n, 1) if n else 0.0,
                "miss": round(sum(misses) / n, 2) if n else 0.0,
                "swing": round(math.sqrt(sum(m * m for m in misses) / n), 2) if n else 0.0,
            }
        out.append({"player_id": int(pid), "team": team[pid], "games": len(played), "dnp": dnp[pid], "stats": stats})
    return out

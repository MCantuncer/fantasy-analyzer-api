"""How many more games an injured player misses, from how many he has missed so far.

ESPN gives an injury status but no return date. A flat guess (every OUT player back
in two weeks) is wrong both ways: most absences end after a game or two, while a
player who has already missed ten games in a row usually misses about as many more.
Game logs of past seasons show that the longer a player has been out, the longer he
is likely to stay out, so the expected games still to miss come from the games he has
missed in a row so far (``REMAINING``).

A player listed OUT who hasn't missed a game yet (hurt in his last game) misses at
least the next one: one game plus what a player out for one game still misses.

The table is the mean (not the median) of what followed in past seasons, so the games
a player is expected to play add up right over a season. It's fitted on the last
finished season: absences have been getting shorter (2024-25 and 2025-26 both came in
below the season before), so older seasons overstate them. ``scripts/absence_backtest.py``
fits it and checks the rule "fit on the last season" on the seasons after. A player's
games-played record (durability.py) was tried as a second input and didn't help: it
says how often a player gets hurt, not how long he stays out.
"""

from __future__ import annotations

import bisect

from app.espn.parse import IR_SLOT, League, Player

# Expected team games still to miss, by team games missed in a row so far (from that
# number up to the next). Fitted on 2025-26 by scripts/absence_backtest.py; made
# non-decreasing, since the dip past 30 games came from absences cut off by the season's end.
REMAINING = (
    (1, 3.1),
    (2, 5.5),
    (3, 7.2),
    (4, 8.5),
    (5, 10.2),
    (7, 12.0),
    (10, 13.7),
    (14, 14.8),
    (20, 16.6),
    (30, 16.6),
)
LONG_OUT = {"OUT", "INJURY_RESERVE"}


def expected_missed(streak: int) -> float:
    """Expected team games still to miss for a player listed out who has missed ``streak`` in a row."""
    if streak <= 0:
        return 1.0 + REMAINING[0][1]
    i = bisect.bisect_right([k for k, _ in REMAINING], streak) - 1
    return REMAINING[i][1]


def is_out(p: Player) -> bool:
    """Listed out or on the IR slot: the statuses the table is for. Day-to-day and suspensions aren't."""
    return p.injury_status in LONG_OUT or p.lineup_slot == IR_SLOT


def missed_in_a_row(team_days: list[int], played: set[int], today: int, before: int = 0) -> int:
    """Team games before ``today`` since the player's last game.

    ``before`` is the run he ended last season with, counted only when he hasn't played
    yet this season.
    """
    run = 0
    for d in sorted((d for d in team_days if d < today), reverse=True):
        if d in played:
            return run
        run += 1
    return run + before


def played_days(logs: list[dict], season: int) -> dict[int, set[int]]:
    """Days each player played in one season, from ESPN's game lines (stat split 5)."""
    out: dict[int, set[int]] = {}
    for e in logs:
        p = e.get("player") or e
        if "id" not in p:
            continue
        days = out.setdefault(int(p["id"]), set())
        for s in p.get("stats") or []:
            if s.get("seasonId") != season or s.get("statSplitTypeId") != 5 or s.get("statSourceId"):
                continue
            if (s.get("stats") or {}).get("42") and s.get("scoringPeriodId") is not None:
                days.add(int(s["scoringPeriodId"]))
    return out


def season_end_run(logs: list[dict], season: int) -> dict[int, int]:
    """Games each player missed in a row at the end of one season, from his game lines.

    ESPN lists a line for every team game a rostered player was on the team for, played or not.
    """
    out: dict[int, int] = {}
    for e in logs:
        p = e.get("player") or e
        if "id" not in p:
            continue
        lines = sorted(
            (int(s["scoringPeriodId"]), bool((s.get("stats") or {}).get("42")))
            for s in p.get("stats") or []
            if s.get("seasonId") == season
            and s.get("statSplitTypeId") == 5
            and not s.get("statSourceId")
            and s.get("scoringPeriodId") is not None
        )
        run = 0
        for _, played in reversed(lines):
            if played:
                break
            run += 1
        out[int(p["id"])] = run
    return out


def attach(league: League, played: dict[int, set[int]], last_season_run: dict[int, int] | None = None) -> None:
    """Sets the expected games still to miss of every player listed out.

    ``played`` holds the days each of them played this season; ``last_season_run`` the
    games they missed in a row at the end of last season.
    """
    cal = league.calendar
    if cal is None:
        return
    last_season_run = last_season_run or {}
    for p in [p for t in league.teams for p in t.roster] + list(league.free_agents):
        if not is_out(p) or p.id not in played:
            continue
        days = sorted(cal.team_days.get(p.pro_team_id, ()))
        streak = missed_in_a_row(days, played[p.id], cal.today, last_season_run.get(p.id, 0))
        p.games_out = expected_missed(streak)

"""This matchup week at a glance: each side's category scores and how many games each roster has.

Fantasy matchup weeks run Monday to Sunday on US dates (the first week starts on opening night).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

from app.espn.parse import IR_SLOT, STAT_NAMES
from app.scores import Schedule

# ESPN marks each category of a matchup side as one of these.
RESULTS = {"WIN": "W", "LOSS": "L", "TIE": "T"}
# Start times are UTC; US evening games land on the next UTC day, so step back 8 hours.
US_OFFSET = timedelta(hours=8)


def us_date(moment: datetime) -> date:
    return (moment - US_OFFSET).date()


def week_bounds(now: datetime) -> tuple[date, date]:
    """Monday and Sunday of the current matchup week, as US dates."""
    today = us_date(now)
    monday = today - timedelta(days=today.weekday())
    return monday, monday + timedelta(days=6)


@dataclass(frozen=True)
class CategoryScore:
    stat: str  # "PTS", "FG%", ...
    value: float
    result: str  # W, L or T


def category_scores(side: dict, stat_ids: list[int]) -> list[CategoryScore]:
    """One matchup side's running total per scoring category, in the league's category order.

    Empty when ESPN sent no per-category scores (the ``mScoreboard`` view is needed for them).
    """
    by_stat = ((side.get("cumulativeScore") or {}).get("scoreByStat")) or {}
    out = []
    for sid in stat_ids:
        item = by_stat.get(str(sid))
        if not item or item.get("ineligible"):
            continue
        out.append(
            CategoryScore(
                stat=STAT_NAMES.get(sid, f"stat{sid}"),
                value=float(item.get("score") or 0.0),
                result=RESULTS.get(str(item.get("result") or "").upper(), "T"),
            )
        )
    return out


@dataclass(frozen=True)
class WeekGames:
    today: int  # games today (US date) by rostered players, started or not
    left: int  # games this week that have not tipped off yet
    total: int  # every game this week


def week_games(pro_team_ids: list[int], schedule: Schedule, now: datetime) -> WeekGames:
    """Player-games for one roster this week: one per rostered player whose NBA team plays.

    Pass the pro team of every player outside the IR slot (see ``roster_pro_teams``).
    """
    monday, sunday = week_bounds(now)
    today = us_date(now)
    on_today = left = total = 0
    for day, games in schedule.games.items():
        d = schedule.date_of(day)
        if d is None or not monday <= d <= sunday:
            continue
        for g in games:
            n = sum(1 for t in pro_team_ids if t in (g.home, g.away))
            if not n:
                continue
            total += n
            if d == today:
                on_today += n
            if g.start > now:
                left += n
    return WeekGames(today=on_today, left=left, total=total)


def roster_pro_teams(players) -> list[int]:
    """The NBA team of each rostered player who can score this week (everyone outside the IR slot)."""
    return [p.pro_team_id for p in players if p.lineup_slot != IR_SLOT and p.pro_team_id]

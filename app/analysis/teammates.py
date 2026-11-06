"""Teammates of an injured player play more minutes while he's out, and the projection should say so.

A player's projection blends ESPN's preseason line with his season so far (parse.py). A
teammate's bigger role during an absence shows up there only as the games without the
injured player pile up, and it keeps counting after the injured player is back. Here the
role change is projected directly. For every regular listed out with games still to miss
(absence.py), the minutes each teammate gains while he's out come from the games the team
has played without him (injury.py). That method beats "no change" on real absences: 3.85
against 4.23 minutes a game of error over 2,084 teammate cases in 2025-26
(scripts/injury_backtest.py).

  * While the injured player is out (until the day he's expected back), the teammate plays
    with the measured gain. After that, he doesn't.
  * With several teammates out at once, only the largest gain counts: they were measured on
    mostly the same games, so adding them up would count those minutes more than once.
  * The part of the gain this season's minutes already hold is taken out: the share of his
    games played without the injured player, times the weight the blend gives this season's
    minutes (MINUTES_WEIGHT_GAMES). Otherwise a role that ends when the injured player returns
    would count for the rest of the season.

``per_game`` gets the rest-of-season average of that, so trade values and season odds use
it. The model's weekly projections use the share of that week's games the injured player
misses instead (CategoryModel.horizon_row). Counting stats follow the minutes, as in the blend.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.analysis.injury import Impact, Season
from app.espn.parse import MINUTES, MINUTES_WEIGHT_GAMES, League, Player, with_minutes

MIN_GAMES_OUT = 3.0  # team games still to miss before a teammate's role is moved
MIN_MINUTES = 20.0  # the injured player's minutes a game: a regular, whose minutes go somewhere
MAX_MINUTES = 40.0
MIN_SHIFT = 0.1  # minutes a game; smaller moves are left alone


@dataclass(frozen=True)
class Shift:
    """One teammate's projected minutes from one absence."""

    absent: int
    mate: int
    change: float  # minutes a game he gains while the injured player is out
    back: int  # day the injured player is expected back
    season: float  # change to his rest-of-season minutes a game, net of what his blend already holds


def without_share(season: Season, team: int, absent: int, mate: int) -> tuple[int, int]:
    """This season: games the teammate played for ``team`` without ``absent``, and games he played in all."""
    mine = [g for g in season.team_games(team) if season.line(g, mate, team)]
    return sum(1 for g in mine if season.line(g, absent, team) is None), len(mine)


def season_days(league: League) -> list[int]:
    """The days left in the regular season, as the model's season horizon counts them."""
    cal = league.calendar
    if cal is None:
        return []
    last = league.regular_season_periods or max(cal.period_days)
    return [d for p in range(league.current_period, last + 1) for d in cal.days_left(p)]


def back_day(league: League, p: Player) -> int:
    """First day an injured player is expected back: his team's first game after the ones he misses."""
    cal = league.calendar
    today = cal.today if cal else 1
    upcoming = sorted(d for d in (cal.team_days.get(p.pro_team_id, ()) if cal else ()) if d >= today)
    missed = round(p.games_out or 0.0)
    return upcoming[missed] if missed < len(upcoming) else (upcoming[-1] + 1 if upcoming else today)


def shifts(league: League, absent: Player, impact: Impact, season: Season) -> list[Shift]:
    """How much each teammate's projected minutes move while ``absent`` is out."""
    cal = league.calendar
    if cal is None or (absent.games_out or 0.0) < MIN_GAMES_OUT or impact.minutes < MIN_MINUTES:
        return []
    team = absent.pro_team_id
    games = cal.team_days.get(team, frozenset())
    left = [d for d in season_days(league) if d in games]
    if not left:
        return []
    back = back_day(league, absent)
    ahead = sum(d < back for d in left) / len(left)
    out = []
    for b in impact.beneficiaries:
        without, played = without_share(season, team, absent.id, b.player_id)
        already = played / (played + MINUTES_WEIGHT_GAMES) * without / played if played else 0.0
        net = b.minutes_change * (ahead - already)
        if abs(b.minutes_change) >= MIN_SHIFT:
            out.append(Shift(absent.id, b.player_id, b.minutes_change, back, net))
    return out


def apply(league: League, moved: list[Shift]) -> None:
    """Moves each teammate's projected minutes, with his counting stats scaled along."""
    # Several teammates out at once: each one's measured gain mostly comes from the same games
    # (the team played without both), so adding them up counts those minutes twice or more.
    # Only the largest counts.
    biggest: dict[int, Shift] = {}
    for s in moved:
        if s.mate not in biggest or abs(s.change) > abs(biggest[s.mate].change):
            biggest[s.mate] = s
    for p in [p for t in league.teams for p in t.roster] + list(league.free_agents):
        s = biggest.get(p.id)
        current = p.per_game.get(MINUTES, 0.0)
        if s is None or current <= 0:
            continue
        p.per_game = with_minutes(p.per_game, min(max(current + s.season, 0.0), MAX_MINUTES))
        p.minute_shifts = ((s.change, s.back),)


def absent_regulars(league: League) -> list[Player]:
    """Players listed out with enough games still to miss to move their teammates' roles."""
    pool = [p for t in league.teams for p in t.roster] + list(league.free_agents)
    return [p for p in pool if p.pro_team_id and (p.games_out or 0.0) >= MIN_GAMES_OUT]

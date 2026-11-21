"""Teammates ruled out of his next game: the minutes they leave behind go mostly to his role.

ESPN lists a player OUT once he is ruled out. teammates.py moves minutes only for absences of
MIN_GAMES_OUT team games or more, measured player by player, so a regular who sits one night
moves nothing there. For the next-game line, each regular (REGULAR_MINUTES or more) listed OUT
who played in the last week gives every teammate a share of his minutes: SAME_ROLE of them to a
player in the same role (guard, wing or big, from assists and rebounds per 36) and OTHER_ROLE to
the others: a 30-minute guard ruled out gives each guard about 2 minutes and everyone else about
1. Fitted on five seasons, one season left out at a time, it lowers the squared miss in his
minutes by 3.6% to 5.2% in every season; in games where a same-role regular sat, the mean miss
goes from 5.40 to 5.15 minutes (scripts/next_game_backtest.py). Once the minutes are counted, his
per-minute line doesn't move: no extra shots or assists when a teammate sits. Opponents' regulars
sitting moved nothing either.
"""

from __future__ import annotations

from collections.abc import Iterable

from app.analysis.gamebox import BACK_MINUTES
from app.espn.parse import IR_SLOT, MINUTES, Player
from app.playchance import DAY_TO_DAY, play_chance

AST, REB = 3, 6
GUARD_ASSISTS = 4.5  # assists per 36 that make a guard
BIG_REBOUNDS = 8.5  # rebounds per 36 that make a big (when not a guard)
REGULAR_MINUTES = 20.0
SAME_ROLE = 0.065  # minutes he gains per minute of a same-role teammate ruled out
OTHER_ROLE = 0.039  # per minute of a teammate in another role
MAX_MINUTES = 40.0


def role(line: dict[int, float]) -> int:
    """0 guard, 1 wing, 2 big, from a per-game line's assists and rebounds per 36 minutes."""
    minutes = line.get(MINUTES, 0.0)
    if minutes <= 0:
        return 1
    if line.get(AST, 0.0) / minutes * 36 >= GUARD_ASSISTS:
        return 0
    return 2 if line.get(REB, 0.0) / minutes * 36 >= BIG_REBOUNDS else 1


def ruled_out(p: Player) -> bool:
    """A regular listed OUT who played in the last week: a fresh absence his teammates' recent minutes don't hold."""
    return (
        p.injury_status == "OUT" and p.per_game.get(MINUTES, 0.0) >= REGULAR_MINUTES and p.recent_minutes[0] is not None
    )


def came_back(p: Player) -> bool:
    """A regular with no injury tag left who missed his team's last game or more: back for his next
    game, as far as ESPN knows. His teammates' minutes and shots lately were partly his
    (gamebox BACK_SHARE, usage_back). A player his coach sat through those games isn't coming back
    from anything."""
    return p.fully_healthy and _returning(p)


# A regular back from an absence with a day-to-day tag left (NBA: probable or questionable) counts as
# back by his chance to play, read TAGGED_BACK_HOURS before tip-off (app/playchance.py), in his
# teammates' minutes only: on 2021-22 to 2025-26 NBA injury reports, counting such returners lowered
# teammates' squared miss in minutes in every season; their points and assists didn't move the same way.
TAGGED_BACK_HOURS = 3.0


def came_back_tagged(p: Player) -> float:
    """His chance to play, for a regular back from an absence who still has a day-to-day tag; 0 otherwise."""
    if p.injury_status != "DAY_TO_DAY" or p.lineup_slot == IR_SLOT or not _returning(p):
        return 0.0
    m = p.game_context
    assert m is not None and m.form is not None
    return play_chance(DAY_TO_DAY, TAGGED_BACK_HOURS, m.form.minutes_long)


def _returning(p: Player) -> bool:
    m = p.game_context
    if m is None or m.form is None:
        return False
    if m.games_missed < 1 or m.form.minutes_long < BACK_MINUTES:
        return False
    return m.sat_ago is None or m.sat_ago >= m.day - m.form.last_day


def minutes_gained(p: Player, teammates: Iterable[Player]) -> float:
    """Minutes his next game gains from teammates ruled out of it."""
    mine = role(p.per_game)
    return sum(
        (SAME_ROLE if role(t.per_game) == mine else OTHER_ROLE) * t.per_game.get(MINUTES, 0.0)
        for t in teammates
        if t.id != p.id and ruled_out(t)
    )

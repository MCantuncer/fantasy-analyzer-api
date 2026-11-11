"""One fantasy team's daily lineup, as ESPN has it for one scoring period (a day).

NBA leagues on ESPN set a lineup each day, and only players in a starting slot count for the
matchup: bench and IR lines count for nothing. The league load keeps rosters for minutes, so
the Scores page reads the lineup on its own (``forTeamId``, about a tenth of the whole league's
rosters) to follow a change made in the ESPN app within a minute.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.scores import Schedule, ScheduledGame

# ESPN's NBA lineup slot ids.
SLOT_NAMES = {
    0: "PG",
    1: "SG",
    2: "SF",
    3: "PF",
    4: "C",
    5: "G",
    6: "F",
    7: "SG/SF",
    8: "G/F",
    9: "PF/C",
    10: "F/C",
    11: "UTIL",
    12: "BE",
    13: "IR",
}
BENCH, IR = 12, 13

# ESPN stat ids in a player's per-game ``averageStats``.
AVG_STATS = {
    "min": 40,
    "fgm": 13,
    "fga": 14,
    "ftm": 15,
    "fta": 16,
    "tpm": 17,
    "reb": 6,
    "ast": 3,
    "stl": 2,
    "blk": 1,
    "to": 11,
    "pts": 0,
}
GAMES_PLAYED = 42


def slot_name(slot_id: int) -> str:
    return SLOT_NAMES.get(slot_id, f"Slot {slot_id}")


@dataclass
class Spot:
    player_id: int
    name: str
    pro_team_id: int
    slot_id: int
    injury_status: str
    game_id: int | None  # his NBA team's game that day, if any
    eligible: tuple[int, ...] = ()  # starting slot ids he can fill
    season_avg: dict[str, float] | None = None  # this season's per-game line, with "gp"; None before his first game

    @property
    def starting(self) -> bool:
        return self.slot_id not in (BENCH, IR)


def season_average(player: dict, season: int) -> dict[str, float] | None:
    """His real per-game line this season, from the stats ESPN sends with the roster (not a projection)."""
    for s in player.get("stats", []):
        if s.get("seasonId") != season or s.get("statSourceId") != 0 or s.get("statSplitTypeId") != 0:
            continue
        avg = {int(k): float(v) for k, v in (s.get("averageStats") or {}).items()}
        gp = int(avg.get(GAMES_PLAYED, 0) or 0)
        if gp <= 0:
            return None
        return {"gp": gp, **{name: round(avg.get(i, 0.0), 2) for name, i in AVG_STATS.items()}}
    return None


def next_game(schedule: Schedule, pro_team_id: int, after: int) -> ScheduledGame | None:
    """His NBA team's first game after a day."""
    for day in sorted(d for d in schedule.games if d > after):
        g = next((g for g in schedule.games[day] if pro_team_id in (g.away, g.home)), None)
        if g:
            return g
    return None


def week_of(schedule: Schedule, day: int) -> list[int]:
    """The days of the Monday-to-Sunday week (US dates) a day falls in, as fantasy matchup weeks run.
    Opening week starts on opening night (day 1)."""
    d = schedule.date_of(day)
    if d is None:
        return [day]
    first = day - d.weekday()
    return [x for x in range(first, first + 7) if x >= 1]


def team_days(schedule: Schedule, pro_team_id: int, days: list[int]) -> list[int]:
    """The days among ``days`` on which his NBA team plays."""
    return [d for d in days if any(pro_team_id in (g.away, g.home) for g in schedule.games.get(d, []))]


def parse_lineup(raw: dict, team_id: int, day: int, schedule: Schedule, season: int = 0) -> list[Spot]:
    """The team's players in ESPN's slot order: starting slots, then the bench, then IR."""
    team = next((t for t in raw.get("teams", []) if int(t.get("id", 0)) == team_id), None)
    if team is None:
        return []
    games = {}
    for g in schedule.games.get(day, []):
        games[g.away] = g.id
        games[g.home] = g.id
    spots = []
    for e in (team.get("roster") or {}).get("entries", []):
        p = (e.get("playerPoolEntry") or {}).get("player") or {}
        if "id" not in p:
            continue
        pro = int(p.get("proTeamId", 0))
        spots.append(
            Spot(
                player_id=int(p["id"]),
                name=p.get("fullName") or str(p["id"]),
                pro_team_id=pro,
                slot_id=int(e.get("lineupSlotId", BENCH)),
                injury_status=p.get("injuryStatus") or "ACTIVE",
                game_id=games.get(pro),
                eligible=tuple(int(x) for x in p.get("eligibleSlots", []) if int(x) not in (BENCH, IR)),
                season_avg=season_average(p, season),
            )
        )
    # Starting slots in ESPN's order, then bench, then IR; players in the same kind of slot by name.
    return sorted(spots, key=lambda s: (s.slot_id, s.name))


def open_slots(spots: list[Spot], counts: dict[int, int]) -> list[int]:
    """Starting slots nobody fills, one entry per empty seat."""
    filled: dict[int, int] = {}
    for s in spots:
        filled[s.slot_id] = filled.get(s.slot_id, 0) + 1
    return [slot for slot, n in sorted(counts.items()) for _ in range(max(0, n - filled.get(slot, 0)))]

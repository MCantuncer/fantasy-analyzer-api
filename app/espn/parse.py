"""Turns raw ESPN JSON into the league model the analysis works on."""

from __future__ import annotations

from dataclasses import dataclass, field

# ESPN stat ids for basketball.
STAT_NAMES = {
    0: "PTS",
    1: "BLK",
    2: "STL",
    3: "AST",
    4: "OREB",
    5: "DREB",
    6: "REB",
    11: "TO",
    13: "FGM",
    14: "FGA",
    15: "FTM",
    16: "FTA",
    17: "3PM",
    18: "3PA",
    19: "FG%",
    20: "FT%",
    21: "3P%",
}
# Percentage categories: stat id -> (makes id, attempts id).
RATIO_STATS = {19: (13, 14), 20: (15, 16), 21: (17, 18)}
GAMES_PLAYED = 42

POSITIONS = {1: "PG", 2: "SG", 3: "SF", 4: "PF", 5: "C"}
IR_SLOT = 13
BENCH_SLOT = 12
UNAVAILABLE = {"OUT", "INJURY_RESERVE", "SUSPENSION"}

# Weight on actual stats grows with games played; projections dominate early.
ACTUAL_WEIGHT_GAMES = 20.0


@dataclass(frozen=True)
class Category:
    stat_id: int
    name: str
    reverse: bool  # lower is better (turnovers)

    @property
    def ratio(self) -> tuple[int, int] | None:
        return RATIO_STATS.get(self.stat_id)


@dataclass
class Player:
    id: int
    name: str
    positions: list[str]
    pro_team_id: int
    injury_status: str
    percent_owned: float
    on_team_id: int  # 0 = free agent
    lineup_slot: int | None
    per_game: dict[int, float]  # stat id -> projected per-game value
    games_played: int  # this season so far
    status: str = "ONTEAM"  # FREEAGENT / WAIVERS for free agents

    @property
    def available(self) -> bool:
        return self.injury_status not in UNAVAILABLE and self.lineup_slot != IR_SLOT


@dataclass
class Team:
    id: int
    name: str
    abbrev: str
    roster: list[Player]
    wins: int
    losses: int
    ties: int
    faab_remaining: int | None


@dataclass
class Matchup:
    period: int
    home: int
    away: int
    winner: str  # HOME / AWAY / TIE / UNDECIDED


@dataclass
class League:
    id: int
    season: int
    name: str
    scoring_type: str  # H2H_CATEGORY (each category) or H2H_MOST_CATEGORIES
    categories: list[Category]
    teams: list[Team]
    schedule: list[Matchup]
    current_period: int
    regular_season_periods: int
    playoff_teams: int
    starter_slots: int
    is_public: bool
    free_agents: list[Player] = field(default_factory=list)

    def team(self, team_id: int) -> Team | None:
        return next((t for t in self.teams if t.id == team_id), None)


def _blend_stats(player: dict, season: int) -> tuple[dict[int, float], int]:
    """Per-game stats: ESPN's season projection blended with this season's actuals."""
    proj: dict[int, float] = {}
    actual: dict[int, float] = {}
    for s in player.get("stats", []):
        if s.get("seasonId") != season or s.get("statSplitTypeId") != 0:
            continue
        avg = {int(k): float(v) for k, v in (s.get("averageStats") or {}).items()}
        if s.get("statSourceId") == 1:
            proj = avg
        elif s.get("statSourceId") == 0:
            actual = avg
    gp = int(actual.get(GAMES_PLAYED, 0) or 0)
    if not proj:
        return actual, gp
    if gp <= 0:
        return proj, gp
    w = gp / (gp + ACTUAL_WEIGHT_GAMES)
    keys = set(proj) | set(actual)
    return {k: (1 - w) * proj.get(k, 0.0) + w * actual.get(k, proj.get(k, 0.0)) for k in keys}, gp


def parse_player(entry: dict, season: int, lineup_slot: int | None = None) -> Player:
    p = entry.get("player") or entry.get("playerPoolEntry", {}).get("player", {})
    per_game, gp = _blend_stats(p, season)
    positions = [POSITIONS[s] for s in p.get("eligibleSlots", []) if s in POSITIONS]
    return Player(
        id=int(p["id"]),
        name=p.get("fullName", str(p["id"])),
        positions=positions,
        pro_team_id=int(p.get("proTeamId", 0)),
        injury_status=p.get("injuryStatus") or "ACTIVE",
        percent_owned=float((p.get("ownership") or {}).get("percentOwned", 0.0)),
        on_team_id=int(entry.get("onTeamId", 0) or 0),
        lineup_slot=lineup_slot,
        per_game=per_game,
        games_played=gp,
        status=entry.get("status", "ONTEAM"),
    )


def _starter_slots(roster_settings: dict) -> int:
    counts = {int(k): v for k, v in roster_settings.get("lineupSlotCounts", {}).items()}
    return sum(v for k, v in counts.items() if k not in (BENCH_SLOT, IR_SLOT))


def parse_league(raw: dict, free_agents: list[dict] | None = None) -> League:
    settings = raw["settings"]
    season = int(raw["seasonId"])
    scoring = settings["scoringSettings"]
    categories = [
        Category(
            stat_id=int(item["statId"]),
            name=STAT_NAMES.get(int(item["statId"]), f"stat{item['statId']}"),
            reverse=bool(item.get("isReverseItem")),
        )
        for item in scoring.get("scoringItems", [])
    ]

    teams: list[Team] = []
    for t in raw.get("teams", []):
        roster = []
        for e in (t.get("roster") or {}).get("entries", []):
            pool = dict(e["playerPoolEntry"])
            pool.setdefault("onTeamId", t["id"])
            roster.append(parse_player(pool, season, lineup_slot=e.get("lineupSlotId")))
        rec = (t.get("record") or {}).get("overall", {})
        spent = (t.get("transactionCounter") or {}).get("acquisitionBudgetSpent")
        acq = settings.get("acquisitionSettings") or {}
        budget = acq.get("acquisitionBudget") if acq.get("isUsingAcquisitionBudget") else None
        teams.append(
            Team(
                id=int(t["id"]),
                name=t.get("name") or f"{t.get('location', '')} {t.get('nickname', '')}".strip(),
                abbrev=t.get("abbrev", ""),
                roster=roster,
                wins=int(rec.get("wins", 0)),
                losses=int(rec.get("losses", 0)),
                ties=int(rec.get("ties", 0)),
                faab_remaining=(budget - spent) if budget is not None and spent is not None else None,
            )
        )

    sched = settings["scheduleSettings"]
    schedule = [
        Matchup(
            period=int(m["matchupPeriodId"]),
            home=int(m["home"]["teamId"]),
            away=int(m["away"]["teamId"]),
            winner=m.get("winner", "UNDECIDED"),
        )
        for m in raw.get("schedule", [])
        if "home" in m and "away" in m and m.get("playoffTierType", "NONE") == "NONE"
    ]

    fas = [parse_player(e, season) for e in (free_agents or [])]

    return League(
        id=int(raw["id"]),
        season=season,
        name=settings.get("name", str(raw["id"])),
        scoring_type=scoring.get("scoringType", "H2H_CATEGORY"),
        categories=categories,
        teams=teams,
        schedule=schedule,
        current_period=int((raw.get("status") or {}).get("currentMatchupPeriod", 1)),
        regular_season_periods=int(sched.get("matchupPeriodCount", 0)),
        playoff_teams=int(sched.get("playoffTeamCount", 4)),
        starter_slots=_starter_slots(settings.get("rosterSettings", {})),
        is_public=bool(settings.get("isPublic", False)),
        free_agents=fas,
    )

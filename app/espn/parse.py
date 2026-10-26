"""Turns raw ESPN JSON into the league model the analysis works on."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

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

POSITIONS = {0: "PG", 1: "SG", 2: "SF", 3: "PF", 4: "C"}  # ESPN lineup slot ids
IR_SLOT = 13
BENCH_SLOT = 12
UTIL_SLOT = 11
UNAVAILABLE = {"OUT", "INJURY_RESERVE", "SUSPENSION"}

# Weight on actual stats grows with games played; projections dominate early.
ACTUAL_WEIGHT_GAMES = 20.0
MIN_PRIOR_GAMES = 10  # last season counts as a prior only with this many games


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
    has_prior: bool = True  # False: no projection or last season, so stats are a small sample
    status: str = "ONTEAM"  # FREEAGENT / WAIVERS for free agents
    # What owners see when they judge a trade:
    espn_rank: int | None = None  # ESPN's preseason rank for category leagues
    espn_auction_value: float = 0.0  # ESPN's suggested auction price ($200 budget)
    avg_auction_value: float = 0.0  # average price paid in ESPN auction drafts
    avg_draft_position: float = 0.0  # average pick in ESPN snake drafts (0 = not drafted)
    season_rank: int | None = None  # ESPN's rank on this season's stats, once games are played
    recent_rank: int | None = None  # ESPN's rank over the last 15 days
    slots: tuple[int, ...] = ()  # lineup slot ids he can fill (PG..C, G, F, UTIL, ...)
    acquisition_type: str = ""  # DRAFT / ADD / TRADE, for rostered players
    acquired_ms: int | None = None  # when his current team got him, epoch milliseconds

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
    # Place in the rolling waiver order (1 = first claim), in leagues without FAAB. None with FAAB,
    # where bids decide claims and the order only breaks ties.
    waiver_priority: int | None = None
    trades: int = 0  # trades made this season
    acquisitions: int = 0  # free-agent and waiver pickups this season
    trade_block: dict[int, str] = field(default_factory=dict)  # player id -> ON_THE_BLOCK / UNTOUCHABLE


@dataclass
class Matchup:
    period: int
    home: int
    away: int
    winner: str  # HOME / AWAY / TIE / UNDECIDED
    # Stats banked so far in a matchup still being played: stat id -> total (makes and attempts too).
    home_totals: dict[int, float] = field(default_factory=dict)
    away_totals: dict[int, float] = field(default_factory=dict)


@dataclass
class Calendar:
    """The NBA schedule in fantasy terms. Days are ESPN scoring periods: day 1 is opening night."""

    opening: date
    today: int  # current scoring period
    team_days: dict[int, frozenset[int]]  # pro team id -> days it plays
    period_days: dict[int, list[int]]  # matchup period -> its days, regular season and playoffs

    def date_of(self, day: int) -> date:
        return self.opening + timedelta(days=day - 1)

    def days_left(self, period: int) -> list[int]:
        return [d for d in self.period_days.get(period, []) if d >= self.today]


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
    is_auction: bool = False
    draft_prices: dict[int, float] = field(default_factory=dict)  # player id -> price paid in this league
    draft_picks: dict[int, int] = field(default_factory=dict)  # player id -> overall pick number
    draft_rounds: int = 0  # rounds in this league's draft
    lineup_slots: dict[int, int] = field(default_factory=dict)  # starting slot id -> count
    calendar: Calendar | None = None  # None: no NBA schedule, assume an even number of games a week

    def team(self, team_id: int) -> Team | None:
        return next((t for t in self.teams if t.id == team_id), None)


def _blend_stats(player: dict, season: int) -> tuple[dict[int, float], int, bool]:
    """Per-game stats: a prior blended with this season's actuals as games are played.

    The prior is ESPN's season projection, or last season's per-game stats for players
    ESPN doesn't project (many free agents). Without either, a hot game or two would
    count at face value, so the third value says whether there was a prior at all.
    """
    proj: dict[int, float] = {}
    last: dict[int, float] = {}
    actual: dict[int, float] = {}
    for s in player.get("stats", []):
        if s.get("statSplitTypeId") != 0:
            continue
        avg = {int(k): float(v) for k, v in (s.get("averageStats") or {}).items()}
        if s.get("seasonId") == season and s.get("statSourceId") == 1:
            proj = avg
        elif s.get("seasonId") == season and s.get("statSourceId") == 0:
            actual = avg
        elif s.get("seasonId") == season - 1 and s.get("statSourceId") == 0:
            last = avg
    prior = proj or (last if last.get(GAMES_PLAYED, 0) >= MIN_PRIOR_GAMES else {})
    gp = int(actual.get(GAMES_PLAYED, 0) or 0)
    if not prior:
        return actual, gp, False
    if gp <= 0:
        return prior, gp, True
    w = gp / (gp + ACTUAL_WEIGHT_GAMES)
    keys = set(prior) | set(actual)
    return {k: (1 - w) * prior.get(k, 0.0) + w * actual.get(k, prior.get(k, 0.0)) for k in keys}, gp, True


def _draft_rank(p: dict) -> tuple[int | None, float]:
    ranks = p.get("draftRanksByRankType") or {}
    r = ranks.get("ROTO") or ranks.get("STANDARD") or {}
    rank = r.get("rank")
    return (int(rank) if rank else None), float(r.get("auctionValue") or 0.0)


def parse_player(entry: dict, season: int, lineup_slot: int | None = None, roster_entry: dict | None = None) -> Player:
    p = entry.get("player") or entry.get("playerPoolEntry", {}).get("player", {})
    per_game, gp, has_prior = _blend_stats(p, season)
    positions = [POSITIONS[s] for s in p.get("eligibleSlots", []) if s in POSITIONS]
    espn_rank, espn_av = _draft_rank(p)
    ratings = entry.get("ratings") or {}
    season_rank = (ratings.get("0") or {}).get("totalRanking") or None
    recent_rank = (ratings.get("2") or {}).get("totalRanking") or None  # last 15 days
    acq = roster_entry or {}
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
        has_prior=has_prior,
        status=entry.get("status", "ONTEAM"),
        espn_rank=espn_rank,
        espn_auction_value=espn_av,
        avg_auction_value=float((p.get("ownership") or {}).get("auctionValueAverage") or 0.0),
        avg_draft_position=float((p.get("ownership") or {}).get("averageDraftPosition") or 0.0),
        season_rank=int(season_rank) if season_rank else None,
        recent_rank=int(recent_rank) if recent_rank else None,
        slots=tuple(int(s) for s in p.get("eligibleSlots", [])),
        acquisition_type=acq.get("acquisitionType") or "",
        acquired_ms=int(acq["acquisitionDate"]) if acq.get("acquisitionDate") else None,
    )


def _trade_block(raw: dict | None) -> dict[int, str]:
    """ESPN's trade block: players an owner lists as available, or marks untouchable."""
    players = (raw or {}).get("players") or {}
    return {int(pid): str(status) for pid, status in players.items() if str(pid).lstrip("-").isdigit()}


def _starter_slots(roster_settings: dict) -> int:
    counts = {int(k): v for k, v in roster_settings.get("lineupSlotCounts", {}).items()}
    return sum(v for k, v in counts.items() if k not in (BENCH_SLOT, IR_SLOT))


def _lineup_slots(roster_settings: dict) -> dict[int, int]:
    counts = {int(k): int(v) for k, v in roster_settings.get("lineupSlotCounts", {}).items()}
    return {k: v for k, v in counts.items() if v > 0 and k not in (BENCH_SLOT, IR_SLOT)}


def parse_calendar(pro_schedule: dict, raw: dict) -> Calendar | None:
    """Days each NBA team plays, and which days make up each fantasy matchup period.

    ESPN doesn't publish the matchup -> day mapping, so it is rebuilt the way ESPN
    builds it: Monday-to-Sunday weeks up to the league's last scoring day, with the
    week the All-Star break starts in (February's longest run of days without games)
    folded together with the week after it, as ESPN did in 2025 and 2026.
    """
    team_days: dict[int, set[int]] = {}
    opening: date | None = None
    for team in (pro_schedule.get("settings") or {}).get("proTeams", []):
        for day, games in (team.get("proGamesByScoringPeriod") or {}).items():
            if not games:
                continue
            team_days.setdefault(int(team["id"]), set()).add(int(day))
            if int(day) == 1:
                # Game times are UTC; US evening games land on the next UTC day, so step back 8 hours.
                ts = games[0]["date"] / 1000 - 8 * 3600
                opening = date(1970, 1, 1) + timedelta(seconds=ts)
    if not team_days or opening is None:
        return None
    status = raw.get("status") or {}
    sched = (raw.get("settings") or {}).get("scheduleSettings") or {}
    last_day = int(status.get("finalScoringPeriod") or max(max(d) for d in team_days.values()))
    n_periods = len(sched.get("matchupPeriods") or {}) or int(sched.get("matchupPeriodCount", 0))
    weeks: list[list[int]] = []
    for day in range(1, last_day + 1):
        d = opening + timedelta(days=day - 1)
        if not weeks or d.weekday() == 0:
            weeks.append([])
        weeks[-1].append(day)
    game_days = set().union(*team_days.values())

    def idle_run(week: list[int]) -> int:
        best = run = 0
        for d in week:
            run = 0 if d in game_days else run + 1
            best = max(best, run)
        return best

    def all_star_week() -> int | None:
        """Index of the week in which February's longest run of days without games begins."""
        best, start, run = 0, None, 0
        for day in range(1, last_day + 1):
            if (opening + timedelta(days=day - 1)).month != 2 or day in game_days:
                run = 0
                continue
            run += 1
            if run > best:
                best, start = run, day - run + 1
        if start is None or best < 2:
            return None
        return next(k for k, w in enumerate(weeks) if start in w)

    # ESPN folds the week the All-Star break starts in together with the week after it.
    k = all_star_week()
    if n_periods and len(weeks) > n_periods and k is not None and k + 1 < len(weeks):
        weeks[k : k + 2] = [weeks[k] + weeks[k + 1]]
    while n_periods and len(weeks) > n_periods:
        # Fallback for other calendars. Early in the season December's NBA Cup games are
        # still unscheduled, so a December week can look emptier than it is.
        feb = [k for k in range(1, len(weeks)) if (opening + timedelta(days=weeks[k][0] - 1)).month == 2]
        i = max(feb or range(1, len(weeks)), key=lambda k: idle_run(weeks[k]))
        weeks[i - 1 : i + 1] = [weeks[i - 1] + weeks[i]]
    return Calendar(
        opening=opening,
        today=int(status.get("latestScoringPeriod") or raw.get("scoringPeriodId") or 1),
        team_days={t: frozenset(d) for t, d in team_days.items()},
        period_days={i + 1: w for i, w in enumerate(weeks)},
    )


def _banked(side: dict) -> dict[int, float]:
    by_stat = (side.get("cumulativeScore") or {}).get("scoreByStat") or {}
    return {int(k): float(v.get("score") or 0.0) for k, v in by_stat.items() if str(k).isdigit()}


def parse_league(
    raw: dict,
    free_agents: list[dict] | None = None,
    pro_schedule: dict | None = None,
    market_adp: dict[int, float] | None = None,
) -> League:
    """``market_adp`` fills in ESPN's average draft position where the league's own views leave it out."""
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
            roster.append(parse_player(pool, season, lineup_slot=e.get("lineupSlotId"), roster_entry=e))
        rec = (t.get("record") or {}).get("overall", {})
        counter = t.get("transactionCounter") or {}
        spent = counter.get("acquisitionBudgetSpent")
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
                waiver_priority=int(t["waiverRank"]) if budget is None and t.get("waiverRank") else None,
                trades=int(counter.get("trades") or 0),
                acquisitions=int(counter.get("acquisitions") or 0),
                trade_block=_trade_block(t.get("tradeBlock")),
            )
        )

    sched = settings["scheduleSettings"]
    schedule = [
        Matchup(
            period=int(m["matchupPeriodId"]),
            home=int(m["home"]["teamId"]),
            away=int(m["away"]["teamId"]),
            winner=m.get("winner", "UNDECIDED"),
            home_totals=_banked(m["home"]) if m.get("winner", "UNDECIDED") == "UNDECIDED" else {},
            away_totals=_banked(m["away"]) if m.get("winner", "UNDECIDED") == "UNDECIDED" else {},
        )
        for m in raw.get("schedule", [])
        if "home" in m and "away" in m and m.get("playoffTierType", "NONE") == "NONE"
    ]

    fas = [parse_player(e, season) for e in (free_agents or [])]

    draft = raw.get("draftDetail") or {}
    picks = [pk for pk in draft.get("picks", []) if pk.get("playerId")]
    is_auction = (settings.get("draftSettings") or {}).get("type") == "AUCTION"
    if market_adp:
        for p in [p for t in teams for p in t.roster] + fas:
            if not p.avg_draft_position:
                p.avg_draft_position = market_adp.get(p.id, 0.0)

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
        is_auction=is_auction,
        draft_prices={int(pk["playerId"]): float(pk.get("bidAmount") or 0) for pk in picks} if is_auction else {},
        draft_picks={int(pk["playerId"]): int(pk.get("overallPickNumber") or 0) for pk in picks},
        draft_rounds=-(-max((int(pk.get("overallPickNumber") or 0) for pk in picks), default=0) // max(len(teams), 1)),
        lineup_slots=_lineup_slots(settings.get("rosterSettings", {})),
        calendar=parse_calendar(pro_schedule, raw) if pro_schedule else None,
    )

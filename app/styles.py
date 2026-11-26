"""How each manager plays, read from what they actually did this season.

Every number here is a plain count from ESPN: pickups, FAAB bids, lineup moves,
starts made against the best lineup they could have set, IR moves, declined
trades and weekly category results. Labels are fixed thresholds against the
league's own median, never tuned to make a team look a certain way, and they
change nothing in the trade or matchup models (those need the 5-season blind
test first).

The day-by-day roster pages are big (about 1.3 MB a league-day), so each one is
boiled down to a ``DayLineup`` straight away and only that is kept.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from app.espn.parse import BENCH_SLOT, IR_SLOT, STAT_NAMES

# Lineup slot -> how many of it start. ESPN's lineupSlotCounts, minus bench and IR.
DAY_SPLIT = 5  # statSplitTypeId of a single day's line
GP_STAT = "42"
MIN_STAT = "40"
SHORT_HOLD_DAYS = 7
# Weekly category result thresholds.
WINS_SHARE = 0.8
GIVES_SHARE = 0.2
# Activity labels.
HAWK_MIN_ADDS = 2.0  # adds per week, and in the league's top 3
STREAMER_SHARE = 0.5  # share of adds dropped again within a week
LOYALIST_MAX_ADDS = 0.5
SET_AND_FORGET_MOVES = 0.6  # lineup moves per week under this share of the league median
# Lineup labels.
SHARP_SHARE = 0.99
LEAK_PER_WEEK = 1.0
LOW_VOLUME_SHARE = 0.9
# Trade door.
CLOSED_DECLINES = 2
# The win path aims at this many categories a week: a majority of nine.
PATH_TARGET = 5
# Weekly calls: about how far one day moves a team's FG% or FT%, and categories chased beyond a bare majority.
PCT_DAY = 0.02
SPARE_CATS = 1
CHASE_MIN_GAMES = 3
CHASE_MIN_POOL_GAMES = 5
# Season per-game line keys kept from a day page: the nine categories plus attempts.
KEEP_STATS = ("0", "1", "2", "3", "6", "11", "13", "14", "15", "16", "17", "18", "19", "20", "21")


@dataclass
class DayLineup:
    starts: int  # games played by players in a starting slot
    best: int  # games the best possible lineup that day would have started


@dataclass
class TeamDays:
    lineups: dict[int, DayLineup] = field(default_factory=dict)  # day -> lineup
    on_roster: dict[int, set[int]] = field(default_factory=dict)  # player id -> days on this roster


def _max_starts(players: list[set[int]], slots: dict[int, int]) -> int:
    """Most players who played that fit the starting slots (bipartite matching)."""
    seats = [s for s, n in slots.items() for _ in range(n)]
    owner = [-1] * len(seats)

    def place(i: int, seen: set[int]) -> bool:
        for j, s in enumerate(seats):
            if s in players[i] and j not in seen:
                seen.add(j)
                if owner[j] < 0 or place(owner[j], seen):
                    owner[j] = i
                    return True
        return False

    return sum(place(i, set()) for i in range(len(players)))


def starter_slots(settings: dict) -> dict[int, int]:
    counts = ((settings.get("rosterSettings") or {}).get("lineupSlotCounts")) or {}
    return {int(k): int(v) for k, v in counts.items() if int(v) > 0 and int(k) not in (BENCH_SLOT, IR_SLOT)}


def _played(player: dict, day: int) -> bool:
    for s in player.get("stats") or []:
        if s.get("scoringPeriodId") == day and s.get("statSplitTypeId") == DAY_SPLIT and s.get("statSourceId") == 0:
            line = s.get("stats") or {}
            return (line.get(GP_STAT) or line.get(MIN_STAT) or 0) > 0
    return False


@dataclass
class DaySummary:
    """One league-day boiled down: lineup per team, who was on each roster, season lines seen."""

    day: int
    lineups: dict[int, DayLineup]
    rosters: dict[int, list[int]]
    averages: dict[int, tuple[dict, int]]  # player id -> (season per-game line, games)


def summarize_day(raw: dict, day: int, slots: dict[int, int], season: int) -> DaySummary:
    lineups: dict[int, DayLineup] = {}
    rosters: dict[int, list[int]] = {}
    averages: dict[int, tuple[dict, int]] = {}
    for t in raw.get("teams") or []:
        tid = int(t["id"])
        played: list[set[int]] = []
        starts = 0
        ids: list[int] = []
        for e in (t.get("roster") or {}).get("entries") or []:
            p = (e.get("playerPoolEntry") or {}).get("player") or {}
            pid = int(p.get("id") or e.get("playerId") or 0)
            ids.append(pid)
            for s in p.get("stats") or []:
                if (
                    s.get("seasonId") == season
                    and s.get("statSplitTypeId") == 0
                    and s.get("statSourceId") == 0
                    and s.get("averageStats")
                ):
                    line = {k: v for k, v in s["averageStats"].items() if k in KEEP_STATS}
                    averages[pid] = (line, int((s.get("stats") or {}).get(GP_STAT) or 0))
            if _played(p, day):
                played.append(set(p.get("eligibleSlots") or []))
                if e.get("lineupSlotId") in slots:
                    starts += 1
        lineups[tid] = DayLineup(starts=starts, best=_max_starts(played, slots))
        rosters[tid] = ids
    return DaySummary(day=day, lineups=lineups, rosters=rosters, averages=averages)


@dataclass
class Pickup:
    day: int
    player_id: int
    bid: int


@dataclass
class TradeTally:
    trades: int = 0
    declined: int = 0


def read_moves(transactions: list[dict]) -> tuple[dict, dict, Counter, dict]:
    """Executed adds and drops per team, waiver bids that lost, and trade activity."""
    seen: dict[str, dict] = {}
    for t in transactions:
        seen.setdefault(t["id"], t)
    adds: dict[int, list[Pickup]] = defaultdict(list)
    drops: dict[int, list[Pickup]] = defaultdict(list)
    lost: Counter = Counter()
    trades: dict[int, TradeTally] = defaultdict(TradeTally)
    for t in seen.values():
        kind, status, tid = t.get("type"), t.get("status") or "", int(t.get("teamId") or 0)
        if kind in ("WAIVER", "FREEAGENT"):
            if status == "EXECUTED":
                for it in t.get("items") or []:
                    move = Pickup(int(t.get("scoringPeriodId") or 0), int(it["playerId"]), int(t.get("bidAmount") or 0))
                    if it.get("type") == "ADD":
                        adds[tid].append(move)
                    elif it.get("type") == "DROP":
                        drops[tid].append(move)
            elif kind == "WAIVER" and status.startswith("FAILED"):
                lost[tid] += 1
        elif kind == "TRADE_DECLINE" and status == "EXECUTED":
            trades[tid].declined += 1  # ESPN files a decline under the team that said no
    return adds, drops, lost, trades


@dataclass
class CategoryRecord:
    cat: str
    won: float  # weeks won, a tie counts half
    weeks: int

    @property
    def share(self) -> float:
        return self.won / self.weeks if self.weeks else 0.0


def category_records(
    schedule: list[dict], cats: dict[str, str], through_period: int
) -> dict[int, list[CategoryRecord]]:
    """Each team's finished weeks, category by category."""
    won: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    weeks: Counter = Counter()
    for m in schedule:
        if int(m.get("matchupPeriodId") or 0) >= through_period or m.get("winner") == "UNDECIDED":
            continue
        for side in ("home", "away"):
            s = m.get(side) or {}
            by = (s.get("cumulativeScore") or {}).get("scoreByStat") or {}
            if not by:
                continue
            tid = int(s["teamId"])
            weeks[tid] += 1
            for sid, cat in cats.items():
                r = (by.get(sid) or {}).get("result")
                won[tid][cat] += 1.0 if r == "WIN" else 0.5 if r == "TIE" else 0.0
    return {tid: [CategoryRecord(cat, won[tid][cat], weeks[tid]) for cat in cats.values()] for tid in weeks}


def scoring_cats(settings: dict) -> dict[str, str]:
    """stat id (as ESPN keys scoreByStat) -> category name, in the league's own order."""
    items = (settings.get("scoringSettings") or {}).get("scoringItems") or []
    return {str(i["statId"]): STAT_NAMES.get(int(i["statId"]), str(i["statId"])) for i in items}


def _z_lines(averages: dict[int, tuple[dict, int]], cats: dict[str, str]) -> dict[int, dict[str, float]]:
    """Per-player category value against the rostered pool (percentages weighted by attempts)."""
    pool = [a for a, gp in averages.values() if gp >= CHASE_MIN_POOL_GAMES]
    if len(pool) < 10:
        return {}
    counting = {sid: cat for sid, cat in cats.items() if int(sid) not in (19, 20, 21)}
    mean = {sid: statistics.fmean(a.get(sid, 0) for a in pool) for sid in counting}
    sd = {sid: statistics.pstdev([a.get(sid, 0) for a in pool]) or 1.0 for sid in counting}
    pct = {"19": ("14", 0.47, 1.2), "20": ("16", 0.78, 0.6), "21": ("18", 0.36, 0.5)}
    out = {}
    for pid, (a, gp) in averages.items():
        if gp < CHASE_MIN_GAMES:
            continue
        line = {}
        for sid, cat in cats.items():
            if sid in pct:
                att, base, scale = pct[sid]
                line[cat] = (a.get(sid, 0) - base) * a.get(att, 0) / scale
            else:
                z = (a.get(sid, 0) - mean[sid]) / sd[sid]
                line[cat] = -z if cat == "TO" else z
        out[pid] = line
    return out


@dataclass
class ManagerStyle:
    team_id: int
    activity: str
    lineup: str
    door: str
    adds: int
    adds_per_week: float
    quick_drops: int  # adds dropped again within a week
    faab_spent: int
    bids_lost: int
    starts_per_week: float
    start_share: float  # starts made / best possible
    missed_starts: int
    lineup_moves_per_week: float
    ir_moves: int
    trades: int
    declined: int
    categories: list[CategoryRecord]
    wins: list[str]
    gives_away: list[str]
    chases: list[str]


@dataclass
class LeagueMedians:
    adds_per_week: float
    starts_per_week: float
    lineup_moves_per_week: float
    start_share: float


def _median(xs: list[float]) -> float:
    return statistics.median(xs) if xs else 0.0


def build_styles(
    teams: list[dict],
    schedule: list[dict],
    settings: dict,
    days: list[DaySummary],
    transactions: list[dict],
    current_day: int,
    current_period: int,
) -> tuple[list[ManagerStyle], LeagueMedians]:
    cats = scoring_cats(settings)
    weeks = max(1.0, (current_day - 1) / 7)
    adds, drops, lost, tallies = read_moves(transactions)
    records = category_records(schedule, cats, current_period)

    on_roster: dict[tuple[int, int], set[int]] = defaultdict(set)
    lineups: dict[int, list[DayLineup]] = defaultdict(list)
    averages: dict[int, tuple[dict, int]] = {}
    for d in sorted(days, key=lambda d: d.day):
        for tid, ids in d.rosters.items():
            for pid in ids:
                on_roster[(tid, pid)].add(d.day)
        for tid, lu in d.lineups.items():
            lineups[tid].append(lu)
        averages.update(d.averages)
    z = _z_lines(averages, cats)

    rows = []
    for t in teams:
        tid = int(t["id"])
        counter = t.get("transactionCounter") or {}
        mine = adds.get(tid, [])
        quick = 0
        for a in mine:
            held = on_roster.get((tid, a.player_id), set())
            run = 0
            while a.day + run in held:
                run += 1
            if a.day + run < current_day and run <= SHORT_HOLD_DAYS:
                quick += 1
        lu = lineups.get(tid, [])
        starts = sum(x.starts for x in lu)
        best = sum(x.best for x in lu)
        recs = records.get(tid, [])
        chases = []
        got = [z[a.player_id] for a in mine if a.player_id in z]
        gave = [z[d.player_id] for d in drops.get(tid, []) if d.player_id in z]
        if got:
            diff = {
                c: statistics.fmean(x[c] for x in got) - (statistics.fmean(x[c] for x in gave) if gave else 0.0)
                for c in cats.values()
            }
            chases = [c for c, v in sorted(diff.items(), key=lambda kv: -kv[1]) if v > 0][:2]
        tally = tallies.get(tid, TradeTally())
        trades = int(counter.get("trades") or 0)
        rows.append(
            ManagerStyle(
                team_id=tid,
                activity="",
                lineup="",
                door="open" if trades else "closed" if tally.declined >= CLOSED_DECLINES else "unknown",
                adds=len(mine),
                adds_per_week=round(len(mine) / weeks, 1),
                quick_drops=quick,
                faab_spent=int(counter.get("acquisitionBudgetSpent") or 0),
                bids_lost=lost.get(tid, 0),
                starts_per_week=round(starts / weeks, 1),
                start_share=round(starts / best, 3) if best else 1.0,
                missed_starts=max(0, best - starts),
                lineup_moves_per_week=round(int(counter.get("moveToActive") or 0) / weeks, 1),
                ir_moves=int(counter.get("moveToIR") or 0),
                trades=trades,
                declined=tally.declined,
                categories=recs,
                wins=[r.cat for r in recs if r.weeks and r.share >= WINS_SHARE],
                gives_away=[r.cat for r in recs if r.weeks and r.share <= GIVES_SHARE],
                chases=chases,
            )
        )
    med = LeagueMedians(
        adds_per_week=_median([r.adds_per_week for r in rows]),
        starts_per_week=_median([r.starts_per_week for r in rows]),
        lineup_moves_per_week=_median([r.lineup_moves_per_week for r in rows]),
        start_share=_median([r.start_share for r in rows]),
    )
    top3 = sorted((r.adds_per_week for r in rows), reverse=True)[:3]
    for r in rows:
        r.activity = activity_label(r, med, top3[-1] if top3 else 0.0)
        r.lineup = lineup_label(r, med, weeks)
    return rows, med


def activity_label(r: ManagerStyle, med: LeagueMedians, third_most: float) -> str:
    if r.adds_per_week < LOYALIST_MAX_ADDS:
        if med.lineup_moves_per_week and r.lineup_moves_per_week < SET_AND_FORGET_MOVES * med.lineup_moves_per_week:
            return "set_and_forget"
        return "draft_loyalist"
    if r.adds and r.quick_drops / r.adds >= STREAMER_SHARE and r.adds_per_week >= HAWK_MIN_ADDS:
        return "streamer"
    if r.adds_per_week >= HAWK_MIN_ADDS and r.adds_per_week >= third_most:
        return "waiver_hawk"
    return "steady_hands"


def lineup_label(r: ManagerStyle, med: LeagueMedians, weeks: float) -> str:
    if r.missed_starts / weeks >= LEAK_PER_WEEK:
        return "leaks_starts"
    if med.starts_per_week and r.starts_per_week < LOW_VOLUME_SHARE * med.starts_per_week:
        return "low_volume"
    if r.start_share >= SHARP_SHARE:
        return "sharp"
    return "steady"


@dataclass
class WinPath:
    lock: list[str]  # won most weeks: keep them
    swing: list[str]  # the next ones up to a majority: where moves pay
    let_go: list[str]
    adds_bring: list[str]  # what this manager's pickups have brought so far
    adds_should_bring: list[str]  # swing categories the pickups are not bringing yet


def win_path(style: ManagerStyle) -> WinPath:
    """Lock what you already win, aim the moves at the next best until a majority, let the rest go."""
    recs = sorted(style.categories, key=lambda r: -r.share)
    lock = [r.cat for r in recs if r.share >= WINS_SHARE][:PATH_TARGET]
    swing = [r.cat for r in recs if r.cat not in lock][: max(0, PATH_TARGET - len(lock))]
    let_go = [r.cat for r in recs if r.cat not in lock and r.cat not in swing]
    return WinPath(
        lock=lock,
        swing=swing,
        let_go=let_go,
        adds_bring=style.chases,
        adds_should_bring=[c for c in swing if c not in style.chases],
    )


@dataclass
class CategoryCall:
    cat: str
    mine: float | None
    theirs: float | None
    status: str  # ahead / behind / tied / not_started
    call: str  # flip / hold / fight / let_go
    their_share: float  # how often they win this category


def _behind_by(cat: str, mine: float, theirs: float, days: float) -> float:
    """How far behind, in days of the two teams' own pace this week (0 when tied or ahead).

    A percentage is measured against PCT_DAY, about what one day can move it.
    """
    gap = (mine - theirs) if cat == "TO" else (theirs - mine)
    if gap <= 0:
        return 0.0
    if cat.endswith("%"):
        return gap / PCT_DAY
    per_day = (abs(mine) + abs(theirs)) / 2 / max(days, 1.0)
    return gap / per_day if per_day else float("inf")


def matchup_calls(
    me: ManagerStyle, them: ManagerStyle, live: dict[str, tuple[float, float]], days: float = 1.0
) -> list[CategoryCall]:
    """This week's cheapest way to a majority of the categories.

    Hold every lead. Rank the rest by how far behind you are (in days of this week's
    pace), cheaper where this opponent usually loses the category, and go after the
    cheapest ones until a majority plus a spare; let the rest go. A category the
    opponent wins in 20% of weeks or less is a flip. Before any stats (a Monday
    morning) the season decides: lock = hold, swing = fight, let go = let go.
    """
    path = win_path(me)
    share = {r.cat: r.share for r in them.categories}
    cats = [r.cat for r in me.categories]
    majority = len(cats) // 2 + 1
    calls: dict[str, str] = {}
    status: dict[str, str] = {}
    cost: dict[str, float] = {}
    for cat in cats:
        mine, theirs = live.get(cat, (None, None))
        if mine is None or theirs is None:
            status[cat] = "not_started"
            calls[cat] = "hold" if cat in path.lock else "let_go" if cat in path.let_go else "fight"
            continue
        if mine == theirs:
            status[cat] = "tied"
        else:
            ahead = mine < theirs if cat == "TO" else mine > theirs
            status[cat] = "ahead" if ahead else "behind"
        if status[cat] == "ahead":
            calls[cat] = "hold"
        else:
            cost[cat] = _behind_by(cat, mine, theirs, days) * (0.5 + share.get(cat, 0.5))
    if cost:
        need = max(0, majority - sum(1 for c in calls.values() if c == "hold")) + SPARE_CATS
        for i, cat in enumerate(sorted(cost, key=lambda c: (cost[c], c))):
            if i < need:
                calls[cat] = "flip" if share.get(cat, 0.5) <= GIVES_SHARE else "fight"
            else:
                calls[cat] = "let_go"
    return [
        CategoryCall(
            cat,
            live.get(cat, (None, None))[0],
            live.get(cat, (None, None))[1],
            status[cat],
            calls[cat],
            round(share.get(cat, 0.5), 2),
        )
        for cat in cats
    ]

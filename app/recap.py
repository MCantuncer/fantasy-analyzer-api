"""The week in review: a finished matchup week looked back on, for the whole league.

Who won each matchup and by which categories, the NBA's best players that week by
9-category value, the free agents who stood out (and who grabbed them, and when),
and each fantasy team's MVP, best add and what sat on its bench.

Everything comes from data that never changes once the week is over: each day's box
scores (``scores.build_day``, saved per day for every league), each day's rosters
(``EspnClient.rosters_on_day``) and the matchup scores. A recap is built once per league
and week, then stored.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field

from app.espn.parse import BENCH_SLOT, IR_SLOT, STAT_NAMES

# Bump when the recap's shape or method changes: stored recaps of another version are rebuilt.
VERSION = 2
# Players of the week and free agents shown.
TOP_PLAYERS = 10
TOP_FREE_AGENTS = 8
# The comparison pool for z-scores: the players who played the most minutes that week,
# about the players a 10-12 team league can roster.
POOL_SIZE = 200

# Box-score fields summed over the week, in the site's stat order.
TOTALS = ("minutes", "pts", "fgm", "fga", "ftm", "fta", "tpm", "reb", "ast", "stl", "blk", "to")
# ESPN stat id -> how the category is valued from a week's totals. Ratios use made minus
# expected makes at the pool's average rate, so volume counts; turnovers count against.
COUNTING = {0: "pts", 6: "reb", 3: "ast", 2: "stl", 1: "blk", 17: "tpm", 11: "to"}
RATIOS = {19: ("fgm", "fga"), 20: ("ftm", "fta")}
LOWER_IS_BETTER = {11}


@dataclass
class Week:
    """The inputs for one finished matchup week."""

    period: int
    days: list[int]
    day_lines: dict[int, list[dict]]  # day -> every box-score line that day (``scores.Line`` fields)
    rosters: dict[int, dict[int, tuple[int, int]]]  # day -> player id -> (fantasy team id, lineup slot)
    owners_now: dict[int, int]  # player id -> fantasy team id today
    categories: list[int]  # the league's scoring stat ids, in its order
    matchups: list[dict] = field(default_factory=list)  # ESPN schedule items of this period


def _totals(week: Week) -> tuple[dict[int, dict[str, float]], dict[int, tuple[str, int]]]:
    tot: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    who: dict[int, tuple[str, int]] = {}
    for day in week.days:
        for ln in week.day_lines.get(day, []):
            t = tot[ln["player_id"]]
            t["games"] += 1
            for k in TOTALS:
                t[k] += ln[k]
            who[ln["player_id"]] = (ln["name"], ln["pro_team_id"])
    return tot, who


class _Scorer:
    """Values a stat line against the week's pool: the sum of category z-scores."""

    def __init__(self, tot: dict[int, dict[str, float]], categories: list[int]):
        self.cats = [c for c in categories if c in COUNTING or c in RATIOS] or [*COUNTING, *RATIOS]
        pool = [tot[p] for p in sorted(tot, key=lambda p: (-tot[p]["minutes"], p))[:POOL_SIZE]]
        self.rate = {}
        for sid, (made, att) in RATIOS.items():
            attempts = sum(t[att] for t in pool)
            self.rate[sid] = sum(t[made] for t in pool) / attempts if attempts else 0.0
        self.mean = {c: statistics.fmean(self._raw(t, c) for t in pool) for c in self.cats} if pool else {}
        self.sd = {c: statistics.pstdev(self._raw(t, c) for t in pool) or 1.0 for c in self.cats} if pool else {}

    def _raw(self, t: dict[str, float], sid: int) -> float:
        if sid in RATIOS:
            made, att = RATIOS[sid]
            return t.get(made, 0) - self.rate[sid] * t.get(att, 0)
        v = t.get(COUNTING[sid], 0)
        return -v if sid in LOWER_IS_BETTER else v

    def z(self, t: dict[str, float]) -> dict[str, float]:
        return {STAT_NAMES[c]: round((self._raw(t, c) - self.mean[c]) / self.sd[c], 2) for c in self.cats}

    def value(self, t: dict[str, float]) -> float:
        return round(sum(self.z(t).values()), 2)


def _team_result(side: dict, categories: list[int]) -> dict:
    score = side.get("cumulativeScore") or {}
    by_stat = score.get("scoreByStat") or {}
    return {
        "team_id": int(side["teamId"]),
        "wins": int(score.get("wins") or 0),
        "losses": int(score.get("losses") or 0),
        "ties": int(score.get("ties") or 0),
        "values": {
            STAT_NAMES[c]: float((by_stat.get(str(c)) or {}).get("score") or 0.0)
            for c in categories
            if str(c) in by_stat and c in STAT_NAMES
        },
    }


def _matchups(week: Week) -> list[dict]:
    out = []
    for m in week.matchups:
        if "home" not in m or "away" not in m:
            continue
        home, away = _team_result(m["home"], week.categories), _team_result(m["away"], week.categories)
        winner = {"HOME": home["team_id"], "AWAY": away["team_id"]}.get(m.get("winner", ""))
        out.append({"home": home, "away": away, "winner_team_id": winner, "margin": abs(home["wins"] - away["wins"])})
    out.sort(key=lambda m: (-m["margin"], m["home"]["team_id"]))
    return out


def is_final(matchups: list[dict]) -> bool:
    """True once ESPN has settled every matchup of the period."""
    return bool(matchups) and all(m.get("winner", "UNDECIDED") != "UNDECIDED" for m in matchups)


def build_recap(week: Week) -> dict:
    """The recap, in the shape of ``schemas.WeekRecap`` minus the league and date fields."""
    tot, who = _totals(week)
    scorer = _Scorer(tot, week.categories)
    z = {p: scorer.z(t) for p, t in tot.items()}
    value = {p: round(sum(z[p].values()), 2) for p in tot}
    order = sorted(tot, key=lambda p: (-value[p], p))
    rank = {p: i + 1 for i, p in enumerate(order)}
    first, last = week.days[0], week.days[-1]
    day_index = {d: i for i, d in enumerate(week.days)}

    def player(p: int) -> dict:
        name, pro = who[p]
        t = tot[p]
        return {
            "player_id": p,
            "name": name,
            "pro_team_id": pro,
            "games": int(t["games"]),
            "value": value[p],
            "rank": rank[p],
            "z": z[p],
            "totals": {k: int(t[k]) for k in TOTALS},
        }

    top = [{**player(p), "owner_team_id": week.rosters.get(last, {}).get(p, (None, 0))[0]} for p in order[:TOP_PLAYERS]]

    by_day = {d: {ln["player_id"]: ln for ln in week.day_lines.get(d, [])} for d in week.days}

    def after(p: int, team_id: int, added: int) -> dict:
        """What he did for ``team_id`` from the day it added him: games while on its roster."""
        t: dict[str, float] = defaultdict(float)
        started = 0
        for d in week.days[day_index[added] :]:
            seat = week.rosters.get(d, {}).get(p)
            ln = by_day[d].get(p)
            if seat is None or seat[0] != team_id or ln is None:
                continue
            t["games"] += 1
            started += seat[1] not in (BENCH_SLOT, IR_SLOT)
            for k in TOTALS:
                t[k] += ln[k]
        return {
            "games": int(t["games"]),
            "started": started,
            "value": scorer.value(t) if t["games"] else 0.0,
            "totals": {k: int(t[k]) for k in TOTALS},
        }

    start = week.rosters.get(first, {})
    free_agents = []
    for p in [p for p in order if p not in start][:TOP_FREE_AGENTS]:
        taken = next((d for d in week.days if p in week.rosters.get(d, {})), None)
        free_agents.append(
            {
                **player(p),
                "picked_up": (
                    {
                        "team_id": week.rosters[taken][p][0],
                        "day": taken,
                        "weekday": day_index[taken],
                        "after": after(p, week.rosters[taken][p][0], taken),
                    }
                    if taken is not None
                    else None
                ),
                "owner_now_team_id": week.owners_now.get(p),
            }
        )

    # Per fantasy team: games its starters played, what its bench played, and the adds that paid off.
    started: dict[int, set[int]] = defaultdict(set)
    bench: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for day in week.days:
        on_day = week.rosters.get(day, {})
        for ln in week.day_lines.get(day, []):
            seat = on_day.get(ln["player_id"])
            if seat is None:
                continue
            team_id, slot = seat
            if slot in (BENCH_SLOT, IR_SLOT):
                b = bench[team_id]
                b["games"] += 1
                for k in ("pts", "reb", "ast", "stl", "blk", "tpm"):
                    b[k] += ln[k]
            else:
                started[team_id].add(ln["player_id"])
    adds: dict[int, list[tuple[int, int]]] = defaultdict(list)  # team -> (player, day added)
    for prev, day in zip(week.days, week.days[1:], strict=False):
        before = week.rosters.get(prev, {})
        for p, (team_id, _) in week.rosters.get(day, {}).items():
            if before.get(p, (None,))[0] != team_id and p in tot and p not in start:
                adds[team_id].append((p, day))
    team_ids = {t for r in week.rosters.values() for t, _ in r.values()}
    teams = []
    for t in sorted(team_ids):
        mvp = max(started.get(t, ()), key=lambda p: (value[p], -p), default=None)
        # The best add is judged on what he did for this team, not on his whole week.
        gains = [(p, day, after(p, t, day)) for p, day in adds.get(t, ())]
        add = max(gains, key=lambda a: (a[2]["games"] > 0, a[2]["value"], -a[0]), default=None)
        teams.append(
            {
                "team_id": t,
                "mvp": player(mvp) if mvp is not None else None,
                "best_add": (
                    {**player(add[0]), "day": add[1], "weekday": day_index[add[1]], "after": add[2]} if add else None
                ),
                "bench": dict(bench[t]) if t in bench else {"games": 0},
            }
        )

    return {
        "version": VERSION,
        "period": week.period,
        "days": week.days,
        "matchups": _matchups(week),
        "players": top,
        "free_agents": free_agents,
        "teams": teams,
    }

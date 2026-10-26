"""Checks the injury-beneficiary model against real absences of a past season.

    uv run python -m scripts.injury_backtest 2026          # test season, with 2025 as last season
    uv run python -m scripts.injury_backtest 2025 --tune   # tuning season (no earlier season used)

An absence is a regular (24+ minutes over his last 10 games) missing 3+ team games
in a row and then playing again for the same team. The model sees only games
before the absence. It is compared with what happened in the first 10 games of the
absence, and with two simple rules: "no change" and "his backup gets his minutes"
(the same-position teammate with the most minutes takes all of them).

Game logs come from ESPN once per team and season and are cached in .cache/gamelogs.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import httpx

from app.analysis.injury import (
    COUNTING,
    LAST_SEASON_WEIGHT,
    LEAGUE_FG,
    LEAGUE_FT,
    MAX_MINUTES,
    GameLine,
    PlayerInfo,
    Season,
    analyze,
    parse_gamelogs,
    value_of,
)

CACHE = Path(".cache/gamelogs")
URL = "https://lm-api-reads.fantasy.espn.com/apis/v3/games/fba/seasons/{season}/players"
REGULAR_MINUTES = 24.0
MIN_RUN = 3
EVAL_GAMES = 10


def load(season: int) -> tuple[list[GameLine], dict[int, PlayerInfo]]:
    CACHE.mkdir(parents=True, exist_ok=True)
    seen: set[tuple[int, int]] = set()
    lines: list[GameLine] = []
    info: dict[int, PlayerInfo] = {}
    for team in range(1, 31):
        path = CACHE / f"{season}-{team}.json"
        if not path.exists():
            flt = {
                "filterProTeamIds": {"value": [team]},
                "filterStatsForSplitTypeIds": {"value": [5]},
                "limit": 60,
                "sortPercOwned": {"sortPriority": 1, "sortAsc": False},
            }
            resp = httpx.get(
                URL.format(season=season),
                params={"view": "kona_player_info", "scoringPeriodId": "0"},
                headers={"X-Fantasy-Filter": json.dumps(flt)},
                timeout=60,
            )
            resp.raise_for_status()
            path.write_text(resp.text)
        got, who = parse_gamelogs(json.loads(path.read_text()))
        info.update(who)
        for ln in got:
            if (ln.player_id, ln.game_id) not in seen:
                seen.add((ln.player_id, ln.game_id))
                lines.append(ln)
    return lines, info


def per_game(lines: list[GameLine]) -> dict[str, float]:
    n = len(lines)
    out = {k: sum(getattr(ln, k) for ln in lines) / n for k in (*COUNTING, "fgm", "fga", "ftm", "fta", "minutes")}
    out["fg_impact"] = out["fgm"] - LEAGUE_FG * out["fga"]
    out["ft_impact"] = out["ftm"] - LEAGUE_FT * out["fta"]
    return out


def episodes(season: Season):
    """(team, absent player, games before, games during) for every qualifying absence."""
    for team, games in season.games.items():
        ids = [g for _, g in games]
        players = {pid for g in ids for pid, ln in season.by_game[g].items() if ln.team == team}
        for x in players:
            played = [i for i, g in enumerate(ids) if season.line(g, x, team)]
            if len(played) < 10:
                continue
            played_set = set(played)
            i = played[0]
            while i <= played[-1]:
                if i in played_set:
                    i += 1
                    continue
                j = i
                while j <= played[-1] and j not in played_set:
                    j += 1
                if j - i >= MIN_RUN and j <= played[-1]:
                    prior = [k for k in played if k < i][-10:]
                    mins = [season.line(ids[k], x, team).minutes for k in prior]
                    if len(prior) == 10 and statistics.mean(mins) >= REGULAR_MINUTES:
                        yield team, x, [ids[k] for k in prior], ids[i : min(j, i + EVAL_GAMES)]
                i = j


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("season", type=int)
    ap.add_argument("--tune", action="store_true", help="use no earlier season")
    args = ap.parse_args()

    lines, info = load(args.season)
    current = Season(lines)
    seasons = [current]
    if not args.tune:
        seasons.append(Season(load(args.season - 1)[0], weight=LAST_SEASON_WEIGHT))
    period_of = {ln.game_id: ln.period for ln in lines}

    err = defaultdict(list)
    hits = defaultdict(int)
    pairs = []
    count = 0
    for team, x, before, during in episodes(current):
        roster_ids = [
            pid
            for pid in {p for g in before for p, ln in current.by_game[g].items() if ln.team == team}
            if pid != x
            and sum(1 for g in during if current.line(g, pid, team)) >= 2
            and sum(1 for g in before if current.line(g, pid, team)) >= 3
        ]
        if len(roster_ids) < 3:
            continue
        roster = [info.get(p) or PlayerInfo(p, str(p), "", team) for p in roster_ids]
        start = period_of[during[0]]
        result = {b.player_id: b for b in analyze(seasons, team, x, roster, before=start).beneficiaries}
        x_recent = statistics.mean(current.line(g, x, team).minutes for g in before)
        x_pos = info[x].position if x in info else ""
        same = [p for p in roster if p.position == x_pos] or roster
        actual = {}
        for p in roster:
            pre = [ln for g in before if (ln := current.line(g, p.id, team))]
            post = [ln for g in during if (ln := current.line(g, p.id, team))]
            a, b = per_game(pre), per_game(post)
            actual[p.id] = (a["minutes"], b["minutes"], value_of({k: b[k] - a[k] for k in b}))
        backup = max(same, key=lambda p: actual[p.id][0]).id
        count += 1
        for pid, (pre_min, post_min, gain) in actual.items():
            b = result.get(pid)
            if b is None:
                continue
            err["no change"].append(abs(post_min - pre_min))
            err["backup"].append(abs((min(pre_min + x_recent, MAX_MINUTES) if pid == backup else pre_min) - post_min))
            miss = abs(min(max(pre_min + b.minutes_change, 0), MAX_MINUTES + 4) - post_min)
            err["model"].append(miss)
            err[f"model, {b.confidence}"].append(miss)
            err[f"no change, where model is {b.confidence}"].append(abs(post_min - pre_min))
            pairs.append((b.value, gain))
        top_actual = sorted(actual, key=lambda p: actual[p][2], reverse=True)[:2]
        top_minutes = sorted(actual, key=lambda p: actual[p][1] - actual[p][0], reverse=True)[:2]
        ranked = [p for p in result if p in actual]
        by_minutes = sorted(ranked, key=lambda p: result[p].minutes_change, reverse=True)
        if ranked and ranked[0] in top_actual:
            hits["model"] += 1
        if backup in top_actual:
            hits["backup"] += 1
        if by_minutes and by_minutes[0] in top_minutes:
            hits["model minutes"] += 1
        if backup in top_minutes:
            hits["backup minutes"] += 1

    if not count:
        sys.exit("no absences found")
    print(f"{args.season}: {count} absences, {len(pairs)} teammate cases")
    for k, v in sorted(err.items()):
        print(f"  minutes error ({k}): {statistics.mean(v):.2f} per game over {len(v)}")
    for k in ("model", "backup"):
        print(f"  top pick among the 2 biggest real gainers in value ({k}): {hits[k] / count:.0%}")
        print(f"  top pick among the 2 biggest real gainers in minutes ({k}): {hits[k + ' minutes'] / count:.0%}")
    xs, ys = zip(*pairs, strict=True)
    print(f"  correlation of predicted and real value change: {statistics.correlation(xs, ys):.2f}")


if __name__ == "__main__":
    main()

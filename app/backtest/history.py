"""A finished season of one league, as ESPN keeps it: the data a backtest replays.

Everything here is public for public leagues: the league with every matchup's final
category totals, each team's lineup on every day, every NBA box score, and the player
pool of that season and the one before (last season's stats are the prior).
Fetched once and cached as gzipped JSON, since a season is a few hundred requests.
"""

from __future__ import annotations

import asyncio
import gzip
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from app.espn.client import EspnClient, EspnCredentials, LeagueNotFound

# Box score stats the model reads: the scoring stats, makes/attempts, minutes, games played.
BOX_STATS = {"0", "1", "2", "3", "6", "11", "13", "14", "15", "16", "17", "40", "42"}
CONCURRENCY = 8


@dataclass
class History:
    league_id: int
    season: int
    league: dict  # ESPN league JSON: settings, teams with final records and seeds, schedule with final scores
    pro_schedule: dict  # NBA games by scoring period
    players: dict[int, dict]  # player id -> {"fullName", "eligibleSlots", "proTeamId"}
    prior: dict[int, dict[str, float]]  # player id -> last season's per-game averageStats (ESPN stat ids)
    box: dict[int, list[dict]]  # day -> [{"pid", "team", "stats"}] for every player who played
    rosters: dict[int, dict[int, list[list[int]]]]  # day -> team id -> [[player id, lineup slot]]

    @property
    def last_day(self) -> int:
        return max(self.box) if self.box else 0

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt") as fh:
            json.dump(asdict(self), fh)

    @classmethod
    def load(cls, path: Path) -> History:
        with gzip.open(path, "rt") as fh:
            raw = json.load(fh)
        return cls(
            league_id=raw["league_id"],
            season=raw["season"],
            league=raw["league"],
            pro_schedule=raw["pro_schedule"],
            players={int(k): v for k, v in raw["players"].items()},
            prior={int(k): v for k, v in raw["prior"].items()},
            box={int(k): v for k, v in raw["box"].items()},
            rosters={int(d): {int(t): r for t, r in teams.items()} for d, teams in raw["rosters"].items()},
        )


def _box_lines(players: list[dict], day: int) -> list[dict]:
    lines = []
    for e in players:
        p = e.get("player") or {}
        for s in p.get("stats", []):
            if s.get("scoringPeriodId") != day or s.get("statSourceId") != 0 or s.get("statSplitTypeId") != 5:
                continue
            stats = s.get("stats") or {}
            if not stats.get("42", 1):  # listed but did not play
                continue
            if stats:
                lines.append(
                    {
                        "pid": int(p["id"]),
                        "team": int(s.get("proTeamId") or p.get("proTeamId") or 0),
                        "stats": {k: float(v) for k, v in stats.items() if k in BOX_STATS},
                    }
                )
    return lines


def _season_averages(players: list[dict], season: int) -> dict[int, dict[str, float]]:
    out = {}
    for e in players:
        p = e.get("player") or {}
        for s in p.get("stats", []):
            if s.get("seasonId") == season and s.get("statSourceId") == 0 and s.get("statSplitTypeId") == 0:
                avg = s.get("averageStats") or {}
                if avg:
                    out[int(p["id"])] = {k: float(v) for k, v in avg.items() if k in BOX_STATS}
    return out


async def fetch_history(
    client: EspnClient, league_id: int, season: int, creds: EspnCredentials | None = None
) -> History:
    sem = asyncio.Semaphore(CONCURRENCY)

    async def limited(coro):
        async with sem:
            return await coro

    league = await client.league(season, league_id, creds)
    pro = await client.pro_schedule(season)
    pool = await client.player_pool(season, league_id, creds)
    try:
        prior_pool = await client.player_pool(season - 1, league_id, creds)
    except LeagueNotFound:  # a league's first season: no prior from the same league
        prior_pool = []
    last = int((league.get("status") or {}).get("finalScoringPeriod") or 0)
    days = range(1, last + 1)
    box_raw = await asyncio.gather(*(limited(client.day_stats(season, league_id, d, creds)) for d in days))
    ros_raw = await asyncio.gather(*(limited(client.rosters_on_day(season, league_id, d, creds)) for d in days))
    rosters = {
        d: {
            int(t["id"]): [
                [int(e["playerId"]), int(e["lineupSlotId"])] for e in (t.get("roster") or {}).get("entries", [])
            ]
            for t in raw.get("teams", [])
        }
        for d, raw in zip(days, ros_raw, strict=True)
    }
    players = {
        int(e["player"]["id"]): {
            "fullName": e["player"].get("fullName", ""),
            "eligibleSlots": e["player"].get("eligibleSlots", []),
            "proTeamId": e["player"].get("proTeamId", 0),
        }
        for e in pool
        if e.get("player")
    }
    return History(
        league_id=league_id,
        season=season,
        league=league,
        pro_schedule=pro,
        players=players,
        prior=_season_averages(prior_pool, season - 1),
        box={d: lines for d, raw in zip(days, box_raw, strict=True) if (lines := _box_lines(raw, d))},
        rosters=rosters,
    )


async def load_history(league_id: int, season: int, cache_dir: Path, client: EspnClient | None = None) -> History:
    path = cache_dir / f"{league_id}-{season}.json.gz"
    if path.exists():
        return History.load(path)
    h = await fetch_history(client or EspnClient(timeout=60.0), league_id, season)
    h.save(path)
    return h

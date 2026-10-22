"""Record a public league from ESPN as a trimmed test fixture.

Usage: uv run python scripts/record_fixture.py LEAGUE_ID SEASON tests/fixtures/NAME.json.gz
Only public leagues: this script never takes cookies.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import sys

from app.espn.client import EspnClient
from app.espn.parse import STAT_NAMES

KEEP_STATS = {str(k) for k in STAT_NAMES} | {"42"}


def trim_player(p: dict, season: int) -> dict:
    stats = [
        {
            "seasonId": s["seasonId"],
            "statSourceId": s["statSourceId"],
            "statSplitTypeId": s["statSplitTypeId"],
            "averageStats": {k: v for k, v in (s.get("averageStats") or {}).items() if k in KEEP_STATS},
        }
        for s in p.get("stats", [])
        if s.get("seasonId") == season and s.get("statSplitTypeId") == 0 and s.get("averageStats")
    ]
    keep = ("id", "fullName", "eligibleSlots", "proTeamId", "injuryStatus")
    out = {k: p[k] for k in keep if k in p}
    out["ownership"] = {"percentOwned": (p.get("ownership") or {}).get("percentOwned", 0)}
    out["stats"] = stats
    return out


async def main(league_id: int, season: int, path: str) -> None:
    client = EspnClient()
    raw = await client.league(season, league_id, None)
    fas = await client.free_agents(season, league_id, None)
    for t in raw.get("teams", []):
        t.pop("valuesByStat", None)
        t["owners"] = []
        t.pop("primaryOwner", None)
        for e in (t.get("roster") or {}).get("entries", []):
            pool = e["playerPoolEntry"]
            pool["player"] = trim_player(pool["player"], season)
            pool.pop("ratings", None)
    raw.pop("members", None)
    raw.pop("draftDetail", None)
    for m in raw.get("schedule", []):
        for side in ("home", "away"):
            if side in m:
                m[side] = {"teamId": m[side]["teamId"]}
    fas = [
        {"onTeamId": f.get("onTeamId", 0), "status": f.get("status"), "player": trim_player(f["player"], season)}
        for f in fas
    ]
    with gzip.open(path, "wt") as fh:
        json.dump({"league": raw, "free_agents": fas}, fh)
    print(f"wrote {path}")


if __name__ == "__main__":
    asyncio.run(main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]))

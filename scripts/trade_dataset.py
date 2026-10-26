"""Build the trade offer dataset from public ESPN leagues.

  uv run python -m scripts.trade_dataset discover --season 2019 --lo 1 --hi 300000 -n 5000 --seed 1
      probe random league ids, keep the public ones in .cache/tradedata/leagues.jsonl
  uv run python -m scripts.trade_dataset crawl [--league 744636428 --seasons 2021 2022 ...]
      download every found league-season that has trade activity (cached, resumable)
  uv run python -m scripts.trade_dataset build [--out DIR]
      turn the cache into offers.jsonl (one offer per line, players with features), offer_sides.csv
      (one line per offer and side) and seasons.json (counts and data quality per league-season)

Read-only and throttled to a few requests a second; no cookies, so only public leagues.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from app.tradedata.dataset import build_league_season, write
from app.tradedata.fetch import PoliteClient, crawl_league_season, discover

LAST_SEASON = 2026  # the last finished season
FIRST_SEASON = 2019  # ESPN's v3 API serves seasons from 2019; older ones need cookies

CACHE = Path(".cache/tradedata")
LEAGUES = CACHE / "leagues.jsonl"


def cmd_discover(args: argparse.Namespace) -> None:
    client = PoliteClient(rate=args.rate)
    total = 0
    for start in range(0, args.n, args.batch):
        n = min(args.batch, args.n - start)
        found = asyncio.run(
            discover(client, args.season, args.lo, args.hi, n, seed=args.seed * 1_000_003 + start, out=LEAGUES)
        )
        total += len(found)
        print(f"probed {start + n}/{args.n}  public found {total}", flush=True)
        for f in found:
            print(f"  {f.league_id} {f.season} {f.scoring_type} size={f.size} seasons={list(f.seasons)} {f.name!r}")


def _targets(args: argparse.Namespace) -> list[tuple[int, int]]:
    if args.league:
        return [(lid, s) for lid in args.league for s in args.seasons]
    seen: dict[int, set[int]] = {}
    for line in LEAGUES.read_text().splitlines() if LEAGUES.exists() else []:
        r = json.loads(line)
        seen.setdefault(r["league_id"], set()).update(s for s in r["seasons"] if s >= FIRST_SEASON)
    # A probe only lists the seasons up to the one probed; later renewals are tried too (one request if absent).
    return [(lid, s) for lid in sorted(seen) for s in range(min(seen[lid]), LAST_SEASON + 1)]


def cmd_crawl(args: argparse.Namespace) -> None:
    client = PoliteClient(rate=args.rate)
    log = CACHE / "crawl.jsonl"
    done = set()
    if log.exists():
        done = {(r["league_id"], r["season"]) for r in map(json.loads, log.read_text().splitlines())}
    for lid, season in _targets(args):
        if (lid, season) in done:
            continue
        try:
            result = asyncio.run(crawl_league_season(client, lid, season, CACHE))
        except Exception as exc:  # noqa: BLE001 - one bad season must not stop the crawl
            result = f"error: {type(exc).__name__}: {exc}"
        print(f"{lid} {season}: {result}  ({client.requests} requests so far)", flush=True)
        if "EspnUnavailable" in result:  # transient: retried on the next run
            continue
        with log.open("a") as fh:
            fh.write(json.dumps({"league_id": lid, "season": season, "result": result}) + "\n")


def cmd_build(args: argparse.Namespace) -> None:
    rows, seasons = [], []
    for path in sorted((CACHE / "history").glob("*.json.gz")):
        lid, season = map(int, path.name.removesuffix(".json.gz").split("-"))
        if args.league and lid not in args.league:
            continue
        try:
            r, stats = build_league_season(lid, season, CACHE)
        except Exception as exc:  # noqa: BLE001
            print(f"{lid} {season}: build failed: {type(exc).__name__}: {exc}", flush=True)
            continue
        rows += r
        seasons.append(stats)
        print(f"{lid} {season}: {stats['outcomes']} declines={stats['declines']}", flush=True)
    write(rows, args.out)
    (args.out / "seasons.json").write_text(json.dumps(seasons, indent=1))
    print(f"{len(rows)} offers from {len(seasons)} league-seasons -> {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("discover")
    d.add_argument("--season", type=int, required=True)
    d.add_argument("--lo", type=int, default=1)
    d.add_argument("--hi", type=int, default=300_000)
    d.add_argument("-n", type=int, default=1000)
    d.add_argument("--seed", type=int, default=1)
    d.add_argument("--batch", type=int, default=500)
    d.add_argument("--rate", type=float, default=4.0)
    c = sub.add_parser("crawl")
    c.add_argument("--league", type=int, nargs="*", help="these leagues instead of the discovered ones")
    c.add_argument("--seasons", type=int, nargs="*", default=[])
    c.add_argument("--rate", type=float, default=3.0)
    b = sub.add_parser("build")
    b.add_argument("--league", type=int, nargs="*")
    b.add_argument("--out", type=Path, default=CACHE / "out")
    args = ap.parse_args()
    {"discover": cmd_discover, "crawl": cmd_crawl, "build": cmd_build}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())

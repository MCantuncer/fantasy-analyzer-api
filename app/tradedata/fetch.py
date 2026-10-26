"""Polite, cached reads of public ESPN leagues: finding them and pulling their transactions.

ESPN has no list of public leagues, so they are found by probing league ids. Every request
here goes through one throttle (a few per second, retries with backoff), reads only, and
never sends cookies: private leagues answer 401 and are skipped.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

from app.espn.client import EspnClient, EspnCredentials, EspnUnavailable, LeagueNotFound, LeaguePrivate

RATE = 4.0  # requests per second, across everything this process sends
RETRIES = 4


class Throttle:
    """At most `rate` request starts per second."""

    def __init__(self, rate: float):
        self.gap = 1.0 / rate
        self.next = 0.0
        self.lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self.lock:
            now = time.monotonic()
            delay = self.next - now
            self.next = max(now, self.next) + self.gap
        if delay > 0:
            await asyncio.sleep(delay)


class PoliteClient(EspnClient):
    """EspnClient with a shared throttle and retries on network errors and 5xx/429."""

    def __init__(self, rate: float = RATE, timeout: float = 60.0, **kw):
        super().__init__(timeout=timeout, **kw)
        self.throttle = Throttle(rate)
        self.requests = 0

    async def _get(self, url, params, creds, headers=None):  # noqa: ANN001 - same signature as the base
        for attempt in range(RETRIES + 1):
            await self.throttle.wait()
            self.requests += 1
            try:
                return await super()._get(url, params, creds, headers)
            except EspnUnavailable:
                if attempt == RETRIES:
                    raise
                await asyncio.sleep(2 ** (attempt + 1) + random.random())
        raise AssertionError("unreachable")


@dataclass
class LeagueSeen:
    league_id: int
    season: int
    status: int  # 200 public, 401 private, 404 none
    name: str = ""
    scoring_type: str = ""
    size: int = 0
    seasons: tuple[int, ...] = ()  # every season this league has existed, the probed one included


async def probe(client: PoliteClient, league_id: int, season: int) -> LeagueSeen:
    try:
        raw = await client._get(
            client._league_url(season, league_id), [("view", "mSettings"), ("view", "mStatus")], None
        )
    except LeaguePrivate:
        return LeagueSeen(league_id, season, 401)
    except LeagueNotFound:
        return LeagueSeen(league_id, season, 404)
    settings = raw.get("settings") or {}
    status = raw.get("status") or {}
    seasons = sorted({season, *status.get("previousSeasons", [])})
    return LeagueSeen(
        league_id,
        season,
        200,
        name=settings.get("name", ""),
        scoring_type=(settings.get("scoringSettings") or {}).get("scoringType", ""),
        size=int(settings.get("size") or 0),
        seasons=tuple(seasons),
    )


async def discover(
    client: PoliteClient, season: int, lo: int, hi: int, n: int, seed: int, out: Path, concurrency: int = 4
) -> list[LeagueSeen]:
    """Probe n random league ids in [lo, hi] for one season; append public ones to `out` (JSON lines)."""
    rng = random.Random(seed)
    ids = rng.sample(range(lo, hi + 1), n)
    sem = asyncio.Semaphore(concurrency)
    found: list[LeagueSeen] = []

    async def one(lid: int) -> LeagueSeen | None:
        async with sem:
            try:
                return await probe(client, lid, season)
            except EspnUnavailable:
                return None

    out.parent.mkdir(parents=True, exist_ok=True)
    for seen in await asyncio.gather(*(one(i) for i in ids)):
        if seen and seen.status == 200:
            found.append(seen)
    with out.open("a") as fh:
        for seen in found:
            fh.write(json.dumps(seen.__dict__) + "\n")
    return found


async def fetch_transactions(
    client: EspnClient, league_id: int, season: int, last_day: int, creds: EspnCredentials | None = None
) -> dict[int, list[dict]]:
    """Every transaction of a season, by scoring period (day 0 is the draft and preseason)."""
    sem = asyncio.Semaphore(4)

    async def one(day: int) -> list[dict]:
        async with sem:
            return await client.transactions(season, league_id, day, creds)

    days = range(0, last_day + 1)
    got = await asyncio.gather(*(one(d) for d in days))
    return dict(zip(days, got, strict=True))


def save_json(path: Path, data) -> None:  # noqa: ANN001
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt") as fh:
        json.dump(data, fh)
    tmp.replace(path)


def load_json(path: Path):  # noqa: ANN201
    with gzip.open(path, "rt") as fh:
        return json.load(fh)


TRADE_TYPES = {"TRADE_PROPOSAL", "TRADE_ACCEPT", "TRADE_DECLINE", "TRADE_VETO", "TRADE_UPHOLD", "TRADE_ERROR"}


def has_trades(transactions: dict[int, list[dict]]) -> bool:
    return any(t.get("type") in TRADE_TYPES for day in transactions.values() for t in day)


def _pool_meta(pool: list[dict], season: int) -> dict[int, dict]:
    """What everyone could see before the season: ESPN's projection, average draft position and auction value."""
    out = {}
    for e in pool:
        p = e.get("player") or {}
        if "id" not in p:
            continue
        own = p.get("ownership") or {}
        proj = next(
            (
                s.get("averageStats") or {}
                for s in p.get("stats", [])
                if s.get("seasonId") == season and s.get("statSourceId") == 1 and s.get("statSplitTypeId") == 0
            ),
            {},
        )
        out[int(p["id"])] = {
            "adp": own.get("averageDraftPosition"),
            "auction_avg": own.get("auctionValueAverage"),
            "pct_owned_end": own.get("percentOwned"),
            "projection": proj,
        }
    return out


async def crawl_league_season(client: PoliteClient, league_id: int, season: int, cache: Path) -> str:
    """Download one league-season into the cache. Returns what happened, for the log."""
    from app.backtest.history import load_history  # heavy: only when a season is worth it

    tx_path = cache / "tx" / f"{league_id}-{season}.json.gz"
    if tx_path.exists():
        tx = {int(k): v for k, v in load_json(tx_path).items()}
    else:
        settings = await client._get(client._league_url(season, league_id), [("view", "mStatus")], None)
        last = int((settings.get("status") or {}).get("finalScoringPeriod") or 0)
        if not last:
            return "no scoring periods"
        tx = await fetch_transactions(client, league_id, season, last)
        save_json(tx_path, tx)
    if not has_trades(tx):
        return "no trade activity"
    meta_path = cache / "meta" / f"{league_id}-{season}.json.gz"
    if not meta_path.exists():
        save_json(meta_path, _pool_meta(await client.player_pool(season, league_id, None), season))
    await load_history(league_id, season, cache / "history", client)
    return "ok"

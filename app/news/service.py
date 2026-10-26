"""Collects news from every source, tags it and keeps it in the store.

A refresh runs at most every ``REFRESH_EVERY``: any request for news starts one
in the background when the last is older than that, so the feed stays current
as long as someone (or a scheduled ping) asks for it. A failing source never
stops the others; its error shows in the source health list.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict
from datetime import UTC, datetime, timedelta

import httpx

from app.espn.client import EspnClient, EspnError
from app.news import facts, sources
from app.news.extract import NameIndex, tag
from app.news.verify import FACT_SOURCE, Item, Story, build_stories
from app.store import Store

log = logging.getLogger(__name__)

REFRESH_EVERY = timedelta(minutes=15)
FIRST_WAIT = 25.0  # seconds a request waits for the very first refresh before answering empty
KEEP = timedelta(days=7)
FUTURE_SLACK = timedelta(hours=1)


def iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")


def parse_iso(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def item_to_dict(item: Item) -> dict:
    d = asdict(item)
    d["published"] = iso(item.published)
    return d


def item_from_dict(d: dict) -> Item:
    return Item(**{**d, "published": parse_iso(d["published"])})


def states_to_dict(states: dict[int, facts.PlayerState]) -> dict:
    return {str(pid): [st.name, st.pro_team_id, st.status] for pid, st in states.items()}


def states_from_dict(body: dict | None) -> dict[int, facts.PlayerState]:
    return {int(k): facts.PlayerState(int(k), v[0], int(v[1]), v[2]) for k, v in (body or {}).items()}


class NewsService:
    def __init__(
        self,
        espn: EspnClient,
        store: Store,
        transport: httpx.AsyncBaseTransport | None = None,
        source_list: tuple[sources.Source, ...] = sources.SOURCES,
    ):
        self.espn = espn
        self.store = store
        self.transport = transport
        self.sources = source_list
        self._lock = asyncio.Lock()
        self._task: asyncio.Task | None = None

    # -- state kept in the store, so a restarted server knows what it has seen
    def _state(self, key: str) -> dict:
        return self.store.get_news_state(key) or {}

    def last_refresh(self) -> datetime | None:
        at = self._state("meta").get("refreshed_at")
        return parse_iso(at) if at else None

    def health(self) -> dict:
        return self._state("health")

    def trending(self) -> list[dict]:
        return self._state("trending").get("players", [])

    def states(self) -> dict[int, facts.PlayerState]:
        return states_from_dict(self.store.get_news_state("players"))

    def _record(self, health: dict, source: sources.Source, now: datetime, items: int, error: str | None):
        entry = health.get(source.id, {})
        entry.update({"last_attempt": iso(now), "ok": error is None, "items": items, "error": error})
        if error is None:
            entry["last_success"] = iso(now)
        health[source.id] = entry

    async def refresh(self, season: int, now: datetime) -> None:
        async with self._lock:
            last = self.last_refresh()
            if last is not None and now - last < REFRESH_EVERY:
                return
            health = self.health()
            fact_source = sources.BY_ID[FACT_SOURCE]
            previous = self.states()
            states = previous
            new_items: list[Item] = []
            try:
                states = facts.parse_states(await self.espn.player_states(season))
                trending = await self.espn.trending(season)
                for ev in facts.diff(previous, states, now) if previous else []:
                    new_items.append(
                        Item(
                            id=ev.id,
                            source_id=FACT_SOURCE,
                            title="",
                            url="",
                            published=ev.at,
                            summary="",
                            kind=ev.kind,
                            player_ids=[ev.player_id],
                            severity=ev.severity,
                            old=ev.old,
                            new=ev.new,
                        )
                    )
                await asyncio.to_thread(self.store.save_news_state, "players", states_to_dict(states))
                await asyncio.to_thread(self.store.save_news_state, "trending", {"players": trending, "at": iso(now)})
                self._record(health, fact_source, now, len(states), None)
            except EspnError as exc:
                self._record(health, fact_source, now, 0, str(exc))

            index = NameIndex({pid: st.name for pid, st in states.items()})
            feeds = [s for s in self.sources if s.url]
            results = await asyncio.gather(*(sources.fetch(s, self.transport) for s in feeds), return_exceptions=True)
            for src, result in zip(feeds, results, strict=True):
                if isinstance(result, BaseException):
                    error = str(result) if isinstance(result, sources.FeedError) else type(result).__name__
                    if not isinstance(result, sources.FeedError):
                        log.warning("news source %s failed: %s", src.id, type(result).__name__)
                    self._record(health, src, now, 0, error)
                    continue
                kept = 0
                for raw in result:
                    if raw.published > now + FUTURE_SLACK or raw.published < now - KEEP:
                        continue
                    espn_ids = [pid for pid in raw.player_ids if pid in states]
                    t = tag(raw.title, raw.summary, index, espn_ids)
                    new_items.append(
                        Item(
                            id=raw.id,
                            source_id=src.id,
                            title=raw.title,
                            url=raw.url,
                            published=raw.published,
                            summary=raw.summary,
                            kind=t.kind,
                            player_ids=t.player_ids,
                            severity=t.severity,
                            body_part=t.body_part,
                        )
                    )
                    kept += 1
                self._record(health, src, now, kept, None)

            if new_items:
                await asyncio.to_thread(self.store.save_news_items, [item_to_dict(i) for i in new_items])
            await asyncio.to_thread(self.store.save_news_state, "health", health)
            await asyncio.to_thread(self.store.save_news_state, "meta", {"refreshed_at": iso(now), "season": season})

    def _start(self, season: int, now: datetime) -> asyncio.Task:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self.refresh(season, now))
        return self._task

    async def ensure_fresh(self, season: int, now: datetime) -> bool:
        """Start a refresh when due. Waits only for the first one ever. True while a refresh is running."""
        last = self.last_refresh()
        if last is not None and now - last < REFRESH_EVERY:
            return False
        task = self._start(season, now)
        if last is None:
            try:
                await asyncio.wait_for(asyncio.shield(task), FIRST_WAIT)
            except TimeoutError:
                return True
        return not task.done()

    def stories(self, teams: dict[int, str], now: datetime) -> list[Story]:
        items = [item_from_dict(d) for d in self.store.news_items(iso(now - KEEP))]
        return build_stories(items, self.states(), teams, now)

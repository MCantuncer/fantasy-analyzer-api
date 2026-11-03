"""One-off: run the real news pipeline once (plus Bluesky desks) and print likely duplicate stories."""

import asyncio
import json
import urllib.parse
from datetime import UTC, datetime, timedelta
from itertools import combinations

import httpx

from app.espn.client import EspnClient
from app.main import current_season
from app.news.extract import NameIndex, tag
from app.news.service import NewsService
from app.news.verify import Item, _overlap, build_stories
from app.store import MemoryStore

BSKY = ["rotoworld-hoops.bsky.social", "rotowirenba.bsky.social", "hoopsrumors.bsky.social"]


async def bluesky(index, states, now):
    out = []
    async with httpx.AsyncClient(timeout=15) as h:
        for handle in BSKY:
            r = await h.get(
                "https://public.api.bsky.app/xrpc/app.bsky.feed.getAuthorFeed?"
                + urllib.parse.urlencode({"actor": handle, "limit": 100, "filter": "posts_no_replies"})
            )
            for p in r.json().get("feed", []):
                if "reason" in p:
                    continue
                rec = p["post"]["record"]
                text = rec.get("text", "").strip()
                ext = (p["post"].get("embed") or {}).get("external") or {}
                title = text.split("\n")[0] or ext.get("title", "")
                at = datetime.fromisoformat(rec["createdAt"].replace("Z", "+00:00"))
                if not title or now - at > timedelta(days=3):
                    continue
                t = tag(title, ext.get("description", ""), index, [])
                out.append(
                    Item(
                        id=f"bsky:{p['post']['uri']}", source_id=f"bsky_{handle.split('.')[0]}", title=title,
                        url=ext.get("uri", ""), published=at, summary=ext.get("description", ""), kind=t.kind,
                        player_ids=t.player_ids, severity=t.severity, body_part=t.body_part,
                    )
                )
    return out


async def main():
    now = datetime.now(UTC)
    season = current_season(now)
    store = MemoryStore()
    svc = NewsService(EspnClient(), store)
    await svc.refresh(season, now)
    states = svc.states()
    teams = {}
    stories = svc.stories(teams, now)
    print("HEALTH", json.dumps({k: (v.get("ok"), v.get("items"), v.get("error")) for k, v in svc.health().items()}))
    print("STORIES", len(stories))
    report(stories, states, "BASE")


def report(stories, states, label):
    name = lambda pid: states[pid].name if pid in states else str(pid)  # noqa: E731
    for s in stories:
        if "Westbrook" in s.headline:
            for i in s.items:
                print(label, "ITEM", s.headline[:30], "|", i.published.isoformat()[:16], i.source_id, i.kind, [name(p) for p in i.player_ids], "|", i.title[:100])
    for s in stories:
        print(label, "STORY", s.kind, s.status, s.published.isoformat()[:16], [name(p) for p in s.player_ids][:4], "|", s.headline[:110], "| n=", len(s.items), sorted({i.source_id for i in s.items}))
    n = 0
    for a, b in combinations(stories, 2):
        shared = set(a.player_ids) & set(b.player_ids)
        ov = _overlap(a.headline, b.headline)
        close = abs(a.published - b.published) <= timedelta(hours=48)
        if close and (shared or ov >= 0.3):
            n += 1
            print(label, "DUP?", "shared=" + ",".join(name(p) for p in shared), f"ov={ov:.2f}", "|", a.kind, a.headline[:90], "||", b.kind, b.headline[:90])
    print(label, "DUP pairs", n)


asyncio.run(main())

"""One-off: what RotoWire and Rotoworld Bluesky posts look like (text, link card, ids)."""

import json

import httpx

for handle in ("rotowirenba.bsky.social", "rotoworld-hoops.bsky.social"):
    url = f"https://public.api.bsky.app/xrpc/app.bsky.feed.getAuthorFeed?actor={handle}&limit=100&filter=posts_no_replies"
    feed = httpx.get(url, timeout=20).json()["feed"]
    print(f"=== {handle}: {len(feed)} posts, {feed[-1]['post']['record']['createdAt']} .. {feed[0]['post']['record']['createdAt']}")
    for e in feed[:3]:
        print(json.dumps({"uri": e["post"]["uri"], "record": e["post"]["record"], "embed": e["post"].get("embed")}, indent=1)[:2500])
    cards: dict[str, int] = {}
    for e in feed:
        card = ((e["post"].get("embed") or {}).get("external") or {}).get("uri", "")
        cards[card] = cards.get(card, 0) + 1
    print("posts sharing a card link:", sum(n for n in cards.values() if n > 1), "of", len(feed))
    for e in feed:
        p = e["post"]
        if "Barnes" in p["record"].get("text", ""):
            ext = (p.get("embed") or {}).get("external") or {}
            print("BARNES", p["record"]["createdAt"], repr(p["record"]["text"]), "| card:", ext.get("uri"), "|", repr(ext.get("title")), "|", repr(ext.get("description")))

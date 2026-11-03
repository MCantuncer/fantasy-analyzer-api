"""Where NBA news comes from, and how each feed is read.

Only the sources listed here are ever read, so a story can't come from a random
site. Each source has a kind:

- ``outlet``: a news organisation. Two outlets from different groups agreeing is
  enough to publish a story; one alone shows as "Reported".
Some outlets are read through their Bluesky account (public API, no key), where
fantasy desks post every player update as it happens.

- ``tip``: social sites (Reddit). Never enough to publish on its own, because that
  is where look-alike insider accounts and made-up trades spread.

ESPN's player data (injury status, NBA team) is the third kind, ``fact``. It is
read through the ESPN client, see ``facts.py``.

Outlets in the same ``group`` are owned by the same company or copy each other,
so they count once.
"""

from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Literal

import httpx

SourceKind = Literal["fact", "outlet", "tip"]
USER_AGENT = "fantasy-analyzer/1.0 (+https://fantasy-analyzer-web.vercel.app)"
# ESPN's site API refuses (403) a user agent that carries a link, while Reddit rate-limits
# browser-like ones, so ESPN gets its own.
ESPN_USER_AGENT = "Mozilla/5.0 (compatible; fantasy-analyzer/1.0)"
MAX_BYTES = 2_000_000  # a feed bigger than this is not a news feed
SUMMARY_CHARS = 400
ESPN_SITE = "https://site.api.espn.com"
BLUESKY_FEED = "https://public.api.bsky.app/xrpc/app.bsky.feed.getAuthorFeed"


def _bluesky(handle: str) -> str:
    return f"{BLUESKY_FEED}?actor={handle}&limit=100&filter=posts_no_replies"


@dataclass(frozen=True)
class Source:
    id: str
    name: str
    kind: SourceKind
    group: str
    url: str = ""
    format: Literal["rss", "espn", "bluesky", "none"] = "rss"


SOURCES: tuple[Source, ...] = (
    Source("espn_players", "ESPN player data", "fact", "espn_data", format="none"),
    Source("espn", "ESPN", "outlet", "espn", f"{ESPN_SITE}/apis/site/v2/sports/basketball/nba/news?limit=50", "espn"),
    # RotoWire's RSS carries a handful of items a day; its Bluesky account posts every player update.
    Source("rotowire", "RotoWire", "outlet", "rotowire", _bluesky("rotowirenba.bsky.social"), "bluesky"),
    Source("rotoworld", "Rotoworld", "outlet", "nbc", _bluesky("rotoworld-hoops.bsky.social"), "bluesky"),
    Source("cbs", "CBS Sports", "outlet", "cbs", "https://www.cbssports.com/rss/headlines/nba/"),
    Source("yahoo", "Yahoo Sports", "outlet", "yahoo", "https://sports.yahoo.com/nba/rss.xml"),
    Source("realgm", "RealGM", "outlet", "realgm", "https://basketball.realgm.com/rss/wiretap/0/0.xml"),
    # HoopsHype now answers every request from a server with 402 (pay to read), so it was replaced.
    Source("hoopsrumors", "Hoops Rumors", "outlet", "hoopsrumors", "https://www.hoopsrumors.com/feed"),
    Source("stein", "The Stein Line", "outlet", "stein", "https://marcstein.substack.com/feed"),
    Source("reddit", "Reddit r/nba", "tip", "reddit", "https://www.reddit.com/r/nba/new/.rss?limit=50"),
)
BY_ID = {s.id: s for s in SOURCES}


@dataclass
class RawItem:
    source_id: str
    title: str
    url: str
    published: datetime
    summary: str = ""
    # ESPN tags its articles with athlete ids, which are the fantasy player ids.
    player_ids: list[int] = field(default_factory=list)

    @property
    def id(self) -> str:
        return f"{self.source_id}:{self.url or self.title}"


class FeedError(Exception):
    pass


_TAG = re.compile(r"<[^>]+>")
_SPACE = re.compile(r"\s+")


def clean(text: str | None, limit: int | None = None) -> str:
    """Plain text from a feed field: no HTML, entities decoded, whitespace collapsed."""
    out = _SPACE.sub(" ", html.unescape(_TAG.sub(" ", text or ""))).strip()
    if limit and len(out) > limit:
        out = out[: limit - 1].rsplit(" ", 1)[0] + "…"
    return out


def _when(text: str | None) -> datetime | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        dt = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _feed_info(root: ET.Element) -> tuple[set[str], set[str]]:
    """The feed's own title, description and home links, which some feeds repeat as an item."""
    channel = next((n for n in root.iter() if _local(n.tag) == "channel"), root)
    texts, links = set(), set()
    for child in channel:
        name = _local(child.tag)
        if name in ("title", "description", "subtitle") and child.text:
            texts.add(clean(child.text).casefold())
        elif name == "link":
            links.add((child.get("href") or child.text or "").strip().rstrip("/"))
    return texts - {""}, links - {""}


def parse_rss(source_id: str, body: bytes) -> list[RawItem]:
    """RSS 2.0 and Atom. Items without a title or date are skipped, and so are items that
    only repeat the feed's own title, description or home page."""
    try:
        root = ET.fromstring(body)  # expat does not fetch external entities
    except ET.ParseError as exc:
        raise FeedError(f"not a feed: {exc}") from None
    feed_texts, feed_links = _feed_info(root)
    out = []
    for node in root.iter():
        if _local(node.tag) not in ("item", "entry"):
            continue
        fields: dict[str, str] = {}
        for child in node:
            name = _local(child.tag)
            if name == "link" and child.get("href"):
                fields.setdefault("link", child.get("href", ""))
            elif child.text and name not in fields:
                fields[name] = child.text
        title = clean(fields.get("title"))
        published = _when(fields.get("pubDate") or fields.get("published") or fields.get("updated"))
        if not title or published is None:
            continue
        url = (fields.get("link") or "").strip()
        if title.casefold() in feed_texts or (url and url.rstrip("/") in feed_links):
            continue
        summary = fields.get("description") or fields.get("summary") or fields.get("content") or ""
        out.append(
            RawItem(
                source_id=source_id,
                title=title,
                url=url,
                published=published,
                summary=clean(summary, SUMMARY_CHARS),
            )
        )
    return out


def parse_espn(source_id: str, data: dict) -> list[RawItem]:
    """ESPN's news API: articles tagged with athlete ids."""
    out = []
    for art in data.get("articles") or data.get("feed") or []:
        title = clean(art.get("headline"))
        published = _when(art.get("published") or art.get("lastModified"))
        if not title or published is None:
            continue
        ids = [
            int(c["athleteId"])
            for c in art.get("categories") or []
            if c.get("type") == "athlete" and str(c.get("athleteId", "")).isdigit()
        ]
        link = ((art.get("links") or {}).get("web") or {}).get("href", "")
        out.append(
            RawItem(
                source_id=source_id,
                title=title,
                url=link,
                published=published,
                summary=clean(art.get("description") or art.get("story"), SUMMARY_CHARS),
                player_ids=ids,
            )
        )
    return out


def parse_bluesky(source_id: str, data: dict) -> list[RawItem]:
    """An account's own posts (no reposts or replies): the first line is the headline, and a
    linked card gives the link and summary."""
    out = []
    for entry in data.get("feed") or []:
        if entry.get("reason"):  # a repost of someone else
            continue
        post = entry.get("post") or {}
        record = post.get("record") or {}
        if record.get("reply"):
            continue
        card = ((post.get("embed") or {}).get("external")) or {}
        text = (record.get("text") or "").strip()
        title = clean(text.split("\n", 1)[0]) or clean(card.get("title"))
        published = _when(record.get("createdAt"))
        if not title or published is None:
            continue
        uri = post.get("uri", "")
        handle = (post.get("author") or {}).get("handle", "")
        link = card.get("uri") or (f"https://bsky.app/profile/{handle}/post/{uri.rsplit('/', 1)[-1]}" if uri else "")
        rest = clean(text.split("\n", 1)[1]) if "\n" in text else ""
        out.append(
            RawItem(
                source_id=source_id,
                title=title,
                url=link,
                published=published,
                summary=clean(rest or card.get("description"), SUMMARY_CHARS),
            )
        )
    return out


def _site(url: str) -> str:
    host = httpx.URL(url).host
    return ".".join(host.split(".")[-2:])


def _same_site(asked: str, got: str) -> bool:
    return _site(asked) == _site(got)


async def fetch(
    source: Source, transport: httpx.AsyncBaseTransport | None = None, timeout: float = 12.0
) -> list[RawItem]:
    async with httpx.AsyncClient(
        transport=transport,
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": ESPN_USER_AGENT if source.format == "espn" else USER_AGENT},
    ) as http:
        try:
            resp = await http.get(source.url)
        except httpx.HTTPError as exc:
            raise FeedError(f"request failed: {type(exc).__name__}") from None
    if not _same_site(source.url, str(resp.url)):
        raise FeedError("redirected to another site")
    if resp.status_code >= 400:
        raise FeedError(f"HTTP {resp.status_code}")
    if len(resp.content) > MAX_BYTES:
        raise FeedError("feed too large")
    if source.format in ("espn", "bluesky"):
        try:
            data = resp.json()
        except ValueError:
            raise FeedError("not JSON") from None
        return parse_espn(source.id, data) if source.format == "espn" else parse_bluesky(source.id, data)
    return parse_rss(source.id, resp.content)

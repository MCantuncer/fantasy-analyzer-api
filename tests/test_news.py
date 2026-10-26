from datetime import UTC, datetime, timedelta

import httpx
from fastapi.testclient import TestClient

from app.accounts import MemoryAccountStore
from app.espn.client import EspnClient
from app.main import create_app
from app.news import facts
from app.news.extract import NameIndex, body_part, kind_of, severity, tag
from app.news.sources import parse_espn, parse_rss
from app.news.verify import Item, build_stories
from app.store import MemoryStore
from tests.conftest import LEAGUE_ID, signed_in

NOW = datetime(2026, 10, 26, 12, 0, tzinfo=UTC)
JWILL, AJAY, DUNN, LAVINE, JAYLIN = 4593803, 4397040, 2993874, 3064440, 4432165
NAMES = {
    JWILL: "Jalen Williams",
    AJAY: "Ajay Mitchell",
    DUNN: "Kris Dunn",
    LAVINE: "Zach LaVine",
    JAYLIN: "Jaylin Williams",
    1: "Wendell Carter Jr.",
}
TEAMS = {25: "OKC", 12: "LAC", 23: "SAC", 28: "TOR"}


def states(**status) -> dict[int, facts.PlayerState]:
    team = {JWILL: 25, AJAY: 25, DUNN: 12, LAVINE: 23, JAYLIN: 25, 1: 19}
    return {pid: facts.PlayerState(pid, name, team[pid], status.get(str(pid), "ACTIVE")) for pid, name in NAMES.items()}


def item(source, title, hours_ago=1.0, players=(), summary="", **kw) -> Item:
    t = tag(title, summary, NameIndex(NAMES), list(players) or None)
    base = dict(
        id=f"{source}:{title}",
        source_id=source,
        title=title,
        url=f"https://example.com/{source}",
        published=NOW - timedelta(hours=hours_ago),
        summary=summary,
        kind=t.kind,
        player_ids=t.player_ids,
        severity=t.severity,
        body_part=t.body_part,
    )
    base.update(kw)
    return Item(**base)


RSS = b"""<?xml version="1.0"?>
<rss version="2.0"><channel><title>NBA</title>
<item><title>Kris Dunn moves into Clippers&#39; starting lineup</title><link>https://cbs.example/dunn</link>
<pubDate>Sun, 26 Oct 2026 08:00:00 GMT</pubDate>
<description>&lt;p&gt;Dunn started &lt;b&gt;again&lt;/b&gt;.&lt;/p&gt;</description></item>
<item><title>No date here</title></item>
</channel></rss>"""

ATOM = b"""<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"><title>r/nba</title>
<entry><title>[Fake] Zach LaVine traded to the Raptors</title><link href="https://reddit.example/1"/>
<updated>2026-10-26T09:00:00+00:00</updated><content type="html">source: trust me</content></entry>
</feed>"""


def test_rss_and_atom_feeds_parse_to_plain_text():
    rss = parse_rss("cbs", RSS)
    assert [i.title for i in rss] == ["Kris Dunn moves into Clippers' starting lineup"]
    assert rss[0].summary == "Dunn started again ."
    assert rss[0].published == datetime(2026, 10, 26, 8, tzinfo=UTC)
    atom = parse_rss("reddit", ATOM)
    assert atom[0].url == "https://reddit.example/1"
    assert atom[0].published.hour == 9


def test_espn_articles_keep_athlete_ids():
    data = {
        "articles": [
            {
                "headline": "Thunder's Jalen Williams out",
                "published": "2026-10-26T07:00:00Z",
                "links": {"web": {"href": "https://espn.example/a"}},
                "categories": [{"type": "athlete", "athleteId": JWILL}, {"type": "team", "teamId": 25}],
            }
        ]
    }
    [art] = parse_espn("espn", data)
    assert art.player_ids == [JWILL] and art.url == "https://espn.example/a"


def test_players_are_found_by_full_name_only():
    index = NameIndex(NAMES)
    assert index.find("Jalen Williams leaves with ankle injury") == [JWILL]
    assert index.find("Jaylin Williams to start for OKC") == [JAYLIN]
    assert index.find("Williams leaves the game") == []
    assert index.find("Wendell Carter gets the start") == [1]
    assert NameIndex({1: "Jalen Green", 2: "Jalen Green"}).find("Jalen Green scores 40") == []


def test_kind_and_severity_come_from_the_words():
    assert kind_of("Report: Kings and Raptors discuss Zach LaVine trade") == "rumor"
    assert kind_of("Zach LaVine traded to Raptors") == "trade"
    assert kind_of("Kris Dunn signs two-year extension") == "signing"
    assert kind_of("Jalen Williams ruled out with ankle sprain") == "injury"
    assert kind_of("Jalen Williams returns to lineup Monday") == "return"
    assert kind_of("Kris Dunn moves into starting lineup") == "role"
    assert severity("fractured left ankle, out indefinitely") == 3
    assert severity("ankle sprain, ruled out Monday") == 2
    assert severity("listed day-to-day with soreness") == 1
    assert body_part("Right ankle; sprain") == "ankle"


def test_roundups_tag_nobody():
    many = "Power rankings: Jalen Williams, Kris Dunn, Zach LaVine and Ajay Mitchell"
    assert tag(many, "", NameIndex(NAMES)).player_ids == []


def test_fact_events_from_two_snapshots():
    before = states()
    after = states(**{str(JWILL): "OUT"})
    after[LAVINE] = facts.PlayerState(LAVINE, "Zach LaVine", 28, "ACTIVE")
    events = {(e.player_id, e.kind) for e in facts.diff(before, after, NOW)}
    assert events == {(JWILL, "injury"), (LAVINE, "trade")}
    back = facts.diff(after, states(), NOW)
    assert (JWILL, "return") in {(e.player_id, e.kind) for e in back}


def test_injury_confirmed_by_espn_status_and_conflict_shown():
    items = [
        item("cbs", "Jalen Williams suffers fractured ankle, out indefinitely", 2),
        item("yahoo", "Thunder's Jalen Williams leaves game with ankle injury", 3),
    ]
    [story] = build_stories(items, states(**{str(JWILL): "DAY_TO_DAY"}), TEAMS, NOW)
    assert story.kind == "injury" and story.status == "confirmed"
    assert story.severity == 3
    assert "ESPN status says day-to-day" in story.conflict and "CBS Sports says serious" in story.conflict


def test_one_outlet_is_reported_and_tips_alone_are_held():
    stories = build_stories(
        [
            item("stein", "Kings, Raptors discuss Zach LaVine trade", 5),
            item("reddit", "Kris Dunn traded to the Lakers for 4 firsts", 4),
        ],
        states(),
        TEAMS,
        NOW,
    )
    by_player = {s.player_ids[0]: s for s in stories}
    assert by_player[LAVINE].status == "reported" and by_player[LAVINE].kind == "rumor"
    assert by_player[DUNN].status == "held"


def test_two_outlets_confirm_but_a_trade_without_a_team_change_did_not_happen():
    fresh = [item("cbs", "Zach LaVine traded to Raptors", 5), item("yahoo", "Raptors acquire Zach LaVine", 4)]
    [story] = build_stories(fresh, states(), TEAMS, NOW)
    assert story.status == "confirmed"
    old = [item("cbs", "Zach LaVine traded to Raptors", 80), item("yahoo", "Raptors acquire Zach LaVine", 79)]
    [story] = build_stories(old, states(), TEAMS, NOW)
    assert story.status == "didnt_happen"


def test_team_change_in_espn_data_turns_a_rumor_into_a_trade():
    fact = Item(
        id="fact:x",
        source_id="espn_players",
        title="",
        url="",
        published=NOW - timedelta(hours=1),
        summary="",
        kind="trade",
        player_ids=[LAVINE],
        old="23",
        new="28",
    )
    [story] = build_stories([item("stein", "Kings, Raptors discuss Zach LaVine trade", 20), fact], states(), TEAMS, NOW)
    assert story.kind == "trade" and story.status == "confirmed"
    assert story.headline == "Kings, Raptors discuss Zach LaVine trade"


def test_fact_only_story_gets_a_plain_data_headline():
    fact = Item(
        id="fact:y",
        source_id="espn_players",
        title="",
        url="",
        published=NOW,
        summary="",
        kind="injury",
        player_ids=[JWILL],
        severity=2,
        old="ACTIVE",
        new="OUT",
    )
    [story] = build_stories([fact], states(**{str(JWILL): "OUT"}), TEAMS, NOW)
    assert story.headline == "ESPN lists Jalen Williams as out" and story.status == "confirmed"


# -- the endpoint, with ESPN's player data and every feed faked


def news_client(recorded, fake_espn, feeds: dict[str, httpx.Response]):
    pool = [
        {"id": JWILL, "fullName": "Jalen Williams", "proTeamId": 25, "injuryStatus": "DAY_TO_DAY"},
        {"id": DUNN, "fullName": "Kris Dunn", "proTeamId": 12, "injuryStatus": "ACTIVE"},
    ]
    trending = [
        {
            "id": DUNN,
            "fullName": "Kris Dunn",
            "proTeamId": 12,
            "injuryStatus": "ACTIVE",
            "ownership": {"percentOwned": 18.7, "percentChange": 10.3},
        }
    ]

    def espn(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/seasons/2027/players"):
            return httpx.Response(200, json=trending if request.url.params["view"] == "kona_player_info" else pool)
        return fake_espn.handler(request)

    def feed(request: httpx.Request) -> httpx.Response:
        return feeds.get(request.url.host, httpx.Response(503))

    app = create_app(
        EspnClient(transport=httpx.MockTransport(espn)),
        store=MemoryStore(),
        accounts=MemoryAccountStore(),
        news_transport=httpx.MockTransport(feed),
    )
    app.state.now = lambda: NOW
    return app


def test_news_endpoint_checks_sources_and_hides_held_stories(recorded, fake_espn):
    rss = RSS.replace(b"Sun, 26 Oct 2026 08:00:00 GMT", b"Sun, 26 Oct 2026 10:00:00 GMT")
    cbs = b"""<rss><channel><item><title>Jalen Williams fractures ankle, out indefinitely</title>
    <link>https://cbs.example/jw</link><pubDate>Sun, 26 Oct 2026 09:00:00 GMT</pubDate></item></channel></rss>"""
    feeds = {
        "www.cbssports.com": httpx.Response(200, content=cbs),
        "www.rotowire.com": httpx.Response(200, content=rss),
        "www.reddit.com": httpx.Response(200, content=ATOM),
    }
    app = news_client(recorded, fake_espn, feeds)
    client: TestClient = signed_in(app)
    body = client.post("/api/news", json={"league_id": LEAGUE_ID, "season": 2027}).json()

    stories = {s["headline"]: s for s in body["stories"]}
    jw = stories["Jalen Williams fractures ankle, out indefinitely"]
    assert jw["status"] == "confirmed" and jw["severity"] == 3 and jw["conflict"]
    assert jw["players"][0]["pro_team"] == "OKC" and jw["players"][0]["injury_status"] == "DAY_TO_DAY"
    assert stories["Kris Dunn moves into Clippers' starting lineup"]["status"] == "reported"
    assert not any("LaVine" in h for h in stories), "a Reddit-only trade stays hidden"

    health = {s["source_id"]: s for s in body["sources"]}
    assert health["cbs"]["ok"] and health["espn_players"]["ok"]
    assert health["yahoo"]["ok"] is False and health["yahoo"]["error"] == "HTTP 503"
    assert body["trending"][0]["percent_change"] == 10.3 and body["trending"][0]["pro_team"] == "LAC"

    held = client.post("/api/news", json={"season": 2027, "include_held": True}).json()
    assert any(s["status"] == "held" for s in held["stories"])


def test_news_needs_sign_in(recorded, fake_espn):
    app = news_client(recorded, fake_espn, {})
    assert TestClient(app).post("/api/news", json={}).status_code == 401


def test_a_redirect_to_another_site_is_refused(recorded, fake_espn):
    feeds = {"www.cbssports.com": httpx.Response(302, headers={"Location": "https://evil.example/feed"})}
    app = news_client(recorded, fake_espn, feeds)
    body = signed_in(app).post("/api/news", json={"season": 2027}).json()
    cbs = next(s for s in body["sources"] if s["source_id"] == "cbs")
    assert cbs["ok"] is False and cbs["error"] == "redirected to another site"


def test_refresh_ping_needs_no_sign_in_and_runs_at_most_every_15_minutes(recorded, fake_espn):
    app = news_client(recorded, fake_espn, {})
    client = TestClient(app)
    assert client.get("/api/news/refresh").json()["ok"] is True
    calls = fake_espn.calls
    client.get("/api/news/refresh")
    assert fake_espn.calls == calls

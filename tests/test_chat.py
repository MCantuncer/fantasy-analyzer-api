import gzip
import json
import os
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from app.espn.client import EspnClient
from app.feed import parse_transactions
from app.main import create_app
from app.social import Challenge, MemorySocialStore, PostgresSocialStore, read_challenge, sign_challenge
from app.store import MemoryStore
from tests.conftest import LEAGUE_ID, MY_TEAM

FIXTURE = Path(__file__).parent / "fixtures" / "chill_primera_2027_feed.json.gz"
SEASON = 2027
OTHER_TEAM = 8  # Eryaman Lakers


@pytest.fixture(scope="module")
def recorded() -> dict:
    with gzip.open(FIXTURE, "rt") as fh:
        return json.load(fh)


class FeedEspn:
    """Serves the recorded feed data. Tests can change a team's abbreviation as its owner would on ESPN."""

    def __init__(self, recorded: dict):
        self.basics = json.loads(json.dumps(recorded["basics"]))
        self.transactions = recorded["transactions"]
        self.players = {p["id"]: p for p in recorded["players"]}
        self.periods_asked: list[int] = []

    def set_abbrev(self, team_id: int, abbrev: str) -> None:
        next(t for t in self.basics["teams"] if t["id"] == team_id)["abbrev"] = abbrev

    def handler(self, request: httpx.Request) -> httpx.Response:
        if f"/leagues/{LEAGUE_ID}" not in request.url.path:
            return httpx.Response(404, json={})
        views = request.url.params.get_list("view")
        if views == ["mTransactions2"]:
            period = int(request.url.params["scoringPeriodId"])
            self.periods_asked.append(period)
            return httpx.Response(200, json={"transactions": self.transactions.get(str(period), [])})
        if views == ["kona_player_info"]:
            ids = json.loads(request.headers["x-fantasy-filter"])["players"]["filterIds"]["value"]
            return httpx.Response(200, json={"players": [self.players[i] for i in ids if i in self.players]})
        return httpx.Response(200, json=self.basics)


@pytest.fixture
def espn(recorded):
    return FeedEspn(recorded)


@pytest.fixture(params=["memory", "postgres"])
def social(request):
    if request.param == "memory":
        return MemorySocialStore()
    url = os.getenv("TEST_DATABASE_URL")
    if not url:
        pytest.skip("set TEST_DATABASE_URL to test against Postgres")
    store = PostgresSocialStore(url)
    with store._connect() as conn:
        conn.execute("truncate team_sessions, league_posts, league_reactions")
    return store


@pytest.fixture
def client(espn, social):
    app = create_app(EspnClient(transport=httpx.MockTransport(espn.handler)), store=MemoryStore(), social=social)
    return TestClient(app)


def link(client, espn, team_id=MY_TEAM) -> dict[str, str]:
    start = client.post("/api/team-link/start", json={"league_id": LEAGUE_ID, "season": SEASON, "team_id": team_id})
    assert start.status_code == 200
    espn.set_abbrev(team_id, start.json()["code"].lower())  # ESPN may store it in any case
    r = client.post("/api/team-link/verify", json={"challenge": start.json()["challenge"]})
    assert r.status_code == 200
    return {"Authorization": f"Bearer {r.json()['token']}"}


def league(**extra) -> dict:
    return {"league_id": LEAGUE_ID, "season": SEASON, **extra}


def test_transactions_keep_roster_moves_and_attach_losing_bids(recorded):
    txs = parse_transactions([t for day in recorded["transactions"].values() for t in day])
    assert len(txs) == 9
    assert {tx.kind for tx in txs} == {"waiver"}  # no lineup shuffles, drafts or cancelled claims
    assert [tx.date_ms for tx in txs] == sorted((tx.date_ms for tx in txs), reverse=True)
    missi = next(tx for tx in txs if 5061589 in tx.player_ids)
    assert (missi.team_id, missi.bid) == (OTHER_TEAM, 4)
    assert [(b.team_id, b.bid) for b in missi.losing_bids] == [(6, 1)]
    assert [m.kind for m in missi.moves] == ["add", "drop"]


def test_parse_trade_and_dedupe():
    item = {"playerId": 1, "type": "TRADE", "fromTeamId": 3, "toTeamId": 7}
    back = {"playerId": 2, "type": "TRADE", "fromTeamId": 7, "toTeamId": 3}
    accept = {
        "id": "a",
        "type": "TRADE_ACCEPT",
        "status": "EXECUTED",
        "teamId": 3,
        "scoringPeriodId": 2,
        "proposedDate": 10,
        "relatedTransactionId": "offer",
        "items": [item, back],
    }
    uphold = {**accept, "id": "b", "type": "TRADE_UPHOLD", "proposedDate": 20}
    [trade] = parse_transactions([accept, uphold, accept])
    assert trade.kind == "trade" and trade.id == "offer"
    assert trade.team_ids == [3, 7]


def test_feed_without_a_link_shows_moves_only(client, espn):
    r = client.post("/api/feed", json=league())
    assert r.status_code == 200
    body = r.json()
    assert body["me"] is None and body["has_more"] is False
    assert sorted(set(espn.periods_asked)) == [1, 2, 3, 4]
    assert len(body["items"]) == 9 and all(i["kind"] == "waiver" for i in body["items"])
    names = {m["player"]["name"] for i in body["items"] for m in i["moves"]}
    assert {"Yves Missi", "Nikola Vucevic", "Peyton Watson"} <= names
    assert body["faab_budget"] == 100
    assert next(t for t in body["teams"] if t["id"] == OTHER_TEAM)["faab_spent"] == 9


def test_link_needs_the_code_on_espn(client, espn):
    start = client.post("/api/team-link/start", json=league(team_id=MY_TEAM)).json()
    assert start["team_name"] == "Kucukesat Aspavas" and start["current_abbrev"] == "KEA"
    assert len(start["code"]) == 4 and start["code"] != "KEA"

    r = client.post("/api/team-link/verify", json={"challenge": start["challenge"]})
    assert r.status_code == 409 and r.json()["detail"]["code"] == "abbrev_mismatch"

    espn.set_abbrev(MY_TEAM, start["code"])
    r = client.post("/api/team-link/verify", json={"challenge": start["challenge"]})
    assert r.status_code == 200
    me = r.json()
    assert (me["team_id"], me["team_name"]) == (MY_TEAM, "Kucukesat Aspavas") and me["token"]

    auth = {"Authorization": f"Bearer {me['token']}"}
    r = client.post("/api/team-link/me", json=league(), headers=auth)
    assert r.status_code == 200 and r.json()["token"] is None
    assert client.post("/api/team-link/me", json=league(season=2026), headers=auth).status_code == 401

    client.post("/api/team-link/unlink", headers=auth)
    assert client.post("/api/team-link/me", json=league(), headers=auth).status_code == 401


def test_challenges_are_signed_and_expire():
    c = Challenge(LEAGUE_ID, SEASON, MY_TEAM, "K7QX", int(time.time()) + 60)
    token = sign_challenge(c)
    assert read_challenge(token) == c
    assert read_challenge(token, now=time.time() + 120) is None
    body, mac = token.split(".")
    forged = sign_challenge(Challenge(LEAGUE_ID, SEASON, OTHER_TEAM, "K7QX", c.expires)).split(".")[0]
    assert read_challenge(f"{forged}.{mac}") is None
    assert read_challenge("garbage") is None


def test_expired_challenge_is_refused(client):
    r = client.post("/api/team-link/verify", json={"challenge": "not-a-real-challenge"})
    assert r.status_code == 410


def test_chat_replies_and_reactions(client, espn):
    me = link(client, espn)
    them = link(client, espn, OTHER_TEAM)

    assert client.post("/api/posts", json=league(body="hi")).status_code == 401
    r = client.post("/api/posts", json=league(body="  Vucevic on the wire, you're welcome  "), headers=me)
    assert r.status_code == 200 and r.json()["body"] == "Vucevic on the wire, you're welcome"

    feed = client.post("/api/feed", json=league(), headers=me).json()
    assert feed["me"]["team_id"] == MY_TEAM
    posts = [i for i in feed["items"] if i["kind"] == "post"]
    assert len(posts) == 1 and feed["items"][0]["kind"] == "post"  # newest first
    missi = next(i for i in feed["items"] if any(m["player"]["name"] == "Yves Missi" for m in i["moves"]))

    client.post("/api/posts", json=league(body="$4 on a backup center. Bold.", target=missi["id"]), headers=them)
    for who in (me, them):
        r = client.post("/api/reactions", json=league(target=missi["id"], reaction="overpay"), headers=who)
        assert r.status_code == 200
    r = client.post("/api/reactions", json=league(target=missi["id"], reaction="steal"), headers=them)
    assert r.json()["reactions"] == {"counts": {"overpay": 2, "steal": 1}, "mine": ["overpay", "steal"]}

    feed = client.post("/api/feed", json=league(), headers=me).json()
    item = next(i for i in feed["items"] if i["id"] == missi["id"])
    assert item["replies"] == 1
    assert item["reactions"] == {"counts": {"overpay": 2, "steal": 1}, "mine": ["overpay"]}

    r = client.post("/api/reactions", json=league(target=missi["id"], reaction="overpay", on=False), headers=me)
    assert r.json()["reactions"]["counts"] == {"overpay": 1, "steal": 1}

    replies = client.post("/api/feed/replies", json=league(target=missi["id"]), headers=me).json()["items"]
    assert [(p["team_id"], p["body"]) for p in replies] == [(OTHER_TEAM, "$4 on a backup center. Bold.")]


def test_trade_vote_is_one_per_team(client, espn):
    me = link(client, espn)
    for team in (3, 7):
        r = client.post("/api/reactions", json=league(target="tx:offer", reaction=f"won:{team}"), headers=me)
    assert r.json()["reactions"] == {"counts": {"won:7": 1}, "mine": ["won:7"]}
    r = client.post("/api/reactions", json=league(target="tx:offer", reaction="won:99"), headers=me)
    assert r.status_code == 422
    r = client.post("/api/reactions", json=league(target="tx:offer", reaction="fire"), headers=me)
    assert r.status_code == 422


def test_only_the_author_deletes_a_post(client, espn):
    me = link(client, espn)
    them = link(client, espn, OTHER_TEAM)
    post_id = int(client.post("/api/posts", json=league(body="mine"), headers=me).json()["id"].split(":")[1])
    assert client.post("/api/posts/delete", json=league(post_id=post_id), headers=them).status_code == 403
    assert client.post("/api/posts/delete", json=league(post_id=post_id), headers=me).status_code == 200
    feed = client.post("/api/feed", json=league(), headers=me).json()
    assert not [i for i in feed["items"] if i["kind"] == "post"]


def test_posting_is_rate_limited(client, espn):
    me = link(client, espn)
    codes = [client.post("/api/posts", json=league(body=f"spam {n}"), headers=me).status_code for n in range(11)]
    assert codes[:10] == [200] * 10 and codes[10] == 429


def test_a_link_only_works_in_its_league(client, espn):
    me = link(client, espn)
    r = client.post("/api/posts", json={"league_id": LEAGUE_ID + 1, "season": SEASON, "body": "hi"}, headers=me)
    assert r.status_code == 401

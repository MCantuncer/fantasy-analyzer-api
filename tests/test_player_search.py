import gzip
import json
from datetime import UTC, datetime

import httpx
from fastapi.testclient import TestClient

from app.accounts import MemoryAccountStore
from app.espn.client import EspnClient
from app.main import create_app
from app.playersearch import injury_tag, on_roster, parse_index
from app.store import MemoryStore

from .conftest import LEAGUE_ID, FakeEspn, signed_in

INDEX = [
    {"id": 1, "fullName": "Bench Guy", "proTeamId": 0, "defaultPositionId": 2, "ownership": {"percentOwned": 0.04}},
    {
        "id": 2,
        "fullName": "Nikola Jokić",
        "proTeamId": 7,
        "defaultPositionId": 5,
        "ownership": {"percentOwned": 100},
        "lastNewsDate": 1794700000000,  # 2026-11-14
    },
    {"id": 3, "fullName": "  ", "proTeamId": 1},
    {"id": 2, "fullName": "Nikola Jokić", "proTeamId": 7, "defaultPositionId": 5},
    {"fullName": "No Id"},
]


def test_parse_index_keeps_named_players_most_rostered_first():
    players = parse_index(INDEX)
    assert [p.id for p in players] == [2, 1]
    assert (players[0].name, players[0].pro_team, players[0].position) == ("Nikola Jokić", "DEN", "C")
    assert (players[1].pro_team, players[1].percent_owned) == ("", 0.0), "no NBA team; ownership rounded"


def test_on_roster_means_recent_news_or_rostered_here():
    now = datetime(2026, 11, 16, tzinfo=UTC)
    jokic, bench = parse_index(INDEX)
    assert jokic.news_at is not None and on_roster(jokic, now)
    long_gone = parse_index([{"id": 9, "fullName": "Jarnell Stokes", "proTeamId": 29}])[0]
    assert not on_roster(long_gone, now), "ESPN keeps him on his old team, with no news"
    assert on_roster(long_gone, now, owned=True), "anyone rostered in the league counts"
    stale = parse_index([{"id": 8, "fullName": "Old News", "proTeamId": 29, "lastNewsDate": 1751328000000}])[0]
    assert not on_roster(stale, now), "news from July 2025"
    assert not on_roster(bench, now, owned=True), "no NBA team, no roster"


def test_injury_tag_only_for_a_real_tag():
    assert [injury_tag(s) for s in (None, "", "ACTIVE", "NORMAL")] == [None] * 4
    assert injury_tag("OUT") == "OUT" and injury_tag("DAY_TO_DAY") == "DAY_TO_DAY"


def search_app(recorded, pro_schedule):
    fake = FakeEspn(recorded, schedule=pro_schedule)
    rostered = recorded["league"]["teams"][0]["roster"]["entries"][0]["playerPoolEntry"]["player"]
    index = [*INDEX, {"id": rostered["id"], "fullName": rostered["fullName"], "proTeamId": 1}]

    def espn(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("view") == "players_wl":
            fake.calls += 1
            return httpx.Response(200, json=index)
        return fake.handler(request)

    app = create_app(
        EspnClient(transport=httpx.MockTransport(espn)), store=MemoryStore(), accounts=MemoryAccountStore()
    )
    app.state.now = lambda: datetime(2026, 11, 16, tzinfo=UTC)
    owner = recorded["league"]["teams"][0]["id"]
    return app, fake, rostered["id"], owner


def test_players_index_endpoint(recorded, pro_schedule):
    app, fake, rostered, owner = search_app(recorded, pro_schedule)
    client = signed_in(app, team_id=None)
    body = {"league_id": LEAGUE_ID, "season": 2027}
    r = client.post("/api/players/index", json=body, headers={"Accept-Encoding": "identity"})
    assert r.status_code == 200, r.text
    assert "content-encoding" not in r.headers
    d = r.json()
    assert d["league_id"] == LEAGUE_ID and d["season"] == 2027
    by_id = {p["player_id"]: p for p in d["players"]}
    assert by_id[2]["owner_team_id"] is None and by_id[2]["pro_team"] == "DEN"
    assert by_id[rostered]["owner_team_id"] == owner, "who has him in this league"
    assert by_id[2]["on_roster"] and by_id[rostered]["on_roster"] and not by_id[1]["on_roster"]

    calls = fake.calls
    zipped = client.post("/api/players/index", json=body, headers={"Accept-Encoding": "gzip"})
    assert zipped.headers["content-encoding"] == "gzip"
    assert zipped.json() == d, "httpx unzips: the same answer, smaller on the wire"
    assert fake.calls == calls, "the index and the league are cached"


def test_players_index_needs_sign_in(recorded, pro_schedule):
    app, *_ = search_app(recorded, pro_schedule)
    r = TestClient(app).post("/api/players/index", json={"league_id": LEAGUE_ID, "season": 2027})
    assert r.status_code == 401


def test_raw_gzip_is_valid_json(recorded, pro_schedule):
    app, *_ = search_app(recorded, pro_schedule)
    client = signed_in(app, team_id=None)
    with client.stream(
        "POST", "/api/players/index", json={"league_id": LEAGUE_ID, "season": 2027}, headers={"Accept-Encoding": "gzip"}
    ) as r:
        raw = b"".join(r.iter_raw())
    assert len(json.loads(gzip.decompress(raw))["players"]) == 3

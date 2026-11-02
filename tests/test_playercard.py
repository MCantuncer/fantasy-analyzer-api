import httpx
from fastapi.testclient import TestClient

from app.accounts import MemoryAccountStore
from app.espn.client import EspnClient
from app.main import create_app
from app.news.service import iso
from app.playercard import parse_card, parse_moves
from app.scores import parse_schedule
from app.store import MemoryStore
from tests.conftest import LEAGUE_ID, signed_in

PID = 7001
NYK, BOS = 18, 2
SCHEDULE = {
    "settings": {
        "proTeams": [
            {
                "id": NYK,
                "abbrev": "NY",
                "location": "New York",
                "name": "Knicks",
                "proGamesByScoringPeriod": {
                    "1": [{"id": 11, "date": 1792800000000, "awayProTeamId": BOS, "homeProTeamId": NYK}],
                    "3": [{"id": 13, "date": 1792972800000, "awayProTeamId": NYK, "homeProTeamId": BOS}],
                    "4": [{"id": 14, "date": 1793059200000, "awayProTeamId": NYK, "homeProTeamId": BOS}],
                },
            },
            {"id": BOS, "abbrev": "BOS", "location": "Boston", "name": "Celtics", "proGamesByScoringPeriod": {}},
        ]
    }
}


def game(gid: int, day: int, minutes: float, season: int = 2027, **stats) -> dict:
    line = {"40": minutes, "0": 20, "6": 8, "3": 4, "2": 1, "1": 2, "17": 1, "11": 3, "13": 8, "14": 15}
    line.update(stats)
    return {
        "statSplitTypeId": 5,
        "statSourceId": 0,
        "externalId": str(gid),
        "seasonId": season,
        "scoringPeriodId": day,
        "proTeamId": NYK,
        "stats": line,
    }


def entry() -> dict:
    return {
        "id": PID,
        "onTeamId": 4,
        "status": "ONTEAM",
        "player": {
            "id": PID,
            "fullName": "Test Center",
            "proTeamId": NYK,
            "defaultPositionId": 5,
            "injuryStatus": "DAY_TO_DAY",
            "ownership": {"percentOwned": 64.25},
            "stats": [
                game(11, 1, 30),
                game(13, 3, 0),  # DNP: no line in the card
                game(14, 4, 34, **{"0": 31}),
                game(900, 170, 28, season=2026),
                {"statSplitTypeId": 0, "statSourceId": 1, "seasonId": 2027, "stats": {"42": 70}},  # projection
                {
                    "statSplitTypeId": 0,
                    "statSourceId": 0,
                    "seasonId": 2027,
                    "stats": {"42": 2, "13": 16, "14": 30, "15": 3, "16": 4},
                    "averageStats": {"40": 32, "0": 25.5, "6": 8},
                },
            ],
        },
        "transactions": [
            {
                "id": "d",
                "type": "DRAFT",
                "status": "EXECUTED",
                "bidAmount": 14,
                "proposedDate": 1000,
                "items": [{"type": "DRAFT", "playerId": PID, "fromTeamId": 0, "toTeamId": 2, "overallPickNumber": 30}],
            },
            {
                "id": "t",
                "type": "TRADE_ACCEPT",
                "status": "EXECUTED",
                "processDate": 3000,
                "items": [
                    {"type": "TRADE", "playerId": PID, "fromTeamId": 2, "toTeamId": 4},
                    {"type": "TRADE", "playerId": 99, "fromTeamId": 4, "toTeamId": 2},
                ],
            },
            {
                "id": "p",
                "type": "TRADE_PROPOSAL",
                "status": "EXECUTED",
                "processDate": 4000,
                "items": [{"type": "TRADE", "playerId": PID, "fromTeamId": 4, "toTeamId": 7}],
            },
            {
                "id": "w",
                "type": "WAIVER",
                "status": "FAILED_INVALIDPLAYERSOURCE",
                "bidAmount": 9,
                "processDate": 5000,
                "items": [{"type": "ADD", "playerId": PID, "fromTeamId": 0, "toTeamId": 6}],
            },
        ],
    }


def test_card_keeps_games_he_played_newest_first_with_opponents():
    card = parse_card(entry(), 2027, {2027: parse_schedule(SCHEDULE)})
    assert [g.game_id for g in card.games] == [14, 11, 900]
    last = card.games[0]
    assert last.opponent_id == BOS and last.home is False and last.stats["pts"] == 31
    assert card.games[1].home is True and card.games[1].opponent_id == BOS
    assert card.games[2].opponent_id == 0 and card.games[2].start_ms is None, "no calendar for last season"
    assert card.season_line.games == 2 and card.season_line.per_game["pts"] == 25.5
    assert round(card.season_line.fg_pct, 3) == 0.533 and card.season_line.ft_pct == 0.75
    assert card.injury_status == "DAY_TO_DAY" and card.on_team_id == 4 and card.percent_owned == 64.2


def test_moves_are_executed_ones_newest_first():
    moves = parse_moves(entry()["transactions"], PID)
    assert [(m.kind, m.via) for m in moves] == [("trade", "trade"), ("draft", "draft")]
    assert (moves[0].from_team_id, moves[0].to_team_id) == (2, 4)
    assert moves[1].pick == 30 and moves[1].bid == 14, "auction price"


def test_waiver_claim_carries_the_bid_and_a_drop_does_not():
    raw = [
        {
            "id": "c",
            "type": "WAIVER",
            "status": "EXECUTED",
            "bidAmount": 7,
            "processDate": 10,
            "items": [
                {"type": "ADD", "playerId": PID, "fromTeamId": 0, "toTeamId": 3},
                {"type": "DROP", "playerId": 5, "fromTeamId": 3, "toTeamId": 0},
            ],
        },
        {
            "id": "f",
            "type": "FREEAGENT",
            "status": "EXECUTED",
            "bidAmount": 0,
            "processDate": 20,
            "items": [{"type": "DROP", "playerId": PID, "fromTeamId": 3, "toTeamId": 0}],
        },
    ]
    moves = parse_moves(raw, PID)
    assert [(m.kind, m.via, m.bid) for m in moves] == [("drop", "free_agent", None), ("add", "waiver", 7)]
    assert moves[1].others == [(5, "drop", 0)], "the drop that made room"
    assert moves[0].others == []


def card_app(store=None, private: bool = False):
    calls = []

    def espn(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if private and "espn_s2=" not in request.headers.get("cookie", ""):
            return httpx.Response(401, json={})
        view = request.url.params.get("view")
        if view == "proTeamSchedules_wl":
            return httpx.Response(200, json=SCHEDULE if "/seasons/2027" in request.url.path else {})
        if view == "kona_playercard":
            wanted = request.headers["x-fantasy-filter"]
            return httpx.Response(200, json={"players": [entry()] if str(PID) in wanted else []})
        if view == "kona_player_info":
            return httpx.Response(200, json={"players": [{"id": 99, "player": {"id": 99, "fullName": "Other Guy"}}]})
        return httpx.Response(404, json={})

    app = create_app(
        EspnClient(transport=httpx.MockTransport(espn)), store=store or MemoryStore(), accounts=MemoryAccountStore()
    )
    return app, calls


def test_player_card_endpoint():
    store = MemoryStore()
    app, calls = card_app(store)
    now = app.state.now()
    store.save_news_items(
        [
            {
                "id": "espn:1",
                "source_id": "espn",
                "title": "Test Center (ankle) questionable Friday",
                "url": "https://example.com/1",
                "published": iso(now),
                "summary": "",
                "player_ids": [PID],
                "kind": "injury",
            }
        ]
    )
    client = signed_in(app, team_id=None)
    r = client.post("/api/player-card", json={"league_id": LEAGUE_ID, "season": 2027, "player_id": PID})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["name"] == "Test Center" and d["position"] == "C" and d["pro_team"] == "NY"
    assert d["injury_status"] == "DAY_TO_DAY" and d["owner_team_id"] == 4
    assert [g["game_id"] for g in d["games"]] == [14, 11], "only this season once he has played in it"
    assert d["games"][0]["opponent"] == "BOS" and d["games"][0]["home"] is False
    assert [m["kind"] for m in d["moves"]] == ["trade", "draft"]
    assert d["moves"][0]["others"] == [{"player_id": 99, "name": "Other Guy", "kind": "trade", "to_team_id": 2}]
    assert d["moves"][0]["at"].endswith("Z")
    assert [n["headline"] for n in d["news"]] == ["Test Center (ankle) questionable Friday"]

    n = len(calls)
    assert client.post("/api/player-card", json={"league_id": LEAGUE_ID, "season": 2027, "player_id": PID}).is_success
    assert not [c for c in calls[n:] if c.url.params.get("view") == "kona_playercard"], "cached"


def test_player_card_unknown_player_and_sign_in():
    app, _ = card_app()
    body = {"league_id": LEAGUE_ID, "season": 2027, "player_id": 1}
    assert TestClient(app).post("/api/player-card", json=body).status_code == 401
    r = signed_in(app, team_id=None).post("/api/player-card", json=body)
    assert r.status_code == 404 and r.json()["detail"]["code"] == "no_player"


def test_private_league_card_needs_cookies():
    app, calls = card_app(private=True)
    client = signed_in(app, team_id=None)
    body = {"league_id": LEAGUE_ID, "season": 2027, "player_id": PID}
    assert client.post("/api/player-card", json=body).status_code == 401
    r = client.post("/api/player-card", json=body, headers={"X-ESPN-S2": "s2", "X-ESPN-SWID": "{swid}"})
    assert r.status_code == 200, r.text

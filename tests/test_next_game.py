import copy

import pytest

from app.analysis.absence import is_out
from app.analysis.form import BLK, MINUTES, PTS, REB, TO
from app.analysis.model import RECENT_MINUTES_WEIGHTS
from app.analysis.report import NEXT_GAME_LEVEL, Analyzer


def _player(league, recent=(None, None)):
    lg = copy.deepcopy(league)
    p = lg.team(3).roster[0]
    p.per_game = {**p.per_game, MINUTES: 22.0, PTS: 11.0, REB: 4.0, BLK: 0.5, TO: 1.2}
    p.minute_shifts = ()
    p.recent_minutes = recent
    p.injury_status = "ACTIVE"
    return lg, p


def test_next_game_is_his_line_at_the_level_players_reach(league):
    lg, p = _player(league)
    ng = Analyzer(lg, 3).ref(p).next_game
    assert ng is not None
    assert ng.minutes == 22.0
    assert ng.pts == pytest.approx(round(11.0 * NEXT_GAME_LEVEL[PTS], 1))
    assert ng.reb == pytest.approx(round(4.0 * NEXT_GAME_LEVEL[REB], 1))


def test_next_game_follows_his_recent_minutes(league):
    lg, p = _player(league, recent=(33.0, 30.0))
    ng = Analyzer(lg, 3).ref(p).next_game
    minutes = 22.0 + RECENT_MINUTES_WEIGHTS[0] * 11.0 + RECENT_MINUTES_WEIGHTS[1] * 8.0
    assert ng.minutes == pytest.approx(round(minutes, 1))
    assert ng.pts == pytest.approx(round(11.0 * minutes / 22.0 * NEXT_GAME_LEVEL[PTS], 1))


def test_no_next_game_while_he_is_out(league):
    lg, p = _player(league)
    p.injury_status = "OUT"
    assert Analyzer(lg, 3).ref(p).next_game is None


def test_every_player_card_has_his_next_game(recorded, pro_schedule, league):
    # Can, 2026-11-15: "How I will check the big players' calculated stats?" Any card, not only FA Targets.
    import httpx

    from app.accounts import MemoryAccountStore
    from app.analysis.model import CategoryModel
    from app.analysis.report import next_game_line
    from app.cache import credentials_key
    from app.espn.client import EspnClient
    from app.main import create_app
    from app.store import MemoryStore
    from tests.conftest import LEAGUE_ID, FakeEspn, signed_in
    from tests.test_playercard import entry

    star = max((p for t in league.teams for p in t.roster if not is_out(p)), key=lambda p: p.per_game.get(PTS, 0))
    # The recorded league predates minutes: give everyone 30 a game.
    recorded = copy.deepcopy(recorded)
    entries = [e["playerPoolEntry"]["player"] for t in recorded["league"]["teams"] for e in t["roster"]["entries"]]
    for player in entries + [e["player"] for e in recorded["free_agents"]]:
        for block in player.get("stats") or []:
            for key in ("averageStats", "stats"):
                if block.get(key):
                    block[key].setdefault("40", 30.0 if key == "averageStats" else 30.0 * block[key].get("42", 1))
    # The card reads game ids from the NBA calendar; the recorded one has no ids or sides, so fill them in.
    schedule = copy.deepcopy(pro_schedule)
    ids = iter(range(1, 10**6))
    for team in schedule["settings"]["proTeams"]:
        for games in (team.get("proGamesByScoringPeriod") or {}).values():
            for g in games:
                g.setdefault("id", next(ids))
                g.setdefault("awayProTeamId", team["id"])
                g.setdefault("homeProTeamId", team["id"])
    fake = FakeEspn(recorded, schedule=schedule)

    def espn(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("view") == "kona_playercard":
            card = entry()
            card["id"] = card["player"]["id"] = star.id
            return httpx.Response(200, json={"players": [card]})
        return fake.handler(request)

    app = create_app(
        EspnClient(transport=httpx.MockTransport(espn)), store=MemoryStore(), accounts=MemoryAccountStore()
    )
    client = signed_in(app, team_id=None)
    got = client.post("/api/player-card/extras", json={"league_id": LEAGUE_ID, "season": 2027, "player_id": star.id})
    assert got.status_code == 200, got.text
    lg = app.state.leagues.get((LEAGUE_ID, 2027, credentials_key(None), True, True))
    me = next(p for t in lg.teams for p in t.roster if p.id == star.id)
    mates = [q for q in [*(p for t in lg.teams for p in t.roster), *lg.free_agents] if q.pro_team_id == me.pro_team_id]
    want = next_game_line(CategoryModel(lg), me, mates)
    assert want is not None and want.minutes > 20
    assert got.json()["next_game"] == want.model_dump()
    # The same line FA Targets would show him with.
    assert Analyzer(lg, 3).next_game(me) == want


def test_no_card_next_game_for_a_player_the_league_doesnt_list(client):
    from tests.conftest import LEAGUE_ID

    got = client.post("/api/player-card/extras", json={"league_id": LEAGUE_ID, "season": 2027, "player_id": 1})
    assert got.status_code in (200, 404)
    if got.status_code == 200:
        assert got.json()["next_game"] is None

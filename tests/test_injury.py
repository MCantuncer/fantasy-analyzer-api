import httpx
from fastapi.testclient import TestClient

from app.accounts import MemoryAccountStore
from app.analysis.injury import GameLine, PlayerInfo, Season, analyze, parse_gamelogs
from app.espn.client import EspnClient
from app.main import create_app
from app.store import MemoryStore
from tests.conftest import LEAGUE_ID, signed_in

TEAM = 9
X, A, B, C = 1, 2, 3, 4  # X goes out; A takes his minutes; B stays put; C is a regular who sometimes sits too
ROSTER = [PlayerInfo(X, "Star", "SF", TEAM), PlayerInfo(A, "Wing", "SF", TEAM), PlayerInfo(B, "Big", "C", TEAM)]


def line(pid: int, game: int, minutes: float, pts: float) -> GameLine:
    return GameLine(pid, game, game, TEAM, minutes, pts=pts, reb=minutes / 6, fgm=pts / 2.4, fga=pts / 2.1)


def season(c_out_too: bool = False) -> Season:
    lines = []
    for g in range(1, 31):
        out = g > 18
        if not out:
            lines.append(line(X, g, 34, 25))
        lines.append(line(A, g, 32 if out else 20, 18 if out else 10))
        lines.append(line(B, g, 28, 12))
        if not (c_out_too and out):
            lines.append(line(C, g, 30, 15))
    return Season(lines)


def test_the_teammate_who_got_the_minutes_comes_first():
    impact = analyze([season()], TEAM, X, ROSTER)
    top = impact.beneficiaries[0]
    assert top.player_id == A and top.confidence == "strong"
    assert 30 < top.minutes_after < 37 and top.gains["pts"] > 5
    big = next(b for b in impact.beneficiaries if b.player_id == B)
    assert abs(big.minutes_change) < 2.5  # only the prior moves him: the games show no change
    assert impact.minutes == 34 and impact.games_missed == 12


def test_another_absence_is_named():
    roster = [*ROSTER, PlayerInfo(C, "Guard", "PG", TEAM)]
    impact = analyze([season(c_out_too=True)], TEAM, X, roster)
    assert impact.beneficiaries[0].note.startswith("Guard was also out")


def test_with_no_games_without_him_the_prior_is_used_and_said():
    lines = [ln for g in range(1, 19) for ln in (line(X, g, 34, 25), line(A, g, 20, 10), line(B, g, 28, 12))]
    impact = analyze([Season(lines)], TEAM, X, ROSTER)
    wing = next(b for b in impact.beneficiaries if b.player_id == A)
    big = next(b for b in impact.beneficiaries if b.player_id == B)
    assert wing.confidence == "estimate" and wing.error == 0
    assert wing.minutes_change > big.minutes_change > 0  # same position and more room to grow


def test_only_games_before_a_date_are_used():
    impact = analyze([season()], TEAM, X, ROSTER, before=19)
    assert impact.games_missed == 0
    assert impact.beneficiaries[0].confidence == "estimate"


def test_parse_gamelogs_reads_game_splits_only():
    raw = [
        {
            "id": X,
            "fullName": "Star",
            "defaultPositionId": 3,
            "proTeamId": TEAM,
            "stats": [
                {
                    "statSplitTypeId": 5,
                    "externalId": "401",
                    "scoringPeriodId": 4,
                    "proTeamId": TEAM,
                    "statSourceId": 0,
                    "stats": {"40": 33, "0": 21, "13": 8, "14": 15},
                },
                {
                    "statSplitTypeId": 5,
                    "externalId": "402",
                    "scoringPeriodId": 5,
                    "proTeamId": TEAM,
                    "statSourceId": 0,
                    "stats": {"40": 0},
                },
                {"statSplitTypeId": 0, "seasonId": 2026, "statSourceId": 0, "stats": {"40": 2000}},
            ],
        }
    ]
    lines, info = parse_gamelogs(raw)
    assert [(ln.game_id, ln.minutes, ln.pts) for ln in lines] == [(401, 33, 21)]
    assert info[X].position == "SF" and info[X].team == TEAM


def raw_logs(n_games: int = 30) -> list[dict]:
    def player(pid, name, pos, minutes):
        stats = []
        for g in range(1, n_games + 1):
            m = minutes(g)
            if m:
                stats.append(
                    {
                        "statSplitTypeId": 5,
                        "externalId": str(1000 + g),
                        "scoringPeriodId": g,
                        "proTeamId": TEAM,
                        "statSourceId": 0,
                        "stats": {"40": m, "0": m * 0.6, "6": m / 5, "13": m / 4, "14": m / 2},
                    }
                )
        return {"id": pid, "fullName": name, "defaultPositionId": pos, "proTeamId": TEAM, "stats": stats}

    return [
        player(X, "Star", 3, lambda g: 34 if g <= 18 else 0),
        player(A, "Wing", 3, lambda g: 32 if g > 18 else 20),
        player(B, "Big", 5, lambda g: 28),
    ]


def test_injury_impact_endpoint(recorded, fake_espn):
    def espn(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/players"):
            if request.url.params["view"] == "kona_player_news":
                return httpx.Response(200, json=[{"id": X, "fullName": "Star", "proTeamId": TEAM}])
            return httpx.Response(200, json=raw_logs() if "/seasons/2027/" in request.url.path else [])
        return fake_espn.handler(request)

    app = create_app(
        EspnClient(transport=httpx.MockTransport(espn)), store=MemoryStore(), accounts=MemoryAccountStore()
    )
    body = signed_in(app).post("/api/injury-impact", json={"player_id": X, "league_id": LEAGUE_ID, "season": 2027})
    assert body.status_code == 200, body.text
    data = body.json()
    assert data["name"] == "Star" and data["games_missed"] == 12
    assert [b["name"] for b in data["beneficiaries"]] == ["Wing"]
    assert data["beneficiaries"][0]["confidence"] == "strong" and data["track_record"]


def test_injury_impact_needs_sign_in(recorded, fake_espn):
    app = create_app(EspnClient(transport=httpx.MockTransport(fake_espn.handler)), store=MemoryStore())
    assert TestClient(app).post("/api/injury-impact", json={"player_id": X}).status_code == 401

import gzip
import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from app.espn.client import EspnClient
from app.espn.parse import parse_league
from app.main import create_app
from app.store import MemoryStore

FIXTURE = Path(__file__).parent / "fixtures" / "chill_primera_2027.json.gz"
SCHEDULE = Path(__file__).parent / "fixtures" / "pro_schedule_2027.json.gz"
LEAGUE_ID = 744636428
MY_TEAM = 3


@pytest.fixture(scope="session")
def recorded() -> dict:
    with gzip.open(FIXTURE, "rt") as fh:
        return json.load(fh)


@pytest.fixture(scope="session")
def pro_schedule() -> dict:
    with gzip.open(SCHEDULE, "rt") as fh:
        return json.load(fh)


@pytest.fixture(scope="session")
def league(recorded, pro_schedule):
    return parse_league(recorded["league"], recorded["free_agents"], pro_schedule)


@pytest.fixture(scope="session")
def flat_league(recorded):
    """The same league without the NBA calendar: every starter plays the same games each week."""
    return parse_league(recorded["league"], recorded["free_agents"])


class FakeEspn:
    """Serves the recorded league. ``private=True`` behaves like ESPN for a private league."""

    def __init__(self, recorded: dict, private: bool = False, schedule: dict | None = None):
        self.recorded = recorded
        self.schedule = schedule
        self.private = private
        self.seen_cookies: list[str] = []
        self.calls = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        cookie = request.headers.get("cookie", "")
        self.seen_cookies.append(cookie)
        if self.private and "espn_s2=" not in cookie:
            return httpx.Response(401, json={"messages": ["not authorized"]})
        if request.url.params.get("view") == "proTeamSchedules_wl" and self.schedule is not None:
            return httpx.Response(200, json=self.schedule)
        if f"/leagues/{LEAGUE_ID}" not in request.url.path:
            return httpx.Response(404, json={})
        if request.url.params.get("view") == "kona_player_info":
            return httpx.Response(200, json={"players": self.recorded["free_agents"]})
        return httpx.Response(200, json=self.recorded["league"])


@pytest.fixture
def fake_espn(recorded, pro_schedule):
    return FakeEspn(recorded, schedule=pro_schedule)


@pytest.fixture
def client(fake_espn):
    app = create_app(EspnClient(transport=httpx.MockTransport(fake_espn.handler)), store=MemoryStore())
    return TestClient(app)

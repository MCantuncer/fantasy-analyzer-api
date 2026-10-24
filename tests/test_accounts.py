import httpx
from fastapi.testclient import TestClient

from app.espn.client import EspnClient
from app.main import create_app
from app.store import MemoryStore
from tests import test_chat
from tests.conftest import LEAGUE_ID, MY_TEAM
from tests.test_chat import SEASON, account, link

# The chat tests' recorded league and stores, shared as fixtures.
recorded, espn, stores = test_chat.recorded, test_chat.espn, test_chat.stores

PAD = "x" * 24
CLIENT_ID = "test-client.apps.googleusercontent.com"


def make_client(espn, stores, google_id=None, google_info=None, monkeypatch=None):
    social, accounts = stores
    if monkeypatch and google_id:
        monkeypatch.setenv("GOOGLE_CLIENT_ID", google_id)

    def tokeninfo(request: httpx.Request) -> httpx.Response:
        info = (google_info or {}).get(request.url.params["id_token"].split(":")[0])
        return httpx.Response(200, json=info) if info else httpx.Response(400, json={"error": "invalid_token"})

    app = create_app(
        EspnClient(transport=httpx.MockTransport(espn.handler)),
        store=MemoryStore(),
        social=social,
        accounts=accounts,
        google_transport=httpx.MockTransport(tokeninfo),
    )
    return TestClient(app)


def test_sign_up_then_sign_in(espn, stores):
    client = make_client(espn, stores)
    creds = {"email": "Can@Example.com", "password": "long enough"}
    r = client.post("/api/auth/signup", json=creds)
    assert r.status_code == 200
    token, me = r.json()["token"], r.json()["me"]
    assert me["email"] == "can@example.com" and me["leagues"] == [] and not me["has_google"]

    assert client.post("/api/auth/signup", json=creds).json()["detail"]["code"] == "email_taken"
    assert client.post("/api/auth/signup", json={**creds, "password": "short"}).status_code == 422

    r = client.post("/api/auth/login", json={"email": "CAN@example.com", "password": "long enough"})
    assert r.status_code == 200 and r.json()["token"] != token
    r = client.post("/api/auth/login", json={**creds, "password": "wrong password"})
    assert r.status_code == 401 and r.json()["detail"]["code"] == "bad_credentials"

    auth = {"Authorization": f"Bearer {token}"}
    assert client.post("/api/me", headers=auth).json()["email"] == "can@example.com"
    client.post("/api/auth/logout", headers=auth)
    assert client.post("/api/me", headers=auth).status_code == 401
    assert client.post("/api/me").status_code == 401


def test_sign_in_is_throttled(espn, stores):
    client = make_client(espn, stores)
    wrong = {"email": "nobody@example.com", "password": "guessing123"}
    codes = [client.post("/api/auth/login", json=wrong).status_code for _ in range(11)]
    assert codes[:10] == [401] * 10 and codes[10] == 429


def test_google_is_off_without_a_client_id(espn, stores, monkeypatch):
    monkeypatch.delenv("GOOGLE_CLIENT_ID", raising=False)
    client = make_client(espn, stores)
    assert client.get("/api/auth/config").json() == {"google_client_id": None}
    r = client.post("/api/auth/google", json={"credential": "x:" + PAD})
    assert r.status_code == 503 and r.json()["detail"]["code"] == "google_off"


def test_google_sign_in(espn, stores, monkeypatch):
    good = {"aud": CLIENT_ID, "iss": "https://accounts.google.com", "sub": "g-1", "email": "Can@Example.com"}
    info = {
        "new": {**good, "email_verified": "true"},
        "other-app": {**good, "aud": "someone-else", "email_verified": "true"},
        "unverified": {**good, "sub": "g-2", "email": "pw@example.com", "email_verified": "false"},
        "existing": {**good, "sub": "g-3", "email": "pw@example.com", "email_verified": "true"},
    }
    client = make_client(espn, stores, CLIENT_ID, info, monkeypatch)
    assert client.get("/api/auth/config").json() == {"google_client_id": CLIENT_ID}

    first = client.post("/api/auth/google", json={"credential": "new:" + PAD})
    assert first.status_code == 200
    assert first.json()["me"]["email"] == "can@example.com" and first.json()["me"]["has_google"]
    again = client.post("/api/auth/google", json={"credential": "new:" + PAD})
    assert again.json()["me"]["id"] == first.json()["me"]["id"]

    for bad in ("other-app", "forged"):
        r = client.post("/api/auth/google", json={"credential": bad + PAD})
        assert r.status_code == 401 and r.json()["detail"]["code"] == "bad_google_token"

    # A Google account joins a password account only when Google vouches for the email.
    pw = client.post("/api/auth/signup", json={"email": "pw@example.com", "password": "long enough"}).json()
    unverified = client.post("/api/auth/google", json={"credential": "unverified:" + PAD}).json()["me"]
    assert unverified["id"] != pw["me"]["id"] and unverified["email"] is None
    joined = client.post("/api/auth/google", json={"credential": "existing:" + PAD}).json()["me"]
    assert joined["id"] == pw["me"]["id"] and joined["has_google"]


def test_my_leagues(espn, stores):
    client = make_client(espn, stores)
    auth = account(client)
    league = {"league_id": LEAGUE_ID, "season": SEASON}
    me = client.post("/api/my-leagues/add", json=league, headers=auth).json()
    assert me["leagues"] == [{**league, "name": "Chill Primera Division", "team_id": None}]
    # Following a league shows the league, not any team in it.
    assert client.post("/api/league-overview", json=league, headers=auth).status_code == 200
    r = client.post("/api/analyze", json={**league, "team_id": MY_TEAM}, headers=auth)
    assert r.status_code == 403 and r.json()["detail"]["code"] == "not_linked"
    assert client.post("/api/my-leagues/remove", json=league, headers=auth).json()["leagues"] == []
    assert client.post("/api/my-leagues/add", json=league).status_code == 401


def test_verifying_unlocks_only_your_team(espn, stores):
    client = make_client(espn, stores)
    auth = link(client, espn)
    league = {"league_id": LEAGUE_ID, "season": SEASON}
    overview = client.post("/api/league-overview", json=league, headers=auth).json()
    assert overview["my_team_id"] == MY_TEAM
    assert [t["id"] for t in overview["teams"] if t["mine"]] == [MY_TEAM]
    r = client.post("/api/analyze", json={**league, "team_id": 8}, headers=auth)
    assert r.status_code == 403 and r.json()["detail"]["code"] == "not_your_team"

import os

import httpx
import pytest
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


class Outbox:
    """Stands in for Resend: keeps what would have been sent."""

    def __init__(self):
        self.sent = []

    async def send(self, email) -> bool:
        self.sent.append(email)
        return True

    def token(self, n=-1) -> str:
        return self.sent[n].text.split("/#reset=")[1].split()[0]


def make_client(espn, stores, google_id=None, google_info=None, monkeypatch=None, mailer=None):
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
        mailer=mailer,
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
    assert client.get("/api/auth/config").json()["google_client_id"] is None
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
    assert client.get("/api/auth/config").json()["google_client_id"] == CLIENT_ID

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
    assert client.post("/api/league-overview", json=league, headers=auth).status_code == 200
    # A team nobody verified is open to an account that added the league; the verified seat is not.
    assert client.post("/api/matchup-projection", json={**league, "team_id": MY_TEAM}, headers=auth).status_code == 200
    r = client.post("/api/matchup-projection", json=league, headers=auth)
    assert r.status_code == 403 and r.json()["detail"]["code"] == "not_linked"
    assert client.post("/api/my-leagues/remove", json=league, headers=auth).json()["leagues"] == []
    r = client.post("/api/analyze", json={**league, "team_id": MY_TEAM}, headers=auth)
    assert r.status_code == 403 and r.json()["detail"]["code"] == "not_following"
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


def test_new_accounts_are_plain_users():
    from app.accounts import MemoryAccountStore

    user = MemoryAccountStore().create_user("someone@example.com", None)
    assert user.user_type == "user" and not user.admin


@pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="set TEST_DATABASE_URL to test against Postgres")
def test_admin_is_set_in_the_database():
    from app.accounts import PostgresAccountStore, hash_password
    from app.social import hash_token

    accounts = PostgresAccountStore(os.environ["TEST_DATABASE_URL"])
    with accounts._connect() as conn:
        conn.execute("truncate users cascade")
    user = accounts.create_user("boss@example.com", hash_password("longenough"))
    assert user.user_type == "user" and not user.admin
    with accounts._connect() as conn:
        conn.execute("update users set user_type = 'admin' where email = %s", ("boss@example.com",))
    accounts.save_session(hash_token("t"), user.id)
    assert accounts.session_user(hash_token("t")).admin
    found, password_hash = accounts.login_lookup("boss@example.com")
    assert found.admin and password_hash.startswith("scrypt$")
    PostgresAccountStore(os.environ["TEST_DATABASE_URL"])  # the column is added once; a restart is fine


def test_keepers_follow_the_account(espn, stores):
    client = make_client(espn, stores)
    auth = link(client, espn)
    team = {"league_id": LEAGUE_ID, "season": SEASON, "team_id": MY_TEAM}
    assert client.post("/api/keepers", json=team, headers=auth).json() == {"player_ids": [], "saved": False}

    r = client.post("/api/keepers/save", json={**team, "player_ids": [11, 22, 11]}, headers=auth)
    assert r.status_code == 200 and r.json() == {"player_ids": [11, 22], "saved": True}
    assert client.post("/api/keepers", json=team, headers=auth).json() == {"player_ids": [11, 22], "saved": True}
    _, accounts = stores
    assert accounts.league_keepers(LEAGUE_ID, SEASON) == {MY_TEAM: [11, 22]}

    # An empty list is a choice too: it is saved, and the league-wide scan sees no keepers.
    assert client.post("/api/keepers/save", json={**team, "player_ids": []}, headers=auth).json()["saved"]
    assert client.post("/api/keepers", json=team, headers=auth).json() == {"player_ids": [], "saved": True}
    assert accounts.league_keepers(LEAGUE_ID, SEASON) == {}


def test_keepers_are_for_the_teams_own_manager(espn, stores):
    client = make_client(espn, stores)
    owner = link(client, espn)
    team = {"league_id": LEAGUE_ID, "season": SEASON, "team_id": MY_TEAM}
    client.post("/api/keepers/save", json={**team, "player_ids": [11]}, headers=owner)

    assert client.post("/api/keepers", json=team).status_code == 401
    other = account(client)
    client.post("/api/my-leagues/add", json={"league_id": LEAGUE_ID, "season": SEASON}, headers=other)
    assert client.post("/api/keepers", json=team, headers=other).status_code == 403
    assert client.post("/api/keepers/save", json={**team, "player_ids": [99]}, headers=other).status_code == 403
    assert client.post("/api/keepers", json=team, headers=owner).json()["player_ids"] == [11]


def test_keepers_saved_before_verifying_count_once_the_team_is_verified(espn, stores):
    client = make_client(espn, stores)
    auth = account(client)
    client.post("/api/my-leagues/add", json={"league_id": LEAGUE_ID, "season": SEASON}, headers=auth)
    team = {"league_id": LEAGUE_ID, "season": SEASON, "team_id": MY_TEAM}
    assert client.post("/api/keepers/save", json={**team, "player_ids": [7]}, headers=auth).status_code == 200
    _, accounts = stores
    assert accounts.league_keepers(LEAGUE_ID, SEASON) == {}  # nobody has proved the team is theirs yet
    link(client, espn, auth=auth)
    assert accounts.league_keepers(LEAGUE_ID, SEASON) == {MY_TEAM: [7]}


def test_password_reset_is_off_without_an_email_service(espn, stores, monkeypatch):
    monkeypatch.delenv("RESEND_API_KEY", raising=False)
    client = make_client(espn, stores)
    assert client.get("/api/auth/config").json()["password_reset"] is False
    r = client.post("/api/auth/forgot", json={"email": "can@example.com"})
    assert r.status_code == 503 and r.json()["detail"]["code"] == "reset_off"


def test_forgot_password_resets_once_and_signs_out_everywhere(espn, stores):
    outbox = Outbox()
    client = make_client(espn, stores, mailer=outbox)
    assert client.get("/api/auth/config").json()["password_reset"] is True
    creds = {"email": "reset@example.com", "password": "old password"}
    old = {"Authorization": f"Bearer {client.post('/api/auth/signup', json=creds).json()['token']}"}

    # Known and unknown addresses get the same answer; only the account's address gets mail.
    known = client.post("/api/auth/forgot", json={"email": "Reset@Example.com"})
    unknown = client.post("/api/auth/forgot", json={"email": "nobody@example.com"})
    assert known.status_code == unknown.status_code == 202 and known.json() == unknown.json()
    assert [m.to for m in outbox.sent] == ["reset@example.com"]
    email = outbox.sent[0]
    assert email.subject == "Reset your COURT password" and "https://courtfantasy.app/#reset=" in email.text
    assert outbox.token() in email.html
    first = outbox.token()

    # A newer link replaces the older one.
    client.post("/api/auth/forgot", json={"email": "reset@example.com"})
    r = client.post("/api/auth/reset", json={"token": first, "password": "new password"})
    assert r.status_code == 400 and r.json()["detail"]["code"] == "bad_reset_link"

    r = client.post("/api/auth/reset", json={"token": outbox.token(), "password": "new password"})
    assert r.status_code == 200 and r.json()["me"]["email"] == "reset@example.com"
    assert client.post("/api/me", headers=old).status_code == 401  # every old session is gone
    fresh = {"Authorization": f"Bearer {r.json()['token']}"}
    assert client.post("/api/me", headers=fresh).status_code == 200
    assert client.post("/api/auth/login", json=creds).status_code == 401
    assert client.post("/api/auth/login", json={**creds, "password": "new password"}).status_code == 200

    # The link works once.
    again = client.post("/api/auth/reset", json={"token": outbox.token(), "password": "third password"})
    assert again.status_code == 400
    assert client.post("/api/auth/reset", json={"token": "x" * 43, "password": "new password"}).status_code == 400
    short = client.post("/api/auth/reset", json={"token": outbox.token(), "password": "short"})
    assert short.status_code == 422


def test_reset_links_expire(espn, stores):
    from datetime import UTC, datetime, timedelta

    outbox = Outbox()
    client = make_client(espn, stores, mailer=outbox)
    client.post("/api/auth/signup", json={"email": "late@example.com", "password": "old password"})
    client.app.state.now = lambda: datetime.now(UTC) - timedelta(minutes=31)
    client.post("/api/auth/forgot", json={"email": "late@example.com"})
    r = client.post("/api/auth/reset", json={"token": outbox.token(), "password": "new password"})
    assert r.status_code == 400 and r.json()["detail"]["code"] == "bad_reset_link"


def test_reset_requests_are_rate_limited(espn, stores):
    outbox = Outbox()
    client = make_client(espn, stores, mailer=outbox)
    client.post("/api/auth/signup", json={"email": "busy@example.com", "password": "old password"})
    codes = [client.post("/api/auth/forgot", json={"email": "busy@example.com"}).status_code for _ in range(4)]
    assert codes == [202, 202, 202, 429] and len(outbox.sent) == 3
    # One visitor can't walk through many addresses either.
    codes = [client.post("/api/auth/forgot", json={"email": f"a{i}@example.com"}).status_code for i in range(8)]
    assert codes[-1] == 429


def test_a_failed_send_still_answers_the_same(espn, stores):
    class Down:
        async def send(self, email) -> bool:
            raise RuntimeError("email service down")

    client = make_client(espn, stores, mailer=Down())
    client.post("/api/auth/signup", json={"email": "down@example.com", "password": "old password"})
    assert client.post("/api/auth/forgot", json={"email": "down@example.com"}).status_code == 202


def test_google_turns_off_a_password_nobody_proved(espn, stores, monkeypatch):
    victim = {"aud": CLIENT_ID, "iss": "https://accounts.google.com", "sub": "g-v", "email": "victim@example.com"}
    victim["email_verified"] = "true"
    outbox = Outbox()
    client = make_client(espn, stores, CLIENT_ID, {"victim": victim}, monkeypatch, mailer=outbox)
    # Someone signs up on another person's address before that person ever came.
    creds = {"email": "victim@example.com", "password": "squatter pw"}
    squatter = {"Authorization": f"Bearer {client.post('/api/auth/signup', json=creds).json()['token']}"}

    me = client.post("/api/auth/google", json={"credential": "victim:" + PAD}).json()["me"]
    assert me["has_google"] and me["notices"][0]["text"].startswith("You signed in with Google")
    assert client.post("/api/me", headers=squatter).status_code == 401
    assert client.post("/api/auth/login", json=creds).status_code == 401


def test_google_keeps_a_proven_password(espn, stores, monkeypatch):
    info = {"aud": CLIENT_ID, "iss": "https://accounts.google.com", "sub": "g-p", "email": "proven@example.com"}
    info["email_verified"] = "true"
    outbox = Outbox()
    client = make_client(espn, stores, CLIENT_ID, {"proven": info}, monkeypatch, mailer=outbox)
    client.post("/api/auth/signup", json={"email": "proven@example.com", "password": "first password"})
    client.post("/api/auth/forgot", json={"email": "proven@example.com"})
    client.post("/api/auth/reset", json={"token": outbox.token(), "password": "proven password"})

    me = client.post("/api/auth/google", json={"credential": "proven:" + PAD}).json()["me"]
    assert me["has_google"] and me["notices"] == []
    creds = {"email": "proven@example.com", "password": "proven password"}
    assert client.post("/api/auth/login", json=creds).status_code == 200


def test_reset_email_escapes_the_link():
    from app.mail import reset_email

    email = reset_email("a@example.com", 'https://x.test/reset-password#token=a"b<c', 30)
    assert 'a"b<c' not in email.html and "a&quot;b&lt;c" in email.html and "30 minutes" in email.text


def test_resend_mailer_sends_the_message():
    import asyncio

    from app.mail import RESEND_URL, ResendMailer, reset_email

    seen = []

    def resend(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"id": "e-1"})

    mailer = ResendMailer("re_test", "COURT <no-reply@courtfantasy.app>", httpx.MockTransport(resend))
    assert asyncio.run(mailer.send(reset_email("a@example.com", "https://x.test/r#token=t", 30)))
    assert str(seen[0].url) == RESEND_URL and seen[0].headers["authorization"] == "Bearer re_test"
    body = seen[0].read().decode()
    assert '"from":"COURT <no-reply@courtfantasy.app>"' in body.replace(": ", ":") and "a@example.com" in body

    refused = ResendMailer("bad", "x@y.z", httpx.MockTransport(lambda r: httpx.Response(403, json={})))
    assert not asyncio.run(refused.send(reset_email("a@example.com", "https://x.test/r", 30)))

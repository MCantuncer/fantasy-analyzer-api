import os
import typing
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from app import plans, schemas
from app.accounts import MemoryAccountStore, User
from app.espn.client import EspnClient
from app.main import create_app
from app.plans import FREE, PRO, Feature, Subscription, entitlement
from app.social import hash_token
from app.store import MemoryStore

NOW = datetime(2026, 11, 22, 21, 0, tzinfo=UTC)
USER = User(1, "fan@example.com", NOW)
ADMIN = User(2, "boss@example.com", NOW, user_type="admin")


def sub(status, days=10, **kw) -> Subscription:
    return Subscription(1, status=status, provider="paddle", period_end=NOW + timedelta(days=days), **kw)


def test_free_without_a_subscription():
    got = entitlement(USER, None, NOW)
    assert got.plan == FREE and got.features == frozenset() and not got.allows(Feature.NEXT_GAME)


def test_pro_holds_every_feature():
    got = entitlement(USER, sub("active"), NOW)
    assert got.plan == PRO and got.features == frozenset(Feature)
    assert got.renews_at == NOW + timedelta(days=10) and got.ends_at is None and got.trial_ends is None


def test_trial_and_cancelled_at_period_end():
    trial = entitlement(USER, sub("trialing", days=7), NOW)
    assert trial.plan == PRO and trial.trial_ends == NOW + timedelta(days=7)
    leaving = entitlement(USER, sub("active", cancel_at_period_end=True), NOW)
    assert leaving.plan == PRO and leaving.ends_at == NOW + timedelta(days=10) and leaving.renews_at is None


def test_grace_after_the_period_then_free():
    assert entitlement(USER, sub("past_due", days=-2), NOW).plan == PRO  # the provider still retries the card
    assert entitlement(USER, sub("past_due", days=-4), NOW).plan == FREE
    assert entitlement(USER, sub("active", days=-4), NOW).plan == FREE  # a missed webhook can't keep Pro forever


@pytest.mark.parametrize("status", ["canceled", "paused", "expired"])
def test_ended_subscriptions_are_free(status):
    assert entitlement(USER, sub(status), NOW).plan == FREE


def test_gifts_and_admins():
    gift = Subscription(1, comp_until=NOW + timedelta(days=30))
    got = entitlement(USER, gift, NOW)
    assert got.plan == PRO and got.gifted and got.ends_at == NOW + timedelta(days=30)
    assert entitlement(USER, Subscription(1, comp_until=NOW - timedelta(seconds=1)), NOW).plan == FREE
    paying = entitlement(USER, sub("active", comp_until=NOW + timedelta(days=30)), NOW)
    assert paying.plan == PRO and not paying.gifted  # the paid plan is what shows
    assert entitlement(ADMIN, None, NOW).features == frozenset(Feature)


def test_every_feature_is_open_until_plans_are_enforced():
    got = entitlement(USER, None, NOW, enforced=False)
    assert got.plan == FREE and got.features == frozenset(Feature)


def test_schema_lists_every_feature():
    assert set(typing.get_args(schemas.PlanFeature)) == {f.value for f in Feature}


def test_memory_store_keeps_a_gift_through_webhook_writes():
    store = plans.MemorySubscriptionStore()
    until = NOW + timedelta(days=5)
    store.gift(1, until)
    store.save(Subscription(1, status="active", period_end=NOW))
    got = store.get(1)
    assert got.status == "active" and got.comp_until == until


# The API --------------------------------------------------------------------------------------


@pytest.fixture
def app(fake_espn):
    return create_app(
        EspnClient(transport=httpx.MockTransport(fake_espn.handler)),
        store=MemoryStore(),
        accounts=MemoryAccountStore(),
        subscriptions=plans.MemorySubscriptionStore(),
    )


def client_for(app, email, admin=False, raise_server_exceptions=True) -> TestClient:
    user = app.state.accounts.create_user(email, None)
    user.user_type = "admin" if admin else "user"
    app.state.accounts.save_session(hash_token(email), user.id)
    headers = {"Authorization": f"Bearer {email}"}
    return TestClient(app, headers=headers, raise_server_exceptions=raise_server_exceptions)


def me(client) -> dict:
    r = client.post("/api/me")
    assert r.status_code == 200, r.text
    return r.json()


def test_me_before_launch_opens_everything(app):
    app.state.plans_enforced = False
    got = me(client_for(app, "fan@example.com"))
    assert got["plan"] == "free" and set(got["features"]) == {f.value for f in Feature}


def test_me_once_enforced(app):
    app.state.plans_enforced = True
    fan = client_for(app, "fan@example.com")
    assert me(fan)["plan"] == "free" and me(fan)["features"] == []
    boss = client_for(app, "boss@example.com", admin=True)
    assert me(boss)["plan"] == "pro" and len(me(boss)["features"]) == len(Feature)


def test_admin_gifts_pro(app):
    app.state.plans_enforced = True
    fan = client_for(app, "fan@example.com")
    assert me(fan)["plan"] == "free"  # now cached for half a minute: a gift must clear it
    boss = client_for(app, "boss@example.com", admin=True)
    r = boss.post("/api/admin/gift-pro", json={"email": " Fan@Example.com ", "days": 30})
    assert r.status_code == 200, r.text
    got = me(fan)
    assert got["plan"] == "pro" and got["gifted"] and got["ends_at"] == r.json()["until"]
    assert boss.post("/api/admin/gift-pro", json={"email": "fan@example.com", "days": 0}).json()["until"] is None
    assert me(fan)["plan"] == "free"


def test_only_admins_gift(app):
    fan = client_for(app, "fan@example.com")
    assert fan.post("/api/admin/gift-pro", json={"email": "fan@example.com", "days": 30}).status_code == 403
    assert TestClient(app).post("/api/admin/gift-pro", json={"email": "a@b.co", "days": 1}).status_code == 401
    boss = client_for(app, "boss@example.com", admin=True)
    assert boss.post("/api/admin/gift-pro", json={"email": "nobody@example.com", "days": 1}).status_code == 404


@pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="set TEST_DATABASE_URL to test against Postgres")
def test_postgres_store():
    from app.accounts import PostgresAccountStore

    accounts = PostgresAccountStore(os.environ["TEST_DATABASE_URL"])
    with accounts._connect() as conn:
        conn.execute("truncate users cascade")
    store = plans.PostgresSubscriptionStore(os.environ["TEST_DATABASE_URL"])
    user = accounts.create_user("fan@example.com", None)
    assert store.get(user.id) is None
    store.gift(user.id, NOW)
    store.save(Subscription(user.id, "active", "paddle", "ctm_1", "sub_1", NOW, True))
    got = store.get(user.id)
    assert (got.status, got.subscription_id, got.period_end, got.cancel_at_period_end) == ("active", "sub_1", NOW, True)
    assert got.comp_until == NOW  # a webhook write leaves the gift alone
    plans.PostgresSubscriptionStore(os.environ["TEST_DATABASE_URL"])  # the table is made once; a restart is fine


# Pro fields and Pro endpoints -------------------------------------------------------------------

LINE = schemas.NextGameLine(
    minutes=34, pts=27, reb=6, ast=7, stl=1, blk=0.5, three_pm=3, to=3, fgm=9, fga=19, ftm=6, fta=7
)


def card_extras() -> schemas.PlayerCardExtras:
    return schemas.PlayerCardExtras(news=[], names={1: "Luka"}, next_game=LINE)


def test_strip_blanks_pro_fields_on_a_copy():
    extras = card_extras()
    free = plans.strip(extras, frozenset())
    assert free.next_game is None and free.names == {1: "Luka"}
    assert extras.next_game == LINE  # a shared, cached original is never changed
    assert plans.strip(extras, frozenset(Feature)) is extras
    assert plans.strip(extras, frozenset({Feature.NEXT_GAME})) is extras
    nested = [{"a": extras}]
    assert plans.strip(nested, frozenset())[0]["a"].next_game is None and nested[0]["a"].next_game == LINE


def test_models_without_pro_fields_pass_untouched():
    other = schemas.Me(id=1, email=None, has_google=False, leagues=[], notices=[])
    assert not plans.reaches_pro(schemas.Me) and plans.strip(other, frozenset()) is other


def all_models():
    return [m for m in vars(schemas).values() if isinstance(m, type) and issubclass(m, BaseModel)]


def test_pro_fields_can_be_blanked():
    marked = [
        (m.__name__, name)
        for m in all_models()
        for name, info in m.model_fields.items()
        if plans.field_feature(info) is not None
    ]
    assert ("PlayerRef", "next_game") in marked and ("Game", "call") in marked
    for model, name in marked:
        assert not getattr(schemas, model).model_fields[name].is_required(), f"{model}.{name} needs a default"


def test_every_route_with_pro_fields_strips_them(app):
    from fastapi.routing import APIRoute

    reaching = [
        r
        for r in app.routes
        if isinstance(r, APIRoute)
        and r.response_model is not None
        and any(plans.reaches_pro(m) for m in plans._models_in(r.response_model))
    ]
    paths = {r.path for r in reaching}
    assert {"/api/analyze", "/api/scores", "/api/player-card/extras", "/api/sample/teams/{team_id}"} <= paths
    assert all(hasattr(r.endpoint, "__wrapped__") for r in reaching)


@pytest.fixture
def card_route(app):
    @app.post("/test/card", response_model=schemas.PlayerCardExtras)
    async def card() -> schemas.PlayerCardExtras:
        return app.state.test_card

    app.state.test_card = card_extras()
    return app


def test_free_accounts_get_no_pro_fields(card_route):
    app = card_route
    app.state.plans_enforced = True
    assert client_for(app, "fan@example.com").post("/test/card").json()["next_game"] is None
    assert TestClient(app).post("/test/card").json()["next_game"] is None  # a visitor too
    assert client_for(app, "boss@example.com", admin=True).post("/test/card").json()["next_game"]["pts"] == 27
    assert app.state.test_card.next_game == LINE


def test_pro_fields_stay_while_plans_are_off(card_route):
    app = card_route
    app.state.plans_enforced = False
    assert client_for(app, "fan@example.com").post("/test/card").json()["next_game"]["pts"] == 27


@pytest.mark.parametrize(
    "path,body,feature",
    [
        ("/api/book-compare", {}, "estimates"),
        ("/api/book-compare/record", {}, "sportsbook"),
        ("/api/book-compare/lost", {}, "estimates"),
        ("/api/scores/pregame", {"league_id": 1, "game_id": 1}, "pregame"),
        ("/api/match-form", {"away": "BOS", "home": "NYK"}, "match_form"),
    ],
)
def test_pro_endpoints_ask_free_accounts_to_upgrade(app, path, body, feature):
    app.state.plans_enforced = True
    r = client_for(app, "fan@example.com").post(path, json=body)
    assert r.status_code == 402, r.text
    assert r.json()["detail"]["code"] == "upgrade_required" and r.json()["detail"]["feature"] == feature
    assert TestClient(app).post(path, json=body).status_code == 401
    boss = client_for(app, "boss@example.com", admin=True, raise_server_exceptions=False)
    assert boss.post(path, json=body).status_code != 402  # past the gate (the test league has no such game)

import os

import pytest
from fastapi.testclient import TestClient

from app.rules import MAX_RULES, PostgresRuleStore, Rule
from tests.conftest import LEAGUE_ID, MY_TEAM, signed_in

SEAT = {"league_id": LEAGUE_ID, "season": 2027, "team_id": MY_TEAM}


def mine(client):
    r = client.post("/api/trade-rules/mine", json=SEAT)
    assert r.status_code == 200, r.text
    return r.json()["rules"]


def test_rules_are_added_listed_and_removed(client: TestClient):
    r = client.post("/api/trade-rules", json=dict(SEAT, kind="protect_category", value=" TO ", why="From feedback."))
    assert r.status_code == 200, r.text
    assert r.json()["label"] == "Protect TO" and r.json()["why"] == "From feedback."
    skip = client.post("/api/trade-rules", json=dict(SEAT, kind="skip_team", value="7", team_name="Los Avokados"))
    assert skip.json()["label"] == "Skip Los Avokados"
    hurt = client.post("/api/trade-rules", json=dict(SEAT, kind="no_injury_prone", value="ignored"))
    assert hurt.json()["value"] == ""
    again = client.post("/api/trade-rules", json=dict(SEAT, kind="protect_category", value="TO"))
    assert again.json()["id"] == r.json()["id"]  # the same rule, not a second one
    assert [x["label"] for x in mine(client)] == ["Protect TO", "Skip Los Avokados", "No injury-prone players in"]

    left = client.post("/api/trade-rules/delete", json=dict(SEAT, rule_id=skip.json()["id"]))
    assert [x["kind"] for x in left.json()["rules"]] == ["protect_category", "no_injury_prone"]


def test_bad_rules_are_refused(client: TestClient):
    assert client.post("/api/trade-rules", json=dict(SEAT, kind="protect_category")).status_code == 400
    assert client.post("/api/trade-rules", json=dict(SEAT, kind="skip_team", value="abc")).status_code == 400
    assert client.post("/api/trade-rules", json=dict(SEAT, kind="skip_team", value=str(MY_TEAM))).status_code == 400
    assert client.post("/api/trade-rules", json=dict(SEAT, kind="made_up")).status_code == 422
    for i in range(MAX_RULES):
        client.post("/api/trade-rules", json=dict(SEAT, kind="protect_category", value=f"C{i}"))
    full = client.post("/api/trade-rules", json=dict(SEAT, kind="protect_category", value="ONE MORE"))
    assert full.status_code == 400 and full.json()["detail"]["code"] == "too_many_rules"


def test_rules_belong_to_the_verified_manager(client: TestClient):
    client.post("/api/trade-rules", json=dict(SEAT, kind="no_injury_prone"))
    other = signed_in(client.app, team_id=MY_TEAM + 1)
    assert other.post("/api/trade-rules/mine", json=dict(SEAT, team_id=MY_TEAM + 1)).json()["rules"] == []
    rule_id = mine(client)[0]["id"]
    other.post("/api/trade-rules/delete", json=dict(SEAT, team_id=MY_TEAM + 1, rule_id=rule_id))
    assert len(mine(client)) == 1  # someone else can't remove it
    assert client.post("/api/trade-rules/mine", json=dict(SEAT, team_id=MY_TEAM + 1)).status_code == 403
    assert TestClient(client.app).post("/api/trade-rules/mine", json=SEAT).status_code == 401


@pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="set TEST_DATABASE_URL to test against Postgres")
def test_postgres_rule_store():
    store = PostgresRuleStore(os.environ["TEST_DATABASE_URL"])
    with store._connect() as conn:
        conn.execute("delete from trade_rules where league_id = 999")
    rule = lambda value: Rule(0, 1, 999, 2027, "protect_category", value, f"Protect {value}", "", None)  # noqa: E731
    first = store.add(rule("TO"))
    assert store.add(rule("TO")).id == first.id
    store.add(rule("FG%"))
    assert [r.value for r in store.mine(1, 999, 2027)] == ["TO", "FG%"]
    assert not store.remove(2, first.id) and store.remove(1, first.id)
    assert [r.value for r in store.mine(1, 999, 2027)] == ["FG%"]

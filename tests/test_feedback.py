import copy
import os

import pytest
from fastapi.testclient import TestClient

from app.feedback import Feedback, MemoryFeedbackStore, PostgresFeedbackStore, deal_key, idea_key
from tests.conftest import LEAGUE_ID, MY_TEAM, signed_in


def player(pid: int, name: str) -> dict:
    return {
        "id": pid,
        "name": name,
        "positions": ["F"],
        "injury_status": "ACTIVE",
        "percent_owned": 50.0,
        "value": 1.0,
        "perceived_value": 1.0,
        "tier": "tradeable",
    }


IDEA = {
    "partner_team_id": 7,
    "partner_team_name": "Antalya Apaches",
    "give": [player(1, "Christian Braun"), player(2, "Isaiah Jackson")],
    "get": [player(3, "Brook Lopez")],
    "my_add": player(4, "Collin Gillespie"),
    "my_cats_gain": 0.21,
    "their_cats_gain": -0.07,
    "edge": 0.28,
    "value_they_get": 2.3,
    "value_they_give": 1.9,
    "acceptance": 0.64,
    "acceptance_label": "likely",
    "pitch": "Helps their FG%.",
    "reasons": ["They aren't one of your direct playoff rivals."],
    "playoff_pct_before": 0.39,
    "playoff_pct_after": 0.54,
    "category_changes": {"FG%": -0.08, "3PM": 0.14},
}


def body(**over) -> dict:
    out = {
        "league_id": LEAGUE_ID,
        "season": 2027,
        "team_id": MY_TEAM,
        "source": "edge",
        "rating": "bad",
        "reasons": ["fake_gain"],
        "text": "  I can just pick up Gillespie  ",
        "idea": copy.deepcopy(IDEA),
    }
    out.update(over)
    return out


def test_feedback_is_saved_with_the_idea_as_shown(client: TestClient):
    r = client.post("/api/trade-feedback", json=body())
    assert r.status_code == 200, r.text
    saved = r.json()
    assert saved["rating"] == "bad" and saved["reasons"] == ["fake_gain"]
    assert saved["text"] == "I can just pick up Gillespie"
    assert saved["idea"]["playoff_pct_after"] == 0.54
    assert saved["idea"]["my_add"]["name"] == "Collin Gillespie"
    stored = next(iter(client.app.state.feedback.rows.values()))
    assert stored.model_version == "dev"
    assert stored.idea_key == "edge:7:1,2>3"

    mine = client.post("/api/trade-feedback/mine", json={"league_id": LEAGUE_ID, "season": 2027, "team_id": MY_TEAM})
    assert mine.status_code == 200
    assert [f["id"] for f in mine.json()["feedback"]] == [saved["id"]]


def test_feedback_on_the_same_deal_replaces_the_earlier_entry(client: TestClient):
    first = client.post("/api/trade-feedback", json=body()).json()
    again = body(rating="good", reasons=[], text="")
    again["idea"]["give"].reverse()  # same deal, listed in another order
    second = client.post("/api/trade-feedback", json=again).json()
    assert second["id"] == first["id"] and second["rating"] == "good"
    other = client.post("/api/trade-feedback", json=body(source="trades")).json()
    assert other["id"] != first["id"]
    mine = client.post("/api/trade-feedback/mine", json={"league_id": LEAGUE_ID, "season": 2027, "team_id": MY_TEAM})
    assert len(mine.json()["feedback"]) == 2


def test_feedback_is_only_for_the_verified_manager(client: TestClient):
    stranger = signed_in(client.app, team_id=None)
    assert stranger.post("/api/trade-feedback", json=body()).status_code == 403
    assert client.post("/api/trade-feedback", json=body(team_id=MY_TEAM + 1)).status_code == 403
    anonymous = TestClient(client.app)
    assert anonymous.post("/api/trade-feedback", json=body()).status_code == 401
    assert not client.app.state.feedback.rows


def test_feedback_input_is_bounded(client: TestClient):
    assert client.post("/api/trade-feedback", json=body(text="x" * 501)).status_code == 422
    assert client.post("/api/trade-feedback", json=body(reasons=["made_up"])).status_code == 422
    assert client.post("/api/trade-feedback", json=body(rating="meh")).status_code == 422


def test_feedback_from_one_manager_is_not_shown_to_another(client: TestClient):
    client.post("/api/trade-feedback", json=body())
    other = signed_in(client.app, team_id=MY_TEAM + 1)
    mine = other.post("/api/trade-feedback/mine", json={"league_id": LEAGUE_ID, "season": 2027, "team_id": MY_TEAM + 1})
    assert mine.json()["feedback"] == []


def test_idea_key_ignores_player_order():
    assert idea_key("trades", 3, [5, 2], [9]) == idea_key("trades", 3, [2, 5], [9])
    assert idea_key("trades", 3, [2], [9]) != idea_key("shop", 3, [2], [9])
    assert isinstance(MemoryFeedbackStore().mine(1, LEAGUE_ID, 2027), list)


@pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="set TEST_DATABASE_URL to test against Postgres")
def test_postgres_feedback_store():
    store = PostgresFeedbackStore(os.environ["TEST_DATABASE_URL"])
    with store._connect() as conn:
        conn.execute("delete from trade_feedback where league_id = 999")
    entry = lambda source, rating: Feedback(  # noqa: E731
        0, 1, 999, 2027, 3, source, idea_key(source, 7, [1, 2], [3]), rating, [], "", IDEA, "dev", None
    )
    first = store.save(entry("edge", "bad"))
    store.save(entry("trades", "bad"))
    assert store.record_checks(1, 999, 2027, deal_key(7, [2, 1], [3]), [{"check": "noise"}]) == 2
    assert all(f.checks == [{"check": "noise"}] and f.checked_at for f in store.mine(1, 999, 2027))
    store.record_reading(first.id, {"kind": "rant"})
    assert [f.reading for f in store.for_deal(1, 999, 2027, deal_key(7, [1, 2], [3])) if f.id == first.id] == [
        {"kind": "rant"}
    ]
    again = store.save(entry("edge", "good"))
    assert again.id == first.id and again.checks is None and again.reading is None and again.rating == "good"

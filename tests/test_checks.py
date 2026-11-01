import pytest
from fastapi.testclient import TestClient

from app import schemas
from app.analysis.checks import ORDER, IdeaChecker, StaleIdea, focus_order
from app.analysis.report import Analyzer
from app.analysis.trades import TradeFinder
from tests.conftest import LEAGUE_ID, MY_TEAM, signed_in

LOOSE = {"min_my_gain": float("-inf"), "min_their_gain": float("-inf"), "min_ratio": 0.0}


def two_for_one(league) -> tuple[Analyzer, schemas.TradeIdea]:
    """A 2-for-1 whose open spot takes the best free agent, shown as a trade idea."""
    a = Analyzer(league, MY_TEAM)
    finder = TradeFinder(league, a.model, a.idx, a.keep, premium=a.premium)
    j = (a.idx + 1) % len(a.teams)
    mine = sorted(a.me.roster, key=a.model.value)
    for get in a.teams[j].roster:
        d = finder._evaluate(j, mine[:2], [get], **LOOSE)
        if d is not None and d.my_add is not None:
            before, cats = a.baseline(d)
            return a, a.trade_idea(finder, d, before, a.playoff_pct(d.dists, d.rosters), cats)
    raise AssertionError("no 2-for-1 in the fixture")


def test_focus_order_puts_the_reasons_checks_first_and_keeps_every_check():
    order = focus_order(["player_concern"])
    assert order[:3] == ["injury", "small_sample", "role_change"]
    assert sorted(order) == sorted(ORDER)
    assert focus_order([]) == ORDER


def test_every_check_returns_a_verdict_with_evidence(league):
    _, idea = two_for_one(league)
    c = IdeaChecker(league, MY_TEAM, idea)
    verdicts = c.run(focus_order(["fake_gain"]))
    assert [v.check for v in verdicts][:3] == ["free_pickup", "noise", "lateral_swap"]
    assert all(v.evidence for v in verdicts)
    built = {"free_pickup", "noise", "lateral_swap", "build_conflict", "small_sample", "injury"}
    assert all((v.status == "cant_check") == (v.check not in built) for v in verdicts)


def test_free_pickup_check_measures_the_pickup_alone(league):
    a, idea = two_for_one(league)
    c = IdeaChecker(league, MY_TEAM, idea)
    status, evidence = c.free_pickup()
    assert idea.my_add.name in evidence
    assert c.after == pytest.approx(idea.playoff_pct_after, abs=1e-3)
    no_add = idea.model_copy(update={"my_add": None})
    assert IdeaChecker(league, MY_TEAM, no_add).free_pickup()[0] == "not_supported"


def test_a_deal_whose_players_moved_is_stale(league):
    _, idea = two_for_one(league)
    moved = idea.model_copy(update={"give": [idea.get[0]]})  # not on my roster
    with pytest.raises(StaleIdea):
        IdeaChecker(league, MY_TEAM, moved)


def test_trade_check_endpoint(client: TestClient, league):
    _, idea = two_for_one(league)
    body = {"league_id": LEAGUE_ID, "season": 2027, "team_id": MY_TEAM, "idea": idea.model_dump(), "reasons": []}
    r = client.post("/api/trade-check", json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["stale"] is None and len(out["verdicts"]) == len(ORDER)

    stale = dict(body, idea=dict(body["idea"], give=[body["idea"]["get"][0]]))
    out = client.post("/api/trade-check", json=stale).json()
    assert out["stale"] and all(v["status"] == "cant_check" for v in out["verdicts"])

    assert signed_in(client.app, team_id=None).post("/api/trade-check", json=body).status_code == 403
    assert client.post("/api/trade-check", json=dict(body, team_id=MY_TEAM + 1)).status_code == 403

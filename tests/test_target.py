import pytest

from app.analysis.target import TargetAnalyzer, TargetError
from tests.conftest import LEAGUE_ID, MY_TEAM


def player(league, name):
    return next(p for t in league.teams for p in t.roster if p.name == name)


def test_fit_and_owner_read(league):
    wemby = player(league, "Victor Wembanyama")
    report = TargetAnalyzer(league, MY_TEAM, wemby.id).target_report()
    assert report.target.name == "Victor Wembanyama"
    assert report.fit.label in {"great", "good"}
    assert "BLK" in report.fit.helps
    assert report.fit.playoff_pct_after > report.fit.playoff_pct_before
    assert report.owner.team_id != MY_TEAM
    assert report.owner.target_is_star
    assert all(w.their_cats_gain > 0 for w in report.owner.would_value)


def test_offers_are_distinct_and_help_me(league):
    giddey = player(league, "Josh Giddey")
    report = TargetAnalyzer(league, MY_TEAM, giddey.id).target_report()
    offers = report.offers
    assert 1 <= len(offers) <= 3
    assert offers[0].angle == "best"
    assert report.note is None
    gives = [(frozenset(p.id for p in o.give), frozenset(p.id for p in o.get)) for o in offers]
    assert len(set(gives)) == len(gives)
    for o in offers:
        assert giddey.id in {p.id for p in o.get}
        assert o.playoff_pct_after >= o.playoff_pct_before
        assert o.pitch


def test_keep_players_are_never_offered(league):
    giddey = player(league, "Josh Giddey")
    amen = player(league, "Amen Thompson")
    report = TargetAnalyzer(league, MY_TEAM, giddey.id, [amen.id]).target_report()
    assert all(amen.id not in {p.id for p in o.give} for o in report.offers)


def test_own_player_rejected(league):
    tatum = player(league, "Jayson Tatum")
    with pytest.raises(TargetError):
        TargetAnalyzer(league, MY_TEAM, tatum.id)


def test_target_endpoint(client, league):
    giddey = player(league, "Josh Giddey")
    r = client.post("/api/target", json={"league_id": LEAGUE_ID, "team_id": MY_TEAM, "player_id": giddey.id})
    assert r.status_code == 200
    assert r.json()["offers"]
    bad = client.post("/api/target", json={"league_id": LEAGUE_ID, "team_id": MY_TEAM, "player_id": 1})
    assert bad.status_code == 400
    assert bad.json()["detail"]["code"] == "player_not_rostered"


def test_report_lists_other_rosters(client):
    r = client.post("/api/analyze", json={"league_id": LEAGUE_ID, "team_id": MY_TEAM, "season": 2027})
    rosters = r.json()["league_rosters"]
    assert len(rosters) == 9
    assert all(t["team_id"] != MY_TEAM for t in rosters)
    assert all(t["players"] for t in rosters)

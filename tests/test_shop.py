import pytest

from app.analysis.shop import MIN_RATIO, ShopAnalyzer, ShopError
from tests.conftest import LEAGUE_ID, MY_TEAM


def ids(league, *names):
    roster = {p.name: p.id for p in league.team(MY_TEAM).roster}
    return [roster[n] for n in names]


def test_deals_send_only_shopped_players(league):
    offered = ids(league, "Jayson Tatum", "Jalen Green")
    rep = ShopAnalyzer(league, MY_TEAM, offered).report()
    assert rep.deals
    for d in rep.deals:
        give = {p.id for p in d.give}
        assert give <= set(offered)
        assert d.uses_all == (give == set(offered))
        # Looks even or better for them on paper, and does more for me than for them.
        assert d.value_they_get >= MIN_RATIO * d.value_they_give - 0.2
        assert d.my_cats_gain >= d.their_cats_gain
        assert d.playoff_pct_after >= d.playoff_pct_before
        assert d.kind in {"win_win", "paper_win"}
    payoff = [d.acceptance * (d.playoff_pct_after - d.playoff_pct_before) for d in rep.deals]
    assert payoff == sorted(payoff, reverse=True)


def test_every_other_team_listed(league):
    rep = ShopAnalyzer(league, MY_TEAM, ids(league, "Jarrett Allen")).report()
    assert len(rep.teams) == len(league.teams) - 1
    assert all(t.team_id != MY_TEAM for t in rep.teams)
    with_deal = [t for t in rep.teams if t.best]
    assert with_deal and rep.teams[: len(with_deal)] == with_deal


def test_shopped_player_overrides_keep(league):
    allen = ids(league, "Jarrett Allen")
    rep = ShopAnalyzer(league, MY_TEAM, allen, keep_player_ids=allen).report()
    assert rep.deals


def test_rejects_players_not_on_my_team(league):
    other = league.team(1).roster[0].id
    with pytest.raises(ShopError):
        ShopAnalyzer(league, MY_TEAM, [other])


def test_shop_endpoint(client, league):
    body = {
        "league_id": LEAGUE_ID,
        "team_id": MY_TEAM,
        "season": 2027,
        "offer_player_ids": ids(league, "Jarrett Allen"),
    }
    r = client.post("/api/shop", json=body)
    assert r.status_code == 200
    assert r.json()["offered"][0]["name"] == "Jarrett Allen"
    bad = client.post("/api/shop", json={**body, "offer_player_ids": [league.team(1).roster[0].id]})
    assert bad.status_code == 422
    assert bad.json()["detail"]["code"] == "not_your_player"
    empty = client.post("/api/shop", json={**body, "offer_player_ids": []})
    assert empty.status_code == 422

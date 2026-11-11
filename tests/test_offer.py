import pytest

from app.analysis.offer import MIN_ODDS_MOVE, OfferChecker, OfferError
from tests.conftest import LEAGUE_ID, MY_TEAM

ALIAGA, AACHEN, JZE = 9, 10, 11


def ids(league, *names):
    found = {p.name: p.id for t in league.teams for p in t.roster}
    return [found[n] for n in names]


def check(league, partner, give, get, **kw):
    return OfferChecker(league, MY_TEAM, partner, ids(league, *give), ids(league, *get), **kw).check()


def test_two_for_two_reads_both_sides(league):
    c = check(league, ALIAGA, ["Jarrett Allen", "Payton Pritchard"], ["Josh Giddey", "Dyson Daniels"])
    i = c.idea
    assert c.partner_team_name == "Aliaga Hermanos"
    assert [p.name for p in i.give] == ["Jarrett Allen", "Payton Pritchard"]
    assert [p.name for p in i.get] == ["Josh Giddey", "Dyson Daniels"]
    assert i.partner_playoff_pct_before is not None and i.partner_playoff_pct_after is not None
    assert len(i.category_detail) == 9
    assert i.finish_before is not None and i.partner_finish_after is not None
    assert not c.my_drops and not c.my_adds and not c.their_drops and not c.their_adds
    assert "AST" in c.helps
    assert c.call_reason and c.partner_reason
    assert 0 < c.odds_noise < 0.02
    assert c.sims >= 10000


def test_call_follows_the_odds(league):
    for partner, give, get in [
        (ALIAGA, ["Jarrett Allen", "Payton Pritchard"], ["Josh Giddey", "Dyson Daniels"]),
        (AACHEN, ["Amen Thompson"], ["Ty Jerome"]),
        (JZE, ["Jarrett Allen", "Ryan Rollins"], ["Desmond Bane"]),
    ]:
        c = check(league, partner, give, get)
        gain = c.idea.playoff_pct_after - c.idea.playoff_pct_before
        if c.call == "take":
            assert gain >= MIN_ODDS_MOVE - 0.001 and gain > c.odds_noise - 0.001
        elif c.call == "pass":
            assert gain <= -MIN_ODDS_MOVE + 0.001
        else:
            assert abs(gain) < MIN_ODDS_MOVE + 0.001 or abs(gain) <= c.odds_noise + 0.001
        theirs = c.idea.partner_playoff_pct_after - c.idea.partner_playoff_pct_before
        assert (c.partner_call == "lose") == (theirs <= -MIN_ODDS_MOVE + 1e-9)
        assert c.paper_win == (c.idea.value_they_get >= c.idea.value_they_give - 0.05)


def test_one_for_two_cuts_my_cheapest_and_lets_me_choose(league):
    c = check(league, JZE, ["Jarrett Allen"], ["Desmond Bane", "Fred VanVleet"])
    assert len(c.my_drops) == 1
    auto = c.my_drops[0]
    assert auto.tier == "droppable"
    assert c.my_drop_options and c.my_drop_options[0].id == auto.id
    assert len(c.their_adds) == 1  # they fill the spot from free agency
    other = next(p for p in c.my_drop_options if p.id != auto.id)
    mine = check(league, JZE, ["Jarrett Allen"], ["Desmond Bane", "Fred VanVleet"], my_drop_ids=[other.id])
    assert [p.id for p in mine.my_drops] == [other.id]
    assert mine.idea.my_drop.id == other.id


def test_two_for_one_is_measured_against_the_free_pickup(league):
    c = check(league, JZE, ["Jarrett Allen", "Ryan Rollins"], ["Desmond Bane"])
    assert len(c.my_adds) == 1 and len(c.their_drops) == 1
    assert c.vs_pickup
    assert c.idea.my_add.id == c.my_adds[0].id
    today = check(league, JZE, ["Jarrett Allen"], ["Desmond Bane"])
    # Without the pickup in the baseline the 2-for-1 would start from today's odds.
    assert c.idea.playoff_pct_before >= today.idea.playoff_pct_before


def test_kept_player_is_flagged(league):
    george = ids(league, "Kyshawn George")
    c = OfferChecker(league, MY_TEAM, AACHEN, george, ids(league, "Ty Jerome"), keep_player_ids=george).check()
    assert any(f.check == "keep" for f in c.flags)


@pytest.mark.parametrize(
    "partner,give,get,drops,code",
    [
        (MY_TEAM, ["Jarrett Allen"], ["Amen Thompson"], [], "own_team"),
        (JZE, ["Desmond Bane"], ["Jarrett Allen"], [], "bad_players"),
        (JZE, ["Jarrett Allen"], ["Ty Jerome"], [], "bad_players"),
        (JZE, ["Jarrett Allen"], ["Desmond Bane"], ["Ryan Rollins"], "bad_drops"),
        (99, ["Jarrett Allen"], ["Desmond Bane"], [], "team_not_found"),
    ],
)
def test_bad_trades_are_refused(league, partner, give, get, drops, code):
    with pytest.raises(OfferError) as exc:
        OfferChecker(league, MY_TEAM, partner, ids(league, *give), ids(league, *get), ids(league, *drops))
    assert exc.value.code == code


def test_offer_check_endpoint(client, league):
    body = {
        "league_id": LEAGUE_ID,
        "team_id": MY_TEAM,
        "partner_team_id": AACHEN,
        "give_player_ids": ids(league, "Amen Thompson"),
        "get_player_ids": ids(league, "Ty Jerome"),
        "direction": "sending",
    }
    r = client.post("/api/offer-check", json=body)
    assert r.status_code == 200
    data = r.json()
    assert data["direction"] == "sending"
    assert data["call"] in {"take", "close", "pass"}
    assert data["idea"]["pitch"]
    assert client.post("/api/offer-check", json=body).json() == data  # cached
    bad = client.post("/api/offer-check", json={**body, "get_player_ids": ids(league, "Jarrett Allen")})
    assert bad.status_code == 400
    assert bad.json()["detail"]["code"] == "bad_players"
    other = client.post("/api/offer-check", json={**body, "team_id": 4})
    assert other.status_code == 403

from fastapi.testclient import TestClient

from app import bookcompare, lostlog
from app.analysis import gamebox, gamescore
from tests.conftest import signed_in
from tests.test_bookcompare import EMBIID, MAXEY, MIL, PHI, book_record, final_day, our_lines
from tests.test_linelog import _app

AWAY, HOME, C, D = 1, 2, 3, 4
AB = {AWAY: "AAA", HOME: "HHH"}
EVEN = {"pf": 110.0, "pa": 110.0, "poss": 100.0, "tpm": 13.0, "tpa": 36.0, "ftm": 17.0, "fta": 22.0}
EVEN |= {"tpm_a": 13.0, "tpa_a": 36.0, "ftm_a": 17.0, "fta_a": 22.0}


def _game(gid, day, team, opp, **stats):
    return gamescore.TeamGame(gid, day, team, opp, EVEN | stats)


def box(away_tpm: float = 5.0, poss: float = 100.0, away_pf: float = 92.0) -> gamebox.Box:
    """Ten even nights for four teams, then AWAY at HOME on day 11 with the away side cold from three."""
    games = []
    for day in range(1, 11):
        for gid, (a, b) in enumerate(((AWAY, C), (HOME, D)), start=day * 10):
            games += [_game(gid, day, a, b), _game(gid, day, b, a)]
    games += [
        _game(999, 11, AWAY, HOME, pf=away_pf, pa=112.0, poss=poss, tpm=away_tpm),
        _game(999, 11, HOME, AWAY, pf=112.0, pa=away_pf, poss=poss),
    ]
    return gamebox.Box(2027, {}, {}, {}, team_games=games)


def test_cold_threes_explain_a_lost_over_and_a_slow_game_explains_another():
    sides = lostlog.team_pieces(box(), 11, AWAY, HOME, (110.0, 110.0), set())
    a = sides[AWAY]
    assert a["tpm"] == 5 and a["tpa"] == 36 and 0.35 < a["exp3"] < 0.37
    assert round(a["pieces"]["three"]) == -24 and abs(a["pieces"]["pace"]) < 1.5
    # the pieces add up to the miss
    assert abs(sum(a["pieces"].values()) - (92 - 110)) < 1e-9
    why = lostlog.game_why(-18 + 2, [(AWAY, "AAA", sides[AWAY], 1.0), (HOME, "HHH", sides[HOME], 1.0)], [])
    assert why["cause"] == "luck" and why["piece"] == "three"
    assert why["text"] == "AAA made 5 of 36 threes (we expected 36%)"

    slow = lostlog.team_pieces(box(away_tpm=13.0, poss=92.0, away_pf=101.0), 11, AWAY, HOME, (110.0, 110.0), set())
    why = lostlog.game_why(-17, [(AWAY, "AAA", slow[AWAY], 1.0), (HOME, "HHH", slow[HOME], 1.0)], [])
    assert why["cause"] == "pace" and why["text"].startswith("Game ran 7 possessions slower than we called (92 vs 99)")


def test_a_margin_counts_the_other_side_against_and_names_the_player_far_off_his_line():
    sides = lostlog.team_pieces(box(away_tpm=13.0), 11, AWAY, HOME, (110.0, 100.0), set())
    # we had AWAY by 10; HOME won by 20 on "other": the player who ran past his line tells it
    star = {"name": "Big Night", "team": HOME, "pts": 45, "line": 21.5}
    counted = [(AWAY, "AAA", sides[AWAY], 1.0), (HOME, "HHH", sides[HOME], -1.0)]
    why = lostlog.game_why(-30, counted, [star])
    assert why["cause"] == "other" and why["text"] == "Big Night (HHH) scored 45 (our line 21.5)"
    assert lostlog.game_why(0, counted, [star]) is None  # nothing to explain


def test_a_player_s_miss_reads_as_minutes_shots_or_shooting():
    line = [34.0, 24.0, 6.0, 4.0, 1.0, 0.5, 2.0, 2.0, 9.0, 18.0, 5.0, 6.0]  # linelog.STATS order
    sat = lostlog.player_why("pts", 24.0, 11.0, line, {"minutes": 22, "fga": 12, "fgm": 4}, 24)
    assert sat["cause"] == "minutes" and sat["text"] == "Played 22 minutes (our line 34) in a 24-point game"
    few = lostlog.player_why("pts", 24.0, 12.0, line, {"minutes": 34, "fga": 9, "fgm": 5}, 4)
    assert few["cause"] == "other" and few["text"] == "Took 9 shots in 34 minutes (about 18 expected)"
    cold = lostlog.player_why("pts", 24.0, 12.0, line, {"minutes": 34, "fga": 19, "fgm": 4}, 4)
    assert cold["cause"] == "luck" and cold["text"] == "Shot 4 of 19"
    boards = lostlog.player_why("reb", 6.0, 2.0, line, {"minutes": 33}, 4)
    assert boards["cause"] == "other" and boards["text"] == "2 rebounds in 33 minutes (our line 6.0)"


def test_the_bet_a_row_takes_mirrors_the_page():
    margin = {"market": "margin", "ours": 2.0, "book": 5.5, "actual": 9.0}
    margin["odds"] = [{"side": "pick", "odd": 1.59}, {"side": "other", "odd": 1.65}]
    got = lostlog.bet_of(margin, "game")
    assert got["bet"] == "under" and got["odd"] == 1.65 and got["won"] is False  # took the points, lost by 9
    assert 0.59 < got["chance"] < 0.61
    game = {"away": AWAY, "home": HOME}
    assert lostlog._words(dict(margin, pick=HOME), got, game, AB) == "AAA +5.5"
    assert lostlog.bet_of(dict(margin, actual=5.5), "game")["won"] is None  # a push
    assert lostlog.bet_of(dict(margin, odds=[{"side": "other", "odd": 1.3}]), "game") is None  # too short
    total = {"market": "total", "ours": 235.0, "book": 227.5, "actual": 205.0}
    total["odds"] = [{"side": "under", "odd": 1.62}, {"side": "over", "odd": 1.62}]
    b = lostlog.bet_of(total, "game")
    assert b["won"] is False and lostlog._words(total, b, game, AB) == "Over 227.5"
    rungs = [
        {"n": 10, "odd": 1.25, "ours": 0.8},
        {"n": 15, "odd": 1.5, "ours": 0.6},
        {"n": 20, "odd": 2.4, "ours": 0.3},
    ]
    player = {"ours": 16.2, "actual": 5.0, "played": True, "rungs": rungs}
    assert lostlog.bet_of(player, "player") == {"bet": "15+", "odd": 1.5, "chance": 0.6, "won": False}
    assert lostlog.bet_of(dict(player, played=False), "player")["won"] is None  # he sat: void


def test_a_cause_that_keeps_coming_back_becomes_a_candidate_luck_never_does():
    def lost(cause, group, piece):
        return {"cause": cause, "group": group, "piece": piece}

    nights = [{"lost": [lost("minutes", "pts", "minutes")] * 8 + [lost("luck", "pts", "shooting")] * 12}]
    got = lostlog.repeats(nights)
    assert got["lost"] == 20 and got["causes"][0] == {"cause": "luck", "count": 12}
    [rep] = got["repeats"]
    assert rep["cause"] == "minutes" and rep["count"] == 8 and rep["share"] == 0.4 and rep["candidate"]
    got = lostlog.repeats([{"lost": [lost("minutes", "pts", "minutes")] * 7}])
    assert not got["repeats"][0]["candidate"]  # 7 times is not yet a repeat


def test_a_settled_night_carries_a_reason_on_each_row_and_lists_its_lost_bets():
    day = final_day(away=104, home=119)
    day["games"][0]["lines"][0] |= {"minutes": 33, "fga": 15, "fgm": 6}
    names = {EMBIID: "Joel Embiid", MAXEY: "Tyrese Maxey"}
    games = bookcompare.compare_day(book_record(), our_lines(), day, names)["games"]
    lostlog.annotate(games, None, 33, set(), our_lines(), day, {PHI: "PHI", MIL: "MIL"})
    [g] = games
    pts = next(p for p in g["players"] if p["name"] == "Joel Embiid" and p["stat"] == "pts")
    assert pts["why"] == {"cause": "luck", "piece": "shooting", "points": -8.1, "text": "Shot 6 of 15"}
    maxey = [p for p in g["players"] if p["name"] == "Tyrese Maxey"]
    assert maxey and all(p["why"] is None for p in maxey)  # he sat: nothing to explain
    assert all(m["why"] is None for m in g["markets"])  # no league box: the game rows wait for it

    n = lostlog.night(games, 33, {PHI: "PHI", MIL: "MIL"})
    assert n["day"] == 33 and n["bets"] >= 1 and n["won"] + len(n["lost"]) == n["bets"]
    embiid = [x for x in n["lost"] if x["bet"].startswith("Joel Embiid")]
    assert embiid and embiid[0]["cause"] == "luck" and embiid[0]["text"] == "Shot 6 of 15"
    assert embiid[0]["group"] == "pts" and embiid[0]["actual"] == 22.0


def test_the_lost_log_is_for_signed_in_users(recorded):
    app, store, _ = _app(recorded)
    store.save_book_lines(2027, 33, book_record())
    store.save_player_lines(2027, 33, our_lines(), False)
    assert TestClient(app).post("/api/book-compare/lost", json={"season": 2027}).status_code == 401
    got = signed_in(app, team_id=None).post("/api/book-compare/lost", json={"season": 2027})
    assert got.status_code == 200
    body = got.json()
    # no results read (no league to read them with): nothing settled, nothing lost
    assert body == {"season": 2027, "nights": [], "lost": 0, "causes": [], "repeats": []}

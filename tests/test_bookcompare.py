from datetime import datetime, timedelta

from fastapi.testclient import TestClient

from app import bookcompare, iddaa, linelog
from app.playersearch import IndexPlayer
from tests.conftest import signed_in
from tests.test_iddaa import EVENT, schedule
from tests.test_linelog import _app
from tests.test_news import boss

EMBIID, MAXEY, OUBRE = 3059318, 4431678, 4066429
PHI, MIL = 20, 15


def book_record() -> dict:
    sched = schedule()
    game = sched.games[33][0]
    rec = iddaa.update({}, sched, [({k: EVENT[k] for k in ("i", "hn", "an", "ci", "d")}, EVENT)], game.start)
    assert rec == {}  # at tip-off it's too late
    rec = iddaa.update(
        {}, sched, [({k: EVENT[k] for k in ("i", "hn", "an", "ci", "d")}, EVENT)], game.start - timedelta(hours=1)
    )
    rec[33]["games"]["401"]["ours"] = [108, 121, 0.85]
    return rec[33]


def our_lines(embiid_pts: float = 31.0) -> dict:
    def packed(team, pts, reb, ast):
        return [team, 34.0, pts, reb, ast, 1.0, 1.0, 2.0, 3.0, 9.0, 18.0, 6.0, 7.0]

    return {
        "teams": {},
        "lines": {str(EMBIID): packed(PHI, embiid_pts, 9.0, 4.0), str(MAXEY): packed(PHI, 24.0, 4.0, 6.0)},
    }


def final_day(away: int = 104, home: int = 119) -> dict:
    def line(pid, name, pts, reb, ast):
        return {
            "player_id": pid,
            "name": name,
            "pro_team_id": PHI,
            "pts": pts,
            "reb": reb,
            "ast": ast,
            "stl": 0,
            "blk": 1,
            "tpm": 2,
        }

    return {
        "games": [
            {
                "id": 401,
                "status": "final",
                "away": {"score": away},
                "home": {"score": home},
                "lines": [line(EMBIID, "Joel Embiid", 22, 11, 3)],  # Maxey sat
            }
        ]
    }


def test_names_match_across_spellings():
    espn = {1: "Jaime Jaquez Jr.", 2: "Nikola Jokić", 3: "Jalen Williams", 4: "Jaylin Williams", 5: "VJ Edgecombe"}
    got = bookcompare.match_names(["Jaime Jaquez", "Nikola Jokic", "J. Williams", "V.J. Edgecombe"], espn)
    assert got == {"Jaime Jaquez": 1, "Nikola Jokic": 2, "V.J. Edgecombe": 5}  # two Williams: no guess


def test_a_number_is_right_when_the_result_lands_near_it_not_on_its_side_of_the_line():
    # Can's case: iddaa 13.5, ours 13.3, he scored 2. Ours was under the line, but far from 2: wrong.
    got = bookcompare.judge(13.3, 13.5, 2.0, bookcompare.BANDS["pts"])
    assert got["ours_hit"] is False and got["book_hit"] is False and got["closer"] == "ours"
    assert got["gap"] == -0.2
    got = bookcompare.judge(7.8, 6.5, 6.0, bookcompare.BANDS["reb"])
    assert got["ours_hit"] is True and got["book_hit"] is True and got["closer"] == "book"
    assert bookcompare.judge(20.0, 18.0, None, 5.0)["ours_hit"] is None  # before the game


def test_a_day_compares_players_and_the_game_against_the_result():
    names = {EMBIID: "Joel Embiid", MAXEY: "Tyrese Maxey"}
    before = bookcompare.compare_day(book_record(), our_lines(), None, names)
    [g] = before["games"]
    embiid = {p["stat"]: p for p in g["players"] if p["name"] == "Joel Embiid"}
    assert embiid["pts"]["ours"] == 31.0 and 25 < embiid["pts"]["book"] < 29 and embiid["pts"]["gap"] > 0
    assert embiid["pts"]["actual"] is None and embiid["pts"]["ours_hit"] is None
    ament = [p for p in g["players"] if p["name"] == "Nate Ament"]
    assert ament and all(p["ours"] is None for p in ament)  # iddaa prices him, our log has no line

    after = bookcompare.compare_day(book_record(), our_lines(), final_day(), names)
    [g] = after["games"]
    embiid = {p["stat"]: p for p in g["players"] if p["name"] == "Joel Embiid"}
    assert embiid["pts"]["actual"] == 22.0 and embiid["pts"]["ours_hit"] is False  # 31 vs 22: off by 9
    assert embiid["reb"]["actual"] == 11.0 and embiid["reb"]["ours_hit"] is True  # 9 vs 11: in, by 2.0
    maxey = [p for p in g["players"] if p["name"] == "Tyrese Maxey"]
    assert maxey and all(p["played"] is False and p["ours_hit"] is None for p in maxey)
    m = {(x["market"], x.get("pick")): x for x in g["markets"]}
    assert m[("winner", PHI)]["ours_hit"] is True and m[("winner", PHI)]["won"] == PHI
    assert m[("margin", PHI)]["ours"] == 13.0 and m[("margin", PHI)]["book"] == 11.5
    assert m[("margin", PHI)]["actual"] == 15 and m[("margin", PHI)]["closer"] == "ours"
    assert m[("total", None)]["ours"] == 229.0 and m[("total", None)]["actual"] == 223.0
    pts = next(t for t in after["tally"] if t["stat"] == "pts")
    assert pts["judged"] == 1 and pts["ours_hits"] == 0 and pts["ours_miss"] == 9.0
    chances = [c for _, c in embiid["pts"]["ladder"]]
    assert len(chances) >= 3 and chances == sorted(chances, reverse=True)  # iddaa's rungs, for the open row
    assert m[("winner", PHI)]["book"] is None and m[("winner", PHI)]["closer"] is None  # iddaa closed PHI


def test_admins_read_a_day_compared(recorded):
    app, store, sched = _app(recorded)
    store.save_book_lines(2027, 33, book_record())
    store.save_player_lines(2027, 33, linelog.settle(our_lines(), []), True)
    app.state.player_index.set(
        2027,
        [IndexPlayer(EMBIID, "Joel Embiid", "PHI", "C", 99.0), IndexPlayer(MAXEY, "Tyrese Maxey", "PHI", "G", 99.0)],
    )
    assert signed_in(app, team_id=None).post("/api/admin/book-compare", json={"season": 2027}).status_code == 403
    got = boss(app).post("/api/admin/book-compare", json={"season": 2027}).json()
    assert got["day"] == 33 and got["days"] == [33] and got["bands"]["pts"] == 5.0
    [g] = got["games"]
    assert g["away"] == "MIL" and g["home"] == "PHI"
    winner = next(m for m in g["markets"] if m["market"] == "winner")
    assert winner["team"] == "PHI" and winner["ours"] == 0.85
    assert any(p["name"] == "Joel Embiid" and p["stat"] == "pts" and p["ours"] == 31.0 for p in g["players"])
    assert {t["stat"] for t in got["tally"]} >= {"pts", "winner", "total"}

    # Anyone signed in reads the same comparison (the Our Estimates page); signed out, nobody.
    assert app.state.book_log is not None
    app.state.book_log.update(ok=False, error="HTTPStatusError: 403")
    admin = TestClient(app, headers={"Authorization": "Bearer admin-token"}).post(
        "/api/admin/book-compare", json={"season": 2027}
    )
    assert admin.json()["look"]["error"]
    user = signed_in(app, team_id=None).post("/api/book-compare", json={"season": 2027}).json()
    assert user["games"] == admin.json()["games"]
    assert TestClient(app).post("/api/book-compare", json={"season": 2027}).status_code == 401
    assert user["look"]["ok"] is False and user["look"]["error"] is None


def test_a_game_not_yet_started_without_a_saved_call_gets_today_s(recorded):
    app, store, _ = _app(recorded)
    rec = book_record()
    del rec["games"]["401"]["ours"]  # the look ran before the box was built
    store.save_book_lines(2027, 33, rec)
    tip = datetime.fromisoformat(rec["games"]["401"]["tip"])
    app.state.now = lambda: tip - timedelta(hours=2)
    got = boss(app).post("/api/admin/book-compare", json={"season": 2027}).json()
    winner = next(m for m in got["games"][0]["markets"] if m["market"] == "winner")
    assert winner["ours"] is not None  # today's call, though the book closed the favourite


def test_each_rung_carries_both_chances_its_value_and_the_result():
    names = {EMBIID: "Joel Embiid", MAXEY: "Tyrese Maxey"}
    after = bookcompare.compare_day(book_record(), our_lines(), final_day(), names)
    [g] = after["games"]
    embiid = {p["stat"]: p for p in g["players"] if p["name"] == "Joel Embiid"}["pts"]
    steps = embiid["rungs"]
    assert len(steps) >= 3 and all(r["odd"] >= iddaa.MIN_ODD for r in steps)
    ours = [r["ours"] for r in steps]
    assert ours == sorted(ours, reverse=True) and all(0 < p < 1 for p in ours)  # higher rung, smaller chance
    for r in steps:
        assert r["value"] == round(r["ours"] * r["odd"] - 1, 3)
        assert r["hit"] == (22 >= r["n"])  # he scored 22
    assert embiid["best_value"] == max(r["value"] for r in steps)
    assert embiid["rung_closer"] in ("ours", "book", "tie")
    maxey = [p for p in g["players"] if p["name"] == "Tyrese Maxey"]
    assert all(r["hit"] is None for p in maxey for r in p["rungs"])  # he sat: nothing settles
    ament = [p for p in g["players"] if p["name"] == "Nate Ament"]
    assert all(r["ours"] is None and r["value"] is None for p in ament for r in p["rungs"])  # no line of ours

    pts = next(t for t in after["tally"] if t["stat"] == "pts")
    assert pts["rungs"] == len(steps)
    assert pts["ours_rung_score"] is not None and pts["book_rung_score"] is not None
    picks = [r for r in steps if r["value"] > 0]
    assert pts["value_picks"] == len(picks)
    assert pts["value_return"] == round(sum(r["odd"] - 1 if r["hit"] else -1 for r in picks), 2)

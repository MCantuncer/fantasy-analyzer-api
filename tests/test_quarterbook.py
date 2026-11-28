"""Quarter and first-half bets beside our quarter numbers (app/quarterbook.py)."""

import httpx
import pytest

from app import linelog, quarterbook
from tests.conftest import signed_in
from tests.test_bookcompare import MIL, PHI, book_record, our_lines
from tests.test_linelog import _app

# iddaa's saved board: quarter totals at two lines (the main one is the closer pair), quarter winners,
# first-half total, handicap and team points.
BOARD = [
    [777, "1|57.5", [["Alt", 1.80], ["Üst", 1.62]]],
    [777, "1|56.5", [["Alt", 1.70], ["Üst", 1.70]]],
    [776, "1", [["1", 1.60], ["0", 13.0], ["2", 2.20]]],
    [776, "4", [["1", 1.30], ["0", 13.0], ["2", 2.90]]],
    [116, "114.5", [["Alt", 1.66], ["Üst", 1.66]]],
    [115, "-3.5", [["1", 1.70], ["2", 1.65]]],
    [746, "60.5", [["Alt", 1.70], ["Üst", 1.62]]],
    [747, "54.5", [["Alt", 1.62], ["Üst", 1.70]]],
    [2, "", [["1", 1.40], ["2", 2.90]]],  # not a quarter market
]
AWAY_Q, HOME_Q = [26.0, 27.0, 27.0, 25.0], [30.0, 31.0, 30.0, 28.0]


def test_the_board_reads_each_market_at_its_main_line():
    m = quarterbook.markets(BOARD)
    assert m[("q_total", 1)] == (56.5, 1.70, 1.70)
    assert m[("q_winner", 1)] == {"1": 1.60, "0": 13.0, "2": 2.20}
    assert m[("h1_total", 0)] == (114.5, 1.66, 1.66) and m[("h1_margin", 0)] == (-3.5, 1.70, 1.65)
    assert ("h1_home", 0) in m and ("h1_away", 0) in m and len(m) == 7


def test_chances_from_the_measured_miss():
    assert quarterbook.over_chance(56.5, 56.5, 8.4) == pytest.approx(0.5)
    assert quarterbook.over_chance(60.0, 56.5, 8.4) > 0.6
    home, tie, away = quarterbook.three_way(0.0, quarterbook.QUARTER_MARGIN_SD)
    assert tie == pytest.approx(0.0486, abs=0.002)  # a quarter ends level about 4.7% of the time
    assert home == pytest.approx(away) and home + tie + away == pytest.approx(1)


def test_each_bet_gets_our_number_our_lean_and_its_result():
    actual = ([20.0, 25.0, 30.0, 30.0], [33.0, 30.0, 25.0, 28.0])
    bets = {(b["market"], b["period"]): b for b in quarterbook.bets(BOARD, MIL, PHI, (AWAY_Q, HOME_Q), actual)}
    q1 = bets[("q_total", 1)]
    assert q1["ours"] == 56.0 and q1["pick"] == "under" and q1["actual"] == 53.0 and q1["hit"] is True
    over = next(s for s in q1["sides"] if s["side"] == "over")
    assert over["book"] == pytest.approx(0.5) and over["value"] == pytest.approx(over["ours"] * 1.70 - 1, abs=1e-3)
    win1 = bets[("q_winner", 1)]
    assert win1["ours"] == 4.0 and win1["pick"] == "home" and win1["result"] == "home" and win1["hit"] is True
    assert [s["side"] for s in win1["sides"]] == ["home", "tie", "away"]
    assert bets[("q_winner", 4)]["pick"] is None  # the home side at 1.30: not the book's number
    h1 = bets[("h1_margin", 0)]
    assert h1["ours"] == 8.0 and h1["pick"] == "home" and h1["actual"] == 18.0 and h1["hit"] is True
    assert bets[("h1_home", 0)]["ours"] == 61.0 and bets[("h1_away", 0)]["actual"] == 45.0
    assert quarterbook.bets(BOARD, MIL, PHI, None, actual) == []
    before = quarterbook.bets(BOARD, MIL, PHI, (AWAY_Q, HOME_Q), None)
    assert all(b["hit"] is None and b["actual"] is None for b in before)


def test_the_record_counts_picks_and_what_they_returned():
    actual = ([20.0, 25.0, 30.0, 30.0], [33.0, 30.0, 25.0, 28.0])
    games = [{"quarters": quarterbook.bets(BOARD, MIL, PHI, (AWAY_Q, HOME_Q), actual)}]
    rec = quarterbook.record(games)
    assert rec["all"]["bets"] == 6 and rec["all"]["won"] + (rec["all"]["bets"] - rec["all"]["won"]) == 6
    assert set(rec["by_market"]) <= {"q_total", "q_winner", "h1_total", "h1_margin", "h1_home", "h1_away"}
    assert rec["edge"]["bets"] <= rec["all"]["bets"]
    assert quarterbook.record([]) == {
        "all": {"bets": 0, "won": 0, "ours": None, "book": None, "returned": 0.0, "need": None},
        "edge": {"bets": 0, "won": 0, "ours": None, "book": None, "returned": 0.0, "need": None},
        "by_market": {},
        "by_odds": {},
    }


def test_the_record_says_how_often_picks_had_to_win_and_splits_them_by_price():
    def pick(odd: float, hit: bool, value: float = 0.1) -> dict:
        side = {"side": "home", "odd": odd, "ours": 0.6, "book": 0.55, "value": value}
        return {"market": "q_total", "pick": "home", "hit": hit, "sides": [side]}

    games = [{"quarters": [pick(1.4, True), pick(1.4, True), pick(1.4, False), pick(2.5, False, -0.1)]}]
    rec = quarterbook.record(games)
    assert rec["all"]["need"] == round(4 / (1.4 * 3 + 2.5), 3)
    assert rec["all"]["returned"] == round(0.4 + 0.4 - 1 - 1, 2)
    assert list(rec["by_odds"]) == ["under 1.50", "2.20+"]
    low = rec["by_odds"]["under 1.50"]
    assert low["all"]["bets"] == 3 and low["all"]["won"] == 2 and low["all"]["need"] == round(1 / 1.4, 3)
    assert rec["by_odds"]["2.20+"]["edge"]["bets"] == 0


def test_the_trial_lists_its_nights_from_the_first_day_on():
    actual = ([20.0, 25.0, 30.0, 30.0], [33.0, 30.0, 25.0, 28.0])
    games = [{"quarters": quarterbook.bets(BOARD, MIL, PHI, (AWAY_Q, HOME_Q), actual)}]
    before = quarterbook.bets(BOARD, MIL, PHI, (AWAY_Q, HOME_Q), None)
    nights = [
        {"date": "2026-11-23", "games": games},  # before the trial
        {"date": "2026-11-24", "games": games},
        {"date": "2026-11-25", "games": [{"quarters": before}]},  # nothing settled yet
        {"date": None, "games": games},  # the schedule didn't read
    ]
    got = quarterbook.trial(nights)
    assert got["start"] == "2026-11-24" and got["review"] == "2026-12-08"
    [night] = got["series"]
    rec = quarterbook.record(games)
    assert night == {
        "date": "2026-11-24",
        "bets": rec["all"]["bets"],
        "won": rec["all"]["won"],
        "returned": rec["all"]["returned"],
        "edge_bets": rec["edge"]["bets"],
        "edge_won": rec["edge"]["won"],
        "edge_returned": rec["edge"]["returned"],
    }
    assert quarterbook.trial([])["series"] == []


def test_the_page_and_the_record_carry_quarter_bets(recorded):
    app, store, _ = _app(recorded)
    rec = book_record()
    rec["games"]["401"]["board"] = BOARD
    store.save_book_lines(2027, 33, rec)
    store.save_player_lines(2027, 33, linelog.settle(our_lines(), []), True)

    def scoreboard(request: httpx.Request) -> httpx.Response:
        event = {
            "id": "401",
            "season": {"type": 2},
            "competitions": [
                {
                    "status": {"type": {"state": "post"}},
                    "competitors": [
                        {
                            "homeAway": "away",
                            "team": {"id": str(MIL)},
                            "linescores": [{"value": v} for v in (20, 25, 30, 30)],
                        },
                        {
                            "homeAway": "home",
                            "team": {"id": str(PHI)},
                            "linescores": [{"value": v} for v in (33, 30, 25, 28)],
                        },
                    ],
                }
            ],
        }
        return httpx.Response(200, json={"events": [event]})

    app.state.clock_transport = httpx.MockTransport(scoreboard)
    user = signed_in(app, team_id=None)
    [g] = user.post("/api/book-compare", json={"season": 2027}).json()["games"]
    q = {(b["market"], b["period"]): b for b in g["quarters"]}
    assert ("q_total", 1) in q and ("h1_margin", 0) in q
    assert q[("q_total", 1)]["actual"] == 53.0 and q[("q_total", 1)]["ours"] > 50
    got = user.post("/api/book-compare/record", json={"season": 2027}).json()
    assert got["quarters"]["all"]["bets"] >= 1
    assert got["quarters"]["trial"]["review"] == "2026-12-08"

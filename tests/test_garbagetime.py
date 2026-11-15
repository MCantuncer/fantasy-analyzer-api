"""Garbage time from ESPN's play-by-play: the rule, the box-score check, and reading each game once."""

import asyncio

import httpx

from app.garbagetime import SummaryBook, clock_seconds, parse_summary, threshold

HOME, AWAY = 1, 2
STARTERS = {11, 12, 13, 14, 15, 21, 22, 23, 24, 25}
KEYS = [
    "minutes",
    "fieldGoalsMade-fieldGoalsAttempted",
    "threePointFieldGoalsMade-threePointFieldGoalsAttempted",
    "freeThrowsMade-freeThrowsAttempted",
    "offensiveRebounds",
    "defensiveRebounds",
    "rebounds",
    "assists",
    "steals",
    "blocks",
    "turnovers",
    "fouls",
    "plusMinus",
    "points",
]


def stat_row(pts=0, fg=(0, 0), tp=(0, 0), ft=(0, 0), reb=0, ast=0, stl=0, blk=0, to=0):
    return ["10", f"{fg[0]}-{fg[1]}", f"{tp[0]}-{tp[1]}", f"{ft[0]}-{ft[1]}", "0", str(reb), str(reb),
            str(ast), str(stl), str(blk), str(to), "0", "+0", str(pts)]  # fmt: skip


def athlete(aid, stats=None):
    return {"athlete": {"id": str(aid)}, "starter": aid in STARTERS, "didNotPlay": stats is None, "stats": stats or []}


def play(period, clock, kind, text, people, home, away, shooting=False, scoring=False, value=0):
    return {
        "type": {"text": kind},
        "text": text,
        "period": {"number": period},
        "clock": {"displayValue": clock},
        "participants": [{"athlete": {"id": str(p)}} for p in people],
        "homeScore": home,
        "awayScore": away,
        "shootingPlay": shooting,
        "scoringPlay": scoring,
        "scoreValue": value,
    }


def summary(box16=None, final=(28, 0)):
    """A blowout: starters score early; with the game decided and the starters out, No. 16 adds a three,
    a rebound and a steal."""
    box16 = box16 or stat_row(pts=3, fg=(1, 1), tp=(1, 1), reb=1, stl=1)
    lines = {
        11: stat_row(pts=3, fg=(1, 1), tp=(1, 1)),
        12: stat_row(pts=2, fg=(1, 1), ast=1),
        16: box16,
        17: stat_row(ast=1),
        21: stat_row(),
        26: stat_row(fg=(0, 1), to=1),
        27: None,  # did not play
    }
    players = [
        {"team": {"id": str(team)}, "statistics": [{"keys": KEYS, "athletes": [athlete(a, lines[a]) for a in ids]}]}
        for team, ids in ((HOME, [11, 12, 16, 17]), (AWAY, [21, 26, 27]))
    ]
    plays = [
        play(1, "11:30", "Jump Shot", "11 makes 25-ft three point jumper (12 assists)", [11, 12], 3, 0, True, True, 3),
        play(4, "7:00", "Layup Shot", "12 makes two point shot", [12], 25, 0, True, True, 2),
        play(4, "6:00", "Substitution", "16 enters the game for 11", [16, 11], 25, 0),
        play(4, "6:00", "Substitution", "26 enters the game for 21", [26, 21], 25, 0),
        play(4, "4:00", "Jump Shot", "16 makes 24-ft three point jumper (17 assists)", [16, 17], 28, 0, True, True, 3),
        play(4, "3:00", "Jump Shot", "26 misses 20-foot jumper", [26], 28, 0, True),
        play(4, "2:55", "Defensive Rebound", "16 defensive rebound", [16], 28, 0),
        play(4, "2:00", "Bad Pass\nTurnover", "26 bad pass (16 steals)", [26, 16], 28, 0),
        play(4, "0.0", "End Game", "End of Game", [], 28, 0),
    ]  # fmt: skip
    comps = [
        {"id": str(HOME), "homeAway": "home", "score": str(final[0])},
        {"id": str(AWAY), "homeAway": "away", "score": str(final[1])},
    ]
    return {"header": {"competitions": [{"competitors": comps}]}, "boxscore": {"players": players}, "plays": plays}


def test_the_rule_and_the_clock():
    assert [threshold(s) for s in (720, 541, 540, 361, 360, 10)] == [25, 25, 20, 20, 10, 10]
    assert clock_seconds("11:39") == 699 and clock_seconds("26.0") == 26 and clock_seconds("") == 0


def test_garbage_time_needs_the_lead_and_the_starters_out():
    notes = parse_summary(summary())
    assert notes.score_for(16) == (28, 0) and notes.score_for(26) == (0, 28)
    assert notes.score_for(27) == (0, 28), "a player who sat out still has his team's score"
    assert notes.score_for(99) is None
    g16 = notes.garbage[16]
    # Up 25 at 7:00 with three starters on: not yet. Two of them leave at 6:00: decided to the end.
    assert (g16["pts"], g16["tpm"], g16["reb"], g16["stl"], g16["minutes"]) == (3, 1, 1, 1, 6.0)
    assert notes.garbage[17]["ast"] == 1
    assert (notes.garbage[26]["fga"], notes.garbage[26]["to"]) == (1, 1)
    assert notes.garbage[12]["pts"] == 0, "his basket came before the game was decided"
    assert notes.garbage[11]["pts"] == 0 and notes.garbage[11]["minutes"] == 0
    assert 27 not in notes.garbage, "no line for a player who didn't play"


def test_a_line_that_doesnt_match_the_box_score_gets_no_garbage_time():
    notes = parse_summary(summary(box16=stat_row(pts=5, fg=(2, 2), tp=(1, 1), reb=1, stl=1)))
    assert 16 not in notes.garbage, "never show a split we can't rebuild exactly"
    assert notes.score_for(16) == (28, 0), "the score still shows"
    assert 11 in notes.garbage


def test_no_final_score_no_notes():
    raw = summary()
    raw["header"]["competitions"][0]["competitors"] = []
    assert parse_summary(raw) is None


def test_each_finished_game_is_read_once():
    calls = []

    def espn(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.params.get("event"))
        if request.url.params.get("event") == "500":
            return httpx.Response(503)
        return httpx.Response(200, json=summary())

    book = SummaryBook()
    transport = httpx.MockTransport(espn)
    got = asyncio.run(book.notes([400, 500], transport))
    assert set(got) == {400} and got[400].score_for(16) == (28, 0)
    again = asyncio.run(book.notes([400, 500], transport))
    assert set(again) == {400}
    assert calls == ["400", "500", "500"], "a read game is kept; one ESPN couldn't give is tried again"

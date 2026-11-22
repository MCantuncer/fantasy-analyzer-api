"""scripts/miss_causes.py: the readings on synthetic games (the season replay itself needs ESPN caches)."""

import random

from scripts.miss_causes import REPEAT_GAMES, summary


def side(called: float, pace: float, three: float, ft: float = 0.0, other: float = 0.0) -> dict:
    pieces = {"pace": pace, "three": three, "free_throws": ft, "other": other}
    return {"called": called, "actual": called + sum(pieces.values()), "pieces": pieces}


def games(n: int, pace_lean: float = 0.0, hot_team: int | None = None, seed: int = 1) -> list[dict]:
    rnd = random.Random(seed)
    out = []
    for i in range(n):
        away, home = rnd.sample(range(1, 31), 2)
        pace = pace_lean + rnd.gauss(0, 4)
        sides = {}
        for t in (away, home):
            hot = 6.0 if t == hot_team else 0.0
            sides[t] = side(110, pace, hot + rnd.gauss(0, 1 if hot else 6), rnd.gauss(0, 1), rnd.gauss(0, 5))
        out.append({"game": i, "day": i // 8, "away": away, "home": home, "split": sides})
    return out


def test_a_quiet_season_names_nothing():
    s = summary(games(600))
    assert s["games"] == 600
    assert abs(sum(s["shares"]["total"].values()) - 1) < 0.005  # the shares (rounded) add to the whole miss
    assert [c for c in s["candidates"] if c["kind"] != "share"] == []  # the made-up shares aren't real ones


def test_a_league_lean_and_one_team_that_keeps_missing_are_named():
    s = summary(games(600, pace_lean=-3, hot_team=4), {4: "BOS"})
    kinds = {c["kind"] for c in s["candidates"]}
    assert kinds - {"share"} == {"league", "team"}
    assert s["league"]["pace"]["mean"] < -2.5
    # the league's pace lean is named once, not again for every team; the one hot team is named alone
    (row,) = s["teams"]
    assert (row["team"], row["piece"], row["whose"]) == ("BOS", "three", "own points")
    assert any("BOS three in its own points" in c["what"] for c in s["candidates"])


def test_too_few_games_name_nothing():
    s = summary(games(REPEAT_GAMES // 2 - 1, pace_lean=-10))  # two team games each
    assert [c for c in s["candidates"] if c["kind"] != "share"] == []

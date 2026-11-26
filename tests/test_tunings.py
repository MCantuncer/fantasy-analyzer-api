import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from app import tunings
from app.analysis import gamescore
from app.lostlog import blend_chance
from tests.conftest import signed_in
from tests.test_linelog import LEAGUE_ID, _app
from tests.test_news import boss

TIP = datetime(2026, 11, 26, 0, 30, tzinfo=UTC)
MIN_, OKC = 16, 25
LINE = {"minutes": 24.0, "pts": 12.0, "reb": 4.0, "ast": 2.0, "three_pm": 1.0}


def form(code: str, won: int, played: int, recent=(), stars=()) -> tunings.TeamForm:
    return tunings.TeamForm(code, won, played, tuple(recent), frozenset(stars))


SLUMP = [(True, 0.7), (False, 0.7), (True, 0.65), (False, 0.7), (False, 0.6)]  # 2 won of 3.35 expected


def inputs(**kw) -> tunings.GameInputs:
    base = dict(away=MIN_, home=OKC, tip=TIP, live_call=(111.1, 109.3), alone_call=(111.1, 109.3))
    return tunings.GameInputs(**(base | kw))


BOOK = {"win": [1.48, 1.87], "margin": [-2.5, 1.65, 1.66], "total": [222.5, 1.68, 1.62]}


def test_playoff_foes_are_the_last_two_seasons_series_either_way_round():
    assert tunings.playoff_foes(2027, "MIN", "OKC")  # 2025 West final
    assert tunings.playoff_foes(2027, "SA", "NY")  # 2026 Finals
    assert not tunings.playoff_foes(2027, "OKC", "BOS")
    assert not tunings.playoff_foes(2028, "MIN", "OKC")  # 2025 is three seasons back by then


def test_the_stack_needs_a_fallen_contender_in_a_slump_against_a_recent_foe():
    okc = form("OKC", 8, 16, SLUMP)
    assert tunings.fallen_and_slumping(2027, okc)
    assert not tunings.fallen_and_slumping(2027, form("OKC", 12, 16, SLUMP))  # still .600+
    assert not tunings.fallen_and_slumping(2027, form("OKC", 8, 16, [(True, 0.5)] * 5))  # no slump
    assert not tunings.fallen_and_slumping(2027, form("OKC", 8, 16, SLUMP[:4]))  # too few games
    assert not tunings.fallen_and_slumping(2027, form("WSH", 3, 16, SLUMP))  # never a contender

    minn = form("MIN", 13, 18, [(True, 0.5)] * 5, stars={7})
    assert tunings.stack(2027, minn, okc, set()) == {"team": "OKC", "edge": 0.12, "star_out": False}
    assert tunings.stack(2027, minn, okc, {7}) == {"team": "OKC", "edge": 0.03, "star_out": True}
    assert tunings.stack(2027, form("BOS", 13, 18), okc, set()) is None  # not a playoff foe
    slumping_min = form("MIN", 7, 18, SLUMP)  # MIN won .598 last season: not a fallen contender
    assert tunings.stack(2027, slumping_min, okc, set())["team"] == "OKC"
    den = form("DEN", 8, 16, SLUMP)  # two fallen contenders in a slump, met in the 2026 playoffs
    assert tunings.stack(2027, den, form("MIN", 8, 16, SLUMP), set())["team"] == "DEN"
    assert tunings.stack(2027, form("OKC", 8, 16, SLUMP), form("SA", 8, 16, SLUMP), set()) is None


def test_calls_blend_live_with_the_book_and_shift_the_book_for_the_stack():
    made, book = tunings.calls(inputs(book=BOOK, stack={"team": "OKC", "edge": 0.12, "star_out": False}))
    fair = round((1 / 1.48) / (1 / 1.48 + 1 / 1.87), 3)
    assert book == [fair, 2.5, 222.5]
    ours = 1 - gamescore.win_chance(109.3 - 111.1)
    assert made["live"] == [111.1, 109.3, round(blend_chance(ours, fair), 3)]
    assert made["ours_alone"][2] == round(ours, 3) and made["ours_alone"][2] < 0.5
    assert made["book"] == [110.0, 112.5, fair]
    assert made["rivalry"][2] == round(fair + 0.12, 3)
    assert made["rivalry"][1] - made["rivalry"][0] == pytest.approx(tunings.margin_of(fair + 0.12), abs=0.11)

    alone, none = tunings.calls(inputs())
    assert none is None and set(alone) == {"live", "ours_alone"}  # no price: no book, no stack
    assert alone["live"][2] == alone["ours_alone"][2]


def test_starting_fives_move_minutes_and_stats_with_them_and_inactives_go_to_zero():
    g = inputs(
        fives=("Confirmed", "Confirmed"),
        starters=frozenset({1, 3}),
        out_nba=frozenset({4}),
        lines={1: (OKC, LINE), 2: (OKC, LINE), 3: (OKC, LINE), 4: (MIN_, LINE), 5: (OKC, LINE)},
        share={1: 0.0, 2: 0.78, 3: 0.95, 4: 1.0, 5: 0.0},
    )
    moved = tunings.moved_lines(g)
    assert moved["1"] == [OKC, 1, 0.0, 28.6, 14.3, 4.8, 2.4, 1.2]  # bench player starts: +4.6 min
    assert moved["2"][:4] == [OKC, 0, 0.78, 20.4]  # usual starter on the bench
    assert "3" not in moved and "5" not in moved  # +0.2 min: stays Live's; 5 is on the bench as usual
    assert moved["4"] == [MIN_, 0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # nba.com lists him inactive

    unlisted = tunings.moved_lines(inputs(lines={1: (OKC, LINE)}, share={1: 0.0}))
    assert unlisted == {}  # no five from nba.com yet: nothing moves


def test_a_look_after_tip_off_keeps_the_last_one_before_it():
    before = tunings.update(None, [(9, inputs(book=BOOK))], TIP - timedelta(minutes=20))
    g = before["games"]["9"]
    assert g["checked"] == (TIP - timedelta(minutes=20)).isoformat(timespec="seconds")
    assert set(g["calls"]) == {"live", "ours_alone", "book"} and g["stack"] is None
    after = tunings.update(before, [(9, inputs(live_call=(90.0, 140.0)))], TIP + timedelta(minutes=1))
    assert after == before


def test_settling_scores_each_tuning_and_leads_only_on_a_clear_edge():
    game = {"calls": {"live": [111.0, 109.0, 0.55], "ours_alone": [111.0, 109.0, 0.44], "book": [110, 112.5, 0.56]}}
    s = tunings.settle_game(game, (100.0, 120.0))  # OKC by 20
    assert s["live"]["hit"] and not s["ours_alone"]["hit"]
    assert s["live"]["margin_miss"] == 22.0 and s["book"]["total_miss"] == pytest.approx(2.5)
    assert s["ours_alone"]["loss"] > s["live"]["loss"]

    sharp = {"live": {**s["live"]}, "book": {**s["book"], "loss": s["live"]["loss"] - 0.1}}
    jitter = {"live": {**s["live"]}, "book": {**s["book"], "loss": s["live"]["loss"] - 0.12}}
    rows = tunings.summary([sharp, jitter, sharp], [])
    book = next(r for r in rows if r["tuning"] == "book")
    assert book["games"] == 3 and book["edge"] > 0 and book["leading"] is True
    lucky = {"live": {**s["live"]}, "book": {**s["book"], "loss": s["live"]["loss"] + 0.4}}
    book = next(r for r in tunings.summary([sharp, lucky, sharp], []) if r["tuning"] == "book")
    assert book["leading"] is False  # the range crosses zero
    assert next(r for r in rows if r["tuning"] == "rivalry") == {"tuning": "rivalry", "games": 0, "leading": False}


def test_moved_player_lines_are_scored_against_lives_on_the_same_players():
    game = {"players": {"1": [OKC, 1, 0.0, 28.6, 14.3, 4.8, 2.4, 1.2], "2": [OKC, 0, 0.8, 20.4, 10.2, 3.4, 1.7, 0.9]}}
    live = {"1": [OKC, 24.0, 12.0, 4.0, 2.0, 0, 0, 1.0, 0, 0, 0, 0, 0], "2": [OKC, 24.0, 12.0, 4.0, 2.0, 0, 0, 1.0]}
    results = {"1": [30.0, 15.0, 5.0, 2.0, 0, 0, 1.0, 0, 0, 0, 0, 0], "2": []}  # 2 sat out
    rows = tunings.settle_players(game, live, results)
    one = next(r for r in rows if r["player"] == 1)
    assert one["tuned"][0] == pytest.approx(1.4) and one["live"][0] == pytest.approx(6.0)
    two = next(r for r in rows if r["player"] == 2)
    assert two["live"][:2] == [24.0, 12.0]
    got = tunings.summary([], rows)
    assert next(r for r in got if r["tuning"] == "ours_alone")["players"]["lines"] == 2


def test_nba_names_match_espns_players_through_accents_and_teams():
    names = tunings.name_index([(1, OKC, "Nikola Topic"), (2, OKC, "Jalen Williams"), (3, 9, "Jalen Williams")])
    assert tunings.match_player("Nikola Topić", OKC, names) == 1
    assert tunings.match_player("Jalen Williams", OKC, names) == 2
    assert tunings.match_player("Jalen Williams", 1, names) is None  # two of them, neither on this team
    assert tunings.match_player("Nobody", OKC, names) is None


def test_the_watch_records_tunings_before_tip_off_and_admins_read_them_scored(recorded):
    app, store, sched = _app(recorded)
    app.state.line_log_league_id = LEAGUE_ID
    day = min(d for d, gs in sched.games.items() if gs and d >= 3)
    game = sched.games[day][0]
    app.state.now = lambda: game.start - timedelta(minutes=45)
    store.save_book_lines(2027, day, {"games": {str(game.id): BOOK}})
    assert asyncio.run(app.state.tuning_sweep()) == [day]
    rec = store.get_tunings(2027, day)["games"][str(game.id)]
    assert set(rec["calls"]) == {"live", "ours_alone", "book"}
    assert rec["fives"] == [None, None] and rec["players"] == {}  # nba.com not read yet: nothing moves

    assert signed_in(app, team_id=None).post("/api/admin/tunings", json={"season": 2027}).status_code == 403
    admin = boss(app)
    got = admin.post("/api/admin/tunings", json={"season": 2027}).json()
    assert got["days"] == [day] and got["settled"] == 0
    assert any(g["game_id"] == game.id for g in got["games"])

    box = app.state.game_boxes.get(2027)
    stats = dict.fromkeys(gamescore.FIELDS, 0.0)
    box.team_games.extend(
        [
            gamescore.TeamGame(game.id, day, game.away, game.home, stats | {"pf": 100.0, "pa": 120.0}),
            gamescore.TeamGame(game.id, day, game.home, game.away, stats | {"pf": 120.0, "pa": 100.0}),
        ]
    )
    got = admin.post("/api/admin/tunings", json={"season": 2027}).json()
    assert got["settled"] == 1
    scores = {s["tuning"]: s for s in got["scores"]}
    assert scores["live"]["games"] == 1 and scores["book"]["hits"] == 1  # the book had the home side
    played = next(g for g in got["games"] if g["game_id"] == game.id)
    assert played["final"] == [100.0, 120.0] and all(c["hit"] is not None for c in played["calls"])

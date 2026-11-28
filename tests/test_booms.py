from datetime import UTC, datetime, time

from app import booms, linelog
from app.analysis.gamebox import Form
from tests.test_linelog import _app
from tests.test_news import boss

STEADY = ((30.0, 20.0, 6.0),) * 10  # minutes, points, rebounds
WILD = tuple((m, p, r) for m, p, r in [(22, 8, 4), (38, 34, 13), (25, 12, 5), (36, 31, 11), (20, 9, 3)] * 2)
LINE = {"minutes": 30.0, "pts": 20.0, "reb": 6.0}


def form(last, team=1) -> Form:
    return Form(rate={}, minutes=30.0, minutes_long=30.0, usage=15.0, last_day=30, team=team, last=tuple(last))


def test_a_boom_is_far_over_the_line_and_short_lines_dont_boom():
    assert booms.bar("pts", 20.0) == 30  # 1.5x
    assert booms.bar("pts", 10.0) == 18  # 8 over beats 1.5x on a small line
    assert booms.bar("pts", 7.9) is None
    assert booms.bar("reb", 10.0) == 16  # 1.6x
    assert booms.bar("reb", 4.0) == 9  # 5 over
    assert booms.bar("reb", 2.5) is None
    assert booms.bar("ast", 9.0) is None  # assists and threes found nothing in the replay


def test_wild_swings_and_a_hot_hand_raise_the_chance_and_say_why():
    steady, steady_why = booms.crazy_chance("pts", 0.1, booms.measures("pts", LINE, STEADY))
    wild, wild_why = booms.crazy_chance("pts", 0.1, booms.measures("pts", LINE, WILD))
    assert steady < 0.1 and steady_why == []  # nothing above the five seasons' average: nothing believed
    assert wild > 0.1
    assert [b[0] for b in wild_why] == ["minutes", "swings"]  # biggest push first
    assert all(b[2] > 0 for b in wild_why)
    hot = WILD[:-3] + ((36, 30, 6),) * 3
    _, why = booms.crazy_chance("pts", 0.1, booms.measures("pts", LINE, hot))
    assert "hot" in [b[0] for b in why] and dict((b[0], b[1]) for b in why)["hot"] == 30.0


def test_rebound_booms_believe_he_has_done_it_lately():
    big = STEADY[:-1] + ((30, 20, 17),)
    _, why = booms.crazy_chance("reb", 0.1, booms.measures("reb", LINE, big))
    assert ["done_it", 17.0] == why[0][:2]


def test_a_game_calls_its_top_five_per_stat_and_skips_the_out_and_the_new():
    lines = {pid: (1, dict(LINE)) for pid in range(1, 9)}
    lines[9] = (1, {"minutes": 10.0, "pts": 9.0, "reb": 3.0})  # short line
    form_of = {pid: form(WILD if pid % 2 else STEADY) for pid in range(1, 10)}
    form_of[8] = form(WILD[:4])  # four games: too few to swing
    calls = booms.game_calls(lines, form_of, {1: "Wild One"}, out={3})
    pts = [c for c in calls if c[3] == "pts"]
    assert len(pts) == 5 and len([c for c in calls if c[3] == "reb"]) == 5
    picked = {c[0] for c in pts}
    assert {1, 5, 7} <= picked and not {3, 8, 9} & picked  # the wild ones first; out, new and short skipped
    first = pts[0]
    assert first[2] in ("Wild One", "") and first[5] == 30 and first[7] > first[6]


def test_a_night_picks_its_top_five_over_all_games_and_its_record_wants_a_clear_edge():
    def call(pid, stat, crazy):
        return [pid, 1, "", stat, 20.0, 30, crazy - 0.02, crazy, []]

    night = [(10, {"booms": [call(i, "pts", 0.1 + i / 100) for i in range(5)]}), (11, {"booms": [call(9, "pts", 0.5)]})]
    picked = booms.night_picks(night)
    assert [c[0] for _, c in picked] == [9, 4, 3, 2, 1] and picked[0][0] == 11
    index = {name: i for i, (name, _) in enumerate(linelog.STATS)}
    got = [0.0] * len(linelog.STATS)
    got[index["pts"]] = 31.0
    assert booms.outcome(call(9, "pts", 0.5), {"9": got}, index) == (31.0, True)
    assert booms.outcome(call(9, "pts", 0.5), {"9": []}, index) == (None, None)  # didn't play
    assert booms.outcome(call(9, "pts", 0.5), {}, index) == (None, None)  # not settled yet
    early = booms.record([[(0.1, True)] + [(0.1, False)] * 4])
    assert early["picks"] == 5 and early["caught"] == 1 and early["edge"] is None and not early["leading"]
    lucky = booms.record([[(0.1, True), (0.1, True)] + [(0.1, False)] * 3] * 6)
    assert lucky["edge"] == 1.5 and lucky["leading"]


def test_crazy_ones_world_shows_the_nights_boom_calls_and_counts_the_settled_ones(recorded):
    app, store, sched = _app(recorded)
    evening = datetime.combine(sched.date_of(31), time(23), UTC)  # 6 pm in New York: night 31 is tonight
    app.state.now = lambda: evening
    index = {name: i for i, (name, _) in enumerate(linelog.STATS)}

    def result(pts, reb):
        row = [0.0] * len(linelog.STATS)
        row[index["pts"]], row[index["reb"]] = pts, reb
        return row

    def game(*calls):
        return {"away": 16, "home": 25, "tip": "2026-11-25T00:30:00+00:00", "booms": [list(c) for c in calls]}

    hot = [["hot", 29.0, 0.02], ["swings", 9.1, 0.01]]
    store.save_tunings(2027, 30, {"games": {"1": game([7, 25, "Shai", "pts", 30.0, 45, 0.1, 0.12, hot])}})
    store.save_player_lines(2027, 30, {"lines": {}, "results": {"7": result(47, 5)}}, True)
    store.save_tunings(
        2027,
        31,
        {
            "games": {
                "2": game(
                    [8, 16, "Ant", "pts", 26.0, 39, 0.1, 0.13, hot], [9, 16, "Rudy", "reb", 11.0, 18, 0.1, 0.11, []]
                )
            }
        },
    )
    client = boss(app)
    got = client.post("/api/crazy-one", json={"season": 2027}).json()
    assert got["day"] == 31
    ant, rudy = got["booms"]
    assert ant["name"] == "Ant" and ant["team"] == "MIN" and ant["opponent"] == "OKC" and not ant["home"]
    assert ant["beliefs"][0] == {"belief": "hot", "value": 29.0, "push": 0.02} and ant["boom"] is None
    assert rudy["stat"] == "reb" and rudy["beliefs"] == []
    rec = got["boom_record"]
    assert rec["nights"] == 1 and rec["picks"] == 1 and rec["caught"] == 1 and rec["expected"] == 0.1
    assert rec["by_night"][0]["picks"][0] == {
        "player": 7,
        "name": "Shai",
        "team": "OKC",
        "stat": "pts",
        "ours": 0.1,
        "boom": True,
    }
    past = client.post("/api/crazy-one", json={"season": 2027, "day": 30}).json()
    assert past["booms"][0]["got"] == 47 and past["booms"][0]["boom"] is True

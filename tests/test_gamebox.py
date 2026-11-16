import copy

import pytest

from app.analysis import gamebox
from app.analysis.gamebox import (
    AST,
    FGA,
    FGM,
    MINUTES,
    PTS,
    REB,
    STL,
    TO,
    Form,
    GameContext,
    GameRow,
    Teammate,
)
from app.analysis.report import Analyzer

COUNTING = gamebox.COUNTING


def _line(minutes: float, **stats: float) -> dict[int, float]:
    ids = {"pts": PTS, "reb": REB, "ast": AST, "stl": STL, "to": TO, "fgm": FGM, "fga": FGA}
    out = dict.fromkeys(COUNTING, 0.0)
    out[MINUTES] = minutes
    for k, v in stats.items():
        out[ids[k]] = v
    return out


def _rows() -> list[GameRow]:
    """Team 1 and 2 play three times. Players 11 and 12 (team 1) score 20 and 0; player 21 (team 2),
    usually 10 in 30 minutes, scores 20 against team 1 every time, and 22 his usual 5: team 1's
    defense gives up points."""
    rows = []
    for game, day in ((101, 1), (102, 3), (103, 4)):
        rows.append(GameRow(11, game, day, 1, _line(30, pts=20, fga=16, fgm=8)))
        rows.append(GameRow(12, game, day, 1, _line(10)))
        rows.append(GameRow(21, game, day, 2, _line(30, pts=20, fga=16, fgm=8)))
        rows.append(GameRow(22, game, day, 2, _line(30, pts=5, fga=6, fgm=2)))
    return rows


def _rate(pts: float) -> dict[int, float]:
    return {k: (pts / 30 if k == PTS else 0.0) for k in COUNTING}


PRIOR = {21: _rate(10), 22: _rate(5), 11: _rate(20), 12: _rate(0)}
SCHEDULE = {
    1: [gamebox.Game(1, 2, True), gamebox.Game(3, 2, False), gamebox.Game(4, 2, True), gamebox.Game(5, 2, True)],
    2: [gamebox.Game(1, 1, False), gamebox.Game(3, 1, True), gamebox.Game(4, 1, False), gamebox.Game(5, 1, False)],
}


def test_defense_is_what_opponents_make_over_what_they_usually_make():
    box = gamebox.build(2027, _rows(), PRIOR, SCHEDULE)
    # Team 1 let player 21 score twice his usual: its points defense reads above zero, shrunk to the league.
    assert box.defense[1][PTS] > 0.05
    # Team 2 let players 11 and 12 make what they usually do.
    assert box.defense[2][PTS] == pytest.approx(0.0, abs=0.02)


def test_margin_form_and_games_missed():
    box = gamebox.build(2027, _rows(), PRIOR, SCHEDULE)
    # Team 1 scored 20 to 25 three times: -5 a game over 3 games + 8 even ones.
    assert box.margin[1] == pytest.approx(-15 / 11)
    assert box.margin[2] == pytest.approx(15 / 11)
    f = box.form[11]
    assert f.rate[PTS] == pytest.approx(20 / 30)
    assert f.minutes == 30
    assert f.usage == pytest.approx(16)
    assert f.last_day == 4
    # He played every game: none missed before day 5; had he stopped after day 1, two by day 5.
    assert box.games_missed(11, 1, 5) == 0
    short = gamebox.build(2027, [r for r in _rows() if not (r.player == 11 and r.day > 1)], PRIOR, SCHEDULE)
    assert short.games_missed(11, 1, 5) == 2


def test_context_finds_the_next_game():
    box = gamebox.build(2027, _rows(), PRIOR, SCHEDULE)
    ctx = gamebox.context(box, 11, 1, 5)
    assert ctx is not None
    assert (ctx.day, ctx.opponent, ctx.home, ctx.back_to_back) == (5, 2, True, True)
    assert ctx.margin == pytest.approx(-30 / 11 + gamebox.HOME_EDGE)
    assert ctx.defense == box.defense[2]
    assert gamebox.context(box, 11, 1, 6) is None


def _ctx(**kw) -> GameContext:
    base = dict(day=5, opponent=2, home=False, back_to_back=False, margin=0.0, defense={}, form=None, games_missed=0)
    return GameContext(**{**base, **kw})


LINE = _line(30.0, pts=20.0, reb=5.0, ast=4.0, stl=1.0, to=2.0, fgm=8.0, fga=16.0)


def test_a_plain_night_is_his_usual_line():
    new, usual, reasons = gamebox.apply(LINE, _ctx(), [])
    assert new[MINUTES] == pytest.approx(30.0 + gamebox.MINUTES_LEVEL)
    # Away: only the home-court terms (assists, FG%) move it.
    assert new[PTS] == pytest.approx(usual[PTS])
    assert new[AST] < usual[AST]
    assert {r.kind for r in reasons} == {"away"}


def test_a_soft_defense_adds_points():
    new, usual, reasons = gamebox.apply(LINE, _ctx(defense={PTS: 0.10}), [])
    base = 20.0 / 30.0 * (30.0 + gamebox.MINUTES_LEVEL)
    assert new[PTS] - usual[PTS] == pytest.approx(base * gamebox.RATE_WEIGHTS[PTS]["opp"] * 0.10)
    opp = next(r for r in reasons if r.kind == "opponent")
    assert opp.stats[PTS] > 0
    assert opp.detail["team"] == 2


def test_teammates_out_and_a_return_from_absence():
    out = [Teammate("Jaylen Brown", 34.0, 22.0)]
    new, usual, reasons = gamebox.apply(LINE, _ctx(games_missed=3), out, gained=2.0)
    kinds = {r.kind: r for r in reasons}
    # Three games missed: about 1.8 minutes fewer in his first game back.
    assert kinds["back_from_absence"].minutes == pytest.approx(3 * gamebox.MISSED_MINUTES)
    # The teammate's minutes (2 from tonight.py, more for a 30-minute player) and his shots.
    assert kinds["teammates_out"].minutes == pytest.approx(2.0 + gamebox.BENCH_SHARE * 34.0 * (1 - 30 / 36))
    assert kinds["teammates_out"].stats[PTS] > 0
    assert kinds["teammates_out"].detail["names"] == "Jaylen Brown"
    assert usual[MINUTES] == pytest.approx(28.0 + gamebox.MINUTES_LEVEL)


def test_his_last_games_carry_about_a_third():
    hot = Form(rate={**{k: v / 30 for k, v in LINE.items()}, PTS: 26 / 30}, minutes=30, minutes_long=30,
               usage=20, last_day=4, team=1)  # fmt: skip
    new, usual, _ = gamebox.apply(LINE, _ctx(form=hot), [])
    base = 20.0 / 30.0 * (30.0 + gamebox.MINUTES_LEVEL)
    assert new[PTS] - usual[PTS] == pytest.approx(base * gamebox.RATE_WEIGHTS[PTS]["form"] * 0.3)


def test_parse_rows_and_schedule():
    raw = [
        {
            "player": {
                "id": 7,
                "proTeamId": 2,
                "stats": [
                    {"statSplitTypeId": 5, "statSourceId": 0, "externalId": "401", "seasonId": 2027,
                     "scoringPeriodId": 3, "proTeamId": 2, "stats": {"0": 12, "40": 25}},
                    {"statSplitTypeId": 5, "statSourceId": 0, "externalId": "402", "seasonId": 2027,
                     "scoringPeriodId": 4, "proTeamId": 2, "stats": {"40": 0}},
                    {"statSplitTypeId": 0, "statSourceId": 0, "seasonId": 2027, "stats": {"0": 12}},
                ],
            }
        }
    ]  # fmt: skip
    rows = gamebox.parse_rows(raw, 2027)
    assert [(r.player, r.game, r.day, r.team, r.line[PTS]) for r in rows] == [(7, 401, 3, 2, 12.0)]
    sched = {
        "settings": {
            "proTeams": [
                {"id": 1, "proGamesByScoringPeriod": {"3": [{"homeProTeamId": 1, "awayProTeamId": 2}]}},
                {"id": 2, "proGamesByScoringPeriod": {"3": [{"homeProTeamId": 1, "awayProTeamId": 2}]}},
            ]
        }
    }
    games = gamebox.parse_schedule(sched)
    assert games[1] == [gamebox.Game(3, 2, True)]
    assert games[2] == [gamebox.Game(3, 1, False)]


def test_the_next_game_line_says_why(league):
    lg = copy.deepcopy(league)
    p = lg.team(3).roster[0]
    p.per_game = {**p.per_game, MINUTES: 30.0, PTS: 20.0, REB: 5.0, AST: 4.0, FGM: 8.0, FGA: 16.0}
    p.minute_shifts = ()
    p.recent_minutes = (None, None)
    p.injury_status = "ACTIVE"
    plain = Analyzer(lg, 3).ref(p).next_game
    assert plain.reasons == [] and plain.usual is None and plain.opponent is None
    p.game_context = _ctx(opponent=21, defense={PTS: 0.08}, home=True)
    ng = Analyzer(lg, 3).ref(p).next_game
    assert ng.opponent == "PHX" and ng.home is True
    assert ng.usual is not None and ng.pts > ng.usual.pts
    top = ng.reasons[0]
    assert top.kind == "opponent"
    assert top.text == "PHX allows 8% more points than usual"
    assert top.pts > 0 and top.value > 0
    assert [abs(r.value) for r in ng.reasons] == sorted((abs(r.value) for r in ng.reasons), reverse=True)

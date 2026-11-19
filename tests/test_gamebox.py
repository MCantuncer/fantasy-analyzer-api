import copy

import pytest

from app.analysis import gamebox
from app.analysis.gamebox import (
    AST,
    FGA,
    FGM,
    FTA,
    FTM,
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
from app.analysis.report import Analyzer, _reason_text

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


def test_context_knows_a_new_team_and_the_games_left():
    box = gamebox.build(2027, _rows(), PRIOR, SCHEDULE)
    # Day 5 is team 1's last game. Player 11 has played for it all along; 21 only for team 2.
    stayed = gamebox.context(box, 11, 1, 5)
    assert (stayed.new_team, stayed.games_left) == (False, 1)
    assert gamebox.context(box, 11, 1, 3).games_left == 3
    moved = gamebox.context(box, 21, 1, 5)
    assert moved.new_team and moved.form is None
    # A player with no games this season isn't new to anyone.
    assert not gamebox.context(box, 99, 1, 5).new_team


def test_a_first_game_for_a_new_team_plays_fewer_minutes():
    new, usual, reasons = gamebox.apply(LINE, _ctx(new_team=True), [])
    assert new[MINUTES] == pytest.approx(30.0 + gamebox.MINUTES_LEVEL + gamebox.NEW_TEAM_MINUTES)
    moved = next(r for r in reasons if r.kind == "new_team")
    assert moved.minutes == pytest.approx(gamebox.NEW_TEAM_MINUTES)
    assert moved.stats[PTS] == pytest.approx(20.0 / 30.0 * gamebox.NEW_TEAM_MINUTES)
    assert usual[MINUTES] == pytest.approx(30.0 + gamebox.MINUTES_LEVEL)


def test_core_players_rest_in_the_last_games_of_the_season():
    core = _line(34.0, pts=24.0)
    new, _, reasons = gamebox.apply(core, _ctx(games_left=gamebox.SEASON_END_GAMES), [])
    heavy = gamebox.HEAVY_LEVEL * max(0.0, 34.0 - gamebox.HEAVY_MINUTES)
    assert new[MINUTES] == pytest.approx(34.0 + gamebox.MINUTES_LEVEL + heavy + gamebox.SEASON_END_MINUTES)
    assert next(r for r in reasons if r.kind == "season_end").minutes == pytest.approx(gamebox.SEASON_END_MINUTES)
    # Not earlier in the season, not for a bench player, and not when the games left aren't known.
    for line, ctx in (
        (core, _ctx(games_left=gamebox.SEASON_END_GAMES + 1)),
        (_line(gamebox.CORE_MINUTES - 1, pts=10.0), _ctx(games_left=1)),
        (core, _ctx()),
    ):
        assert all(r.kind != "season_end" for r in gamebox.apply(line, ctx, [])[2])


def test_context_knows_how_long_ago_his_coach_sat_him():
    sat = {11: [2, 4, 5], 12: [9]}  # day 5 is the game day itself, day 9 is still to come
    box = gamebox.build(2027, _rows(), PRIOR, SCHEDULE, sat=sat)
    assert gamebox.context(box, 11, 1, 5).sat_ago == 1
    assert gamebox.context(box, 12, 1, 5).sat_ago is None
    assert gamebox.context(gamebox.build(2027, _rows(), PRIOR, SCHEDULE), 11, 1, 5).sat_ago is None


def test_a_player_his_coach_just_sat_plays_fewer_minutes():
    new, usual, reasons = gamebox.apply(LINE, _ctx(sat_ago=gamebox.COACH_SAT_DAYS), [])
    assert new[MINUTES] == pytest.approx(30.0 + gamebox.MINUTES_LEVEL + gamebox.COACH_SAT_MINUTES)
    sat = next(r for r in reasons if r.kind == "coach_decision")
    assert sat.minutes == pytest.approx(gamebox.COACH_SAT_MINUTES) and sat.detail["days"] == gamebox.COACH_SAT_DAYS
    assert sat.stats[PTS] == pytest.approx(20.0 / 30.0 * gamebox.COACH_SAT_MINUTES)
    assert usual[MINUTES] == pytest.approx(30.0 + gamebox.MINUTES_LEVEL)
    # Not once it is over COACH_SAT_DAYS old, and not for a line too short to lose from.
    short = _line(gamebox.COACH_SAT_LINE - 1, pts=4.0)
    for line, ctx in ((LINE, _ctx(sat_ago=gamebox.COACH_SAT_DAYS + 1)), (short, _ctx(sat_ago=1)), (LINE, _ctx())):
        assert all(r.kind != "coach_decision" for r in gamebox.apply(line, ctx, [])[2])


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


def _form(**kw) -> Form:
    base = dict(rate={k: v / 30 for k, v in LINE.items()}, minutes=30.0, minutes_long=30.0, usage=20.0, last_day=4,
                team=1)  # fmt: skip
    return Form(**{**base, **kw})


def test_recent_form_is_decayed_and_shrunk_by_what_his_line_expects():
    # 26 points over 30 (decayed) minutes against a line of 20: a 30% gap, trusted by E / (E + shrink).
    hot = _form(decayed={PTS: (26.0, 30.0), STL: (2.0, 30.0)})
    new, usual, reasons = gamebox.apply(LINE, _ctx(form=hot), [])
    minutes = 30.0 + gamebox.MINUTES_LEVEL + gamebox.RECENT_MINUTES * 0.0
    base = 20.0 / 30.0 * minutes
    shrink = gamebox.FORM_DECAY[PTS][1]
    assert new[PTS] - usual[PTS] == pytest.approx(base * gamebox.RATE_WEIGHTS[PTS]["form"] * 0.3 * 20 / (20 + shrink))
    # Steals: line 1, he made 2: the gap is 100%, but one expected steal is little to go on.
    stl = gamebox._form(hot, STL, 1.0 / 30.0)
    assert stl == pytest.approx(1.0 * 1 / (1 + gamebox.FORM_DECAY[STL][1]))
    assert next(r for r in reasons if r.kind == "form").stats[PTS] > 0
    # Without decayed sums (no games), no form.
    assert gamebox._form(_form(), PTS, 20 / 30) == 0.0


def test_decayed_sums_weigh_recent_games_more():
    rows = [GameRow(11, 100 + d, d, 1, _line(30, pts=p)) for d, p in ((1, 10.0), (2, 20.0), (3, 30.0))]
    box = gamebox.build(2027, rows, PRIOR, SCHEDULE)
    h = gamebox.FORM_DECAY[PTS][0]
    w = [0.5 ** (2 / h), 0.5 ** (1 / h), 1.0]
    pts, minutes = box.form[11].decayed[PTS]
    assert pts == pytest.approx(10 * w[0] + 20 * w[1] + 30 * w[2])
    assert minutes == pytest.approx(30 * sum(w))
    # Minutes lately: decayed over this season's games.
    assert box.form[11].minutes == pytest.approx(30.0)
    assert box.form[11].last_minutes == 30.0


def test_form_runs_across_the_summer():
    last = [GameRow(11, 900 + d, d, 1, _line(30, pts=30.0)) for d in range(1, 21)]
    rows = [GameRow(11, 101, 1, 1, _line(30, pts=10.0))]
    carried = gamebox.carry(last)
    box = gamebox.build(2027, rows, PRIOR, SCHEDULE, carried)
    h = gamebox.FORM_DECAY[PTS][0]
    w_last = sum(0.5 ** (k / h) for k in range(20)) * 0.5 ** (1 / h)  # one game of this season since
    pts, minutes = box.form[11].decayed[PTS]
    assert pts == pytest.approx(10 + 30 * w_last)
    assert minutes == pytest.approx(30 + 30 * w_last)
    # Without last season, only his one game.
    assert gamebox.build(2027, rows, PRIOR, SCHEDULE).form[11].decayed[PTS] == pytest.approx((10.0, 30.0))
    # The longer minutes run counts last season too; minutes lately don't.
    long = [GameRow(11, 900 + d, d, 1, _line(20, pts=10.0)) for d in range(1, 21)]
    box = gamebox.build(2027, rows, PRIOR, SCHEDULE, gamebox.carry(long))
    assert 20 < box.form[11].minutes_season < 30
    assert box.form[11].minutes == pytest.approx(30.0)


def test_a_new_starter_plays_more_and_a_benched_one_less():
    promoted = _form(started=True, start_share=0.25)
    new, usual, reasons = gamebox.apply(LINE, _ctx(form=promoted), [])
    switch = next(r for r in reasons if r.kind == "starter_switch")
    assert switch.minutes == pytest.approx(gamebox.SWITCH_MINUTES * 0.75)
    # His usual line holds his usual role (started a quarter of his games), not the switch.
    assert usual[MINUTES] == pytest.approx(30.0 + gamebox.MINUTES_LEVEL + gamebox.ROLE_MINUTES * (0.25 - 0.5))
    benched = _form(started=False, start_share=1.0)
    _, _, reasons = gamebox.apply(LINE, _ctx(form=benched), [])
    assert next(r for r in reasons if r.kind == "starter_switch").minutes < 0


def test_reason_texts():
    def text(kind, **detail):
        return _reason_text(gamebox.Reason(kind, detail=detail), _ctx())

    assert text("form") == "His recent form"
    assert text("starter_switch", started=1) == "Started his last game"
    assert text("starter_switch", started=0) == "Moved to the bench"
    assert text("new_team") == "First game for his new team"
    assert text("season_end") == "Last games of the season"
    assert text("coach_decision", days=1) == "Coach sat him yesterday"
    assert text("coach_decision", days=3) == "Coach sat him 3 days ago"
    assert text("back_from_absence", games=0, game_back=2) == "Second game back"
    assert text("back_from_absence", games=3, game_back=1) == "First game back after 3 missed"
    assert text("teammates_out", names="Jaylen Brown", points=24.6) == "Teammates out: 25 points a game to share"
    assert text("teammates_out", names="Jaylen Brown", points=0.0) == "Brown out"


def _ramp_rows(days: tuple[int, ...]) -> list[GameRow]:
    """Teams 1 and 2 play days 1-6; player 11 (team 1) plays ``days``; player 12 every game."""
    rows = []
    for day in range(1, 7):
        rows.append(GameRow(12, 200 + day, day, 1, _line(30, pts=10)))
        rows.append(GameRow(21, 200 + day, day, 2, _line(30, pts=10)))
        if day in days:
            rows.append(GameRow(11, 200 + day, day, 1, _line(30, pts=20)))
    return rows


RAMP_SCHEDULE = {
    1: [gamebox.Game(d, 2, True) for d in range(1, 8)],
    2: [gamebox.Game(d, 1, False) for d in range(1, 8)],
}


def test_minutes_come_back_over_three_games():
    # Missed days 2 and 3, back on day 4.
    first = gamebox.build(2027, [r for r in _ramp_rows((1, 4, 5)) if r.day < 4], PRIOR, RAMP_SCHEDULE)
    assert gamebox.context(first, 11, 1, 4).game_back == 1
    second = gamebox.build(2027, [r for r in _ramp_rows((1, 4, 5)) if r.day < 5], PRIOR, RAMP_SCHEDULE)
    assert second.form[11].since_return == 1
    ctx = gamebox.context(second, 11, 1, 5)
    assert (ctx.games_missed, ctx.game_back) == (0, 2)
    third = gamebox.build(2027, _ramp_rows((1, 4, 5)), PRIOR, RAMP_SCHEDULE)
    assert gamebox.context(third, 11, 1, 6).game_back == 3
    after = gamebox.build(2027, _ramp_rows((1, 4, 5, 6)), PRIOR, RAMP_SCHEDULE)
    assert gamebox.context(after, 11, 1, 7).game_back == 0
    # One game missed starts no ramp.
    once = gamebox.build(2027, _ramp_rows((1, 2, 4)), PRIOR, RAMP_SCHEDULE)
    assert gamebox.context(once, 11, 1, 5).game_back == 0
    _, _, reasons = gamebox.apply(LINE, _ctx(game_back=2), [])
    back = next(r for r in reasons if r.kind == "back_from_absence")
    assert back.minutes == pytest.approx(gamebox.RAMP_MINUTES * gamebox.RAMP[2])
    assert back.detail["game_back"] == 2


def test_points_of_teammates_out_go_to_those_left():
    out = [Teammate("Jaylen Brown", 34.0, 22.0, 25.0)]
    plain = gamebox.apply(LINE, _ctx(team_points=100.0), [Teammate("Jaylen Brown", 34.0, 22.0)])[0]
    new, _, reasons = gamebox.apply(LINE, _ctx(team_points=100.0), out)
    base = 20.0 / 30.0 * new[MINUTES]
    assert new[PTS] - plain[PTS] == pytest.approx(base * gamebox.RATE_WEIGHTS[PTS]["vac"] * 0.25)
    assert next(r for r in reasons if r.kind == "teammates_out").detail["points"] == 25.0


def test_a_regular_coming_back_takes_minutes_and_shots_from_the_bench():
    back = [Teammate("Jaylen Brown", 34.0, 22.0, 25.0, 2)]
    plain = gamebox.apply(LINE, _ctx(), [])[0]
    new, usual, reasons = gamebox.apply(LINE, _ctx(), [], 0.0, back)
    moved = gamebox.BACK_SHARE * 34.0 * (1 - 30.0 / gamebox.BENCH_MINUTES)
    assert new[MINUTES] - plain[MINUTES] == pytest.approx(moved)
    base = 20.0 / 30.0 * new[MINUTES]
    shots = base * gamebox.RATE_WEIGHTS[PTS]["usage_back"] * 0.22
    assert new[PTS] - plain[PTS] == pytest.approx(20.0 / 30.0 * moved + shots)
    reason = next(r for r in reasons if r.kind == "teammates_back")
    assert reason.minutes == pytest.approx(moved) and reason.detail["names"] == "Jaylen Brown"
    assert _reason_text(reason, _ctx()) == "Brown back"
    assert usual[MINUTES] == pytest.approx(plain[MINUTES])
    # A starter at 36 minutes or more has no bench minutes to give back.
    star = _line(36.0, pts=27.0)
    assert gamebox.minute_signals(36.0, _ctx(), [], back)["bench_back"] == 0.0
    assert all(r.minutes == 0 for r in gamebox.apply(star, _ctx(), [], 0.0, back)[2] if r.kind == "teammates_back")


def test_an_absence_already_a_game_old_gives_fewer_new_minutes():
    fresh = gamebox.apply(LINE, _ctx(), [Teammate("Jaylen Brown", 34.0, 22.0)])[0]
    old = gamebox.apply(LINE, _ctx(), [Teammate("Jaylen Brown", 34.0, 22.0, 0.0, 2)])[0]
    bench = 1 - 30.0 / gamebox.BENCH_MINUTES
    assert old[MINUTES] - fresh[MINUTES] == pytest.approx(gamebox.OLD_OUT_SHARE * 34.0 * bench)
    assert old[MINUTES] > gamebox.apply(LINE, _ctx(), [])[0][MINUTES]


def test_free_throws_follow_the_opponents_fouls():
    line = {**LINE, FTA: 6.0, FTM: 4.8}
    soft = gamebox.apply(line, _ctx(defense={FTA: 0.2}), [])[0]
    plain = gamebox.apply(line, _ctx(), [])[0]
    assert soft[FTA] > plain[FTA]
    assert soft[FTM] / soft[FTA] == pytest.approx(0.8)


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
                {"id": 1, "proGamesByScoringPeriod": {"3": [{"homeProTeamId": 1, "awayProTeamId": 2, "date": 5}]}},
                {"id": 2, "proGamesByScoringPeriod": {"3": [{"homeProTeamId": 1, "awayProTeamId": 2, "date": 5}]}},
            ]
        }
    }
    games = gamebox.parse_schedule(sched)
    assert games[1] == [gamebox.Game(3, 2, True, 5)]
    assert games[2] == [gamebox.Game(3, 1, False, 5)]


def test_the_next_game_line_says_why(league):
    lg = copy.deepcopy(league)
    p = lg.team(3).roster[0]
    p.per_game = {**p.per_game, MINUTES: 30.0, PTS: 20.0, REB: 5.0, AST: 4.0, FGM: 8.0, FGA: 16.0}
    p.minute_shifts = ()
    p.recent_minutes = (None, None)
    p.injury_status = "ACTIVE"
    plain = Analyzer(lg, 3).ref(p).next_game
    assert plain.reasons == [] and plain.usual is None and plain.opponent is None
    p.game_context = _ctx(opponent=21, defense={PTS: 0.08}, home=True, tip=1763335800000)
    ng = Analyzer(lg, 3).ref(p).next_game
    assert ng.opponent == "PHX" and ng.home is True and ng.tip == "2025-11-16T23:30:00Z"
    assert ng.usual is not None and ng.pts > ng.usual.pts
    top = ng.reasons[0]
    assert top.kind == "opponent"
    assert top.text == "Against PHX"
    assert top.pts > 0 and top.value > 0
    assert [abs(r.value) for r in ng.reasons] == sorted((abs(r.value) for r in ng.reasons), reverse=True)


def test_a_team_expected_to_outscore_its_lines_lifts_them(league):
    lg = copy.deepcopy(league)
    p = lg.team(3).roster[0]
    p.per_game = {**p.per_game, MINUTES: 30.0, PTS: 20.0, REB: 5.0, AST: 4.0, FGM: 8.0, FGA: 16.0}
    p.minute_shifts = ()
    p.recent_minutes = (None, None)
    p.injury_status = "ACTIVE"
    p.games_played = 10
    p.season_line = {PTS: 20.0}
    p.game_context = _ctx(opponent=21, team_points=110.0)
    even = Analyzer(lg, 3).ref(p).next_game
    for call, higher in ((140.0, True), (90.0, False)):
        p.game_context = _ctx(opponent=21, team_points=110.0, team_score=call)
        ng = Analyzer(lg, 3).ref(p).next_game
        assert (ng.pts > even.pts) is higher and (ng.minutes > even.minutes) is higher
        assert ng.usual.pts == even.usual.pts  # the usual line is before the game: unchanged


def test_week_contexts_are_his_teams_games_on_the_days_left():
    box = gamebox.build(2027, _rows(), PRIOR, SCHEDULE)
    # Team 1 plays on days 4 and 5 of days 4..7; a team with no games has none.
    games = gamebox.week_contexts(box, 11, 1, [4, 5, 6, 7])
    assert [m.day for m in games] == [4, 5]
    assert games[1] == gamebox.context(box, 11, 1, 5)
    assert gamebox.week_contexts(box, 11, 9, [4, 5]) == ()


def test_a_week_line_is_his_games_one_by_one():
    easy = _ctx(day=5, defense={PTS: 0.2, REB: 0.1})
    hard = _ctx(day=6, defense={PTS: -0.2, REB: -0.1}, home=True)
    week = gamebox.week_line(LINE, [easy, hard])
    a, b, level = gamebox.apply(LINE, easy, [])[0], gamebox.apply(LINE, hard, [])[0], gamebox.level_line(LINE)
    for k in (MINUTES, PTS, REB, AST, FGM, FGA):
        assert week[k] == pytest.approx(LINE[k] + (a[k] + b[k]) / 2 - level[k])
    assert a[PTS] > b[PTS]  # the two games differ, the week holds both
    # No games, or a line with no minutes: the line as it was.
    assert gamebox.week_line(LINE, []) is LINE
    assert gamebox.week_line(_line(0.0), [easy]) == _line(0.0)


def test_a_week_line_keeps_the_level_of_his_line():
    # Plain games (no opponent or form in them), one at home and one away, move nothing: apply's levels
    # stay out of a week.
    week = gamebox.week_line(LINE, [_ctx(home=True), _ctx(day=6)])
    for k in (MINUTES, PTS, REB, AST, STL, TO, FGM, FGA):
        assert week[k] == pytest.approx(LINE[k])
    # A soft defense then shows in full.
    soft = gamebox.week_line(LINE, [_ctx(defense={PTS: 0.2})])
    assert soft[PTS] > LINE[PTS] and soft[MINUTES] == pytest.approx(LINE[MINUTES])


def test_a_week_line_keeps_free_throws_and_one_game_signals_to_the_first_game():
    line = {**LINE, gamebox.FTM: 4.0, gamebox.FTA: 5.0}
    week = gamebox.week_line(line, [_ctx(defense={gamebox.FTA: 0.3})])
    assert (week[gamebox.FTM], week[gamebox.FTA]) == (4.0, 5.0)
    # A coach who just sat him takes minutes off his next game only.
    sat = _ctx(sat_ago=1)
    one = gamebox.week_line(LINE, [sat])[MINUTES]
    two = gamebox.week_line(LINE, [sat, _ctx(day=6, sat_ago=2)])[MINUTES]
    plain = gamebox.week_line(LINE, [_ctx()])[MINUTES]
    assert one == pytest.approx(plain + gamebox.COACH_SAT_MINUTES)
    assert two == pytest.approx(plain + gamebox.COACH_SAT_MINUTES / 2)

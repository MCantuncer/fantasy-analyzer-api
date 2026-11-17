import pytest

from app.analysis import gamebox, gamescore
from app.analysis.gamebox import AST, FGA, FGM, MINUTES, PTS, REB, GameRow
from app.analysis.report import TEAM_SUM_CAP, TEAM_SUM_SHARE, team_adjust, team_sum_gap

OREB, TO, FTA, FTM, TPM, TPA = gamescore.OREB, gamescore.TO, gamescore.FTA, gamescore.FTM, gamescore.TPM, gamescore.TPA


def _row(player: int, game: int, day: int, team: int, pts: float, threes: tuple[float, float]) -> GameRow:
    """A fifth of a team's game: 5 of them make its 240 minutes."""
    line = dict.fromkeys((*gamebox.COUNTING, *gamescore.TEAM_STATS, gamebox.GS), 0.0)
    line |= {PTS: pts, FGA: 18.0, FTA: 4.0, FTM: 3.0, OREB: 2.0, TO: 3.0, TPM: threes[0], TPA: threes[1]}
    return GameRow(player, game, day, team, line | {MINUTES: 48.0})


def _game(game, day, a, b, pts_a, pts_b, threes_a=(13, 36), threes_b=(13, 36)) -> list[GameRow]:
    rows = []
    for k in range(5):
        rows.append(_row(a * 100 + k, game, day, a, pts_a / 5, (threes_a[0] / 5, threes_a[1] / 5)))
        rows.append(_row(b * 100 + k, game, day, b, pts_b / 5, (threes_b[0] / 5, threes_b[1] / 5)))
    return rows


def test_team_games_sum_each_side_and_leave_out_games_with_rows_missing():
    rows = _game(1, 1, 1, 2, 110, 100)
    rows += [r for r in _game(2, 2, 3, 4, 100, 100) if r.player != 300]  # team 3 short a player: 192 minutes
    games = gamescore.team_games(rows)
    assert [(g.team, g.opponent) for g in games] == [(1, 2), (2, 1)]
    one = games[0].stats
    assert one["pf"] == pytest.approx(110) and one["pa"] == pytest.approx(100)
    assert one["poss"] == pytest.approx(5 * (18 + 0.44 * 4 - 2 + 3))


def test_a_team_that_scores_more_is_expected_to_score_more():
    rows = []
    for g, day in enumerate(range(1, 11)):
        rows += _game(10 + g, day, 1, 2, 125, 105)
        rows += _game(30 + g, day, 3, 4, 110, 110)
    s = gamescore.build(gamescore.team_games(rows))
    strong, even = gamescore.expected(s, 1, 4, None, False), gamescore.expected(s, 3, 4, None, False)
    weak = gamescore.expected(s, 2, 4, None, False)
    assert strong > even > weak


def test_home_court_and_a_back_to_back():
    rows = _game(1, 1, 1, 2, 110, 110)
    s = gamescore.build(gamescore.team_games(rows))
    base = gamescore.expected(s, 1, 2, None, False)
    assert gamescore.expected(s, 1, 2, True, False) - base == pytest.approx(gamescore.HOME)
    assert gamescore.expected(s, 1, 2, False, False) - base == pytest.approx(-gamescore.HOME)
    assert gamescore.expected(s, 1, 2, None, True) - base == pytest.approx(gamescore.BACK_TO_BACK)


def test_a_hot_three_point_run_is_not_all_kept():
    """Team 1 made its points on threes far above the league's rate: the call keeps only part of it."""
    hot, n = [], 0
    for day in range(1, 11):
        hot += _game(n := n + 1, day, 1, 2, 120, 110, threes_a=(20, 36))
        hot += _game(n := n + 1, day, 3, 4, 120, 110, threes_a=(13, 36))
    s = gamescore.build(gamescore.team_games(hot))
    x = gamescore.signals(s, 1, 2)
    assert x["own_three_luck"] > 0
    assert gamescore.expected(s, 1, 2, None, False) < gamescore.expected(s, 3, 2, None, False)


def test_last_seasons_end_is_where_teams_start():
    last = []
    for g, day in enumerate(range(1, 21)):
        last += _game(g + 1, day, 1, 2, 125, 100)
    start = gamescore.season_end(gamescore.team_games(last))
    s = gamescore.build([], start)
    assert gamescore.expected(s, 1, 2, None, False) > gamescore.expected(s, 2, 1, None, False) + 5


def test_what_is_left_of_last_season_moves_with_the_league():
    """Last season everyone scored 110. This season the league scores 125 a game: teams 5 and 6, who
    haven't played yet, are called near this season's level, not last season's (or lower)."""
    last = []
    for g, day in enumerate(range(1, 21)):
        last += _game(100 + g, day, 1, 2, 110, 110) + _game(200 + g, day, 5, 6, 110, 110)
    start = gamescore.season_end(gamescore.team_games(last))
    now = []
    for g, day in enumerate(range(1, 31)):
        now += _game(300 + g, day, 1, 2, 125, 125) + _game(400 + g, day, 3, 4, 125, 125)
    s = gamescore.build(gamescore.team_games(now), start)
    called = gamescore.expected(s, 5, 6, None, False) - gamescore.LEVEL
    assert called == pytest.approx(s.rate("pf"), rel=0.01)
    assert called > 120
    # Before any game of this season the start is just last season's.
    s0 = gamescore.build([], start)
    assert gamescore.expected(s0, 5, 6, None, False) - gamescore.LEVEL == pytest.approx(110, rel=0.01)


def test_the_box_gives_each_game_context_its_teams_expected_score():
    rows = _game(1, 1, 1, 2, 120, 100)
    schedule = {
        1: [gamebox.Game(1, 2, True), gamebox.Game(3, 2, True)],
        2: [gamebox.Game(1, 1, False), gamebox.Game(3, 1, False)],
    }
    box = gamebox.build(2026, rows, {}, schedule)
    one, two = gamebox.context(box, 100, 1, 2), gamebox.context(box, 200, 2, 2)
    assert one.team_score is not None and two.team_score is not None
    assert one.team_score > two.team_score


def test_lines_short_of_the_teams_call_move_up_by_their_shares():
    # Lines add to 90, the rest of the team 30 a game (team 110 less these players' 80): 120 expected 126.
    gap = team_sum_gap(126.0, lines=90.0, season=80.0, team_points=110.0)
    assert gap == pytest.approx(126 / 120 - 1)
    line = {MINUTES: 30.0, PTS: 20.0, REB: 5.0, AST: 4.0, FGM: 7.0, FGA: 15.0}
    moved = team_adjust(line, gap)
    assert moved[PTS] == pytest.approx(20 * (1 + TEAM_SUM_SHARE[PTS] * gap))
    assert moved[MINUTES] > 30 and moved[REB] > 5 and moved[AST] > 4
    assert moved[FGM] / moved[FGA] == pytest.approx(7 / 15)  # FG% stays


def test_the_gap_is_capped_and_empty_without_lines():
    assert team_sum_gap(200.0, 60.0, 60.0, 60.0) == pytest.approx(TEAM_SUM_CAP)
    assert team_sum_gap(110.0, 0.0, 0.0, 110.0) == 0.0
    assert team_adjust({PTS: 10.0}, 0.0) == {PTS: 10.0}


def test_win_chance_grows_with_the_margin_and_is_even_at_zero():
    assert gamescore.win_chance(0.0) == pytest.approx(0.5)
    assert gamescore.win_chance(5.0) == pytest.approx(gamescore.win_chance(-5.0))
    assert 0.65 < gamescore.win_chance(5.0) < 0.67
    assert 0.83 < gamescore.win_chance(12.0) < 0.85


def test_a_call_is_never_tied():
    c = gamescore.call(112.4, 112.6)
    assert (c.away, c.home) == (112, 113)
    assert 0.5 < c.chance < 0.51
    c = gamescore.call(118.2, 104.6)
    assert (c.away, c.home) == (118, 105) and c.chance > 0.8


def test_a_days_call_comes_from_the_games_before_it():
    rows = _game(1, 1, 1, 2, 125, 100) + _game(2, 3, 1, 2, 125, 100)
    schedule = {
        1: [gamebox.Game(1, 2, True), gamebox.Game(3, 2, True), gamebox.Game(5, 2, True)],
        2: [gamebox.Game(1, 1, False), gamebox.Game(3, 1, False), gamebox.Game(5, 1, False)],
    }
    box = gamebox.build(2026, rows, {}, schedule)
    before, after = box.call(3, 2, 1), box.call(5, 2, 1)
    assert after.home - after.away > before.home - before.away > 0  # one more blowout counted by day 5
    assert box.call(3, 2, 1) == before  # the same call after the game

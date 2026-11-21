from app.analysis import gamebox, gamescore, lineupcall, preview
from app.analysis.gamebox import MINUTES, PTS, GameRow
from app.pregame import ruled_out

STAR, STARTER, BENCH = 1, 2, 3  # player ids on team 1


def _line(minutes: float, pts: float) -> dict[int, float]:
    keys = (*gamebox.COUNTING, MINUTES, gamebox.GS, *gamescore.TEAM_STATS)
    return dict.fromkeys(keys, 0.0) | {MINUTES: minutes, PTS: pts}


def _rows(days: int, star_until: int | None = None) -> list[GameRow]:
    """Team 1 against team 2 every day: a 30-minute star scoring 30, starters and a bench."""
    rows = []
    for day in range(1, days + 1):
        game = 100 + day
        if star_until is None or day <= star_until:
            rows.append(GameRow(STAR, game, day, 1, _line(30.0, 30.0)))
        rows.append(GameRow(STARTER, game, day, 1, _line(30.0, 12.0)))
        for k in range(4):
            rows.append(GameRow(10 + k, game, day, 1, _line(45.0, 15.0)))
        rows.append(GameRow(BENCH, game, day, 1, _line(12.0, 4.0)))
        for k in range(6):
            rows.append(GameRow(20 + k, game, day, 2, _line(40.0, 18.0)))
    return rows


def _lineups(rows: list[GameRow]) -> lineupcall.Lineups:
    prior = {STAR: {PTS: 1.0}, STARTER: {PTS: 0.4}, BENCH: {PTS: 0.2}}
    return lineupcall.build(rows, prior, PTS, MINUTES)


def test_a_lineup_without_its_star_has_less_talent():
    lu = _lineups(_rows(12))
    everyone = lu.rotation(1, 13)
    assert set(everyone) == {STAR, STARTER, BENCH, 10, 11, 12, 13}
    full = lu.talent(1, 13, everyone)
    without = lu.talent(1, 13, [p for p in everyone if p != STAR])
    assert full is not None and without is not None
    assert without < full


def test_players_without_a_last_season_count_at_the_replacement_rate():
    lu = _lineups(_rows(5))
    assert lu.floor == lu.rate[BENCH] + (lu.rate[STARTER] - lu.rate[BENCH]) * 0.5  # 25th percentile of three
    assert lu.rate.get(10) is None


def test_a_regular_ruled_out_who_always_plays_costs_his_points():
    lu = _lineups(_rows(12))
    assert lu.absent(1, 13, set(), gamescore.HALF_LIFE) == 0.0  # everyone who always plays, plays
    missing = lu.absent(1, 13, {STAR}, gamescore.HALF_LIFE)
    assert missing == -30.0
    assert lu.absent(1, 13, {BENCH}, gamescore.HALF_LIFE) == 0.0  # not a regular: 12 minutes


def test_a_regular_already_missing_lately_costs_less():
    lu = _lineups(_rows(12, star_until=8))  # out the last 4 games: the ratings partly know
    missing = lu.absent(1, 13, {STAR}, gamescore.HALF_LIFE)
    assert -30.0 < missing < 0.0


def test_the_usual_gap_waits_for_enough_games():
    rows = _rows(12)
    lu = _lineups(rows)
    games = gamescore.team_games(rows)
    assert lineupcall.usual_gap(lu, games, 5, lambda d, t, o: 100.0) is None  # 8 team games
    lots = _rows(40)
    lu = _lineups(lots)
    gap = lineupcall.usual_gap(lu, gamescore.team_games(lots), 40, lambda d, t, o: 100.0)
    assert gap is not None


def test_box_calls_move_when_a_star_is_ruled_out():
    rows = _rows(40)
    sched = {1: [gamebox.Game(d, 2, True) for d in range(1, 42)], 2: [gamebox.Game(d, 1, False) for d in range(1, 42)]}
    prior = {STAR: _line(1.0, 1.0), STARTER: _line(1.0, 0.4)}
    box = gamebox.build(2027, rows, prior, sched)
    plain = box.call_parts(41, 2, 1)
    without = box.call_parts(41, 2, 1, {STAR})
    assert set(plain[1]) >= {"lineup", "absences"}
    assert sum(without[1].values()) < sum(plain[1].values())
    assert sum(without[0].values()) == sum(plain[0].values())  # the other side is unchanged
    # the pre-game box folds both pieces into the matchup term
    m = preview.margin(*without, 100, 101)
    assert {t.kind for t in m.terms} == set(preview.MARGIN_TERMS)


def test_ruled_out_reads_out_tags_only():
    record = {"players": {"1": {"team": 1, "status": "OUT"}, "2": {"team": 1, "status": "DAY_TO_DAY"}}}
    assert ruled_out(record) == {1}
    assert ruled_out(None) == set()

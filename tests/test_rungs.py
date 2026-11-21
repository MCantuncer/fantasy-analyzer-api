from app.analysis import rungs


def test_a_higher_rung_is_less_likely_and_a_higher_line_more_likely():
    for stat, line in (("pts", 20.0), ("reb", 7.0), ("ast", 5.0), ("stl", 1.1), ("blk", 1.0), ("three_pm", 2.2)):
        ladder = [rungs.chance(stat, line, n) for n in range(1, 12)]
        assert ladder == sorted(ladder, reverse=True)
        assert rungs.chance(stat, line + 1, 3) >= rungs.chance(stat, line, 3)
    assert rungs.chance("pts", 20.0, 0) == 1.0
    assert 0.4 < rungs.chance("pts", 20.0, 20) < 0.6  # a 20-point line gets to 20 about half the time
    assert rungs.chance("pts", 20.0, 40) < 0.02


def test_no_line_no_chance_and_value_is_chance_times_odds_less_one():
    assert rungs.chance("pts", None, 20) is None and rungs.chance("fga", 10.0, 5) is None
    assert rungs.value(0.25, 5.0) == 0.25 and rungs.value(0.5, 1.8) == -0.1 and rungs.value(None, 3.0) is None


def test_past_the_grid_the_end_values_hold():
    assert rungs.chance("pts", 80.0, 30) == rungs.chance("pts", 40.0, 30)
    assert rungs.chance("reb", 7.0, 500) == 0.0

import json
import math
from pathlib import Path

import numpy as np
import pytest

from app.analysis import blackbox, gamebox
from app.analysis.blackbox import FGA, FGM, MINUTES, PTS, REB
from tests.test_gamebox import PRIOR, SCHEDULE, _rows

SAMPLE = Path(__file__).parent / "data" / "blackbox_sample.json"


def test_trees_read_as_lightgbm_does():
    """The shipped trees, read by blackbox.py, give LightGBM's own answers on a sample of real inputs
    (written by scripts/black_box_backtest.py --export with the trees)."""
    sample = json.loads(SAMPLE.read_text())
    assert set(sample["calls"]) == set(blackbox.models()) | set(blackbox.with_misses())
    assert any(x.get("pm_n", 0) >= blackbox.MIN_MISSES for x in sample["inputs"])  # both kinds of trees
    assert any(x.get("pm_n", 0) < blackbox.MIN_MISSES for x in sample["inputs"])
    for i, x in enumerate(sample["inputs"]):
        got = blackbox.calls(x)
        for cat, want in sample["calls"].items():
            if want[i] is None or math.isnan(want[i]):
                assert cat not in got
            else:
                assert got[cat] == pytest.approx(want[i], abs=1e-9)


def _tree(**kw) -> blackbox.Trees:
    """Two trees on inputs a, b: the first splits a at 1 (missing goes right), then b at 0 on its left;
    the second is a single leaf of 0.5."""
    spec = {
        "features": ["a", "b"],
        "pull": 1.0,
        "depth": 2,
        "trees": [
            {
                "feature": [0, 1],
                "threshold": [1.0, 0.0],
                "default_left": [False, True],
                "left": [1, ~0],
                "right": [~2, ~1],
                "leaf": [1.0, 2.0, 3.0],
            },
            {"feature": [0], "threshold": [0.0], "default_left": [True], "left": [~0], "right": [~0], "leaf": [0.5]},
        ],
    } | kw
    return blackbox._trees(spec)


def test_trees_walk_splits_and_missing_inputs():
    t = _tree()
    assert t.predict(np.array([0.0, -1.0])) == 1.5  # a <= 1, b <= 0: leaf 0
    assert t.predict(np.array([0.0, 1.0])) == 2.5  # a <= 1, b > 0: leaf 1
    assert t.predict(np.array([2.0, -1.0])) == 3.5  # a > 1: leaf 2
    assert t.predict(np.array([np.nan, -1.0])) == 3.5  # a missing: right
    assert t.predict(np.array([0.0, np.nan])) == 1.5  # b missing: left
    assert _tree(pull=0.5).predict(np.array([0.0, -1.0])) == 0.75


def test_inputs_use_only_games_before_the_day():
    rows = _rows()  # days 1, 3, 4
    last = blackbox.per_game([gamebox.GameRow(11, 1, 1, 1, {MINUTES: 20.0, PTS: 10.0})])
    box = gamebox.build(2027, rows, PRIOR, SCHEDULE, last_lines=last)
    m = gamebox.context(box, 11, 1, 5)
    x = m.box_inputs
    assert x["gp"] == 3 and x["h1_pts"] == 20 and x["h10_min"] == 30
    assert x["ly_pts"] == 10 and x["ly_gp"] == 1
    assert x["t_pf5"] == 20 and x["o_pf5"] == 25 and x["t_mg_last"] == -5
    early = blackbox.game_inputs(box.history, 11, 1, 2, 3)  # his game on day 3: only day 1 counts
    assert early["gp"] == 1 and "t_pf5" not in early  # too few team games for team inputs
    assert math.isnan(blackbox.game_inputs(box.history, 99, 1, 2, 5)["p_rest"])  # no games yet


def test_inputs_hold_lines_and_teammates():
    box = gamebox.build(2027, _rows(), PRIOR, SCHEDULE)
    m = gamebox.context(box, 11, 1, 5)
    line = {MINUTES: 30.0, PTS: 18.0, FGA: 14.0, FGM: 7.0}
    out = [gamebox.Teammate("", 25.0, 12.0, 14.0)]
    back = [gamebox.Teammate("", 20.0, 10.0, 8.0, chance=0.5)]
    x = blackbox.inputs(m, line, {**line, PTS: 19.0}, out, back, 2.0, 0.05)
    assert x["l0_1"] == 18 and x["n_PTS"] == 19 and x["n_MIN"] == 30
    assert x["l0_8"] == pytest.approx(7 - blackbox.FG0 * 14)
    assert x["out_min"] == 25 and x["out_use"] == 12 and x["n_back"] == 0.5 and x["back_min"] == 10
    assert x["gained"] == 2 and x["team_gap"] == 0.05 and x["home"] == 1.0


def test_adjust_moves_lines_by_the_calls():
    line = {MINUTES: 30.0, PTS: 18.0, REB: 1.0, FGA: 10.0, FGM: 5.0}
    out = blackbox.adjust(line, {"MIN": 1.5, "PTS": -0.5, "REB": -3.0, "FG": 0.4})
    assert out[MINUTES] == 31.5 and out[PTS] == 17.5
    assert out[REB] == 0.0  # never below zero
    assert out[FGM] == 5.4 and out[FGA] == 10.0  # FG impact through the makes
    assert blackbox.adjust(line, {"FG": 9.0})[FGM] == 10.0  # no more makes than shots


def test_no_trees_no_change(monkeypatch, tmp_path):
    monkeypatch.setattr(blackbox, "MODELS", tmp_path / "missing.json")
    for f in (blackbox._raw, blackbox.models, blackbox.with_misses):
        f.cache_clear()
    try:
        assert blackbox.calls({"n_MIN": 30.0, "pm_n": 9.0}) == {}
    finally:
        for f in (blackbox._raw, blackbox.models, blackbox.with_misses):
            f.cache_clear()


def _night(lines: dict, results: dict, engine: dict | None = None) -> dict:
    """One settled line-log night: lines {pid: [team, *12 stats]}, results {pid: [12 stats] | []}."""
    rec = {
        "teams": [],
        "lines": {str(p): v for p, v in lines.items()},
        "results": {str(p): v for p, v in results.items()},
    }
    if engine is not None:
        rec["engine"] = {str(p): v for p, v in engine.items()}
    return rec


def _stats(mins: float, pts: float) -> list[float]:
    return [mins, pts, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def test_past_misses_read_the_engine_line_once_the_trees_are_in(monkeypatch):
    monkeypatch.setitem(blackbox.TREES_FROM, 2027, 3)
    records = [
        (1, _night({11: [1, *_stats(30, 20)]}, {11: _stats(32, 25)})),  # before the trees: the saved line
        (2, _night({11: [1, *_stats(30, 20)]}, {11: []})),  # did not play: no miss
        (3, _night({11: [1, *_stats(31, 21)]}, {11: _stats(30, 20)}, engine={11: _stats(30, 18)})),
        (4, _night({11: [1, *_stats(31, 21)]}, {11: _stats(30, 20)})),  # trees in, no engine line: skipped
    ]
    ms = blackbox.past_misses(records, 2027)
    assert ms.day.tolist() == [1, 3] and ms.player.tolist() == [11, 11]
    col = {c: i for i, c in enumerate(blackbox.CATS)}
    assert ms.miss[0][col["MIN"]] == 2 and ms.miss[0][col["PTS"]] == 5
    assert ms.miss[1][col["PTS"]] == 2  # against the engine's 18, not the saved 21


def test_miss_inputs_are_his_own_from_nights_before_the_day(monkeypatch):
    monkeypatch.setitem(blackbox.TREES_FROM, 2027, 100)
    records = []
    for day in range(1, 9):  # 11 beats his points line by the day number, 12 by 1 every night
        lines = {11: [1, *_stats(30, 20)], 12: [1, *_stats(30, 10)]}
        records.append((day, _night(lines, {11: _stats(30, 20 + day), 12: _stats(30, 11)})))
    ms = blackbox.past_misses(records, 2027)
    x = blackbox.miss_inputs(ms, 11, 7)  # days 1..6 only
    assert x["pm_n"] == 6
    assert x["pms_PTS"] == pytest.approx(3.5) and x["pm5_PTS"] == pytest.approx(4)  # days 2..6
    assert x["pm5_MIN"] == 0
    assert blackbox.miss_inputs(ms, 12, 7)["pms_PTS"] == 1
    assert blackbox.miss_inputs(ms, 11, 1) == {"pm_n": 0.0}
    assert blackbox.miss_inputs(None, 11, 7) == {"pm_n": 0.0}

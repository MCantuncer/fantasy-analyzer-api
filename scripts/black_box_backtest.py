"""Trains and judges the trees of app/analysis/blackbox.py: patterns in the engine's own misses.

    uv run --with lightgbm --with pyarrow python -m scripts.black_box_backtest [--collect] [--export]

1. Collect (--collect, or when a season's cache is missing): every case of scripts/next_game_matchup_backtest
   (2018-19..2026-27), the line that ships (gamebox.apply, then each team's lines held to its expected
   score, as scripts/game_score_backtest.py replays report.next_game_line), what he did, and
   blackbox.inputs: the same function report.next_game_line calls live. Kept in .cache/blackbox/.
2. Judge, per category: gradient-boosted trees (LightGBM, settings fixed in P) learn the line's miss.
   Their 40 strongest inputs, the number of trees and a pull toward zero are picked inside the training
   seasons only (the last of them held back). A category is kept only if the trees lower the squared
   miss in every test: each of 2021-22..2025-26 left out (trained on the other four), each trained on
   earlier seasons only, 2026-27 so far (trained on all five), and, frozen, 2018-19..2020-21, which the
   engine was never fitted on; and in each by more than the same recipe gets from misses shuffled within
   each season (which can only learn a season's average miss).
3. Export (--export): the kept categories' trees, trained on the five seasons, to
   app/analysis/blackbox_models.json, plus a sample of inputs with LightGBM's own answers
   (tests/data/blackbox_sample.json) that tests/test_blackbox.py checks blackbox.py's reading against.
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from app.analysis import blackbox, gamebox
from app.analysis.report import team_adjust, team_sum_gap
from scripts.next_game_matchup_backtest import cases

SEASONS = (2022, 2023, 2024, 2025, 2026)
NOW = 2027
OLD = (2019, 2020, 2021)
CACHE = Path(".cache/blackbox")
SAMPLE = Path("tests/data/blackbox_sample.json")
TOP = 40
P = dict(
    objective="regression",
    learning_rate=0.03,
    num_leaves=15,
    min_data_in_leaf=400,
    feature_fraction=0.7,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=10.0,
    verbose=-1,
    num_threads=4,
    seed=7,
)
GAMES_PTS = gamebox.PTS


def collect(season: int) -> dict:
    """Every case of a season: inputs, the shipped line and what he did (blackbox.cats order)."""
    raw = []
    for _box, c in cases(season):
        new, _, _ = gamebox.apply(c.line, c.context, c.mates, c.gained, c.back, c.absent)
        raw.append((c, new))
    groups: dict[tuple[int, int], list] = defaultdict(list)
    for c, new in raw:
        groups[(c.day, c.team)].append((c, new))
    rows = []
    records: dict[int, dict] = defaultdict(lambda: {"lines": {}, "engine": {}, "results": {}})
    for group in groups.values():
        lines_pts = sum(new.get(GAMES_PTS, 0.0) for _, new in group)
        season_pts = sum(c.so_far.get(GAMES_PTS, 0.0) for c, _ in group)
        for c, new in group:
            m = c.context
            gap = 0.0 if m.team_score is None else team_sum_gap(m.team_score, lines_pts, season_pts, m.team_points)
            shipped = team_adjust(new, gap)
            x = blackbox.inputs(m, c.line, shipped, c.mates, c.back, c.gained, gap)
            rows.append([c.day, c.player, x, blackbox.cats(shipped), blackbox.cats(c.actual)])
            # the night as the line log would hold it (app/linelog.py), for the past misses below
            rec = records[c.day]
            saved = [round(shipped.get(k, 0.0), 1) for k in blackbox.LOG_STATS]
            rec["lines"][str(c.player)] = [c.team, *saved]
            rec["engine"][str(c.player)] = saved
            rec["results"][str(c.player)] = [round(c.actual.get(k, 0.0), 1) for k in blackbox.LOG_STATS]
    # the engine's own misses on earlier nights, counted by the code the app counts the line log's with
    misses = blackbox.past_misses(sorted(records.items()), season)
    for r in rows:
        r[2] = r[2] | blackbox.miss_inputs(misses, r[1], r[0])
    rows.sort(key=lambda r: (r[0], r[1]))
    names = sorted({k for r in rows for k in r[2]})
    return {
        "day": np.array([r[0] for r in rows]),
        "player": np.array([r[1] for r in rows]),
        "names": names,
        "x": np.array([[r[2].get(k, np.nan) for k in names] for r in rows], dtype=float),
        "new": np.array([r[3] for r in rows]),
        "act": np.array([r[4] for r in rows]),
    }


def load(refresh: bool) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    CACHE.mkdir(parents=True, exist_ok=True)
    per = {}
    for s in (*OLD, *SEASONS, NOW):
        path = CACHE / f"{s}.pkl"
        if refresh or not path.exists():
            path.write_bytes(pickle.dumps(collect(s)))
            print(f"collected {s}", flush=True)
        per[s] = pickle.loads(path.read_bytes())
    names = sorted({k for d in per.values() for k in d["names"]})
    xs, season, day, new, act = [], [], [], [], []
    for s, d in per.items():
        idx = [d["names"].index(k) if k in d["names"] else -1 for k in names]
        x = np.where(np.array(idx) >= 0, d["x"][:, np.maximum(idx, 0)], np.nan)
        xs.append(x)
        season.append(np.full(len(x), s))
        day.append(d["day"])
        new.append(d["new"])
        act.append(d["act"])
    return (np.vstack(xs), names, np.concatenate(season), np.concatenate(day), np.vstack(new), np.vstack(act))


def fit(lgb, X, y, tr, val):
    """Trees on ``tr``: inputs, tree count and pull picked with ``val`` (inside ``tr``) held back."""
    part = tr & ~val
    b0 = lgb.train(
        P,
        lgb.Dataset(X[part], y[part]),
        2000,
        valid_sets=[lgb.Dataset(X[val], y[val])],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    cols = np.argsort(-b0.feature_importance("gain"))[:TOP]
    b = lgb.train(
        P,
        lgb.Dataset(X[part][:, cols], y[part]),
        2000,
        valid_sets=[lgb.Dataset(X[val][:, cols], y[val])],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    it = max(1, b.best_iteration)
    pv = b.predict(X[val][:, cols], num_iteration=it)
    pull = float(np.clip(pv @ y[val] / max(pv @ pv, 1e-12), 0.0, 1.0))
    full = lgb.train(P, lgb.Dataset(X[tr][:, cols], y[tr]), it)
    return full, cols, pull


def validation(season, day, tr):
    seen = sorted(set(season[tr]))
    return tr & (season == seen[-1]) if len(seen) >= 2 else tr & (day > np.median(day[tr]))


def judge(lgb, X, season, day, y, rows=None, evals=None) -> list[tuple[str, int, float, list[float]]]:
    """Per test: L (left out), F (forward), N (2026-27), O (older): % less squared miss than the line, and
    the squared miss of the trees on each of ``evals`` (row masks; all rows if none). ``rows``: the rows
    trees may train on (all if none)."""
    rows = np.ones(len(y), bool) if rows is None else rows
    evals = evals or [np.ones(len(y), bool)]
    five = np.isin(season, SEASONS) & rows
    tests = [("L", s, five & (season != s)) for s in SEASONS]
    tests += [("F", s, five & (season < s)) for s in SEASONS[1:]]
    tests += [("N", NOW, five)]
    out = []
    for mode, s, tr in tests:
        model, cols, pull = fit(lgb, X, y, tr, validation(season, day, tr))
        for t in (s, *OLD) if mode == "N" else (s,):
            te = (season == t) & evals[0]
            p = pull * model.predict(X[te][:, cols])
            e0 = float(np.mean(y[te] ** 2))
            more = []
            for ev in evals[1:]:
                tt = (season == t) & ev
                more.append(float(np.mean((y[tt] - pull * model.predict(X[tt][:, cols])) ** 2)))
            out.append((mode if t == s else "O", t, (e0 - float(np.mean((y[te] - p) ** 2))) / e0 * 100, more))
    return out


def shuffled(y, season, rng):
    z = y.copy()
    for s in np.unique(season):
        i = np.where(season == s)[0]
        z[i] = y[rng.permutation(i)]
    return z


def export_tree(t: dict, names: list[str]) -> dict:
    """One LightGBM tree (dump_model) as blackbox.Trees' flat arrays."""
    feature, threshold, default_left, left, right = [], [], [], [], []
    leaf: list[float] = []

    def walk(n: dict) -> int:
        if "leaf_index" in n:
            j = int(n["leaf_index"])
            while len(leaf) <= j:
                leaf.append(0.0)
            leaf[j] = float(n["leaf_value"])
            return ~j
        i = int(n["split_index"])
        while len(feature) <= i:
            for a in (feature, threshold, default_left, left, right):
                a.append(0)
        if n["decision_type"] != "<=" or n["missing_type"] == "Zero":
            raise ValueError(f"split not read by blackbox.py: {n['decision_type']} {n['missing_type']}")
        feature[i] = int(n["split_feature"])
        threshold[i] = float(n["threshold"])
        # missing type None: LightGBM reads a missing input as 0
        default_left[i] = bool(n["default_left"]) if n["missing_type"] == "NaN" else 0.0 <= threshold[i]
        left[i] = walk(n["left_child"])
        right[i] = walk(n["right_child"])
        return i

    root = walk(t["tree_structure"])
    if root < 0:  # a single leaf: a node whose both ways lead to it
        feature, threshold, default_left, left, right = [0], [0.0], [True], [root], [root]

    def depth(n: dict) -> int:
        return 0 if "leaf_index" in n else 1 + max(depth(n["left_child"]), depth(n["right_child"]))

    return {
        "feature": feature,
        "threshold": threshold,
        "default_left": default_left,
        "left": left,
        "right": right,
        "leaf": leaf,
        "depth": max(1, depth(t["tree_structure"])),
    }


MISS_INPUTS = ("pm_n", "pm5_", "pms_")


def _show(r, mode, k=2):
    return " ".join(f"{t[k]:+.2f}" for t in r if t[0] == mode)


def main() -> None:
    import lightgbm as lgb  # only here: the app reads the exported trees with numpy

    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true", help="rebuild the cached cases")
    ap.add_argument("--export", action="store_true", help="write the kept categories' trees")
    ap.add_argument("--cats", default=",".join(blackbox.CATS))
    args = ap.parse_args()
    X, names, season, day, new, act = load(args.collect)
    res = act - new
    plain = [i for i, k in enumerate(names) if not k.startswith(MISS_INPUTS)]
    known = X[:, names.index("pm_n")] >= blackbox.MIN_MISSES
    print(f"{len(X):,} player games, {len(names)} inputs ({len(plain)} without past misses)", flush=True)
    print(f"{int(known.sum()):,} games with {blackbox.MIN_MISSES}+ past misses saved\n", flush=True)
    rng = np.random.default_rng(1)
    kept, kept_misses = [], []
    for cat in args.cats.split(","):
        y = res[:, blackbox.CATS.index(cat)]
        # 1. trees without past misses, on every game
        real = judge(lgb, X[:, plain], season, day, y, evals=[np.ones(len(y), bool), known])
        fake = judge(lgb, X[:, plain], season, day, shuffled(y, season, rng))
        ok = all(g > 0 and g > f[2] for (_, _, g, _), f in zip(real, fake, strict=True))
        print(
            f"{cat:4s} left out {_show(real, 'L')} | forward {_show(real, 'F')} | 2026-27 {_show(real, 'N')}"
            f" | 2018-19..2020-21 {_show(real, 'O')}{'  KEPT' if ok else ''}\n"
            f"     shuffled misses: left out {_show(fake, 'L')} | forward {_show(fake, 'F')}"
            f" | 2026-27 {_show(fake, 'N')} | older {_show(fake, 'O')}",
            flush=True,
        )
        if ok:
            kept.append(cat)
        # 2. trees that also read his past misses, on games with MIN_MISSES of them: they must beat the
        # line by more than shuffled misses do, and beat (1) where (1) is kept, in every test
        real2 = judge(lgb, X, season, day, y, rows=known, evals=[known])
        fake2 = judge(lgb, X, season, day, shuffled(y, season, rng), rows=known, evals=[known])
        base = []  # the squared miss there of the line, or of (1) when kept
        for _, t, _, more in real:
            e_line = float(np.mean(y[(season == t) & known] ** 2))
            base.append(more[0] if ok else e_line)
        e2 = [float(np.mean(y[(season == t) & known] ** 2)) * (1 - g / 100) for _, t, g, _ in real2]
        vs = [(b - e) / b * 100 for b, e in zip(base, e2, strict=True)]
        ok2 = all(g > 0 and g > f[2] and v > 0 for (_, _, g, _), f, v in zip(real2, fake2, vs, strict=True))
        rows_vs = [(m, t, v) for (m, t, _, _), v in zip(real2, vs, strict=True)]
        print(
            f"     with past misses ({blackbox.MIN_MISSES}+ saved), against the line: left out {_show(real2, 'L')}"
            f" | forward {_show(real2, 'F')} | 2026-27 {_show(real2, 'N')} | older {_show(real2, 'O')}\n"
            f"     ... against {'(1)' if ok else 'the line'}: left out {_show(rows_vs, 'L')}"
            f" | forward {_show(rows_vs, 'F')} | 2026-27 {_show(rows_vs, 'N')} | older {_show(rows_vs, 'O')}"
            f"{'  KEPT' if ok2 else ''}\n"
            f"     shuffled: left out {_show(fake2, 'L')} | forward {_show(fake2, 'F')} | 2026-27 {_show(fake2, 'N')}"
            f" | older {_show(fake2, 'O')}",
            flush=True,
        )
        if ok2:
            kept_misses.append(cat)
    print(f"\nkept: {', '.join(kept) or 'none'}; with past misses: {', '.join(kept_misses) or 'none'}")
    if not args.export:
        return
    five = np.isin(season, SEASONS)
    now = np.where(season == NOW)[0]
    rows = np.concatenate([now[~known[now]][:20], now[known[now]][:20]])
    out: dict[str, dict] = {"models": {}, "with_misses": {}}
    sample: dict = {"inputs": [], "calls": {}}
    calls = np.full((len(rows), len(blackbox.CATS)), np.nan)
    for key, cats_, cols_all, train in (
        ("models", kept, plain, five),
        ("with_misses", kept_misses, None, five & known),
    ):
        for cat in cats_:
            c = blackbox.CATS.index(cat)
            y = res[:, c]
            Xs = X if cols_all is None else X[:, cols_all]
            sub = names if cols_all is None else [names[i] for i in cols_all]
            model, cols, pull = fit(lgb, Xs, y, train, validation(season, day, train))
            feats = [sub[i] for i in cols]
            trees = [export_tree(t, feats) for t in model.dump_model()["tree_info"]]
            out[key][cat] = {
                "features": feats,
                "pull": pull,
                "trees": trees,
                "depth": max(t.pop("depth") for t in trees),
            }
            p = pull * model.predict(Xs[rows][:, cols])
            use = known[rows] if key == "with_misses" else ~(known[rows] & (cat in kept_misses))
            calls[use, c] = p[use]
    for cat in {*kept, *kept_misses}:
        sample["calls"][cat] = [None if np.isnan(v) else float(v) for v in calls[:, blackbox.CATS.index(cat)]]
    sample["inputs"] = [{k: float(v) for k, v in zip(names, X[i], strict=True) if np.isfinite(v)} for i in rows]
    out |= {"kept": kept, "kept_with_misses": kept_misses}
    blackbox.MODELS.write_text(json.dumps(out, separators=(",", ":")))
    SAMPLE.write_text(json.dumps(sample, separators=(",", ":")))
    print(f"wrote {blackbox.MODELS} ({blackbox.MODELS.stat().st_size // 1024} KB) and {SAMPLE}")


if __name__ == "__main__":
    main()

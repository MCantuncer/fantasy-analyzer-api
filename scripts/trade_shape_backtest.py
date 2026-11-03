"""How often real trade offers get accepted, by shape, at the same value for the other owner.

    uv run python -m scripts.trade_shape_backtest [offer_sides.csv]

Reads the offer dataset of scripts/trade_dataset.py (one line per offer and side; the
default path is .cache/tradedata/offer_sides.csv). Accepted offers are labelled 1,
withdrawn or expired ones 0. Each offer is seen from the proposer's side, so in the shape
"2-for-1" the proposer sends two players and the other owner has to cut one to fit them.

The fit is a logistic regression of acceptance on the other owner's net value (z-scores
summed over categories, clipped to +-4, the same scale as CategoryModel.value) plus one
term per shape, with 1-for-1 as the reference. Every league counts the same: a few leagues
made most of the offers. Shape effects are odds ratios against a 1-for-1 of the same value.
A 90% interval comes from resampling whole leagues. Only category leagues (H2H categories,
most categories, roto) are used; points leagues trade on a different scale.

Validation is leave-one-league-out: each league in turn is predicted by a model fitted on
the others, by log loss and by how well it ranks its own accepted offers above the rest (AUC).
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

DEFAULT = Path(".cache/tradedata/offer_sides.csv")
SHAPES = ("1-for-1", "2-for-2", "2-for-1", "1-for-2")  # proposer sends-for-gets; 1-for-1 is the reference
CATEGORY_LEAGUES = ("H2H_CATEGORY", "H2H_MOST_CATEGORIES", "ROTO")
CLIP = 4.0
L2 = 0.1
BOOTSTRAP = 300


def load(path: Path) -> list[dict]:
    """One row per offer: league, shape, the other owner's net value, accepted or not."""
    sides: dict[str, dict[str, dict]] = defaultdict(dict)
    with path.open() as fh:
        for r in csv.DictReader(fh):
            if r["outcome"] not in ("accepted", "withdrawn") or r["scoring_type"] not in CATEGORY_LEAGUES:
                continue
            sides[r["offer_id"]].setdefault(r["role"], r)
    rows = []
    for s in sides.values():
        p, r = s.get("proposer"), s.get("receiver")
        if p is None or r is None:
            continue
        shape = f"{p['n_give']}-for-{p['n_get']}"
        if shape not in SHAPES:
            continue
        rows.append(
            {
                "league": p["league_id"],
                "shape": shape,
                "net": max(-CLIP, min(CLIP, float(r["net_season"] or 0.0))),
                "accepted": p["outcome"] == "accepted",
            }
        )
    return rows


def design(rows: list[dict], with_shape: bool) -> np.ndarray:
    cols = [np.ones(len(rows)), np.array([r["net"] for r in rows])]
    if with_shape:
        cols += [np.array([r["shape"] == s for r in rows], float) for s in SHAPES[1:]]
    return np.column_stack(cols)


def weights(rows: list[dict]) -> np.ndarray:
    """Every league weighs the same, scaled so the weights sum to the number of offers."""
    count: dict[str, int] = defaultdict(int)
    for r in rows:
        count[r["league"]] += 1
    w = np.array([1.0 / count[r["league"]] for r in rows])
    return w * len(rows) / w.sum()


def fit(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    b = np.zeros(x.shape[1])
    ridge = L2 * np.eye(len(b))
    ridge[0, 0] = 0.0
    for _ in range(60):
        mu = 1 / (1 + np.exp(-(x @ b)))
        h = x.T @ (x * (w * mu * (1 - mu))[:, None]) + ridge
        b += np.linalg.solve(h, x.T @ (w * (y - mu)) - ridge @ b)
    return b


def auc(score: np.ndarray, y: np.ndarray) -> float:
    pos, neg = score[y == 1], score[y == 0]
    if not len(pos) or not len(neg):
        return math.nan
    return float(np.mean(pos[:, None] > neg[None, :]) + 0.5 * np.mean(pos[:, None] == neg[None, :]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", type=Path, default=DEFAULT)
    args = ap.parse_args()
    rows = load(args.path)
    y = np.array([r["accepted"] for r in rows], float)
    w = weights(rows)
    leagues = sorted({r["league"] for r in rows})
    print(f"{len(rows)} offers, {int(y.sum())} accepted, {len(leagues)} category leagues\n")

    print(f"{'shape':9} {'offers':>7} {'accepted':>9} {'rate':>6} {'league-balanced':>16} {'share of accepted':>18}")
    for s in SHAPES:
        m = np.array([r["shape"] == s for r in rows])
        bal = float(np.average(y[m], weights=w[m])) if m.any() else math.nan
        print(f"{s:9} {m.sum():7d} {int(y[m].sum()):9d} {y[m].mean():6.1%} {bal:16.1%} {y[m].sum() / y.sum():18.1%}")

    b = fit(design(rows, True), y, w)
    rng = np.random.default_rng(0)
    by_league = defaultdict(list)
    for i, r in enumerate(rows):
        by_league[r["league"]].append(i)
    boot = []
    for _ in range(BOOTSTRAP):
        idx = np.concatenate([by_league[lg] for lg in rng.choice(leagues, len(leagues))])
        sub = [rows[i] for i in idx]
        boot.append(np.exp(fit(design(sub, True), y[idx], weights(sub))[2:]))
    lo, hi = np.percentile(np.array(boot), [5, 95], axis=0)
    print(f"\nnet value slope {b[1]:.3f} per z-point; odds of acceptance against a 1-for-1 of the same value:")
    for k, s in enumerate(SHAPES[1:]):
        print(f"  {s}: x{math.exp(b[2 + k]):.2f}  (90% interval {lo[k]:.2f}-{hi[k]:.2f})")

    print("\nleave-one-league-out (leagues with 30+ offers and 3+ accepted):")
    results = {False: [], True: []}
    for lg in leagues:
        test = np.array([r["league"] == lg for r in rows])
        if test.sum() < 30 or y[test].sum() < 3:
            continue
        train = [r for r, t in zip(rows, test, strict=True) if not t]
        held = [r for r, t in zip(rows, test, strict=True) if t]
        for with_shape in (False, True):
            bb = fit(design(train, with_shape), y[~test], weights(train))
            score = design(held, with_shape) @ bb
            mu = np.clip(1 / (1 + np.exp(-score)), 1e-6, 1 - 1e-6)
            loss = float(-np.mean(y[test] * np.log(mu) + (1 - y[test]) * np.log(1 - mu)))
            results[with_shape].append((auc(score, y[test]), loss))
    for with_shape, label in ((False, "value only"), (True, "value + shape")):
        a = np.array(results[with_shape])
        print(f"  {label:14} leagues {len(a)}  AUC {np.nanmean(a[:, 0]):.3f}  log loss {a[:, 1].mean():.4f}")


if __name__ == "__main__":
    main()

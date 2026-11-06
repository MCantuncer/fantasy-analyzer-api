"""Fit the trade acceptance odds on real offers, and check them on leagues they never saw.

    uv run python -m scripts.offer_replay        # first: replays every offer through the finder
    uv run python -m scripts.acceptance_fit [replayed.jsonl]

Reads the offers scripts/offer_replay.py replayed through the trade finder (default
.cache/tradedata/replayed.jsonl): accepted ones are labelled 1, withdrawn or expired ones 0.
Each offer carries what the finder computes for every deal, seen from the proposer:

  * paper: log of the trade value the other owner gets over what they send (perceived.py,
    owner premium included), clipped to +-PAPER_CLIP
  * their gain / my gain: expected categories won per week, the other owner's in the
    categories they compete in
  * shape: 2-for-2, 2-for-1 (they must cut someone) and 1-for-2, against a 1-for-1

Logistic regressions on several sets of these are compared leave-one-league-out: each league
is predicted by a model fitted on the others, by log loss averaged over leagues and by how well
it ranks that league's accepted offers above the rest (AUC, 0.5 = chance). Every league counts
the same: a few leagues made most of the offers. The chosen model's coefficients go into
app/analysis/trades.py (ACCEPT_COEF); the script prints them, the ones in use, and the
calibration next to the hand-tuned odds the app used before.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from app.analysis.trades import ACCEPT_COEF, PAPER_CLIP, acceptance_odds
from scripts.offer_replay import REPLAYED
from scripts.trade_shape_backtest import auc, fit, weights

SHAPES = ("1-for-1", "2-for-2", "2-for-1", "1-for-2")  # players the proposer sends-for-gets
GAIN_CLIP = 1.0
COLUMNS = ["intercept", "paper", "their gain", "my gain", "2-for-2", "2-for-1", "1-for-2"]
MODELS = {
    "base rate": [0],
    "shape": [0, 4, 5, 6],
    "paper": [0, 1],
    "paper + shape": [0, 1, 4, 5, 6],
    "their gain + shape": [0, 2, 4, 5, 6],
    "everything": [0, 1, 2, 3, 4, 5, 6],
}
CHOSEN = "paper + shape"


def load(path: Path) -> list[dict]:
    with path.open() as fh:
        rows = [json.loads(line) for line in fh]
    return [r for r in rows if r["shape"] in SHAPES]


def design(rows: list[dict]) -> np.ndarray:
    def clip(v: float, c: float) -> float:
        return float(np.clip(v, -c, c))

    return np.column_stack(
        [np.ones(len(rows))]
        + [np.array([clip(math.log(max(r["ratio"], 1e-6)), PAPER_CLIP) for r in rows])]
        + [np.array([clip(r["their_gain"], GAIN_CLIP) for r in rows])]
        + [np.array([clip(r["my_gain"], GAIN_CLIP) for r in rows])]
        + [np.array([r["shape"] == s for r in rows], float) for s in SHAPES[1:]]
    )


def old_odds(r: dict) -> float:
    """The hand-tuned odds the app used before."""
    p = float(np.interp(r["ratio"], [0.85, 0.95, 1.05, 1.2, 1.4], [0.2, 0.4, 0.55, 0.7, 0.8]))
    p += 0.1 if r["their_gain"] >= 0.15 else -0.1 if r["their_gain"] < 0.05 else 0.0
    p = float(np.clip(p, 0.05, 0.9))
    k = {"2-for-2": 0.63, "2-for-1": 0.40}.get(r["shape"], 1.0)
    return float(np.clip(p * k / (1 - p + p * k), 0.05, 0.9))


def held_out(rows: list[dict], x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Each league's odds from a model fitted on every other league."""
    leagues = np.array([r["league"] for r in rows])
    p = np.zeros(len(rows))
    for lg in set(leagues):
        test = leagues == lg
        b = fit(x[~test], y[~test], weights([r for r, t in zip(rows, test, strict=True) if not t]))
        p[test] = 1 / (1 + np.exp(-(x[test] @ b)))
    return p


def by_league(rows: list[dict], p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Log loss and AUC, each averaged over leagues."""
    groups: dict[int, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        groups[r["league"]].append(i)
    losses, aucs = [], []
    for idx in groups.values():
        q, t = np.clip(p[idx], 1e-6, 1 - 1e-6), y[idx]
        losses.append(float(-np.mean(t * np.log(q) + (1 - t) * np.log(1 - q))))
        aucs.append(auc(q, t))
    return float(np.mean(losses)), float(np.nanmean(aucs))


def calibration(name: str, p: np.ndarray, y: np.ndarray, w: np.ndarray) -> None:
    print(f"  {name}:")
    for lo, hi in ((0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.45), (0.45, 0.6), (0.6, 1.01)):
        m = (p >= lo) & (p < hi)
        if m.sum() >= 10:
            shown, real = np.average(p[m], weights=w[m]), np.average(y[m], weights=w[m])
            band = f"{lo:.0%}-{min(hi, 1):.0%}"
            print(f"    shown {band}: {m.sum():5d} offers, avg shown {shown:5.1%}, accepted {real:5.1%}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", type=Path, default=REPLAYED)
    args = ap.parse_args()
    rows = load(args.path)
    y = np.array([r["accepted"] for r in rows], float)
    w = weights(rows)
    x = design(rows)
    print(f"{len(rows)} replayed offers, {int(y.sum())} accepted, {len({r['league'] for r in rows})} category leagues")
    print(f"league-balanced acceptance rate {np.average(y, weights=w):.1%}\n")

    print("leave-one-league-out, averaged over leagues (log loss: lower is better; AUC: 0.5 = chance):")
    preds = {}
    for name, cols in MODELS.items():
        preds[name] = held_out(rows, x[:, cols], y)
        loss, rank = by_league(rows, preds[name], y)
        coef = ", ".join(f"{COLUMNS[c]} {v:+.2f}" for c, v in zip(cols, fit(x[:, cols], y, w), strict=True))
        print(f"  {name:20} log loss {loss:.4f}  AUC {rank:.3f}  [{coef}]")
    old = np.array([old_odds(r) for r in rows])
    loss, rank = by_league(rows, old, y)
    print(f"  {'old hand-tuned':20} log loss {loss:.4f}  AUC {rank:.3f}")

    b = fit(x[:, MODELS[CHOSEN]], y, w)
    print(f"\n{CHOSEN}, fitted on every league (ACCEPT_COEF: intercept, paper, 2-for-2, 2-for-1, 1-for-2):")
    print("  (" + ", ".join(f"{v:.3f}" for v in b) + ")")
    print("  in use: " + str(ACCEPT_COEF))
    in_use = np.array([acceptance_odds(r["ratio"], *map(int, r["shape"].split("-for-"))) for r in rows])

    print("\ncalibration, league-weighted:")
    calibration(f"{CHOSEN}, held out", preds[CHOSEN], y, w)
    calibration("in use", in_use, y, w)
    calibration("old hand-tuned", old, y, w)


if __name__ == "__main__":
    main()

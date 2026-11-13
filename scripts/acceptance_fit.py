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

Declined offers count too. ESPN records that an offer was declined but never what was in it,
so they can't be replayed; the dataset keeps their number per league-season (seasons.json from
scripts/trade_dataset.py build). Each withdrawn offer is weighted to stand for its league's
declined ones as well: (withdrawn + declined) / withdrawn. That assumes a declined offer looked
like a withdrawn one, so it moves how often the answer is yes, not what tells offers apart.

Logistic regressions on several sets of these are compared leave-one-league-out: each league
is predicted by a model fitted on the others, by log loss averaged over leagues and by how well
it ranks that league's accepted offers above the rest (AUC, 0.5 = chance). Every league counts
the same: a few leagues made most of the offers. The chosen model's coefficients go into
app/analysis/trades.py (ACCEPT_COEF); the script prints them, the ones in use, the same fit
without the declines, and how far the odds shown are from each league's real rate.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from app.analysis.trades import ACCEPT_COEF, PAPER_CLIP, acceptance_odds
from scripts.offer_replay import CACHE, REPLAYED
from scripts.trade_shape_backtest import auc, fit, weights

SEASONS = CACHE / "seasons.json"  # offers and declines per league-season, from trade_dataset build
LARGE_LEAGUE = 40  # offers a league needs for its own rate to mean something

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


def decline_factors(rows: list[dict], seasons: list[dict]) -> dict[int, float]:
    """Offers each withdrawn one stands for, by league: itself plus its share of the declined ones.

    Counted over the seasons the league has replayed offers in.
    """
    replayed = {(r["league"], r["season"]) for r in rows}
    withdrawn: dict[int, int] = defaultdict(int)
    declined: dict[int, int] = defaultdict(int)
    for s in seasons:
        if (s["league_id"], s["season"]) in replayed:
            withdrawn[s["league_id"]] += s["outcomes"].get("withdrawn", 0)
            declined[s["league_id"]] += s["declines"]
    return {lg: 1.0 + declined[lg] / withdrawn[lg] if withdrawn[lg] else 1.0 for lg in {r["league"] for r in rows}}


def decline_weights(rows: list[dict], factors: dict[int, float]) -> np.ndarray:
    """Weights that count declines: a withdrawn offer weighs its league's factor, an accepted one 1,
    then every league weighs the same and the weights sum to the number of offers."""
    raw = np.array([1.0 if r["accepted"] else factors[r["league"]] for r in rows])
    total: dict[int, float] = defaultdict(float)
    for r, v in zip(rows, raw, strict=True):
        total[r["league"]] += v
    w = np.array([v / total[r["league"]] for r, v in zip(rows, raw, strict=True)])
    return w * len(rows) / w.sum()


def held_out(rows: list[dict], x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Each league's odds from a model fitted on every other league."""
    leagues = np.array([r["league"] for r in rows])
    p = np.zeros(len(rows))
    for lg in set(leagues):
        test = leagues == lg
        b = fit(x[~test], y[~test], w[~test] * (~test).sum() / w[~test].sum())
        p[test] = 1 / (1 + np.exp(-(x[test] @ b)))
    return p


def by_league(rows: list[dict], p: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float, float]:
    """Log loss and AUC averaged over leagues, and how far the odds shown are from the real rate
    in a league (mean absolute gap over the large leagues). ``w`` weights offers inside a league."""
    groups: dict[int, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        groups[r["league"]].append(i)
    losses, aucs, gaps = [], [], []
    for idx in groups.values():
        q, t, v = np.clip(p[idx], 1e-6, 1 - 1e-6), y[idx], w[idx]
        losses.append(float(np.average(-(t * np.log(q) + (1 - t) * np.log(1 - q)), weights=v)))
        aucs.append(auc(q, t))
        if len(idx) >= LARGE_LEAGUE:
            gaps.append(abs(float(np.average(q, weights=v) - np.average(t, weights=v))))
    return float(np.mean(losses)), float(np.nanmean(aucs)), float(np.mean(gaps))


def calibration(name: str, p: np.ndarray, y: np.ndarray, w: np.ndarray) -> None:
    print(f"  {name}:")
    for lo, hi in ((0, 0.05), (0.05, 0.08), (0.08, 0.12), (0.12, 0.2), (0.2, 0.35), (0.35, 1.01)):
        m = (p >= lo) & (p < hi)
        if m.sum() >= 10:
            shown, real = np.average(p[m], weights=w[m]), np.average(y[m], weights=w[m])
            band = f"{lo:.0%}-{min(hi, 1):.0%}"
            print(f"    shown {band}: {m.sum():5d} offers, avg shown {shown:5.1%}, accepted {real:5.1%}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", type=Path, default=REPLAYED)
    ap.add_argument("--seasons", type=Path, default=SEASONS, help="seasons.json from trade_dataset build")
    args = ap.parse_args()
    rows = load(args.path)
    y = np.array([r["accepted"] for r in rows], float)
    factors = decline_factors(rows, json.loads(args.seasons.read_text()))
    w, plain = decline_weights(rows, factors), weights(rows)
    x = design(rows)
    print(f"{len(rows)} replayed offers, {int(y.sum())} accepted, {len(factors)} category leagues")
    print(f"declined offers per withdrawn one, league by league: {np.mean(list(factors.values())) - 1:.1f} on average")
    print(f"league-balanced acceptance rate: {np.average(y, weights=plain):.1%} of accepted and withdrawn offers,")
    print(f"  {np.average(y, weights=w):.1%} with the declined ones counted\n")

    print("leave-one-league-out with declines counted (log loss: lower is better; AUC: 0.5 = chance;")
    print("gap: odds shown against the real rate, large leagues):")
    preds = {}
    for name, cols in MODELS.items():
        preds[name] = held_out(rows, x[:, cols], y, w)
        loss, rank, gap = by_league(rows, preds[name], y, w)
        coef = ", ".join(f"{COLUMNS[c]} {v:+.2f}" for c, v in zip(cols, fit(x[:, cols], y, w), strict=True))
        print(f"  {name:20} log loss {loss:.4f}  AUC {rank:.3f}  gap {gap:5.1%}  [{coef}]")
    in_use = np.array([acceptance_odds(r["ratio"], *map(int, r["shape"].split("-for-"))) for r in rows])
    loss, rank, gap = by_league(rows, in_use, y, w)
    print(f"  {'in use':20} log loss {loss:.4f}  AUC {rank:.3f}  gap {gap:5.1%}")

    cols = MODELS[CHOSEN]
    print(f"\n{CHOSEN}, fitted on every league (ACCEPT_COEF: intercept, paper, 2-for-2, 2-for-1, 1-for-2):")
    print("  (" + ", ".join(f"{v:.3f}" for v in fit(x[:, cols], y, w)) + ")")
    print("  in use: " + str(ACCEPT_COEF))
    print("  without declines: (" + ", ".join(f"{v:.3f}" for v in fit(x[:, cols], y, plain)) + ")")

    print("\ncalibration, league-weighted, declines counted:")
    calibration(f"{CHOSEN}, held out", preds[CHOSEN], y, w)
    calibration("in use", in_use, y, w)


if __name__ == "__main__":
    main()

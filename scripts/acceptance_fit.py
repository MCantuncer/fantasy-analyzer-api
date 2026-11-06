"""Fit the trade acceptance model on real offers, and check it on leagues it never saw.

    uv run python -m scripts.acceptance_fit [offers.jsonl.gz]

Reads the offer dataset of scripts/trade_dataset.py (default .cache/tradedata/offers.jsonl.gz).
Accepted offers are labelled 1, withdrawn or expired ones 0; only category leagues are used.

Each offer is described by what the trade finder knows about every deal, seen from the owner
who gets it:

  * paper: log of the trade value they get over the trade value they send (perceived.py's
    blend of ESPN rank, draft price, this season's ranks; % owned isn't in the data)
  * net: the 9-category value they gain, z-scores summed over categories (the scale of
    CategoryModel.value), clipped to +-4. Projection early on, this season's stats later.
  * shape: 2-for-2, 2-for-1 (they must cut someone) and 1-for-2, against a 1-for-1
  * chasing: they sit outside the playoff spots

Logistic regressions on several sets of these are compared leave-one-league-out: each league
is predicted by a model fitted on the others, by log loss and by how well it ranks its own
accepted offers (AUC, 0.5 = chance). Every league counts the same: a few leagues made most of
the offers. The set that predicts held-out leagues best is the one the app uses; the script
prints its odds for app/analysis/trades.py (ACCEPT_BY_SHAPE) and its calibration next to the
hand-tuned odds the app used before.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from functools import cache
from pathlib import Path

import numpy as np

from app.analysis.perceived import (
    DRAFT_FADE_GAMES,
    RECENT_FULL_GAMES,
    SEASON_FULL_GAMES,
    W_AVG_AUCTION,
    W_ESPN,
    W_ESPN_FLOOR,
    W_PAID,
    W_PROJECTION,
    W_RECENT,
    W_SEASON,
    PerceivedValue,
)
from app.analysis.trades import ACCEPT_BY_SHAPE
from app.espn.parse import parse_league
from scripts.trade_shape_backtest import auc, fit, weights

DEFAULT = Path(".cache/tradedata/offers.jsonl.gz")
CATEGORY_LEAGUES = ("H2H_CATEGORY", "H2H_MOST_CATEGORIES", "ROTO")
SHAPES = ("1-for-1", "2-for-2", "2-for-1", "1-for-2")  # sender-for-receiver counts, from the proposer
NET_FULL_SEASON = 1 / 3  # share of the season after which this season's stats replace the projection
MIN_DOLLARS = 0.5  # a package worth less than this counts as this much, so ratios stay finite
PAPER_CLIP = 1.5  # log value ratio, about 0.22x to 4.5x
NET_CLIP = 4.0
CHASING_GAP = 0.0  # playoff_line_gap below this: outside the playoff spots
# Columns of design(): which ones each candidate model uses.
COLUMNS = ["intercept", "paper", "net", "2-for-2", "2-for-1", "1-for-2", "chasing"]
MODELS = {
    "base rate": [0],
    "shape": [0, 3, 4, 5],
    "shape + chasing": [0, 3, 4, 5, 6],
    "net + shape": [0, 2, 3, 4, 5],
    "paper + shape + chasing": [0, 1, 3, 4, 5, 6],
    "everything": [0, 1, 2, 3, 4, 5, 6],
}
CHOSEN = "shape"

# ESPN's rank -> auction dollars ($200 budget), the curve perceived.py prices every rank on,
# read off the recorded League A fixture so every run prices ranks the same way.
FIXTURE = Path("tests/fixtures/chill_primera_2027.json.gz")


@cache
def _curve() -> PerceivedValue:
    with gzip.open(FIXTURE, "rt") as fh:
        rec = json.load(fh)
    return PerceivedValue(parse_league(rec["league"], rec["free_agents"]))


def dollars(rank: float | None) -> float:
    return _curve().dollars_at_rank(rank) if rank else 0.0


def perceived(p: dict, auction: bool) -> float:
    """A player's trade value from what owners could see that day, perceived.py's weights."""
    gp = (p.get("season") or {}).get("gp") or 0
    fade = max(1 - gp / DRAFT_FADE_GAMES, 0.0)
    pre, mine = p.get("preseason") or {}, p.get("league_draft") or {}
    sig: list[tuple[float, float]] = []
    if pre.get("projection_rank"):
        sig.append((W_ESPN_FLOOR + (W_ESPN - W_ESPN_FLOOR) * fade, dollars(pre["projection_rank"])))
        sig.append((W_PROJECTION, dollars(pre["projection_rank"])))
    if fade:
        if auction and pre.get("espn_auction_avg"):
            sig.append((W_AVG_AUCTION * fade, pre["espn_auction_avg"]))
        elif pre.get("espn_adp"):
            sig.append((W_AVG_AUCTION * fade, dollars(pre["espn_adp"])))
        if auction and mine.get("bid"):
            sig.append((W_PAID * fade, mine["bid"]))
        elif mine.get("pick"):
            sig.append((W_PAID * fade, dollars(mine["pick"])))
    season, recent = p.get("season") or {}, p.get("last15") or {}
    if season.get("rank") and gp:
        sig.append((W_SEASON * min(gp / SEASON_FULL_GAMES, 1.0), dollars(season["rank"])))
    if recent.get("rank") and gp:
        sig.append((W_RECENT * min(gp / RECENT_FULL_GAMES, 1.0), dollars(recent["rank"])))
    total = sum(w for w, _ in sig)
    return sum(w * x for w, x in sig) / total if total else 0.0


def package(values: list[float]) -> float:
    return sum(max(v, 0.0) ** 1.5 for v in values) ** (1 / 1.5) if values else 0.0


def load(path: Path) -> list[dict]:
    rows = []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            d = json.loads(line)
            if d["outcome"] not in ("accepted", "withdrawn") or d["league"]["scoring_type"] not in CATEGORY_LEAGUES:
                continue
            if len(d["sides"]) != 2 or d["proposer"] < 0:
                continue
            proposer = str(d["proposer"])
            receiver = next(k for k in d["sides"] if k != proposer)
            side = d["sides"][receiver]
            shape = f"{len(d['sides'][proposer]['gives'])}-for-{len(d['sides'][proposer]['gets'])}"
            players = {p["pid"]: p for p in d["players"]}
            auction = d["league"].get("draft_type") == "AUCTION"
            gets = [perceived(players[i], auction) for i in side["gets"] if i in players]
            gives = [perceived(players[i], auction) for i in side["gives"] if i in players]
            if shape not in SHAPES or not gets or not gives:
                continue
            w = min(d["season_fraction"] / NET_FULL_SEASON, 1.0)
            net = (1 - w) * side["net_projection_value"] + w * side["net_season"]
            team = next((t for t in d["teams"] if str(t["team_id"]) == receiver), None)
            chasing = bool(team and team["periods_done"] > 0 and team["playoff_line_gap"] < CHASING_GAP)
            rows.append(
                {
                    "league": d["league_id"],
                    "season": d["season"],
                    "paper": max(package(gets), MIN_DOLLARS) / max(package(gives), MIN_DOLLARS),
                    "net": net,
                    "shape": shape,
                    "chasing": chasing,
                    "accepted": d["outcome"] == "accepted",
                }
            )
    return rows


def design(rows: list[dict]) -> np.ndarray:
    return np.column_stack(
        [np.ones(len(rows))]
        + [np.array([np.clip(math.log(r["paper"]), -PAPER_CLIP, PAPER_CLIP) for r in rows])]
        + [np.array([np.clip(r["net"], -NET_CLIP, NET_CLIP) for r in rows])]
        + [np.array([r["shape"] == s for r in rows], float) for s in SHAPES[1:]]
        + [np.array([r["chasing"] for r in rows], float)]
    )


def old_odds(r: dict) -> float:
    """The hand-tuned odds the app used before, with net value standing in for their gain."""
    p = float(np.interp(r["paper"], [0.85, 0.95, 1.05, 1.2, 1.4], [0.2, 0.4, 0.55, 0.7, 0.8]))
    p += 0.1 if r["net"] >= 1.0 else -0.1 if r["net"] < 0 else 0.0
    p += 0.05 if r["chasing"] else 0.0
    p = float(np.clip(p, 0.05, 0.9))
    k = {"2-for-2": 0.63, "2-for-1": 0.40}.get(r["shape"], 1.0)
    return float(np.clip(p * k / (1 - p + p * k), 0.05, 0.9))


def calibration(name: str, p: np.ndarray, y: np.ndarray, w: np.ndarray) -> None:
    print(f"  {name}:")
    for lo, hi in ((0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.45), (0.45, 0.6), (0.6, 1.01)):
        m = (p >= lo) & (p < hi)
        if m.sum() >= 10:
            print(
                f"    shown {lo:.0%}-{min(hi, 1):.0%}: {m.sum():5d} offers,"
                f" avg shown {np.average(p[m], weights=w[m]):5.1%}, accepted {np.average(y[m], weights=w[m]):5.1%}"
            )


def log_loss(p: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.average(y * np.log(p) + (1 - y) * np.log(1 - p), weights=w))


def held_out(rows: list[dict], x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Each league's odds from a model fitted on every other league."""
    leagues = np.array([r["league"] for r in rows])
    p = np.zeros(len(rows))
    for lg in set(leagues):
        test = leagues == lg
        train = [r for r, t in zip(rows, test, strict=True) if not t]
        b = fit(x[~test], y[~test], weights(train))
        p[test] = 1 / (1 + np.exp(-(x[test] @ b)))
    return p


def league_auc(rows: list[dict], p: np.ndarray, y: np.ndarray) -> float:
    leagues = np.array([r["league"] for r in rows])
    return float(np.nanmean([auc(p[leagues == lg], y[leagues == lg]) for lg in set(leagues)]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", type=Path, default=DEFAULT)
    args = ap.parse_args()
    rows = load(args.path)
    y = np.array([r["accepted"] for r in rows], float)
    w = weights(rows)
    x = design(rows)
    print(f"{len(rows)} offers, {int(y.sum())} accepted, {len({r['league'] for r in rows})} category leagues")
    print(f"league-balanced acceptance rate {np.average(y, weights=w):.1%}\n")

    print("leave-one-league-out (log loss: lower is better; AUC within each league, 0.5 = chance):")
    preds = {}
    for name, cols in MODELS.items():
        preds[name] = held_out(rows, x[:, cols], y)
        b = fit(x[:, cols], y, w)
        coef = ", ".join(f"{COLUMNS[c]} {v:+.2f}" for c, v in zip(cols, b, strict=True))
        loss, rank = log_loss(preds[name], y, w), league_auc(rows, preds[name], y)
        print(f"  {name:24} log loss {loss:.4f}  AUC {rank:.3f}  [{coef}]")
    old = np.array([old_odds(r) for r in rows])
    print(f"  {'old hand-tuned':24} log loss {log_loss(old, y, w):.4f}  AUC {league_auc(rows, old, y):.3f}")

    b = fit(x[:, MODELS[CHOSEN]], y, w)
    odds = {
        (1, 1): b[0],
        (2, 2): b[0] + b[1],
        (2, 1): b[0] + b[2],
        (1, 2): b[0] + b[3],
    }
    print(f"\n{CHOSEN} model, fitted on every league (ACCEPT_BY_SHAPE):")
    print("  {" + ", ".join(f"{k}: {1 / (1 + math.exp(-v)):.2f}" for k, v in odds.items()) + "}")
    print(f"  in use: {ACCEPT_BY_SHAPE}")

    print("\ncalibration, league-weighted:")
    calibration(f"{CHOSEN}, held out", preds[CHOSEN], y, w)
    calibration("old hand-tuned", old, y, w)
    recent = np.array([r["season"] >= 2025 for r in rows])
    if recent.sum():
        print(f"\nlast two seasons only ({recent.sum()} offers):")
        calibration(f"{CHOSEN}, held out", preds[CHOSEN][recent], y[recent], w[recent])
        calibration("old hand-tuned", old[recent], y[recent], w[recent])


if __name__ == "__main__":
    main()

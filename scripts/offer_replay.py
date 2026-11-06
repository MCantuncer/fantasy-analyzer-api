"""Replay real trade offers through the trade finder, on the league as it stood that day.

    uv run python -m scripts.offer_replay [--limit N] [--workers N] [--out FILE]

Every accepted or withdrawn offer in a category league (scripts/trade_dataset.py, default
.cache/tradedata/offers.jsonl.gz, histories in .cache/tradedata/history/) is scored the way
the finder scores its own deals: the league is rebuilt on the offer's day (app/backtest/replay.py:
that day's rosters, stats from earlier box scores only, records so far), the players go back
to the teams that held them before the offer, and TradeFinder._evaluate runs with the
proposer as "me". Its checks are measured, not applied:

  * my gain: the proposer's expected categories won per week
  * their gain: the other owner's, in the categories they compete in (not their punts)
  * paper ratio: trade value they get over what they send, owner premium included

The report answers two questions about the finder's rules, every league weighted the same:
would the deals it suggests (both gains and the paper band, MIN_* in trades.py) be accepted
more often than other offers, and does each input rank accepted offers above withdrawn ones
within a league (AUC, 0.5 = chance)? The season simulation's playoff-odds check is left out:
it takes seconds per deal. Offers before any game was played are skipped (no spreads yet).

The replayed rows go to .cache/tradedata/replayed.jsonl, which scripts/acceptance_fit.py
fits the acceptance odds on.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import cache
from pathlib import Path

import numpy as np

import app.analysis.trades as trades
from app.analysis.model import CategoryModel
from app.backtest.history import History
from app.backtest.replay import Replay
from app.espn.parse import League
from scripts.trade_shape_backtest import auc

CACHE = Path(".cache/tradedata")
OFFERS = CACHE / "offers.jsonl.gz"
REPLAYED = CACHE / "replayed.jsonl"  # read by scripts/acceptance_fit.py
CATEGORY_LEAGUES = ("H2H_CATEGORY", "H2H_MOST_CATEGORIES", "ROTO")
GAIN_BINS = (-np.inf, -0.1, 0.0, trades.MIN_THEIR_GAIN, 0.1, 0.2, np.inf)
PAPER_MAX = trades.MAX_VALUE_RATIO  # the finder's paper band tops out here; the replay lifts the cap to measure
RATIO_BINS = (0.0, 0.7, trades.MIN_VALUE_RATIO, 1.0, 1.2, PAPER_MAX, np.inf)


def load_offers(path: Path) -> list[dict]:
    out = []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            d = json.loads(line)
            if d["outcome"] not in ("accepted", "withdrawn") or d["league"]["scoring_type"] not in CATEGORY_LEAGUES:
                continue
            if len(d["sides"]) != 2 or d["proposer"] < 0:
                continue
            moves = [(p["pid"], p["fantasy_team"], p["to_team"]) for p in d["players"]]
            ranks = {
                p["pid"]: ((p.get("season") or {}).get("rank"), (p.get("last15") or {}).get("rank"))
                for p in d["players"]
            }
            if not moves:
                continue
            out.append(
                {
                    "league": d["league_id"],
                    "season": d["season"],
                    "day": d["day"],
                    "proposer": d["proposer"],
                    "receiver": next(int(k) for k in d["sides"] if int(k) != d["proposer"]),
                    "moves": moves,
                    "ranks": ranks,
                    "accepted": d["outcome"] == "accepted",
                }
            )
    return out


@cache
def _replay(league_id: int, season: int) -> Replay | None:
    path = CACHE / "history" / f"{league_id}-{season}.json.gz"
    if not path.exists():
        return None
    try:
        return Replay(History.load(path), absences=False)
    except ValueError:  # no NBA schedule for that season
        return None


@cache
def _meta(league_id: int, season: int) -> dict[int, dict]:
    path = CACHE / "meta" / f"{league_id}-{season}.json.gz"
    if not path.exists():
        return {}
    with gzip.open(path, "rt") as fh:
        return {int(k): v for k, v in json.load(fh).items()}


def _draft(h: History) -> tuple[dict[int, float], dict[int, int]]:
    picks = [p for p in (h.league.get("draftDetail") or {}).get("picks", []) if p.get("playerId", -1) > 0]
    prices = {int(p["playerId"]): float(p["bidAmount"]) for p in picks if p.get("bidAmount")}
    order = {int(p["playerId"]): int(p["overallPickNumber"]) for p in picks if p.get("overallPickNumber")}
    return prices, order


def enrich(lg: League, h: History, offer: dict) -> None:
    """What owners saw that the replayed league lacks: ESPN's preseason market and the draft.

    The pool meta has ESPN's average draft position and auction price; its preseason rank is
    taken as the ADP order (ESPN's own rank isn't kept). Season and last-15 ranks come from the
    offer's player features for the players in it. % owned is left out: the data only has it
    at season's end.
    """
    meta = _meta(offer["league"], offer["season"])
    by_adp = sorted((m["adp"], pid) for pid, m in meta.items() if m.get("adp"))
    rank = {pid: r for r, (_, pid) in enumerate(by_adp, 1)}
    for t in lg.teams:
        for p in t.roster:
            m = meta.get(p.id) or {}
            p.avg_draft_position = float(m.get("adp") or 0.0)
            p.avg_auction_value = float(m.get("auction_avg") or 0.0)
            p.espn_auction_value = p.avg_auction_value
            p.espn_rank = rank.get(p.id)
            p.percent_owned = 0.0
            season, recent = offer["ranks"].get(p.id, (None, None))
            p.season_rank, p.recent_rank = season, recent
    lg.draft_prices, lg.draft_picks = _draft(h)


def score(offer: dict) -> dict | None:
    """The finder's view of one real offer, or None when the league can't be rebuilt that day."""
    rp = _replay(offer["league"], offer["season"])
    if rp is None or offer["day"] < 1:
        return None
    lg = rp.snapshot(rp.period_of(offer["day"]), offer["day"])
    enrich(lg, rp.h, offer)
    by_id = {p.id: p for t in lg.teams for p in t.roster}
    teams = {t.id: t for t in lg.teams}
    for pid, frm, _ in offer["moves"]:  # each player back on the team that held him before the offer
        p = by_id.get(pid)
        if p is None or frm not in teams:
            return None
        for t in lg.teams:
            t.roster[:] = [x for x in t.roster if x.id != pid]
        p.on_team_id = frm
        teams[frm].roster.append(p)
    idx = {t.id: i for i, t in enumerate(lg.teams)}
    if offer["proposer"] not in idx or offer["receiver"] not in idx:
        return None
    give = [by_id[pid] for pid, frm, _ in offer["moves"] if frm == offer["proposer"]]
    get = [by_id[pid] for pid, frm, _ in offer["moves"] if frm == offer["receiver"]]
    if not give or not get:
        return None
    trades.MAX_VALUE_RATIO = float("inf")  # measure the ratio instead of filtering on it
    finder = trades.TradeFinder(lg, CategoryModel(lg), idx[offer["proposer"]], keep=set())
    d = finder._evaluate(
        idx[offer["receiver"]], give, get, min_my_gain=-np.inf, min_their_gain=-np.inf, min_ratio=0.0,
        min_margin=-np.inf,
    )  # fmt: skip
    if d is None or not np.isfinite(d.my_gain) or not np.isfinite(d.their_gain):
        return None  # before any games the category spreads are zero and gains undefined
    ratio = d.give_value / d.get_value
    return {
        "league": offer["league"],
        "season": offer["season"],
        "accepted": offer["accepted"],
        "shape": f"{len(give)}-for-{len(get)}",
        "my_gain": d.my_gain,
        "their_gain": d.their_gain,
        "ratio": ratio,
        "suggested": d.my_gain >= trades.MIN_MY_GAIN
        and d.their_gain >= trades.MIN_THEIR_GAIN
        and trades.MIN_VALUE_RATIO <= ratio <= PAPER_MAX,
    }


def rate(rows: list[dict]) -> tuple[float, int]:
    """League-weighted acceptance rate and the number of offers."""
    by: dict[int, list[bool]] = defaultdict(list)
    for r in rows:
        by[r["league"]].append(r["accepted"])
    return (float(np.mean([np.mean(v) for v in by.values()])) if by else float("nan")), len(rows)


def within_league_auc(rows: list[dict], key: str) -> float:
    by: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by[r["league"]].append(r)
    vals = [
        auc(np.array([r[key] for r in rs], float), np.array([r["accepted"] for r in rs], float)) for rs in by.values()
    ]
    return float(np.nanmean(vals))


def table(rows: list[dict], key: str, bins: tuple[float, ...], label: str) -> None:
    print(f"\n{label}:")
    for lo, hi in zip(bins, bins[1:], strict=False):
        p, n = rate([r for r in rows if lo <= r[key] < hi])
        if n:
            print(f"  {lo:>6.2f} to {hi:<6.2f} {n:5d} offers  accepted {p:5.1%}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", type=Path, default=OFFERS)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=REPLAYED, help="where the replayed rows go, one JSON object a line")
    args = ap.parse_args()
    offers = load_offers(args.path)
    offers.sort(key=lambda o: (o["league"], o["season"], o["day"]))  # one league's replays per worker run
    if args.limit:
        offers = offers[: args.limit]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = [r for r in pool.map(score, offers, chunksize=32) if r is not None]
    if not args.limit:
        args.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(offers)} offers, {len(rows)} replayed, {sum(r['accepted'] for r in rows)} accepted")
    base, _ = rate(rows)
    print(f"league-weighted acceptance {base:.1%}")
    yes, no = [r for r in rows if r["suggested"]], [r for r in rows if not r["suggested"]]
    print(f"  passes the finder's checks:  {rate(yes)[1]:5d} offers, accepted {rate(yes)[0]:5.1%}")
    print(f"  fails them:                  {rate(no)[1]:5d} offers, accepted {rate(no)[0]:5.1%}")
    for key in ("their_gain", "my_gain", "ratio"):
        print(f"  within-league AUC of {key}: {within_league_auc(rows, key):.3f}")
    table(rows, "their_gain", GAIN_BINS, "their gain (expected categories won per week)")
    table(rows, "ratio", RATIO_BINS, "paper ratio (what they get / what they send)")
    both = [r for r in rows if r["their_gain"] >= trades.MIN_THEIR_GAIN and r["my_gain"] >= trades.MIN_MY_GAIN]
    print(f"\nboth sides gain: {rate(both)[1]} offers, accepted {rate(both)[0]:.1%}")
    in_band = [r for r in both if trades.MIN_VALUE_RATIO <= r["ratio"] <= PAPER_MAX]
    print(f"  and inside the paper band: {rate(in_band)[1]} offers, accepted {rate(in_band)[0]:.1%}")


if __name__ == "__main__":
    main()

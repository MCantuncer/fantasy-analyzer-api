"""Score the model on finished seasons of a public league, replayed week by week.

Usage: uv run python -m scripts.backtest LEAGUE_ID SEASON [SEASON ...] [--json OUT.json] [--flat-durability]
The first run of a season downloads it from ESPN (a few hundred requests, a few minutes)
into .cache/backtest/; later runs read the cache. Only public leagues: no cookies.

Judge a model change by whether it improves every season, not by its best one:
tuning to one season's noise is what this tool is meant to catch.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from app.backtest.history import load_history, load_past_games
from app.backtest.replay import Replay

CACHE = Path(".cache/backtest")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("league_id", type=int)
    ap.add_argument("seasons", type=int, nargs="+")
    ap.add_argument("--json", type=Path, help="also write the full scores here")
    ap.add_argument("--sims", type=int, default=2000, help="season simulations per cut week")
    ap.add_argument(
        "--flat-durability", action="store_true", help="ignore games-played history (every player equally durable)"
    )
    args = ap.parse_args()

    out = {}
    for season in args.seasons:
        h = asyncio.run(load_history(args.league_id, season, CACHE))
        past = None if args.flat_durability else asyncio.run(load_past_games(season, CACHE))
        replay = Replay(h, past)
        weekly, season_score = replay.weekly().summary(), replay.season(n_sims=args.sims).summary()
        out[season] = {"weekly": weekly, "season": season_score}
        print(f"\n{season}: weekly categories n={weekly['n']}")
        print(f"  brier {weekly['brier']:.4f}  log loss {weekly['log_loss']:.4f}  spread {weekly['spread']:.3f}")
        print(f"  {'cat':5} {'brier':>7} {'spread':>7} {'bias %':>7}")
        for cat, s in weekly["categories"].items():
            print(f"  {cat:5} {s['brier']:7.4f} {s['spread']:7.3f} {s['bias_pct']:7.2f}")
        print(f"{season}: playoff odds n={season_score['n']}")
        print(
            f"  brier {season_score['brier']:.4f}  log loss {season_score['log_loss']:.4f}"
            f"  record spread {season_score['spread']:.3f}"
        )
        for cut, s in season_score["by_cut"].items():
            print(f"  from week {cut:2}: brier {s['brier']:.4f}  spread {s['spread']:.3f}")
    if args.json:
        args.json.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

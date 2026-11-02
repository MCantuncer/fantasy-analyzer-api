"""Checks on past seasons that vision-trade targets really are worth more later.

    uv run python -m scripts.vision_backtest LEAGUE_ID SEASON [SEASON ...]

Vision trades (app/analysis/trades.py, mode "vision") buy a player for what the model
projects he'll do the rest of the way, while his owner sees what he has done so far. That
only pays if the projection beats the season so far for those players. At the start of
matchup periods 4 to 12, the league is rebuilt as it stood that day (app/backtest/replay.py)
and every rostered player is scored against what he really did from then to the end of
the fantasy playoffs, by the value of his per-game line in this league's categories
(z-scores summed, CategoryModel.line_value).

  * Slow starts: healthy players with 5+ games whose projection is 1+ above their season so
    far (the vision signal), against every other healthy player as a control. Players with
    fewer than 10 games left are skipped.
  * Players out (missed their team's last 3 games, the replay's stand-in for an injury tag)
    and expected to miss 6+ more: games missed and played against expected, and value per
    game once back against projected. Healthy players projected as high are the control for
    plain regression to the mean.

Uses the season cache of scripts/backtest.py (.cache/backtest/).
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
from dataclasses import dataclass
from pathlib import Path

from app.analysis.absence import is_out
from app.analysis.model import CategoryModel
from app.analysis.trades import VISION_MIN_GAMES, VISION_SLOW_GAP
from app.backtest.history import load_history, load_past_games
from app.backtest.replay import Replay

CACHE = Path(".cache/backtest")
CUTS = (4, 6, 8, 10, 12)
MIN_LEFT = 10  # games left for a healthy player to be scored
LONG_OUT = 6.0  # expected games still to miss
GOOD = 2.0  # projected value of a "good" player


@dataclass
class Healthy:
    projected: float
    so_far: float
    actual: float  # rest of the season, per game

    @property
    def slow(self) -> bool:
        return self.projected - self.so_far >= VISION_SLOW_GAP


@dataclass
class Out:
    projected: float
    missed_expected: float
    missed: int
    played_expected: int
    played: int
    actual: float | None  # per game once back


def _avg(lines: list[dict]) -> dict[int, float]:
    keys = {k for s in lines for k in s}
    return {int(k): sum(s.get(k, 0.0) for s in lines) / len(lines) for k in keys}


def run(league_id: int, season: int) -> tuple[list[Healthy], list[Out]]:
    h = asyncio.run(load_history(league_id, season, CACHE))
    replay = Replay(h, asyncio.run(load_past_games(season, CACHE)))
    healthy, out = [], []
    for period in CUTS:
        league = replay.snapshot(period)
        model = CategoryModel(league)
        cal = league.calendar
        day, end = cal.today, max(d for days in cal.period_days.values() for d in days)
        for p in (p for t in league.teams for p in t.roster):
            later = [(d, s) for d, _, s in replay.lines.get(p.id, []) if day <= d <= end]
            actual = model.line_value(_avg([s for _, s in later])) if later else None
            if is_out(p) and (p.games_out or 0.0) >= LONG_OUT:
                team = [d for d in cal.team_days.get(p.pro_team_id, ()) if day <= d <= end]
                back = later[0][0] if later else end + 1
                out.append(
                    Out(
                        model.value(p),
                        p.games_out,
                        sum(d < back for d in team),
                        max(len(team) - round(p.games_out), 0),
                        len(later),
                        actual,
                    )
                )
            elif p.available and p.games_played >= VISION_MIN_GAMES and p.season_line and len(later) >= MIN_LEFT:
                healthy.append(Healthy(model.value(p), model.line_value(p.season_line), actual))
    return healthy, out


def _mean(xs) -> float:
    xs = list(xs)
    return statistics.fmean(xs) if xs else float("nan")


def report_healthy(name: str, rows: list[Healthy]) -> None:
    print(
        f"  {name:24} n={len(rows):4}  projected {_mean(r.projected for r in rows):+5.2f}"
        f"  so far {_mean(r.so_far for r in rows):+5.2f}  actual {_mean(r.actual for r in rows):+5.2f}"
        f"  | mean abs error: projection {_mean(abs(r.projected - r.actual) for r in rows):4.2f}"
        f", season so far {_mean(abs(r.so_far - r.actual) for r in rows):4.2f}"
    )


def report_out(name: str, rows: list[Out]) -> None:
    back = [r for r in rows if r.actual is not None]
    print(
        f"  {name:24} n={len(rows):4}  missed {_mean(r.missed for r in rows):4.1f} (expected"
        f" {_mean(r.missed_expected for r in rows):4.1f})  played {_mean(r.played for r in rows):4.1f}"
        f" (expected {_mean(r.played_expected for r in rows):4.1f})  per game once back"
        f" {_mean(r.actual for r in back):+5.2f} (projected {_mean(r.projected for r in back):+5.2f})"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("league_id", type=int)
    ap.add_argument("seasons", type=int, nargs="+")
    args = ap.parse_args()
    for season in args.seasons:
        healthy, out = run(args.league_id, season)
        print(f"\n{season}: value per game (z-scores summed), from the cut day to the playoffs' end")
        report_healthy("slow start (signal)", [r for r in healthy if r.slow])
        report_healthy("other healthy (control)", [r for r in healthy if not r.slow])
        report_healthy(f"healthy projected >{GOOD:g}", [r for r in healthy if r.projected > GOOD])
        report_out("out 6+ more games", out)
        report_out(f"out, projected >{GOOD:g}", [r for r in out if r.projected > GOOD])


if __name__ == "__main__":
    main()

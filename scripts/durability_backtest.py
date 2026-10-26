"""Checks the expected-games model (app/analysis/durability.py) against past seasons.

    uv run python -m scripts.durability_backtest            # fit on 2024, test on 2025+2026
    uv run python -m scripts.durability_backtest --fit      # also search the constants on the fit seasons

For each player ESPN projected for 50+ games, the model sees his last three seasons'
games played and his first N team games of the season (N = 0, 10,
20, 40). It predicts the share of his team's remaining games he plays once any absence
running at that point is over (the injury status covers that one), which game logs
then tell. Compared with: the league's typical share for everyone, ESPN's projection
alone, and the model without this season's games.

Season pools (games played, ESPN projections) are cached in .cache/season_pools and
game logs in .cache/gamelogs (shared with scripts/injury_backtest.py).
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

import httpx

from app.analysis import durability as dur
from app.analysis.durability import PastSeason, expected_share
from app.espn.parse import GAMES_PLAYED

URL = "https://lm-api-reads.fantasy.espn.com/apis/v3/games/fba/seasons/{season}/players"
POOLS = Path(".cache/season_pools")
LOGS = Path(".cache/gamelogs")
POOL_SIZE = 500
FIT_SEASONS = (2024,)  # 2023's stored ESPN projections are rest-of-season ones, so it can't be filtered alike
TEST_SEASONS = (2025, 2026)
CUTOFFS = (0, 10, 20, 40)
MIN_PROJECTED = 50
MIN_REST = 30  # team games left after the cutoff, so the outcome isn't a handful of games
GAME_SPLIT = 5


def _get(season: int, flt: dict, path: Path) -> list[dict]:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        resp = httpx.get(
            URL.format(season=season),
            params={"view": "kona_player_info", "scoringPeriodId": "0"},
            headers={"X-Fantasy-Filter": json.dumps(flt)},
            timeout=90,
        )
        resp.raise_for_status()
        path.write_text(resp.text)
    return json.loads(path.read_text())


def season_pool(season: int) -> tuple[dict[int, int], dict[int, float]]:
    """Games played and ESPN's preseason projected games, by player, for one season."""
    flt = {"limit": POOL_SIZE, "sortPercOwned": {"sortPriority": 1, "sortAsc": False}}
    played: dict[int, int] = {}
    projected: dict[int, float] = {}
    for e in _get(season, flt, POOLS / f"{season}.json"):
        p = e.get("player") or e
        for s in p.get("stats") or []:
            if s.get("seasonId") != season or s.get("statSplitTypeId") != 0:
                continue
            gp = (s.get("stats") or {}).get(str(GAMES_PLAYED))
            if gp is None:
                continue
            if s.get("statSourceId") == 0:
                played[int(p["id"])] = int(gp)
            elif s.get("statSourceId") == 1:
                projected[int(p["id"])] = float(gp)
    return played, projected


def game_logs(season: int) -> dict[int, list[bool]]:
    """Each player's team games in order: True when he played. Games he sat on a roster count too."""
    days: dict[int, dict[int, bool]] = {}
    for team in range(1, 31):
        flt = {
            "filterProTeamIds": {"value": [team]},
            "filterStatsForSplitTypeIds": {"value": [GAME_SPLIT]},
            "limit": 60,
            "sortPercOwned": {"sortPriority": 1, "sortAsc": False},
        }
        for e in _get(season, flt, LOGS / f"{season}-{team}.json"):
            p = e.get("player") or e
            mine = days.setdefault(int(p["id"]), {})
            for s in p.get("stats") or []:
                if s.get("seasonId") != season or s.get("statSplitTypeId") != GAME_SPLIT or s.get("statSourceId"):
                    continue
                if s.get("scoringPeriodId") is not None:
                    mine[int(s["scoringPeriodId"])] = bool((s.get("stats") or {}).get(str(GAMES_PLAYED)))
    return {pid: [g[d] for d in sorted(g)] for pid, g in days.items()}


@dataclass
class Case:
    season: int
    cutoff: int
    past: list[PastSeason | None]
    projected: float
    played: int
    rest: float  # share of the remaining team games he played


def season_length(played: dict[int, int]) -> int:
    """Games in that NBA season: the most anyone played (82, fewer in a shortened season; traded players can top 82)."""
    return min(max(played.values(), default=dur.FULL_SEASON), dur.FULL_SEASON)


def cases(season: int) -> list[Case]:
    pools = {s: season_pool(s) for s in range(season - dur.HISTORY_SEASONS, season + 1)}
    lengths = {s: season_length(pools[s][0]) for s in pools}
    _, projected = pools[season]
    logs = game_logs(season)
    out = []
    for pid, proj in projected.items():
        games = logs.get(pid)
        if proj < MIN_PROJECTED or not games:
            continue
        past = [
            PastSeason(pools[s][0][pid], lengths[s]) if pid in pools[s][0] else None
            for s in range(season - 1, season - 1 - dur.HISTORY_SEASONS, -1)
        ]
        for n in CUTOFFS:
            # An absence still running at the cutoff is a known injury, which the model
            # handles from the injury status; the outcome starts from his next game.
            back = next((i for i in range(n, len(games)) if games[i]), len(games))
            rest = games[back:]
            if len(rest) < MIN_REST:
                continue
            out.append(Case(season, n, past, proj, sum(games[:n]), sum(rest) / len(rest)))
    return out


def rmse(cs: list[Case], predict) -> float:
    """Root mean squared error: fitting it gets the expected share right, which values add up."""
    return statistics.fmean((predict(c) - c.rest) ** 2 for c in cs) ** 0.5


def model(**kw):
    return lambda c: expected_share(c.past, c.played, c.cutoff, **kw)


def report(name: str, cs: list[Case], kw: dict) -> None:
    rules = {
        "typical share": lambda c: kw.get("league_share", dur.LEAGUE_SHARE),
        "ESPN projection": lambda c: min(c.projected / dur.FULL_SEASON, 1.0),
        "model, no season games": lambda c: expected_share(c.past, **kw),
        "model": model(**kw),
    }
    print(f"\n{name}: root mean squared error in share of remaining games (lower is better)")
    print(f"  {'after N games':24}" + "".join(f"{n:>8}" for n in CUTOFFS) + f"{'all':>8}")
    for label, rule in rules.items():
        cols = [rmse([c for c in cs if c.cutoff == n], rule) for n in CUTOFFS]
        print(f"  {label:24}" + "".join(f"{v:8.4f}" for v in cols) + f"{rmse(cs, rule):8.4f}")
    # Calibration: players split into fifths by predicted share, against what they played.
    for n in (0, 10):
        ranked = sorted((model(**kw)(c), c.rest) for c in cs if c.cutoff == n)
        q = len(ranked) // 5
        cells = [ranked[i * q : (i + 1) * q] for i in range(5)]
        print(
            f"  after {n:2} games, by fifth: "
            + "  ".join(
                f"{statistics.fmean(p for p, _ in g):.2f}->{statistics.fmean(a for _, a in g):.2f}" for g in cells
            )
            + "  (predicted->played)"
        )
    # The case asked about: injury-prone players off to a healthy start.
    prone = [c for c in cs if c.cutoff == 10 and expected_share(c.past, **kw) < dur.PRONE_SHARE]
    for label, group in (
        ("prone, 9+ of first 10", [c for c in prone if c.played >= 9]),
        ("prone, missed 2+ of 10", [c for c in prone if c.played < 9]),
    ):
        if group:
            actual = statistics.fmean(c.rest for c in group)
            pred = statistics.fmean(model(**kw)(c) for c in group)
            print(f"  {label:24} n={len(group):3}  played {actual:.3f} of the rest, model said {pred:.3f}")


def fit(cs: list[Case]) -> dict:
    share = statistics.fmean(c.rest for c in cs if c.cutoff == 0)
    grid = itertools.product((82.0, 120.0, 160.0, 240.0, 320.0), (0.3, 0.5, 0.7, 1.0), (2.0, 3.0, 5.0, 8.0, 12.0))
    best = min(
        grid, key=lambda g: rmse(cs, model(league_share=share, prior_games=g[0], decay=g[1], current_weight=g[2]))
    )
    return dict(league_share=round(share, 3), prior_games=best[0], decay=best[1], current_weight=best[2])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fit", action="store_true", help="search the constants on the fit seasons first")
    args = ap.parse_args()
    fit_cases = [c for s in FIT_SEASONS for c in cases(s)]
    test_cases = [c for s in TEST_SEASONS for c in cases(s)]
    kw: dict = {}
    if args.fit:
        kw = fit(fit_cases)
        print("fitted:", kw)
    report(f"fit seasons {FIT_SEASONS}", fit_cases, kw)
    report(f"test seasons {TEST_SEASONS}", test_cases, kw)


if __name__ == "__main__":
    main()

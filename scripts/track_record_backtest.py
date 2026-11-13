"""Checks how fast this season should take over from the projection for players with a track record.

    uv run python -m scripts.track_record_backtest

A proven player has PROVEN_GAMES or more in each of the last PROVEN_SEASONS seasons
(app/espn/parse.py). For players ESPN projected for 20+ games, after N of their games
(10, 15, 20, 30), the projection as the app builds it (parse._blend_stats, age trend
included) predicts their per-game line over the rest of the season (20+ more games),
which game logs then tell. Scored by the value of the line in standard 9-cat terms, as
scripts/age_backtest.py does.

Printed:
  * value error for each slowdown tried, relative to none (below 1 is better), by how
    many of the last three seasons a player was a regular in, on every season. A
    slowdown is kept only if it wins on the fit season (2023-24) and on both after.
  * the chosen rule's error at each checkpoint.
  * slow and hot starters (value so far 1.5 or more away from the projection after 15
    games): how much of the gap they closed over the rest of the season, against what
    the plain blend and the chosen rule expect.

Uses the caches of scripts/durability_backtest.py (.cache/season_pools, .cache/gamelogs).
"""

from __future__ import annotations

from dataclasses import dataclass

from app.espn import parse
from app.espn.ages import season_age
from app.espn.parse import GAMES_PLAYED, PROVEN_GAMES, PROVEN_SEASONS, _blend_stats
from scripts.age_backtest import STATS, _entry, _names, scales, value
from scripts.durability_backtest import season_pool
from scripts.minutes_backtest import season_lines
from scripts.projection_backtest import _total, game_lines, projections

FIT, TEST = 2024, (2025, 2026)
CUTS = (10, 15, 20, 30)
MIN_REST = 20
TRIED = (1.0, 1.5, 2.0, 3.0)
FAR = 1.5  # a slow or hot start: value so far this far from the projection
STARTER_CUT = 15


@dataclass
class Case:
    season: int
    cut: int
    regular_seasons: int  # of the last PROVEN_SEASONS seasons, how many he played PROVEN_GAMES in
    prior: float  # value of the projection before any games
    so_far: float
    rest: float
    left: int  # games after the cut, the weight of the case
    predicted: dict[float, float]  # slowdown -> value of the blend

    @property
    def proven(self) -> bool:
        return self.regular_seasons == PROVEN_SEASONS


def _blend_at(player: dict, season: int, age: float, slowdown: float) -> dict[int, float]:
    chosen = parse.PROVEN_SLOWDOWN
    parse.PROVEN_SLOWDOWN = slowdown
    try:
        return _blend_stats(player, season, age, proven=True)[0]
    finally:
        parse.PROVEN_SLOWDOWN = chosen


def cases(season: int) -> list[Case]:
    names, lines, last, sc = _names(season), game_lines(season), season_lines(season - 1), scales(season)
    past = [season_pool(season - k)[0] for k in range(1, PROVEN_SEASONS + 1)]
    out = []
    for pid, proj in projections(season).items():
        age, games = season_age(names.get(pid, ""), season), lines.get(pid, [])
        if age is None:
            continue
        regular = sum(played.get(pid, 0) >= PROVEN_GAMES for played in past)
        before = [_entry(season, 1, proj)] + ([_entry(season - 1, 0, last[pid])] if pid in last else [])
        prior = value(_blend_stats({"stats": before}, season, age)[0], sc)
        for n in CUTS:
            if len(games) < n + MIN_REST:
                continue
            so_far = {s: _total(games[:n], s) / n for s in STATS} | {GAMES_PLAYED: float(n)}
            rest = {s: _total(games[n:], s) / (len(games) - n) for s in STATS}
            player = {"stats": [*before, _entry(season, 0, so_far)]}
            predicted = {m: value(_blend_at(player, season, age, m), sc) for m in TRIED}
            out.append(Case(season, n, regular, prior, value(so_far, sc), value(rest, sc), len(games) - n, predicted))
    return out


def rmse(cs: list[Case], slowdown) -> float:
    """Value error weighted by games left; ``slowdown`` is a number or a function of the case."""
    pick = slowdown if callable(slowdown) else (lambda _: slowdown)
    weight = sum(c.left for c in cs)
    return (sum(c.left * (c.predicted[pick(c)] - c.rest) ** 2 for c in cs) / weight) ** 0.5 if weight else 0.0


def chosen(c: Case) -> float:
    """The slowdown the app applies to this player."""
    return parse.PROVEN_SLOWDOWN if c.proven else 1.0


def closed(cs: list[Case], slowdown) -> tuple[float, float]:
    """Share of the gap between the start and the projection that the rest of the season closed,
    and the share the blend expects."""
    pick = slowdown if callable(slowdown) else (lambda _: slowdown)
    gap = sum(c.prior - c.so_far for c in cs)
    real = sum(c.rest - c.so_far for c in cs) / gap
    return real, sum(c.predicted[pick(c)] - c.so_far for c in cs) / gap


def main() -> None:
    if parse.PROVEN_SLOWDOWN not in TRIED:
        raise SystemExit(f"PROVEN_SLOWDOWN {parse.PROVEN_SLOWDOWN} is not among the slowdowns tried {TRIED}")
    data = {s: cases(s) for s in (FIT, *TEST)}
    print(f"PROVEN_GAMES {PROVEN_GAMES}  PROVEN_SEASONS {PROVEN_SEASONS}  PROVEN_SLOWDOWN {parse.PROVEN_SLOWDOWN}")
    print("\nvalue error by slowdown, relative to none (below 1 is better)")
    for k in range(PROVEN_SEASONS, -1, -1):
        for season, cs in data.items():
            here = [c for c in cs if c.regular_seasons == k]
            base = rmse(here, 1.0)
            cells = "  ".join(f"x{m:g} {rmse(here, m) / base:.3f}" for m in TRIED[1:])
            print(f"  regular in {k} of the last {PROVEN_SEASONS} seasons  {season} (n={len(here):4})  {cells}")
    print("\nthe chosen rule, relative to no slowdown (below 1 is better)")
    for season, cs in data.items():
        cells = []
        for n in CUTS:
            here = [c for c in cs if c.cut == n]
            cells.append(f"after {n}: {rmse(here, chosen) / rmse(here, 1.0):.3f}")
        proven = [c for c in cs if c.proven]
        only = rmse(proven, chosen) / rmse(proven, 1.0)
        overall = rmse(cs, chosen) / rmse(cs, 1.0)
        print(f"  {season}  " + "  ".join(cells) + f"  all: {overall:.3f}  proven only: {only:.3f}")
    print(f"\nproven players after {STARTER_CUT} games, all seasons: share of the gap to the projection closed")
    everyone = [c for cs in data.values() for c in cs if c.proven and c.cut == STARTER_CUT]
    groups = (
        ("slow start, projected above average", [c for c in everyone if c.so_far - c.prior <= -FAR and c.prior > 0]),
        ("slow start, anyone", [c for c in everyone if c.so_far - c.prior <= -FAR]),
        ("hot start, anyone", [c for c in everyone if c.so_far - c.prior >= FAR]),
    )
    for label, cs in groups:
        real, plain = closed(cs, 1.0)
        _, slowed = closed(cs, chosen)
        print(f"  {label:36} n={len(cs):3}  real {real:.2f}  plain blend expects {plain:.2f}  chosen rule {slowed:.2f}")


if __name__ == "__main__":
    main()

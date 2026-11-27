"""His own three-point luck in next-game lines (gamebox own_three_luck), replayed with the app's code.

    uv run python -m scripts.own_three_luck_backtest            # five seasons + the season so far
    uv run python -m scripts.own_three_luck_backtest 2024       # one season's rows only (cached in .cache)

Each case is the shipped line with the signal off, the signal's piece at weight 1 (base x signal, as
gamebox.apply adds it) and what he did. The weight for points and for threes is fitted blind: leave
one season out (fit on the other four of 2021-22..2025-26) and forward (fit on earlier seasons only),
then once on all five for 2026-27 so far. Can's rule: it gets weight only if every one of those is
better. Prints the % less squared miss and the weights.
"""

from __future__ import annotations

import pickle
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from app.analysis import gamebox
from app.analysis.gamebox import MINUTES, PTS, TPM
from scripts.durability_backtest import season_pool
from scripts.next_game_matchup_backtest import SEASONS, cases

CURRENT = 2027
CACHE = Path(".cache/own_three_luck")
STATS = {"PTS": PTS, "3PM": TPM}


def season_rows(season: int) -> dict[str, np.ndarray]:
    path = CACHE / f"{season}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    for s in range(season - 4, season + 1):
        season_pool(s)
    cols: dict[str, list[float]] = {f"{k}_{x}": [] for k in STATS for x in ("off", "piece", "act")}
    for _box, c in cases(season):
        if c.line.get(MINUTES, 0.0) < 8:
            continue
        off_ctx = replace(c.context, own_three_luck=0.0)
        off, _, _ = gamebox.apply(c.line, off_ctx, c.mates, c.gained, c.back, c.absent)
        minutes = c.line[MINUTES]
        rate = {k: c.line.get(k, 0.0) / minutes for k in gamebox.COUNTING}
        for name, stat in STATS.items():
            base = rate[stat] * off[MINUTES]
            sig = gamebox.rate_signal("own_three_luck", stat, rate[stat], c.context, c.mates, c.back, rate)
            cols[f"{name}_off"].append(off.get(stat, 0.0))
            cols[f"{name}_piece"].append(base * sig)
            cols[f"{name}_act"].append(c.actual.get(stat, 0.0))
    out = {k: np.array(v) for k, v in cols.items()}
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(out))
    return out


def gain(y: np.ndarray, x: np.ndarray, w: float) -> float:
    e0 = float(np.mean(y**2))
    return (e0 - float(np.mean((y - w * x) ** 2))) / e0 * 100 if e0 > 0 else 0.0


def fit(y: np.ndarray, x: np.ndarray) -> float:
    d = float(x @ x)
    return float(x @ y) / d if d > 0 else 0.0


def main() -> None:
    if len(sys.argv) > 1:
        for s in sys.argv[1:]:
            print(s, len(season_rows(int(s))["PTS_off"]), flush=True)
        return
    data = {s: season_rows(s) for s in (*SEASONS, CURRENT)}
    for name in STATS:
        y = {s: d[f"{name}_act"] - d[f"{name}_off"] for s, d in data.items()}
        x = {s: d[f"{name}_piece"] for s, d in data.items()}
        loso, fwd, ws = [], [], []
        for s in SEASONS:
            rest = [q for q in SEASONS if q != s]
            w = fit(np.concatenate([y[q] for q in rest]), np.concatenate([x[q] for q in rest]))
            ws.append(w)
            loso.append(gain(y[s], x[s], w))
        for s in SEASONS[1:]:
            before = [q for q in SEASONS if q < s]
            w = fit(np.concatenate([y[q] for q in before]), np.concatenate([x[q] for q in before]))
            fwd.append(gain(y[s], x[s], w))
        w_all = fit(np.concatenate([y[q] for q in SEASONS]), np.concatenate([x[q] for q in SEASONS]))
        shipped = gamebox.RATE_WEIGHTS[STATS[name]]["own_three_luck"]
        print(
            f"{name}: left out {' '.join(f'{g:+.3f}' for g in loso)} | forward {' '.join(f'{g:+.3f}' for g in fwd)}"
            f" | weights {min(ws):.3f}..{max(ws):.3f}, all five {w_all:.3f}"
        )
        print(
            f"     shipped weight {shipped}: "
            + " ".join(f"{s} {gain(y[s], x[s], shipped):+.3f}" for s in (*SEASONS, CURRENT))
        )


if __name__ == "__main__":
    main()

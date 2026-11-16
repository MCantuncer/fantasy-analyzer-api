"""Checks the next-game weights (app/analysis/gamebox.py) and tests new signals, the way they were chosen.

    uv run python -m scripts.next_game_tuning                # five past seasons
    uv run python -m scripts.next_game_tuning --current      # plus the season so far (its cache refreshed)
    uv run python -m scripts.next_game_tuning --out report.md

Can's rule (2026-11-15): a change gets weight only if it lowers the squared miss in EVERY season, tested
blind, both ways:
  * leave one season out: fitted on the other seasons, scored on the one left out;
  * forward: fitted only on the seasons before, scored on the next one (from the second season on).

Part 1, the weights. Every weight in gamebox.py is fitted again on the cases of
scripts/next_game_matchup_backtest.py (the app's own code, day by day). Per category and season: the %
less squared miss of the refit against the shipped weights. A refit is worth a pull request only when it
passes in every season, both ways, and gains at least MIN_GAIN % on average in that category.

Part 2, new signals. Each signal in CANDIDATES is added on top of the refit spec for the categories it
names. It passes a category only when it lowers that category's squared miss in every season, both ways.
To test a new idea, add one function to CANDIDATES (it gets the day's league box and the case) and run
this again.

Writes the report (markdown) to stdout or --out, and the numbers to .cache/next_game_tuning.json.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from app.analysis import gamebox
from app.analysis.gamebox import AST, BLK, FGA, FGM, FTA, FTM, MINUTES, PTS, REB, STL, TO, TPM
from scripts.durability_backtest import LOGS, POOLS
from scripts.next_game_matchup_backtest import FG0, FT0, SCHEDULES, SEASONS, Case, cases, cats

CURRENT = 2027
MIN_GAIN = 0.3  # % less squared miss, averaged over seasons, before a refit is worth shipping
CATS = ("MIN", "PTS", "REB", "AST", "STL", "BLK", "3PM", "TO", "FG", "FT")
RATE_STATS = {"pts": PTS, "reb": REB, "ast": AST, "stl": STL, "blk": BLK, "tpm": TPM, "to": TO, "fga": FGA, "fta": FTA}
CAT_OF = {"pts": "PTS", "reb": "REB", "ast": "AST", "stl": "STL", "blk": "BLK", "tpm": "3PM", "to": "TO", "fta": "FT"}
MINUTE_SIGNALS = tuple(gamebox.MINUTE_WEIGHTS)
MINUTE_WEIGHTS = gamebox.MINUTE_WEIGHTS


# New signals to test. Each: the categories it may move ("min" for minutes, or keys of RATE_STATS, or
# "fg" for FG%) and its value for a case, from the league box of that morning. A minutes signal is added
# to the minutes fit; a per-minute signal multiplies the stat's per-minute rate, like gamebox's weights.
def _blowout(box: gamebox.Box, c: Case) -> float:
    """How lopsided the game looks: starters sit late in blowouts."""
    return abs(c.context.margin) / 10 * min(c.line.get(MINUTES, 0.0) / 36, 1.0)


def _rested(box: gamebox.Box, c: Case) -> float:
    """Two or more days off before the game."""
    before = [g.day for g in box.schedule.get(c.team, []) if g.day < c.day]
    return 1.0 if before and c.day - max(before) >= 3 else 0.0


def _opp_back_to_back(box: gamebox.Box, c: Case) -> float:
    """The opponent played yesterday."""
    return 1.0 if box.back_to_back(c.opponent, c.day) else 0.0


CANDIDATES: dict[str, tuple[tuple[str, ...], Callable[[gamebox.Box, Case], float]]] = {
    "blowout_minutes": (("min",), _blowout),
    "rested": (("min", "pts", "fga"), _rested),
    "opp_back_to_back": (("pts", "fga", "ast", "to"), _opp_back_to_back),
}


def refresh_current(season: int) -> None:
    """The season so far changes every day: drop its cached game logs, pool and schedule."""
    for path in [*LOGS.glob(f"{season}-*.json"), POOLS / f"{season}.json", SCHEDULES / f"{season}.json"]:
        path.unlink(missing_ok=True)


def collect(seasons: tuple[int, ...]) -> dict[str, np.ndarray]:
    """One row per case: the signals gamebox uses, the candidates, what shipped and what he did."""
    cols: dict[str, list[float]] = {}

    def put(name: str, value: float) -> None:
        cols.setdefault(name, []).append(float(value))

    for season in seasons:
        n = 0
        for box, c in cases(season):
            line, m = c.line, c.context
            lm = line.get(MINUTES, 0.0)
            new, _, _ = gamebox.apply(line, m, c.mates, c.gained)
            put("season", season)
            put("lm", lm)
            put("gained", c.gained)
            put("act_min", c.actual.get(MINUTES, 0.0))
            for k, v in gamebox.minute_signals(lm, m, c.mates).items():
                put(f"m_{k}", v)
            put("home", 0.5 if m.home else -0.5)
            for name, stat in RATE_STATS.items():
                rate = line.get(stat, 0.0) / lm if lm > 0 else 0.0
                put(f"rate_{name}", rate)
                for sig in gamebox.RATE_WEIGHTS[stat]:
                    put(f"r_{name}_{sig}", gamebox.rate_signal(sig, stat, rate, m, c.mates))
                put(f"act_{name}", c.actual.get(stat, 0.0))
            for stat, name in ((FGM, "fgm"), (FTM, "ftm")):
                put(f"rate_{name}", line.get(stat, 0.0) / lm if lm > 0 else 0.0)
                put(f"act_{name}", c.actual.get(stat, 0.0))
            fga = line.get(FGA, 0.0)
            put("pct", line.get(FGM, 0.0) / fga if fga > 0 else FG0)
            put("ft_pct", gamebox.ft_line_pct(line))
            put("opp_pct", (1 + m.defense.get(FGM, 0.0)) / (1 + m.defense.get(FGA, 0.0)) - 1)
            for i, v in enumerate(cats(new)):
                put(f"ship_{i}", v)
            for i, v in enumerate(cats(c.actual)):
                put(f"act_cat_{i}", v)
            for name, (_, fn) in CANDIDATES.items():
                put(f"cand_{name}", fn(box, c))
            n += 1
        print(f"{season}: {n} games", flush=True)
    return {k: np.array(v) for k, v in cols.items()}


def ols(x: np.ndarray, y: np.ndarray, intercept: bool = True) -> np.ndarray:
    a = np.column_stack([np.ones(len(x)), x]) if intercept else x
    return np.linalg.lstsq(a, y, rcond=None)[0]


def rate_spec(extra: dict[str, list[str]] | None = None) -> dict[str, list[str]]:
    spec = {name: list(gamebox.RATE_WEIGHTS[stat]) for name, stat in RATE_STATS.items()}
    for name, signals in (extra or {}).items():
        if name in spec:
            spec[name] += signals
    return spec


def _rate_x(d: dict[str, np.ndarray], name: str, signal: str) -> np.ndarray:
    key = f"r_{name}_{signal}"
    return d[key] if key in d else d[f"cand_{signal}"]


def _minutes_x(d: dict[str, np.ndarray], minute_extra: list[str]) -> np.ndarray:
    return np.column_stack([d[f"m_{s}"] for s in MINUTE_SIGNALS] + [d[f"cand_{s}"] for s in minute_extra])


def fit(d: dict[str, np.ndarray], train: np.ndarray, minute_extra: list[str], rate: dict[str, list[str]]) -> dict:
    """The two-stage model of gamebox.apply, fitted on the ``train`` rows."""
    xm = _minutes_x(d, minute_extra)
    w = {"minutes": ols(xm[train], (d["act_min"] - d["lm"])[train])}
    for name, signals in rate.items():
        ea = d[f"rate_{name}"] * d["act_min"]
        x = np.column_stack([_rate_x(d, name, s) for s in signals]) * ea[:, None]
        w[name] = ols(x[train], (d[f"act_{name}"] - ea)[train])
    xf = np.column_stack([d["opp_pct"], d["home"]]) * d["act_fga"][:, None]
    w["fg"] = ols(xf[train], (d["act_fgm"] - d["pct"] * d["act_fga"])[train], intercept=False)
    return w


def predict(d: dict[str, np.ndarray], w: dict, minute_extra: list[str], rate: dict[str, list[str]]) -> np.ndarray:
    """Predicted cats() per row (MIN, 7 counting stats, FG and FT impact), as gamebox.apply makes them."""
    xm = _minutes_x(d, minute_extra)
    mhat = np.clip(d["lm"] + np.column_stack([np.ones(len(xm)), xm]) @ w["minutes"], 0, gamebox.MAX_MINUTES)
    out = {"min": mhat}
    for name, signals in rate.items():
        base = d[f"rate_{name}"] * mhat
        x = np.column_stack([_rate_x(d, name, s) for s in signals])
        out[name] = np.maximum(base * (1 + x @ w[name][1:]) + w[name][0], 0)
    pct = np.clip(d["pct"] + np.column_stack([d["opp_pct"], d["home"]]) @ w["fg"], 0, 1)
    fg = out["fga"] * pct - FG0 * out["fga"]
    ft = out["fta"] * (d["ft_pct"] - FT0)
    return np.column_stack([mhat, *(out[k] for k in ("pts", "reb", "ast", "stl", "blk", "tpm", "to")), fg, ft])


def shipped_weights() -> dict:
    """gamebox's weights in fit()'s layout, to check predict() reproduces gamebox.apply."""
    w = {"minutes": np.array([gamebox.MINUTES_LEVEL, *(MINUTE_WEIGHTS[s] for s in MINUTE_SIGNALS)])}
    for name, stat in RATE_STATS.items():
        weights = gamebox.RATE_WEIGHTS[stat]
        w[name] = np.array([gamebox.RATE_LEVEL[stat], *weights.values()])
    w["fg"] = np.array([gamebox.FG_PCT_WEIGHTS["opp"], gamebox.FG_PCT_WEIGHTS["home"]])
    return w


def blind(d: dict[str, np.ndarray], seasons: tuple[int, ...], minute_extra: list[str], rate: dict[str, list[str]]):
    """Per season, the squared miss per category: left out (``loso``) and fitted on earlier seasons only
    (``forward``, None for the first season)."""
    act = np.column_stack([d[f"act_cat_{i}"] for i in range(len(CATS))])
    loso, forward = {}, {}
    for i, s in enumerate(seasons):
        test = d["season"] == s
        w = fit(d, ~test, minute_extra, rate)
        loso[s] = np.mean((act[test] - predict(d, w, minute_extra, rate)[test]) ** 2, axis=0)
        if i:
            w = fit(d, np.isin(d["season"], seasons[:i]), minute_extra, rate)
            forward[s] = np.mean((act[test] - predict(d, w, minute_extra, rate)[test]) ** 2, axis=0)
        else:
            forward[s] = None
    return loso, forward


def gains(base: dict, new: dict) -> dict[int, np.ndarray | None]:
    return {s: None if base[s] is None else (base[s] - new[s]) / base[s] * 100 for s in base}


def passes(g_loso: dict, g_fwd: dict, c: int) -> bool:
    every = [g[c] for g in g_loso.values()] + [g[c] for g in g_fwd.values() if g is not None]
    return min(every) > 0


def row(g: dict, c: int) -> str:
    return " ".join("  –  " if v is None else f"{v[c]:+.1f}%" for v in g.values())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--current", action="store_true", help=f"add the {CURRENT - 1}-{CURRENT % 100} season so far")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    seasons = SEASONS + ((CURRENT,) if args.current else ())
    if args.current:
        refresh_current(CURRENT)
    d = collect(seasons)
    act = np.column_stack([d[f"act_cat_{i}"] for i in range(len(CATS))])
    shipped = np.column_stack([d[f"ship_{i}"] for i in range(len(CATS))])
    # predict() with gamebox's own weights must give what gamebox.apply gives.
    check = predict(d, shipped_weights(), [], rate_spec())
    drift = float(np.max(np.abs(check - shipped)))
    ship = {s: np.mean((act[d["season"] == s] - shipped[d["season"] == s]) ** 2, axis=0) for s in seasons}

    base_l, base_f = blind(d, seasons, [], rate_spec())
    g_l, g_f = (
        gains(ship, base_l),
        gains({s: ship[s] if base_f[s] is not None else None for s in seasons}, {s: base_f[s] for s in seasons}),
    )
    w_all = fit(d, np.ones(len(d["lm"]), dtype=bool), [], rate_spec())
    lines = [
        "# Next-game weights check",
        "",
        f"Seasons: {', '.join(f'{s - 1}-{s % 100:02d}' for s in seasons)}; {len(d['lm']):,} games."
        f" predict() vs gamebox.apply: largest gap {drift:.4f} (should be ~0).",
        "",
        "## 1. Refit weights vs the shipped ones",
        "",
        "% less squared miss of the refit, per season: left out | fitted on earlier seasons only.",
        "",
        "| Cat | Left out | Forward | Ship? |",
        "| --- | --- | --- | --- |",
    ]
    report: dict = {"seasons": seasons, "games": int(len(d["lm"])), "drift": drift, "weights": {}, "candidates": {}}
    ship_cats = []
    for c, cat in enumerate(CATS):
        mean = float(np.mean([v[c] for v in g_l.values()]))
        ok = passes(g_l, g_f, c) and mean >= MIN_GAIN
        if ok:
            ship_cats.append(cat)
        lines.append(f"| {cat} | {row(g_l, c)} | {row(g_f, c)} | {'yes' if ok else 'no'} |")
    report["weights"]["ship"] = ship_cats
    report["weights"]["refit"] = {k: [round(float(x), 4) for x in v] for k, v in w_all.items()}
    report["weights"]["shipped"] = {k: [round(float(x), 4) for x in v] for k, v in shipped_weights().items()}
    lines += ["", "Refit on all seasons (intercept first; minutes: " + ", ".join(MINUTE_SIGNALS) + "):", ""]
    for k, v in w_all.items():
        old = report["weights"]["shipped"][k]
        names = {"minutes": ["level", *MINUTE_SIGNALS], "fg": ["opp", "home"]}.get(k) or ["level", *rate_spec()[k]]
        pairs = ", ".join(f"{n} {o:+.4f} -> {x:+.4f}" for n, o, x in zip(names, old, v, strict=False))
        lines.append(f"- {k}: {pairs}")

    lines += ["", "## 2. New signals", "", "On top of the refit spec; % less squared miss per season.", ""]
    for name, (targets, _) in CANDIDATES.items():
        minute_extra = [name] if "min" in targets else []
        extra = {t: [name] for t in targets if t in RATE_STATS}
        loso, fwd = blind(d, seasons, minute_extra, rate_spec(extra))
        cl, cf = gains(base_l, loso), gains(base_f, fwd)
        moved = ["MIN"] if "min" in targets else []
        moved += [CAT_OF[t] for t in targets if t in CAT_OF] + (["FG"] if "fga" in targets else [])
        result = {}
        lines.append(f"**{name}** ({', '.join(targets)})")
        lines.append("")
        for cat in moved:
            c = CATS.index(cat)
            ok = passes(cl, cf, c)
            result[cat] = bool(ok)
            lines.append(f"- {cat}: left out {row(cl, c)} | forward {row(cf, c)} {'PASS' if ok else ''}")
        lines.append("")
        report["candidates"][name] = result
    text = "\n".join(lines) + "\n"
    if args.out:
        args.out.write_text(text)
    print(text)
    Path(".cache").mkdir(exist_ok=True)
    Path(".cache/next_game_tuning.json").write_text(json.dumps(report, indent=1, default=list))


if __name__ == "__main__":
    main()

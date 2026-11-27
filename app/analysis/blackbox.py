"""A machine-found correction to the next-game line: patterns in our own misses nobody has to explain.

The rest of the engine is built from signals with a reason (gamebox.py, tonight.py, gamescore.py). This
module adds the opposite: gradient-boosted trees, one per category, that learned the engine's miss (what
a player did minus report.next_game_line's line) from about 150 facts known before tip-off, and give
back the part of the miss they expect. Nobody reads the trees; the check is strict instead
(scripts/black_box_backtest.py, Can's rule): each category was kept only if its trees lowered the squared
miss in every season 2021-22..2025-26 left out (trained on the other four), trained on earlier seasons
only, on 2026-27 so far, and, frozen, on three seasons the engine never saw (2018-19..2020-21); with the
misses shuffled, the same recipe finds nothing. The inputs are each tree's 40 strongest, picked inside
the training seasons only, and the trees' output is pulled toward zero by a factor fitted on a season
they didn't train on.

The inputs come from the same places live and in the replay: the league box (gamebox.build, for his and
both teams' games so far, through ``History``), his game context, today's line before the game
(``line``) and after it (``new``), and the teammates ruled out or back. ``inputs`` builds them all;
``adjust`` moves a line by the trees' call. The trees live in blackbox_models.json, written by the script.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from app.analysis.gamebox import GameContext, GameRow, Teammate

PTS, BLK, STL, AST, REB, TO, FGM, FGA, FTM, FTA, TPM, TPA = 0, 1, 2, 3, 6, 11, 13, 14, 15, 16, 17, 18
MINUTES = 40
MODELS = Path(__file__).with_name("blackbox_models.json")
FG0, FT0 = 0.4756, 0.7973  # league FG% and FT% the FG and FT impacts count makes against (as the replays)
CATS = ("MIN", "PTS", "REB", "AST", "STL", "BLK", "3PM", "TO", "FG", "FT")
HIST = {"min": MINUTES, "pts": PTS, "reb": REB, "ast": AST, "tpm": TPM, "fga": FGA, "fta": FTA, "tpa": TPA,
        "stl": STL, "blk": BLK, "to": TO}  # fmt: skip
_COLS = (MINUTES, PTS, REB, AST, STL, BLK, TPM, TO, FGM, FGA, FTM, FTA, TPA)
_COL = {k: i for i, k in enumerate(_COLS)}
TEAM_STATS = {"reb": REB, "ast": AST, "tpm": TPM, "tpa": TPA, "fta": FTA, "to": TO, "stl": STL, "blk": BLK}
FORM_STATS = {"pts": PTS, "reb": REB, "ast": AST, "stl": STL, "blk": BLK, "tpm": TPM, "to": TO, "fga": FGA,
              "fta": FTA}  # fmt: skip
NAN = float("nan")


def cats(line: Mapping[int, float]) -> list[float]:
    """Minutes, the seven counting categories, then FG and FT as makes over the league's rate."""
    return [line.get(s, 0.0) for s in (MINUTES, PTS, REB, AST, STL, BLK, TPM, TO)] + [
        line.get(FGM, 0.0) - FG0 * line.get(FGA, 0.0),
        line.get(FTM, 0.0) - FT0 * line.get(FTA, 0.0),
    ]


@dataclass(frozen=True)
class History:
    """The games so far the trees read beyond the box's own summaries: each player's games this season
    (day, then _COLS), each team's games (day, its _COLS sums, its opponent's), last season's per game."""

    players: dict[int, np.ndarray] = field(default_factory=dict)  # player -> rows of [day, *_COLS], by day
    teams: dict[int, np.ndarray] = field(default_factory=dict)  # team -> rows of [day, *own, *opponent's], by day
    last: dict[int, dict[str, float]] = field(default_factory=dict)  # player -> last season's per game (+ "gp")
    misses: Misses | None = None  # the engine's own misses this season (past_misses)


def per_game(rows: list[GameRow]) -> dict[int, dict[str, float]]:
    """Each player's per-game line over a season's rows (last season's, for History.last)."""
    sums: dict[int, np.ndarray] = {}
    games: dict[int, int] = defaultdict(int)
    for r in rows:
        sums[r.player] = sums.get(r.player, 0.0) + np.array([r.line.get(k, 0.0) for k in _COLS])
        games[r.player] += 1
    return {
        pid: {name: float(s[_COL[k]] / games[pid]) for name, k in HIST.items()} | {"gp": float(games[pid])}
        for pid, s in sums.items()
    }


def history(
    rows: list[GameRow], last: dict[int, dict[str, float]] | None = None, misses: Misses | None = None
) -> History:
    by_player: dict[int, list[list[float]]] = defaultdict(list)
    team_game: dict[tuple[int, int], np.ndarray] = {}
    game_day: dict[int, int] = {}
    teams_in: dict[int, set[int]] = defaultdict(set)
    for r in rows:
        x = [r.line.get(k, 0.0) for k in _COLS]
        by_player[r.player].append([r.day, *x])
        key = (r.game, r.team)
        team_game[key] = team_game.get(key, 0.0) + np.array(x)
        game_day[r.game] = r.day
        teams_in[r.game].add(r.team)
    teams: dict[int, list[list[float]]] = defaultdict(list)
    for game, ts in teams_in.items():
        if len(ts) != 2:
            continue
        a, b = sorted(ts)
        for t, o in ((a, b), (b, a)):
            teams[t].append([game_day[game], *team_game[(game, t)], *team_game[(game, o)]])
    return History(
        players={p: np.array(sorted(v, key=lambda x: x[0])) for p, v in by_player.items()},
        teams={t: np.array(sorted(v, key=lambda x: x[0])) for t, v in teams.items()},
        last=last or {},
        misses=misses,
    )


def _poss(x: np.ndarray, off: int) -> np.ndarray:
    return x[:, off + _COL[FGA]] + 0.44 * x[:, off + _COL[FTA]] + x[:, off + _COL[TO]]


def _team(h: History, team: int, day: int, who: str) -> dict[str, float]:
    games = h.teams.get(team)
    if games is None:
        return {}
    games = games[games[:, 0] < day]
    if len(games) < 3:
        return {}
    n = len(_COLS)
    own, opp = 1, 1 + n
    out = {}
    for tail, nm in ((5, "5"), (len(games), "s")):
        g = games[-tail:]
        court = np.maximum(g[:, own + _COL[MINUTES]], 240.0) / 240.0
        out[f"{who}_pace{nm}"] = float(np.mean((_poss(g, own) + _poss(g, opp)) / 2 / court))
        out[f"{who}_pf{nm}"] = float(np.mean(g[:, own + _COL[PTS]]))
        out[f"{who}_pa{nm}"] = float(np.mean(g[:, opp + _COL[PTS]]))
        for name, k in TEAM_STATS.items():
            out[f"{who}_{name}_for{nm}"] = float(np.mean(g[:, own + _COL[k]]))
            out[f"{who}_{name}_ag{nm}"] = float(np.mean(g[:, opp + _COL[k]]))
    last = games[-1]
    out[f"{who}_games"] = float(len(games))
    out[f"{who}_rest"] = float(min(day - last[0], 6))
    out[f"{who}_g7"] = float(np.sum(games[:, 0] >= day - 7))
    out[f"{who}_ot_last"] = float(last[own + _COL[MINUTES]] > 245)
    out[f"{who}_mg_last"] = float(last[own + _COL[PTS]] - last[opp + _COL[PTS]])
    return out


# --- the engine's own misses -----------------------------------------------------------------------------
# Each settled night of the line log (app/linelog.py) holds each player's line as the engine had it before
# the trees ("engine"; before TREES_FROM[season] the saved line itself had no trees in it) and what he did.
# His own misses (result less line, in cats() order) are inputs: his last MISS_LAST games and his season.
# Only his own: the log holds one league's pool, not every player, so a team's or an opponent's misses
# would be counted over other players live than in the replay. The replay builds the same records from
# its own lines (scripts/black_box_backtest.py), so the same code counts them. Trees that read them only
# answer for a player with MIN_MISSES misses saved.
TREES_FROM = {2027: 39}  # the first day whose saved lines have the trees in them (api #227, 2026-11-27)
MISS_LAST = 5
MIN_MISSES = 5
LOG_STATS = (MINUTES, PTS, REB, AST, STL, BLK, TPM, TO, FGM, FGA, FTM, FTA)  # linelog.STATS order


def _log_cats(a: list[float]) -> list[float]:
    return cats(dict(zip(LOG_STATS, a, strict=True)))


@dataclass(frozen=True)
class Misses:
    """The engine's misses this season, by day: who, and cats() of result less line."""

    day: np.ndarray
    player: np.ndarray
    miss: np.ndarray  # (n, len(CATS))


def past_misses(records: list[tuple[int, dict]], season: int) -> Misses:
    """Every settled night's misses from the line log's (day, record) pairs."""
    rows = []
    before_trees = TREES_FROM.get(season, 10**6)
    for day, rec in records:
        engine = rec.get("engine") or {}
        for pid, got in (rec.get("results") or {}).items():
            saved = (rec.get("lines") or {}).get(pid)
            if not got or saved is None:
                continue
            line = engine.get(pid) or (saved[1:] if day < before_trees else None)
            if line is not None:
                rows.append((day, int(pid), np.array(_log_cats(got)) - np.array(_log_cats(line))))
    rows.sort(key=lambda r: r[0])
    return Misses(
        day=np.array([r[0] for r in rows], dtype=np.int64),
        player=np.array([r[1] for r in rows], dtype=np.int64),
        miss=np.array([r[2] for r in rows], dtype=float).reshape(-1, len(CATS)),
    )


def miss_inputs(ms: Misses | None, player: int, day: int) -> dict[str, float]:
    """His misses before ``day``: how many (the MIN_MISSES gate), his last MISS_LAST and his season."""
    out: dict[str, float] = {"pm_n": 0.0}
    if ms is None or not len(ms.day):
        return out
    mine = np.flatnonzero((ms.day < day) & (ms.player == player))
    out["pm_n"] = float(len(mine))
    for name, idx in (("pm5", mine[-MISS_LAST:]), ("pms", mine)):
        if len(idx):
            for c, v in zip(CATS, ms.miss[idx].mean(0), strict=True):
                out[f"{name}_{c}"] = float(v)
    return out


def game_inputs(h: History | None, player: int, team: int, opponent: int, day: int) -> dict[str, float]:
    """What the league box knows of his game on ``day``: his games so far, last season, both teams' games."""
    if h is None:
        return {}
    out: dict[str, float] = {}
    games = h.players.get(player)
    games = games[games[:, 0] < day] if games is not None else np.zeros((0, 1 + len(_COLS)))
    out["gp"] = float(len(games))
    for tail in (1, 3, 10):
        g = games[-tail:]
        for name, k in HIST.items():
            out[f"h{tail}_{name}"] = float(np.mean(g[:, 1 + _COL[k]])) if len(g) else NAN
    g = games[-10:]
    if len(g) >= 3:
        for name, k in (("min", MINUTES), ("pts", PTS), ("reb", REB)):
            out[f"sd10_{name}"] = float(np.std(g[:, 1 + _COL[k]]))
    s = games[:, 1:].sum(0) if len(games) else np.zeros(len(_COLS))
    out["s_fgp"] = float(s[_COL[FGM]] / s[_COL[FGA]]) if s[_COL[FGA]] else NAN
    out["s_ftp"] = float(s[_COL[FTM]] / s[_COL[FTA]]) if s[_COL[FTA]] else NAN
    out["s_tpp"] = float(s[_COL[TPM]] / s[_COL[TPA]]) if s[_COL[TPA]] else NAN
    out["s_fga"] = float(s[_COL[FGA]] / max(1, len(games)))
    out["p_g4"] = float(np.sum(games[:, 0] >= day - 4)) if len(games) else 0.0
    out["p_g7"] = float(np.sum(games[:, 0] >= day - 7)) if len(games) else 0.0
    out["p_rest"] = float(min(day - games[-1, 0], 15)) if len(games) else NAN
    last = h.last.get(player)
    for name in (*HIST, "gp"):
        if name != "tpa":
            out[f"ly_{name}"] = last[name] if last else NAN
    out |= _team(h, team, day, "t") | _team(h, opponent, day, "o")
    out |= miss_inputs(h.misses, player, day)
    return out


def inputs(
    m: GameContext,
    line: Mapping[int, float],
    new: Mapping[int, float],
    out: Collection[Teammate],
    back: Collection[Teammate],
    gained: float,
    gap: float,
) -> dict[str, float]:
    """Every input of a game, from his game context (with its league-box part, ``m.box_inputs``), today's
    line before the game (``line``) and the line the trees correct (``new``: after gamebox.apply and the
    team's sum, report.team_adjust by ``gap``), the teammates ruled out (``out``) or back (``back``,
    counted by their chance to play) and the minutes ``gained`` from those out."""
    x: dict[str, float] = dict(m.box_inputs)
    for i, v in enumerate(cats(line)):
        x[f"l0_{i}"] = v
    for c, v in zip(CATS, cats(new), strict=True):
        x[f"n_{c}"] = v
    x["home"] = float(m.home)
    x["b2b"] = float(m.back_to_back)
    x["margin"] = m.margin
    x["games_missed"] = float(m.games_missed)
    x["game_back"] = float(m.game_back)
    x["team_points"] = m.team_points
    x["team_score"] = m.team_score if m.team_score is not None else NAN
    x["new_team"] = float(m.new_team)
    x["games_left"] = float(m.games_left) if m.games_left is not None else NAN
    x["sat_ago"] = float(min(m.sat_ago, 60)) if m.sat_ago is not None else NAN
    x["rot_n"] = float(len(m.rotation))
    x["rot_sum"] = float(sum(c for _, c in m.rotation))
    x["opp3luck"] = m.opp_three_luck
    x["own3luck"] = m.own_three_luck
    for name, k in FORM_STATS.items():
        x[f"def_{name}"] = m.defense.get(k, NAN)
    for name, k in (("reb", REB), ("ast", AST), ("blk", BLK)):
        x[f"tm_made_{name}"] = m.team_made.get(k, NAN)
    f = m.form
    if f is not None:
        for name, k in FORM_STATS.items():
            x[f"rate_{name}"] = f.rate.get(k, NAN)
            d = f.fast.get(k)
            x[f"fast_{name}"] = d[0] / d[1] if d and d[1] > 0 else NAN
            d = f.decayed.get(k)
            x[f"dec_{name}"] = d[0] / d[1] if d and d[1] > 0 else NAN
        x["f_min"] = f.minutes
        x["f_min_long"] = f.minutes_long
        x["f_usage"] = f.usage
        x["f_min_season"] = f.minutes_season if f.minutes_season is not None else NAN
        x["f_last_min"] = f.last_minutes if f.last_minutes is not None else NAN
        x["f_started"] = float(f.started) if f.started is not None else NAN
        x["f_start_share"] = f.start_share if f.start_share is not None else NAN
        x["f_since_return"] = float(f.since_return)
        x["f_decided_by"] = f.decided_by if f.decided_by is not None else NAN
        x["f_ot"] = float(f.overtime)
        x["f_min_recent"] = f.minutes_recent if f.minutes_recent is not None else NAN
        x["f_games"] = float(f.games)
        x["f_rest"] = float(min(m.day - f.last_day, 10))
    x["gained"] = gained
    x["n_out"] = float(len(out))
    x["out_min"] = sum(t.minutes for t in out)
    x["out_use"] = sum(t.usage for t in out)
    x["out_pts"] = sum(t.points for t in out)
    x["n_back"] = sum(t.chance for t in back)
    x["back_min"] = sum(t.minutes * t.chance for t in back)
    x["back_use"] = sum(t.usage * t.chance for t in back)
    x["team_gap"] = gap
    x["day"] = float(m.day)
    return x


# --- the trees -------------------------------------------------------------------------------------------
# LightGBM's trees as flat arrays (the script exports them): a node splits on ``feature`` at
# ``threshold`` (x <= threshold goes left); a missing input goes the node's ``default_left`` way; a
# child below zero is a leaf (~child). All trees of a category step down together, one level a step.


@dataclass(frozen=True)
class Trees:
    features: tuple[str, ...]
    pull: float  # the trees' output is multiplied by this (fitted on a season they didn't train on)
    feature: np.ndarray  # (trees, nodes) int
    threshold: np.ndarray  # (trees, nodes) float
    default_left: np.ndarray  # (trees, nodes) bool
    left: np.ndarray  # (trees, nodes) int
    right: np.ndarray  # (trees, nodes) int
    leaf: np.ndarray  # (trees, leaves) float
    depth: int

    def predict(self, x: np.ndarray) -> float:
        """The trees' sum for one input row (in ``features`` order), times the pull."""
        n = self.feature.shape[0]
        rows = np.arange(n)
        node = np.zeros(n, dtype=np.int64)  # >= 0: a node; < 0: ~leaf
        for _ in range(self.depth):
            live = node >= 0
            if not live.any():
                break
            nd = np.where(live, node, 0)
            v = x[self.feature[rows, nd]]
            go_left = np.where(np.isnan(v), self.default_left[rows, nd], v <= self.threshold[rows, nd])
            nxt = np.where(go_left, self.left[rows, nd], self.right[rows, nd])
            node = np.where(live, nxt, node)
        return float(self.leaf[rows, ~node].sum() * self.pull)


def _trees(spec: dict) -> Trees:
    trees = spec["trees"]
    nodes = max(1, max(len(t["feature"]) for t in trees))
    leaves = max(len(t["leaf"]) for t in trees)
    shape = (len(trees), nodes)

    def pad(key: str, fill: float | int | bool, dtype: type) -> np.ndarray:
        a = np.full(shape, fill, dtype=dtype)
        for i, t in enumerate(trees):
            a[i, : len(t[key])] = t[key]
        return a

    leaf = np.zeros((len(trees), leaves))
    for i, t in enumerate(trees):
        leaf[i, : len(t["leaf"])] = t["leaf"]
    # a one-leaf tree has no node: its root points straight to leaf 0
    left, right = pad("left", -1, np.int64), pad("right", -1, np.int64)
    return Trees(
        features=tuple(spec["features"]),
        pull=float(spec["pull"]),
        feature=pad("feature", 0, np.int64),
        threshold=pad("threshold", 0.0, float),
        default_left=pad("default_left", True, bool),
        left=left,
        right=right,
        leaf=leaf,
        depth=int(spec["depth"]),
    )


@cache
def _raw() -> dict:
    return json.loads(MODELS.read_text()) if MODELS.exists() else {}


@cache
def models() -> dict[str, Trees]:
    """The kept categories' trees (empty if the file is missing: lines then stay as they are)."""
    return {cat: _trees(spec) for cat, spec in _raw().get("models", {}).items()}


@cache
def with_misses() -> dict[str, Trees]:
    """The kept categories' trees that also read his past misses: they answer once he has MIN_MISSES."""
    return {cat: _trees(spec) for cat, spec in _raw().get("with_misses", {}).items()}


def calls(x: Mapping[str, float]) -> dict[str, float]:
    """The trees' expected miss per kept category for one game's inputs: the trees that read his past
    misses where they are kept and he has MIN_MISSES of them, else the others."""
    out = {}
    known = x.get("pm_n", 0.0) >= MIN_MISSES
    for cat in models().keys() | with_misses().keys():
        t = with_misses().get(cat) if known else None
        t = t or models().get(cat)
        if t is None:
            continue
        row = np.array([x.get(name, NAN) for name in t.features], dtype=float)
        v = t.predict(row)
        if math.isfinite(v):
            out[cat] = v
    return out


def adjust(new: Mapping[int, float], miss: Mapping[str, float]) -> dict[int, float]:
    """``new`` moved by the trees' expected miss: minutes and counting stats by their call (never below
    zero), FG and FT impacts through the makes (attempts stay)."""
    out = dict(new)
    keys = {"MIN": MINUTES, "PTS": PTS, "REB": REB, "AST": AST, "STL": STL, "BLK": BLK, "3PM": TPM, "TO": TO}
    for cat, v in miss.items():
        if cat in keys:
            k = keys[cat]
            out[k] = max(out.get(k, 0.0) + v, 0.0)
        elif cat == "FG" and out.get(FGA, 0.0) > 0:
            out[FGM] = min(max(out.get(FGM, 0.0) + v, 0.0), out[FGA])
        elif cat == "FT" and out.get(FTA, 0.0) > 0:
            out[FTM] = min(max(out.get(FTM, 0.0) + v, 0.0), out[FTA])
    return out

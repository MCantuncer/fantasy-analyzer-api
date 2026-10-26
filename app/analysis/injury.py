"""Who gains when a player is out: measured from the games his team played without him.

For an injured player X, each teammate's minutes and per-minute production are
compared between games X played and games he missed, this season and last
(last season counts a quarter). Depth-chart guesses ("his backup gets his minutes")
are only a fallback, and every number is checked before it is trusted:

- Sample size. With few games without X, the measured change is pulled toward a
  prior that shares X's minutes among teammates by position and current role.
  The weight of the evidence grows with the number of games (``MINUTES_K``).
- Noise. Each change comes with a standard error; "strong" evidence needs at
  least ``STRONG_GAMES`` games and a change clear of its error band.
- Other absences. When another regular also sat out most of those games, the
  gain may belong to that absence, and the result says so.
- Per-minute rates move less than minutes, so they are pulled toward the
  player's normal rates until ``RATE_K`` minutes of evidence are in.

The fantasy gain is the change in a teammate's per-game line, in z-score units
of a 9-category league (``CATEGORY_SD``). ``scripts/injury_backtest.py`` checks
the method against real absences of past seasons.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

GAME_SPLIT = 5
# ESPN stat ids in a game line.
STAT_IDS = {"pts": 0, "blk": 1, "stl": 2, "ast": 3, "reb": 6, "to": 11, "fgm": 13, "fga": 14, "ftm": 15, "fta": 16}
TPM, MIN = 17, 40
COUNTING = ("pts", "reb", "ast", "stl", "blk", "tpm", "to")

# Tuned on 2024-25 absences, checked on 2025-26 (see scripts/injury_backtest.py).
MINUTES_K = 8.0  # games of evidence that weigh as much as the prior
RATE_K = 150.0  # minutes of evidence that weigh as much as the player's normal per-minute rates
PRIOR_SHARE = 0.85  # of the absent player's minutes, the share the prior hands to the listed teammates
LAST_SEASON_WEIGHT = 0.25
HALF_LIFE: float | None = 10.0  # games: this season's games count half as much this many team games ago
STRONG_GAMES = 8
SOME_GAMES = 3
MIN_SD = 6.0  # per-game minutes spread used when a sample is too small to measure one
CORE = 5  # a team's regulars: its top players by minutes with X on the floor
MAX_MINUTES = 38.0
# Spread of per-game averages among the top 150 players (2025-26), for z-scores.
CATEGORY_SD = {"pts": 5.6, "reb": 2.5, "ast": 1.9, "stl": 0.35, "blk": 0.45, "tpm": 0.85, "to": 0.75}
FG_IMPACT_SD = 0.95  # makes above league rate per game
FT_IMPACT_SD = 0.65
LEAGUE_FG = 0.475
LEAGUE_FT = 0.785
# Per-minute rates of an average rotation player (500+ minutes, 2025-26). A player's own rates are
# pulled toward these until ``BASE_RATE_K`` minutes are in, so 40 garbage-time minutes don't project as a star.
AVERAGE_RATES = {
    "pts": 0.486,
    "reb": 0.181,
    "ast": 0.112,
    "stl": 0.035,
    "blk": 0.020,
    "tpm": 0.056,
    "to": 0.057,
    "fgm": 0.177,
    "fga": 0.373,
    "ftm": 0.078,
    "fta": 0.099,
}
BASE_RATE_K = 300.0
POSITIONS = {1: "PG", 2: "SG", 3: "SF", 4: "PF", 5: "C"}
ORDER = ["PG", "SG", "SF", "PF", "C"]


@dataclass(frozen=True)
class GameLine:
    player_id: int
    game_id: int
    period: int
    team: int
    minutes: float
    pts: float = 0
    reb: float = 0
    ast: float = 0
    stl: float = 0
    blk: float = 0
    tpm: float = 0
    to: float = 0
    fgm: float = 0
    fga: float = 0
    ftm: float = 0
    fta: float = 0


@dataclass
class PlayerInfo:
    id: int
    name: str
    position: str
    team: int  # current NBA team


def parse_gamelogs(players: list[dict]) -> tuple[list[GameLine], dict[int, PlayerInfo]]:
    """Every game line in ESPN's per-game splits (``filterStatsForSplitTypeIds`` 5), and who each player is."""
    lines: list[GameLine] = []
    info: dict[int, PlayerInfo] = {}
    for entry in players:
        p = entry.get("player") or entry
        if "id" not in p:
            continue
        pid = int(p["id"])
        info[pid] = PlayerInfo(
            pid,
            p.get("fullName", ""),
            POSITIONS.get(int(p.get("defaultPositionId") or 0), ""),
            int(p.get("proTeamId") or 0),
        )
        for st in p.get("stats") or []:
            if st.get("statSplitTypeId") != GAME_SPLIT or st.get("statSourceId") or not st.get("externalId"):
                continue
            s = st.get("stats") or {}
            minutes = float(s.get(str(MIN), 0) or 0)
            if minutes <= 0:
                continue
            lines.append(
                GameLine(
                    player_id=pid,
                    game_id=int(st["externalId"]),
                    period=int(st.get("scoringPeriodId") or 0),
                    team=int(st.get("proTeamId") or p.get("proTeamId") or 0),
                    minutes=minutes,
                    tpm=float(s.get(str(TPM), 0) or 0),
                    **{k: float(s.get(str(v), 0) or 0) for k, v in STAT_IDS.items()},
                )
            )
    return lines, info


class Season:
    """One season's game lines, indexed by team and game."""

    def __init__(self, lines: list[GameLine], weight: float = 1.0):
        self.weight = weight
        self.by_game: dict[int, dict[int, GameLine]] = defaultdict(dict)  # game -> player -> line
        self.games: dict[int, list[tuple[int, int]]] = defaultdict(list)  # team -> [(period, game)]
        seen: set[tuple[int, int]] = set()
        for ln in lines:
            self.by_game[ln.game_id][ln.player_id] = ln
            if (ln.team, ln.game_id) not in seen:
                seen.add((ln.team, ln.game_id))
                self.games[ln.team].append((ln.period, ln.game_id))
        for g in self.games.values():
            g.sort()

    def team_games(self, team: int, before: int | None = None) -> list[int]:
        return [gid for period, gid in self.games.get(team, []) if before is None or period < before]

    def line(self, game: int, player: int, team: int) -> GameLine | None:
        ln = self.by_game.get(game, {}).get(player)
        return ln if ln is not None and ln.team == team else None


@dataclass
class Split:
    """A teammate's games with and without the absent player, weighted by season."""

    with_lines: list[tuple[float, GameLine]] = field(default_factory=list)
    without_lines: list[tuple[float, GameLine]] = field(default_factory=list)
    also_out: dict[int, float] = field(default_factory=dict)  # regular -> weighted games he also missed


def _mean_sd(xs: list[tuple[float, float]]) -> tuple[float, float, float]:
    """Weighted mean, sd and effective count of (weight, value) pairs."""
    w = sum(a for a, _ in xs)
    if w <= 0:
        return 0.0, MIN_SD, 0.0
    m = sum(a * v for a, v in xs) / w
    var = sum(a * (v - m) ** 2 for a, v in xs) / w
    n = w**2 / sum(a * a for a, _ in xs)
    return m, (math.sqrt(var) if n >= 2 else MIN_SD), n


def _rates(lines: list[tuple[float, GameLine]]) -> tuple[dict[str, float], float]:
    """Per-minute rates of every stat, and the minutes behind them."""
    mins = sum(w * ln.minutes for w, ln in lines)
    if mins <= 0:
        return {}, 0.0
    keys = (*COUNTING, "fgm", "fga", "ftm", "fta")
    return {k: sum(w * getattr(ln, k) for w, ln in lines) / mins for k in keys}, mins


def _affinity(a: str, b: str) -> float:
    if not a or not b:
        return 0.4
    gap = abs(ORDER.index(a) - ORDER.index(b))
    return {0: 1.0, 1: 0.6}.get(gap, 0.25)


def prior_weight(affinity: float, minutes: float) -> float:
    """How much of the absent player's minutes the prior hands a teammate: same position, room to grow."""
    return affinity * max(MAX_MINUTES - minutes, 0.0) * math.sqrt(max(minutes, 5.0))


@dataclass
class Beneficiary:
    player_id: int
    name: str
    minutes_before: float
    minutes_after: float
    minutes_change: float
    error: float  # +/- around the measured minutes change (80% band); 0 when nothing was measured
    games_without: float  # effective games of evidence
    gains: dict[str, float]  # per-game change in each category (pts, reb, ..., fg%, ft% as makes over league rate)
    value: float  # change in 9-category z-score per game
    confidence: str  # "strong", "some" or "estimate"
    note: str | None = None


def value_of(gains: dict[str, float]) -> float:
    z = sum((-1 if k == "to" else 1) * gains.get(k, 0.0) / CATEGORY_SD[k] for k in COUNTING)
    return z + gains.get("fg_impact", 0.0) / FG_IMPACT_SD + gains.get("ft_impact", 0.0) / FT_IMPACT_SD


def _per_game(rates: dict[str, float], minutes: float) -> dict[str, float]:
    out = {k: rates.get(k, 0.0) * minutes for k in (*COUNTING, "fgm", "fga", "ftm", "fta")}
    out["fg_impact"] = out["fgm"] - LEAGUE_FG * out["fga"]
    out["ft_impact"] = out["ftm"] - LEAGUE_FT * out["fta"]
    return out


@dataclass
class Impact:
    minutes: float  # the absent player's recent minutes per game
    games_missed: float  # team games without him, weighted like the evidence
    beneficiaries: list[Beneficiary]


def analyze(
    seasons: list[Season],
    team: int,
    absent: int,
    roster: list[PlayerInfo],
    before: int | None = None,
    positions: dict[int, str] | None = None,
) -> Impact:
    """Teammates in ``roster`` ranked by what they gain per game while ``absent`` is out.

    ``seasons`` run newest first; only games before scoring period ``before`` of the
    newest season are used (for backtests). ``positions`` maps player id to PG..C.
    """
    positions = positions or {p.id: p.position for p in roster}
    names = {p.id: p.name for p in roster}
    mates = [p for p in roster if p.id != absent]
    splits = {p.id: Split() for p in mates}
    x_minutes: list[tuple[float, float]] = []
    missed = 0.0
    for i, season in enumerate(seasons):
        games = season.team_games(team, before if i == 0 else None)
        played = [g for g in games if season.line(g, absent, team)]
        if not played:
            continue
        first = games.index(played[0])
        last = len(games) if i == 0 else games.index(played[-1]) + 1  # still on the team now
        tenure = games[first:last]
        x_set = set(played)
        missed += season.weight * sum(1 for g in tenure if g not in x_set)
        x_minutes += [(season.weight, season.line(g, absent, team).minutes) for g in played[-20:]]
        # The regulars: most minutes alongside X.
        totals: dict[int, float] = defaultdict(float)
        for g in played:
            for pid, ln in season.by_game[g].items():
                if ln.team == team and pid != absent:
                    totals[pid] += ln.minutes
        core = sorted(totals, key=totals.get, reverse=True)[:CORE]
        for age, g in enumerate(reversed(tenure)):
            without = g not in x_set
            # Recent games tell more about today's rotation.
            weight = season.weight * (0.5 ** (age / HALF_LIFE) if HALF_LIFE and i == 0 else 1.0)
            for p in mates:
                ln = season.line(g, p.id, team)
                if ln is None:
                    continue
                sp = splits[p.id]
                if without:
                    sp.without_lines.append((weight, ln))
                    for c in core:
                        if c != p.id and season.line(g, c, team) is None:
                            sp.also_out[c] = sp.also_out.get(c, 0.0) + weight
                else:
                    sp.with_lines.append((weight, ln))

    x_min, _, _ = _mean_sd(x_minutes)
    base = {p.id: _mean_sd([(w, ln.minutes) for w, ln in splits[p.id].with_lines]) for p in mates}
    raw = {}
    for p in mates:
        m = base[p.id][0]
        if base[p.id][2] < 1:
            continue
        x_pos = positions.get(absent, "")
        raw[p.id] = prior_weight(_affinity(x_pos, positions.get(p.id, "")), m)
    total = sum(raw.values()) or 1.0

    out = []
    for p in mates:
        sp = splits[p.id]
        m_with, sd_with, n_with = base[p.id]
        if n_with < 1:
            continue  # never played alongside X: no baseline
        m_wo, sd_wo, n_wo = _mean_sd([(w, ln.minutes) for w, ln in sp.without_lines])
        prior = min(PRIOR_SHARE * x_min * raw.get(p.id, 0.0) / total, MAX_MINUTES - m_with)
        observed = m_wo - m_with if n_wo > 0 else 0.0
        change = (n_wo * observed + MINUTES_K * prior) / (n_wo + MINUTES_K)
        se = math.sqrt(sd_with**2 / max(n_with, 1) + sd_wo**2 / max(n_wo, 1)) if n_wo > 0 else 0.0
        after = max(0.0, min(m_with + change, MAX_MINUTES + 4))

        r_own, mins_with = _rates(sp.with_lines)
        r_with = {
            k: (mins_with * r_own.get(k, v) + BASE_RATE_K * v) / (mins_with + BASE_RATE_K)
            for k, v in AVERAGE_RATES.items()
        }
        r_wo, mins_wo = _rates(sp.without_lines)
        rates = {k: (mins_wo * r_wo.get(k, v) + RATE_K * v) / (mins_wo + RATE_K) for k, v in r_with.items()}
        now, then = _per_game(r_with, m_with), _per_game(rates, after)
        gains = {k: then[k] - now[k] for k in (*COUNTING, "fg_impact", "ft_impact")}

        if n_wo >= STRONG_GAMES and abs(observed) > 1.28 * se:
            confidence = "strong"
        elif n_wo >= SOME_GAMES:
            confidence = "some"
        else:
            confidence = "estimate"
        note = None
        total_wo = sum(w for w, _ in sp.without_lines)
        if n_wo > 0 and sp.also_out:
            other, shared = max(sp.also_out.items(), key=lambda kv: kv[1])
            if shared / total_wo > 0.5 and other in names:
                note = f"{names[other]} was also out in most of those games, so part of the gain may be his"
        out.append(
            Beneficiary(
                player_id=p.id,
                name=p.name,
                minutes_before=round(m_with, 1),
                minutes_after=round(after, 1),
                minutes_change=round(after - m_with, 1),
                error=round(1.28 * se, 1),
                games_without=round(n_wo, 1),
                gains={k: round(v, 2) for k, v in gains.items()},
                value=round(value_of(gains), 2),
                confidence=confidence,
                note=note,
            )
        )
    out.sort(key=lambda b: (b.value, b.minutes_change), reverse=True)
    return Impact(minutes=round(x_min, 1), games_missed=round(missed, 1), beneficiaries=out)

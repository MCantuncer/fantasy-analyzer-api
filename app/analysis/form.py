"""Is a player's hot run real? What his last 15 days were made of, and how much of it tends to last.

A run over ESPN's "last 15 days" (about 7 games) is compared with his usual level: last season
counted as LAST_SEASON_GAMES games, blended with this season's games before the run. The gap, in
standard 9-category value per game, splits exactly into three parts:

  * minutes: his usual per-minute line played at the run's minutes,
  * production: more shots, boards, assists and steals per minute than usual, at his usual shooting,
  * shooting: the run's FG%, 3P% and FT% above his usual on the same attempts.

Over the next 10 games, about half of the minutes part stays and about a fifth of the other two
(KEEP), in each of five seasons (2021-22 to 2025-26, every run of every player; fitted on all
players and the same for those outside the top ~120). A run made of shooting on the same minutes
is a straw fire, and so is one made of more shots and boards per minute (that fades as fast);
one made of a minutes jump is a bigger role. The projection already prices this
(this season counts in proportion to games played), so this explains a run and never moves a value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app import schemas

GP, MINUTES = 42, 40
PTS, REB, AST, STL, BLK, TPM, TPA, TO, FGM, FGA, FTM, FTA = 0, 6, 3, 2, 1, 17, 18, 11, 13, 14, 15, 16
SEASON_SPLIT, LAST_15_DAYS = 0, 2
LAST_SEASON_GAMES = 15  # weight of last season in his usual level, in games
MIN_LAST_SEASON_GAMES = 20  # last season counts only with this many games
MIN_USUAL_GAMES = 5  # without last season, games before the run his usual level needs
MIN_RUN_GAMES = 3
HOT_GAP = 2.5  # value per game above his usual that makes a run hot (a quarter of all runs this long are)
# Share of each part still there over the next 10 games, fitted on every run of five seasons
# (scripts/hot_run_backtest.py; one season at a time: minutes 0.43-0.60, production 0.17-0.21,
# shooting 0.14-0.20). Hot runs that read as a bigger role kept 50% of the gap, straw fires 27%.
KEEP = {"minutes": 0.51, "production": 0.19, "shooting": 0.17}
ROLE_SHARE = 0.5  # minutes this big a share of the gap: a bigger role
FIRE_SHARE = 0.25  # minutes this small a share: a straw fire (in between: mixed)
# 9-category value of a per-game line: mean and spread of the 180 players with the most minutes
# (40+ games) in 2025-26. Fixed, so a run reads the same in every league and on the player card.
SCALE = {
    PTS: (15.545, 5.516),
    REB: (5.201, 2.240),
    AST: (3.523, 1.948),
    STL: (0.993, 0.329),
    BLK: (0.546, 0.427),
    TPM: (1.725, 0.913),
    TO: (1.799, 0.764),
}
FG_SCALE = (0.4756, 0.5617)  # league FG% and the spread of FG impact (makes over league rate)
FT_SCALE = (0.7973, 0.2751)
LINE = (MINUTES, PTS, REB, AST, STL, BLK, TPM, TPA, TO, FGM, FGA, FTM, FTA)


@dataclass(frozen=True)
class HotRun:
    verdict: Literal["straw_fire", "bigger_role", "mixed"]
    games: int  # in the last 15 days
    gap: float  # value per game above his usual
    minutes: float  # the three parts, adding up to gap
    production: float
    shooting: float
    likely_kept: float  # share of the gap expected to last over his next 10 games
    usual: dict[int, float]  # his usual per-game line
    recent: dict[int, float]  # his last 15 days, per game


def value(line: dict[int, float]) -> float:
    total = 0.0
    for stat, (mean, sd) in SCALE.items():
        z = (line.get(stat, 0.0) - mean) / sd
        total += -z if stat == TO else z
    total += (line.get(FGM, 0.0) - FG_SCALE[0] * line.get(FGA, 0.0)) / FG_SCALE[1]
    total += (line.get(FTM, 0.0) - FT_SCALE[0] * line.get(FTA, 0.0)) / FT_SCALE[1]
    return total


def _averages(stats: list[dict], season: int, split: int) -> tuple[dict[int, float], int]:
    for s in stats:
        if s.get("seasonId") == season and s.get("statSourceId") == 0 and s.get("statSplitTypeId") == split:
            avg = {int(k): float(v) for k, v in (s.get("averageStats") or {}).items()}
            return avg, int(avg.get(GP, 0) or 0)
    return {}, 0


def _pct(line: dict[int, float], made: int, tried: int) -> float:
    return line.get(made, 0.0) / line[tried] if line.get(tried, 0.0) > 0 else 0.0


def split_run(usual: dict[int, float], recent: dict[int, float]) -> tuple[float, float, float]:
    """The gap between the two lines as (minutes, production, shooting), adding up exactly."""
    base = value(usual)
    scale = recent.get(MINUTES, 0.0) / usual[MINUTES] if usual.get(MINUTES, 0.0) > 0 else 1.0
    at_minutes = {k: v * scale for k, v in usual.items()}
    # The run's volume at his usual shooting.
    usual_shot = dict(recent)
    for made, tried in ((FGM, FGA), (TPM, TPA), (FTM, FTA)):
        usual_shot[made] = _pct(usual, made, tried) * recent.get(tried, 0.0)
    usual_shot[PTS] = (
        recent.get(PTS, 0.0)
        + 2 * (usual_shot[FGM] - recent.get(FGM, 0.0))
        + (usual_shot[TPM] - recent.get(TPM, 0.0))
        + (usual_shot[FTM] - recent.get(FTM, 0.0))
    )
    v1, v2 = value(at_minutes), value(usual_shot)
    return v1 - base, v2 - v1, value(recent) - v2


def hot_run(stats: list[dict], season: int) -> HotRun | None:
    """His last 15 days against his usual level, from ESPN's stat lines; None unless it's a hot run."""
    season_avg, season_gp = _averages(stats, season, SEASON_SPLIT)
    recent, run_gp = _averages(stats, season, LAST_15_DAYS)
    last, last_gp = _averages(stats, season - 1, SEASON_SPLIT)
    before = season_gp - run_gp
    if run_gp < MIN_RUN_GAMES or before < 0 or not recent.get(MINUTES):
        return None
    prior = LAST_SEASON_GAMES if last_gp >= MIN_LAST_SEASON_GAMES else 0
    if prior == 0 and before < MIN_USUAL_GAMES:
        return None
    usual = {
        k: (season_avg.get(k, 0.0) * season_gp - recent.get(k, 0.0) * run_gp + prior * last.get(k, 0.0))
        / (before + prior)
        for k in LINE
    }
    if usual[MINUTES] <= 0:
        return None
    recent_line = {k: recent.get(k, 0.0) for k in LINE}
    minutes, production, shooting = split_run(usual, recent_line)
    gap = minutes + production + shooting
    if gap < HOT_GAP:
        return None
    if minutes >= ROLE_SHARE * gap:
        verdict = "bigger_role"
    elif minutes <= FIRE_SHARE * gap:
        verdict = "straw_fire"
    else:
        verdict = "mixed"
    kept = (KEEP["minutes"] * minutes + KEEP["production"] * production + KEEP["shooting"] * shooting) / gap
    return HotRun(
        verdict=verdict,
        games=run_gp,
        gap=gap,
        minutes=minutes,
        production=production,
        shooting=shooting,
        likely_kept=min(max(kept, 0.0), 1.0),
        usual=usual,
        recent=recent_line,
    )


def hot_run_ref(run: HotRun | None) -> schemas.HotRun | None:
    if run is None:
        return None
    u, r = run.usual, run.recent

    def pct(line: dict[int, float], made: int, tried: int) -> float | None:
        return round(line[made] / line[tried], 3) if line.get(tried, 0.0) > 0 else None

    return schemas.HotRun(
        verdict=run.verdict,
        games=run.games,
        gap=round(run.gap, 2),
        minutes_part=round(run.minutes, 2),
        production_part=round(run.production, 2),
        shooting_part=round(run.shooting, 2),
        likely_kept=round(run.likely_kept, 2),
        minutes_recent=round(r[MINUTES], 1),
        minutes_usual=round(u[MINUTES], 1),
        pts_recent=round(r[PTS], 1),
        pts_usual=round(u[PTS], 1),
        fg_recent=pct(r, FGM, FGA),
        fg_usual=pct(u, FGM, FGA),
        three_recent=pct(r, TPM, TPA),
        three_usual=pct(u, TPM, TPA),
        ft_recent=pct(r, FTM, FTA),
        ft_usual=pct(u, FTM, FTA),
    )

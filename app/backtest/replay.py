"""Replay a finished season through the live model, one week at a time, without peeking.

To predict matchup period P, the league is rebuilt as it stood on P's first day: that
day's rosters and lineup slots, each player's stats from box scores before that day
(blended with last season by the same code the app uses), records from the matchups
already played, and P onward undecided. The model then projects P and the season
simulation plays out the rest, and both are scored against what really happened.

ESPN keeps only today's injury status, so a player counts as out when his NBA team
played its last three games without him.

Scores (lower is better unless noted):
  * weekly Brier and log loss of every category win probability, ties counted as half;
  * spread check per category, the sd of (actual - projected) / projected sd: 1.0 means
    the model is as unsure as it should be, above 1 overconfident;
  * season Brier and log loss of playoff odds at several cut weeks, and the same spread
    check on final winning percentage.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from app.analysis.model import CategoryModel
from app.analysis.sim import EACH_CATEGORY, simulate_season
from app.backtest.history import History
from app.espn.parse import GAMES_PLAYED, League, parse_calendar, parse_league

OUT_AFTER_MISSED = 3  # team games missed in a row before a player counts as out
SEASON_SIMS = 2000
DEFAULT_CUTS = (2, 4, 6, 8, 10, 12, 14)


def _outcome(result: str | None) -> float:
    return 1.0 if result == "WIN" else 0.0 if result == "LOSS" else 0.5


class Replay:
    def __init__(self, h: History):
        self.h = h
        settings = h.league["settings"]
        self.n_periods = int(settings["scheduleSettings"]["matchupPeriodCount"])
        self.playoff_teams = int(settings["scheduleSettings"].get("playoffTeamCount", 4))
        self.each_cat = settings["scoringSettings"].get("scoringType") == EACH_CATEGORY
        self.calendar = parse_calendar(h.pro_schedule, h.league)
        if self.calendar is None:
            raise ValueError("No NBA schedule for this season.")
        self.matchups = [
            m
            for m in h.league.get("schedule", [])
            if "home" in m and "away" in m and int(m["matchupPeriodId"]) <= self.n_periods
        ]
        self.days = sorted(h.box)
        # Days each NBA team played, and each player's game lines in day order.
        self.team_days = {t: sorted(d) for t, d in self.calendar.team_days.items()}
        self.lines: dict[int, list[tuple[int, int, dict]]] = defaultdict(list)
        for d in self.days:
            for line in h.box[d]:
                self.lines[line["pid"]].append((d, line["team"], line["stats"]))

    # ---- the league as it stood on one day -------------------------------------------------

    def first_day(self, period: int) -> int:
        return self.calendar.period_days[period][0]

    def _player(self, pid: int, day: int) -> dict:
        info = self.h.players.get(pid, {})
        before = [(d, t, s) for d, t, s in self.lines.get(pid, []) if d < day]
        team = before[-1][1] if before else next((t for _, t, _ in self.lines.get(pid, [])), info.get("proTeamId", 0))
        stats = []
        prior = self.h.prior.get(pid)
        if prior:
            stats.append(
                {"seasonId": self.h.season - 1, "statSourceId": 0, "statSplitTypeId": 0, "averageStats": prior}
            )
        if before:
            total: dict[str, float] = defaultdict(float)
            for _, _, s in before:
                for k, v in s.items():
                    total[k] += v
            avg = {k: v / len(before) for k, v in total.items()}
            avg[str(GAMES_PLAYED)] = float(len(before))
            stats.append({"seasonId": self.h.season, "statSourceId": 0, "statSplitTypeId": 0, "averageStats": avg})
        recent = [d for d in self.team_days.get(team, []) if d < day][-OUT_AFTER_MISSED:]
        played = {d for d, _, _ in before}
        out = len(recent) == OUT_AFTER_MISSED and not played.intersection(recent)
        return {
            "id": pid,
            "fullName": info.get("fullName", str(pid)),
            "eligibleSlots": info.get("eligibleSlots", []),
            "proTeamId": team,
            "injuryStatus": "OUT" if out else "ACTIVE",
            "stats": stats,
        }

    def _records(self, period: int) -> dict[int, list[float]]:
        rec: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
        for m in self.matchups:
            if int(m["matchupPeriodId"]) >= period:
                continue
            for side in ("home", "away"):
                tid = int(m[side]["teamId"])
                if self.each_cat:
                    for v in (m[side].get("cumulativeScore") or {}).get("scoreByStat", {}).values():
                        r = v.get("result")
                        if r is not None:
                            rec[tid][0 if r == "WIN" else 1 if r == "LOSS" else 2] += 1
                else:
                    w = m.get("winner")
                    won = (w == "HOME") == (side == "home")
                    rec[tid][2 if w == "TIE" else 0 if won else 1] += 1
        return rec

    def snapshot(self, period: int) -> League:
        """The league on the first day of `period`, knowing only what had happened by then."""
        day = self.first_day(period)
        rosters = self.h.rosters.get(day) or self.h.rosters[min(self.h.rosters, key=lambda d: abs(d - day))]
        records = self._records(period)
        teams = []
        for t in self.h.league["teams"]:
            tid = int(t["id"])
            w, lo, ti = records[tid]
            entries = [
                {"lineupSlotId": slot, "playerPoolEntry": {"player": self._player(pid, day), "onTeamId": tid}}
                for pid, slot in rosters.get(tid, [])
            ]
            teams.append(
                {
                    "id": tid,
                    "abbrev": t.get("abbrev", ""),
                    "name": t.get("name", ""),
                    "roster": {"entries": entries},
                    "record": {"overall": {"wins": w, "losses": lo, "ties": ti}},
                }
            )
        schedule = []
        for m in self.matchups:
            past = int(m["matchupPeriodId"]) < period
            schedule.append(
                {
                    "matchupPeriodId": m["matchupPeriodId"],
                    "home": {"teamId": m["home"]["teamId"]},
                    "away": {"teamId": m["away"]["teamId"]},
                    "winner": m.get("winner", "UNDECIDED") if past else "UNDECIDED",
                }
            )
        raw = {
            "id": self.h.league_id,
            "seasonId": self.h.season,
            "settings": self.h.league["settings"],
            "status": {
                **(self.h.league.get("status") or {}),
                "latestScoringPeriod": day,
                "currentMatchupPeriod": period,
            },
            "teams": teams,
            "schedule": schedule,
        }
        return parse_league(raw, [], self.h.pro_schedule)

    # ---- scoring --------------------------------------------------------------------------

    def weekly(self, first: int = 2) -> WeeklyScore:
        """Project every period from its first day and score each category against the final result."""
        score = WeeklyScore()
        for period in range(first, self.n_periods + 1):
            lg = self.snapshot(period)
            model = CategoryModel(lg)
            idx = {t.id: i for i, t in enumerate(lg.teams)}
            dists = [model.dist(t.roster, period) for t in lg.teams]
            probs = model.win_matrix(dists)
            for m in self.matchups:
                if int(m["matchupPeriodId"]) != period:
                    continue
                for side, other in (("home", "away"), ("away", "home")):
                    i, j = idx[int(m[side]["teamId"])], idx[int(m[other]["teamId"])]
                    actual = (m[side].get("cumulativeScore") or {}).get("scoreByStat", {})
                    for c, cat in enumerate(model.cats):
                        a = actual.get(str(cat.stat_id))
                        if a is None:
                            continue
                        score.add(
                            cat.name,
                            probs[i, j, c],
                            _outcome(a.get("result")),
                            a["score"],
                            dists[i].mu[c],
                            dists[i].sd[c],
                        )
        return score

    def season(self, cuts: tuple[int, ...] = DEFAULT_CUTS, n_sims: int = SEASON_SIMS) -> SeasonScore:
        """Playoff odds and final records simulated from each cut week, against the final standings."""
        final = {int(t["id"]): t for t in self.h.league["teams"]}
        score = SeasonScore()
        for period in cuts:
            if period > self.n_periods:
                continue
            lg = self.snapshot(period)
            model = CategoryModel(lg)
            dists = [model.dist(t.roster) for t in lg.teams]
            period_dists = {p: [model.dist(t.roster, p) for t in lg.teams] for p in range(period, self.n_periods + 1)}
            res = simulate_season(lg, dists, model.reverse, n_sims=n_sims, period_dists=period_dists)
            for i, t in enumerate(lg.teams):
                ft = final[t.id]
                seed = ft.get("playoffSeed") or 99
                pct = float(((ft.get("record") or {}).get("overall") or {}).get("percentage", 0.0))
                score.add(
                    period,
                    float(res.playoff_pct[i]),
                    float(seed <= self.playoff_teams),
                    pct,
                    res.win_pct[i],
                    res.win_pct_sd[i],
                )
        return score


def _log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 0.01, 0.99)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


@dataclass
class WeeklyScore:
    rows: list[tuple[str, float, float, float, float, float]] = field(default_factory=list)

    def add(self, cat: str, p: float, outcome: float, actual: float, mu: float, sd: float) -> None:
        self.rows.append((cat, float(p), outcome, float(actual), float(mu), float(sd)))

    def summary(self) -> dict:
        if not self.rows:
            return {}
        cats = np.array([r[0] for r in self.rows])
        p, y, act, mu, sd = (np.array([r[k] for r in self.rows]) for k in range(1, 6))
        z = (act - mu) / np.maximum(sd, 1e-9)
        per_cat = {
            c: {
                "brier": float(((p[m] - y[m]) ** 2).mean()),
                "spread": float(z[m].std()),
                "bias_pct": float(((act[m] - mu[m]) / np.maximum(np.abs(mu[m]), 1e-9)).mean() * 100),
            }
            for c in dict.fromkeys(cats)
            for m in [cats == c]
        }
        return {
            "n": len(self.rows),
            "brier": float(((p - y) ** 2).mean()),
            "log_loss": _log_loss(p, y),
            "spread": float(z.std()),
            "categories": per_cat,
        }


@dataclass
class SeasonScore:
    rows: list[tuple[int, float, float, float, float, float]] = field(default_factory=list)

    def add(self, cut: int, odds: float, made: float, pct: float, sim_pct: float, sim_sd: float) -> None:
        self.rows.append((cut, odds, made, pct, float(sim_pct), float(sim_sd)))

    def summary(self) -> dict:
        if not self.rows:
            return {}
        cut, odds, made, pct, sim, sd = (np.array([r[k] for r in self.rows]) for k in range(6))
        z = (pct - sim) / np.maximum(sd, 1e-3)
        return {
            "n": len(self.rows),
            "brier": float(((odds - made) ** 2).mean()),
            "log_loss": _log_loss(odds, made),
            "spread": float(np.sqrt((z**2).mean())),
            "by_cut": {
                int(c): {
                    "brier": float(((odds[m] - made[m]) ** 2).mean()),
                    "spread": float(np.sqrt((z[m] ** 2).mean())),
                }
                for c in sorted(set(cut.tolist()))
                for m in [cut == c]
            },
        }

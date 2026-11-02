"""What each side could see when an offer was decided: player and team snapshots as of that morning.

Everything is computed from data dated before the decision day, so nothing leaks from the
future. The one exception is marked: ESPN's preseason projection and draft position are as
ESPN served them before the season, which is fine, but ESPN only keeps today's injury status,
so availability is inferred from games missed while the NBA team played.

Player value is a 9-category z-score against the players rostered in that league that day
(FG% and FT% weighted by attempts, turnovers counted against). It is a reference scale, not
the app's model: the acceptance model should learn its own weights on top of these columns.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from app.backtest.history import History
from app.espn.parse import parse_calendar
from app.tradedata.offers import Offer

# ESPN stat ids in the box lines.
PTS, BLK, STL, AST, REB, TO, FGM, FGA, FTM, FTA, TPM, MIN, GP = (
    "0", "1", "2", "3", "6", "11", "13", "14", "15", "16", "17", "40", "42",
)  # fmt: skip
COLS = [PTS, BLK, STL, AST, REB, TO, FGM, FGA, FTM, FTA, TPM, MIN]
IDX = {k: i for i, k in enumerate(COLS)}
CATS = ["PTS", "REB", "AST", "STL", "BLK", "3PM", "FG%", "FT%", "TO"]
COUNTING = {"PTS": PTS, "REB": REB, "AST": AST, "STL": STL, "BLK": BLK, "3PM": TPM, "TO": TO}
WINDOWS = (7, 15)  # days, as ESPN's "last 7 / last 15" splits
OUT_AFTER_MISSED = 3


def _vec(stats: dict) -> np.ndarray:
    return np.array([float(stats.get(k, 0.0)) for k in COLS])


class Snapshot:
    """A league-season's box scores, indexed so 'as of day D' sums are cheap."""

    def __init__(self, h: History, meta: dict[int, dict] | None = None):
        self.h = h
        self.meta = meta or {}
        self.calendar = parse_calendar(h.pro_schedule, h.league)
        days: dict[int, list[int]] = defaultdict(list)
        rows: dict[int, list[np.ndarray]] = defaultdict(list)
        for d in sorted(h.box):
            for line in h.box[d]:
                days[line["pid"]].append(d)
                rows[line["pid"]].append(_vec(line["stats"]))
        self.days = {p: np.array(v) for p, v in days.items()}
        self.cum = {p: np.vstack([np.zeros(len(COLS)), np.cumsum(np.vstack(r), axis=0)]) for p, r in rows.items()}
        self.team_days = {t: np.array(sorted(d)) for t, d in (self.calendar.team_days if self.calendar else {}).items()}
        self.draft = {
            int(p["playerId"]): p
            for p in (h.league.get("draftDetail") or {}).get("picks", [])
            if p.get("playerId", -1) > 0
        }
        self._z_cache: dict[tuple[int, int | None], tuple[dict, dict]] = {}

    # ---- totals -------------------------------------------------------------------------

    def totals(self, pid: int, day: int, window: int | None = None) -> tuple[int, np.ndarray]:
        """Games and summed stats from box scores strictly before `day` (within `window` days if given)."""
        if pid not in self.days:
            return 0, np.zeros(len(COLS))
        d, c = self.days[pid], self.cum[pid]
        hi = int(np.searchsorted(d, day, side="left"))
        lo = int(np.searchsorted(d, day - window, side="left")) if window else 0
        return hi - lo, c[hi] - c[lo]

    def last_game(self, pid: int, day: int) -> dict | None:
        if pid not in self.days:
            return None
        hi = int(np.searchsorted(self.days[pid], day, side="left"))
        if hi == 0:
            return None
        row = self.cum[pid][hi] - self.cum[pid][hi - 1]
        return {"day": int(self.days[pid][hi - 1]), **_cats_of(row, 1)}

    def nba_team(self, pid: int, day: int) -> int:
        for d in range(day - 1, 0, -1):
            for line in self.h.box.get(d, ()):
                if line["pid"] == pid:
                    return int(line["team"])
            if d < day - 30:
                break
        return int(self.h.players.get(pid, {}).get("proTeamId", 0))

    def missed(self, pid: int, day: int, window: int = 14) -> tuple[int, int]:
        """(team games, games the player missed) in the last `window` days before `day`."""
        team = self.nba_team(pid, day)
        td = self.team_days.get(team)
        if td is None:
            return 0, 0
        team_games = int(np.sum((td >= day - window) & (td < day)))
        played, _ = self.totals(pid, day, window)
        return team_games, max(0, team_games - played)

    def missed_in_a_row(self, pid: int, day: int) -> int:
        team = self.nba_team(pid, day)
        td = self.team_days.get(team)
        if td is None:
            return 0
        played = set(self.days.get(pid, np.array([])).tolist())
        streak = 0
        for d in reversed(td[td < day].tolist()):
            if d in played:
                break
            streak += 1
        return streak

    # ---- league-relative value ------------------------------------------------------------

    def rostered(self, day: int) -> dict[int, int]:
        """player -> fantasy team the day before `day`."""
        r = self.h.rosters.get(self._roster_day(day), {})
        return {pid: tid for tid, entries in r.items() for pid, _ in entries}

    def z_table(self, day: int, window: int | None = None) -> tuple[dict[int, dict[str, float]], dict[int, int]]:
        """Per-category z and overall value for every player with games, and the value rank."""
        key = (day, window)
        if key in self._z_cache:
            return self._z_cache[key]
        per_game: dict[int, np.ndarray] = {}
        for pid in self.days:
            gp, tot = self.totals(pid, day, window)
            if gp:
                per_game[pid] = tot / gp
        pool = [p for p in self.rostered(day) if p in per_game] or list(per_game)
        out: dict[int, dict[str, float]] = {}
        if len(pool) >= 10:
            out = _z(per_game, pool)
        order = sorted(out, key=lambda p: -out[p]["value"])
        ranks = {p: i + 1 for i, p in enumerate(order)}
        self._z_cache[key] = (out, ranks)
        return out, ranks

    def prior_z(self) -> tuple[dict[int, dict[str, float]], dict[int, int]]:
        if ("prior", None) not in self._z_cache:
            per_game = {p: _vec(s) for p, s in self.h.prior.items() if s}
            pool = sorted(per_game, key=lambda p: -per_game[p][IDX[MIN]])[
                : max(10, len(self.h.league.get("teams", [])) * 13)
            ]
            out = _z(per_game, pool) if len(pool) >= 10 else {}
            order = sorted(out, key=lambda p: -out[p]["value"])
            self._z_cache[("prior", None)] = (out, {p: i + 1 for i, p in enumerate(order)})
        return self._z_cache[("prior", None)]

    def projection_z(self) -> tuple[dict[int, dict[str, float]], dict[int, int]]:
        if ("proj", None) not in self._z_cache:
            per_game = {p: _vec(m["projection"]) for p, m in self.meta.items() if m.get("projection")}
            n = max(10, len(self.h.league.get("teams", [])) * 13)
            pool = sorted(per_game, key=lambda p: -per_game[p][IDX[MIN]])[:n]
            out = _z(per_game, pool) if len(pool) >= 10 else {}
            order = sorted(out, key=lambda p: -out[p]["value"])
            self._z_cache[("proj", None)] = (out, {p: i + 1 for i, p in enumerate(order)})
        return self._z_cache[("proj", None)]

    # ---- one player, one team -------------------------------------------------------------

    def player(self, pid: int, day: int, fantasy_team: int) -> dict:
        info = self.h.players.get(pid, {})
        out: dict = {"pid": pid, "name": info.get("fullName", ""), "fantasy_team": fantasy_team}
        gp, tot = self.totals(pid, day)
        out["season"] = {"gp": gp, **(_cats_of(tot, gp) if gp else {})}
        z, ranks = self.z_table(day)
        out["season"]["value"] = z.get(pid, {}).get("value")
        out["season"]["rank"] = ranks.get(pid)
        for w in WINDOWS:
            g, t = self.totals(pid, day, w)
            zw, rw = self.z_table(day, w)
            out[f"last{w}"] = {"gp": g, **(_cats_of(t, g) if g else {}), "value": zw.get(pid, {}).get("value"),
                               "rank": rw.get(pid)}  # fmt: skip
        out["last_game"] = self.last_game(pid, day)
        pz, pr = self.prior_z()
        out["prior_season"] = {"value": pz.get(pid, {}).get("value"), "rank": pr.get(pid)}
        jz, jr = self.projection_z()
        m = self.meta.get(pid, {})
        out["preseason"] = {
            "projection_value": jz.get(pid, {}).get("value"),
            "projection_rank": jr.get(pid),
            "espn_adp": m.get("adp"),
            "espn_auction_avg": m.get("auction_avg"),
        }
        pick = self.draft.get(pid)
        out["league_draft"] = (
            {
                "pick": pick.get("overallPickNumber"),
                "bid": pick.get("bidAmount"),
                "team": pick.get("teamId"),
                "keeper": pick.get("keeper", False),
            }  # fmt: skip
            if pick
            else None
        )
        team_games, missed = self.missed(pid, day)
        out["availability"] = {
            "team_games_14d": team_games,
            "missed_14d": missed,
            "missed_in_a_row": self.missed_in_a_row(pid, day),
        }
        out["availability"]["likely_out"] = out["availability"]["missed_in_a_row"] >= OUT_AFTER_MISSED
        out["days_on_team"] = self.days_on_team(pid, fantasy_team, day)
        return out

    def days_on_team(self, pid: int, team: int, day: int) -> int | None:
        n = 0
        for d in sorted((x for x in self.h.rosters if x < day), reverse=True):
            if not any(p == pid for p, _ in self.h.rosters[d].get(team, [])):
                return n
            n += 1
        return n  # since the draft

    def team(self, tid: int, day: int, events_before: list[dict]) -> dict:
        roster = [pid for pid, _ in self.h.rosters.get(self._roster_day(day), {}).get(tid, [])]
        z, _ = self.z_table(day)
        value = [z[p]["value"] for p in roster if p in z]
        cat_strength = {c: round(sum(z[p][c] for p in roster if p in z), 3) for c in CATS}

        def count(types: tuple[str, ...]) -> int:
            return sum(1 for e in events_before if e.get("teamId") == tid and e.get("type") in types)

        return {
            "team_id": tid,
            **self.standing(tid, day),
            "roster_size": len(roster),
            "roster_value": round(sum(value), 3),
            "top_player_value": round(max(value), 3) if value else None,
            "likely_out": sum(self.missed_in_a_row(p, day) >= OUT_AFTER_MISSED for p in roster),
            "category_strength": cat_strength,
            "moves_before": count(("FREEAGENT", "WAIVER")),
            "trade_offers_made_before": count(("TRADE_PROPOSAL",)),
            "declines_before": count(("TRADE_DECLINE",)),
            "accepts_before": count(("TRADE_ACCEPT",)),
        }

    def _roster_day(self, day: int) -> int:
        days = [d for d in self.h.rosters if d < day] or [min(self.h.rosters)]
        return max(days)

    def standing(self, tid: int, day: int) -> dict:
        """Record from matchups finished before `day`, the team's rank, and how far from a playoff spot."""
        settings = self.h.league["settings"]
        sched = settings["scheduleSettings"]
        n_reg = int(sched["matchupPeriodCount"])
        playoff_teams = int(sched.get("playoffTeamCount", 4))
        each_cat = settings["scoringSettings"].get("scoringType") == "H2H_CATEGORY"
        period_days = self.calendar.period_days if self.calendar else {}
        rec: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
        done = 0
        for m in self.h.league.get("schedule", []):
            p = int(m["matchupPeriodId"])
            if p > n_reg or "home" not in m or "away" not in m or p not in period_days:
                continue
            if period_days[p][-1] >= day:
                continue
            done = max(done, p)
            for side, other in (("home", "away"), ("away", "home")):
                t = int(m[side]["teamId"])
                if each_cat:
                    sbs = (m[side].get("cumulativeScore") or {}).get("scoreByStat") or {}
                    for v in sbs.values():
                        r = v.get("result")
                        rec[t][0 if r == "WIN" else 1 if r == "LOSS" else 2] += r in ("WIN", "LOSS", "TIE")
                else:
                    w = m.get("winner")
                    rec[t][0 if w == side.upper() else 1 if w == other.upper() else 2] += 1
        teams = [int(t["id"]) for t in self.h.league.get("teams", [])]
        pct = {t: (rec[t][0] + 0.5 * rec[t][2]) / max(1.0, sum(rec[t])) for t in teams}
        order = sorted(teams, key=lambda t: -pct[t])
        rank = order.index(tid) + 1 if tid in order else None
        line = pct[order[playoff_teams - 1]] if len(order) >= playoff_teams else 0.0
        return {
            "wins": rec[tid][0],
            "losses": rec[tid][1],
            "ties": rec[tid][2],
            "win_pct": round(pct.get(tid, 0.0), 4),
            "rank": rank,
            "playoff_line_gap": round(pct.get(tid, 0.0) - line, 4),
            "periods_done": done,
            "periods_left": n_reg - done,
        }


def _cats_of(tot: np.ndarray, gp: int) -> dict[str, float]:
    pg = tot / gp
    fga, fta = tot[IDX[FGA]], tot[IDX[FTA]]
    out = {c: round(float(pg[IDX[s]]), 3) for c, s in COUNTING.items()}
    out["FG%"] = round(float(tot[IDX[FGM]] / fga), 4) if fga else None
    out["FT%"] = round(float(tot[IDX[FTM]] / fta), 4) if fta else None
    out["FGA"] = round(float(pg[IDX[FGA]]), 3)
    out["FTA"] = round(float(pg[IDX[FTA]]), 3)
    out["MIN"] = round(float(pg[IDX[MIN]]), 2)
    return out


def _z(per_game: dict[int, np.ndarray], pool: list[int]) -> dict[int, dict[str, float]]:
    m = np.vstack([per_game[p] for p in pool])
    fg_pct = m[:, IDX[FGM]].sum() / max(1e-9, m[:, IDX[FGA]].sum())
    ft_pct = m[:, IDX[FTM]].sum() / max(1e-9, m[:, IDX[FTA]].sum())

    def impact(x: np.ndarray) -> dict[str, float]:
        d = {c: x[IDX[s]] for c, s in COUNTING.items()}
        d["FG%"] = x[IDX[FGM]] - fg_pct * x[IDX[FGA]]
        d["FT%"] = x[IDX[FTM]] - ft_pct * x[IDX[FTA]]
        return d

    pool_impacts = [impact(per_game[p]) for p in pool]
    mu = {c: float(np.mean([r[c] for r in pool_impacts])) for c in CATS}
    sd = {c: float(np.std([r[c] for r in pool_impacts])) or 1.0 for c in CATS}
    out = {}
    for p, x in per_game.items():
        imp = impact(x)
        z = {c: (imp[c] - mu[c]) / sd[c] * (-1 if c == "TO" else 1) for c in CATS}
        z = {c: round(float(v), 3) for c, v in z.items()}
        z["value"] = round(sum(z[c] for c in CATS), 3)
        out[p] = z
    return out


def offer_row(snap: Snapshot, offer: Offer, events_before: list[dict]) -> dict:
    """One offer with every player's and team's snapshot as of the decision day."""
    day = offer.day
    players = [snap.player(pid, day, frm) | {"to_team": to} for pid, frm, to in offer.moves]
    teams = [snap.team(t, day, events_before) for t in offer.teams if t > 0]
    sides = {}
    for t in offer.teams:
        if t <= 0:
            continue
        gets = [p for p in players if p["to_team"] == t]
        gives = [p for p in players if p["fantasy_team"] == t]
        sides[t] = {
            "gives": [p["pid"] for p in gives],
            "gets": [p["pid"] for p in gets],
            **{f"net_{k}": _net(gets, gives, k) for k in ("season", "last7", "last15", "prior_season")},
            "net_projection_value": round(
                _sum(gets, "preseason", "projection_value") - _sum(gives, "preseason", "projection_value"), 3
            ),
            "best_player_given_rank": min((p["season"]["rank"] for p in gives if p["season"]["rank"]), default=None),
            "best_player_got_rank": min((p["season"]["rank"] for p in gets if p["season"]["rank"]), default=None),
            "roster_spots_freed": len(gives) - len(gets),
        }  # fmt: skip
    settings = snap.h.league["settings"]
    return {
        "league_id": offer.league_id,
        "season": offer.season,
        "offer_id": offer.offer_id,
        "day": day,
        "season_fraction": round(day / max(1, snap.h.last_day), 3),
        "proposed_ms": offer.proposed_ms,
        "outcome": offer.outcome,
        "source": offer.source,
        "proposer": offer.proposer,
        "receivers": offer.receivers,
        "n_players": len(offer.moves),
        "drops": offer.drops,
        "vetoes": offer.vetoes,
        "prior_offers_same_teams": offer.prior_offers,
        "repeat": offer.repeat,
        "notes": offer.notes,
        "league": {
            "size": len(snap.h.league.get("teams", [])),
            "scoring_type": settings["scoringSettings"].get("scoringType"),
            "categories": sorted(int(i["statId"]) for i in settings["scoringSettings"].get("scoringItems", [])),
            "draft_type": (settings.get("draftSettings") or {}).get("type"),
            "veto_votes_required": (settings.get("tradeSettings") or {}).get("vetoVotesRequired"),
        },
        "sides": sides,
        "teams": teams,
        "players": players,
    }


def _sum(ps: list[dict], block: str, key: str) -> float:
    return round(sum(p[block][key] or 0.0 for p in ps if p.get(block)), 3)


def _net(gets: list[dict], gives: list[dict], block: str) -> float:
    return round(_sum(gets, block, "value") - _sum(gives, block, "value"), 3)

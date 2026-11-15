"""Checks what a player's next-game line could add from the game's context (schemas.NextGameLine).

    uv run python -m scripts.next_game_backtest

Before each of a player's games (from his 4th on) in five seasons (2021-22 to 2025-26), his line is
the one the app builds: parse._blend_stats at that point, with the minutes his last 7 and 15 days
point to (model.RECENT_MINUTES_WEIGHTS). Its miss in each category (PTS, REB, AST, STL, BLK, 3PM,
TO, FG and FT impact, and the 9-category value of the whole line) is then fitted, one season left
out at a time, on each context signal:

  * vs this opponent: his average miss in earlier games against the same team (this and past seasons),
  * vs his type of center: the same against the opponent's lead big of the same type (four types,
    clustered on rebounds, blocks, threes, steals, assists and free throws per 36),
  * his own miss: his average miss in all his earlier games,
  * opponent win% and pace, former team, his team lost its last game, his team's win%.

A signal passes when it lowers the squared miss in every one of the five left-out seasons. Then the
matchup signals are fitted again on top of his own miss: what they add beyond who he is.

Printed: the signals that pass for each category, for every player and for stars (a line of 22+
points), then the matchup signals on top of his own miss, and the mean miss of each stat with and
without the opponent's win% (a multiplicative factor per stat).

Uses the caches of scripts/durability_backtest.py (.cache/season_pools, .cache/gamelogs).
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from app.analysis.form import FG_SCALE, FT_SCALE, SCALE
from app.analysis.model import RECENT_MINUTES_WEIGHTS
from app.analysis.tonight import OTHER_ROLE, REGULAR_MINUTES, SAME_ROLE, role
from app.espn.ages import season_age
from app.espn.parse import GAMES_PLAYED, MINUTES, PROVEN_GAMES, PROVEN_SEASONS, _blend_stats, with_minutes
from scripts.age_backtest import _entry, _names
from scripts.durability_backtest import LOGS, game_logs, season_pool
from scripts.minutes_backtest import season_lines
from scripts.projection_backtest import projections

SEASONS = (2022, 2023, 2024, 2025, 2026)
FIRST = 3  # games played before the first case
STAR_POINTS = 22.0
PTS, REB, AST, STL, BLK, TPM, TO, FGM, FGA, FTM, FTA, TPA = 0, 6, 3, 2, 1, 17, 11, 13, 14, 15, 16, 18
STATS = (MINUTES, PTS, REB, AST, STL, BLK, TPM, TO, FGM, FGA, FTM, FTA, TPA)
CATS = ("PTS", "REB", "AST", "STL", "BLK", "3PM", "TO", "FG", "FT", "VALUE")
BIG_TYPES = 4


def cats(line: dict[int, float]) -> list[float]:
    """The line in each category, FG and FT as impact (makes over the league rate), then its value."""
    fg = line[FGM] - FG_SCALE[0] * line[FGA]
    ft = line[FTM] - FT_SCALE[0] * line[FTA]
    out = [line[s] for s in (PTS, REB, AST, STL, BLK, TPM, TO)] + [fg, ft]
    value = sum((line[s] - m) / sd * (-1 if s == TO else 1) for s, (m, sd) in SCALE.items())
    return [*out, value + fg / FG_SCALE[1] + ft / FT_SCALE[1]]


def season_games(season: int) -> dict[int, dict[str, tuple[int, int, dict[int, float]]]]:
    """Each player's games: game id -> (day, his team, his line)."""
    game_logs(season)
    out: dict[int, dict[str, tuple[int, int, dict[int, float]]]] = defaultdict(dict)
    for team in range(1, 31):
        for e in json.loads((LOGS / f"{season}-{team}.json").read_text()):
            p = e.get("player") or e
            for s in p.get("stats") or []:
                if s.get("seasonId") != season or s.get("statSplitTypeId") != 5 or s.get("statSourceId"):
                    continue
                line = {int(k): float(v) for k, v in (s.get("stats") or {}).items()}
                if line.get(GAMES_PLAYED):
                    out[int(p["id"])][s["externalId"]] = (int(s["scoringPeriodId"]), int(s["proTeamId"]), line)
    return out


def big_types(games: dict[int, dict]) -> tuple[dict[tuple[int, int], int], np.ndarray]:
    """(season, player) -> type of big, for players with 9+ rebounds or 1.5+ blocks per 36 that season."""
    profiles = {}
    for season, players in games.items():
        for pid, gl in players.items():
            lines = [x[2] for x in gl.values()]
            minutes = sum(x.get(MINUTES, 0.0) for x in lines)
            if minutes >= 200:
                per36 = np.array(
                    [sum(x.get(k, 0.0) for x in lines) / minutes * 36 for k in (REB, BLK, TPA, STL, AST, FTA)]
                )
                if per36[0] >= 9 or per36[1] >= 1.5:
                    profiles[(season, pid)] = per36
    x = np.array(list(profiles.values()))
    mu, sd = x.mean(0), x.std(0)
    z = (x - mu) / sd
    best = None
    for seed in range(8):
        cen = z[np.random.default_rng(seed).choice(len(z), BIG_TYPES, replace=False)]
        for _ in range(100):
            lab = ((z[:, None] - cen[None]) ** 2).sum(-1).argmin(1)
            cen = np.array([z[lab == j].mean(0) if (lab == j).any() else cen[j] for j in range(BIG_TYPES)])
        sse = ((z - cen[lab]) ** 2).sum()
        if best is None or sse < best[0]:
            best = (sse, cen, lab)
    _, cen, lab = best
    return dict(zip(profiles, (int(v) for v in lab), strict=True)), cen * sd + mu


_ROLES: dict[int, dict[int, int]] = {}


def season_roles(g: dict[int, dict], season: int) -> dict[int, int]:
    """Each player's role (tonight.role) from his season's per-game line."""
    if season not in _ROLES:
        roles = {}
        for pid, gl in g.items():
            n = len(gl)
            line = {k: sum(x[2].get(k, 0.0) for x in gl.values()) / n for k in (MINUTES, AST, REB)}
            roles[pid] = role(line)
        _ROLES[season] = roles
    return _ROLES[season]


def regulars_out(g: dict[int, dict], sides: dict) -> dict[tuple[str, int], list[tuple[int, float]]]:
    """(game, team) -> the regulars who sat it: (player, minutes a game). A regular played 5 of the
    team's last 10 games at REGULAR_MINUTES or more, and one of its last 3 (a fresh absence)."""
    order: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for gid, teams in sides.items():
        for team, side in teams.items():
            order[team].append((g[side[0][0]][gid][0], gid))
    out = {}
    for team, games in order.items():
        games.sort()
        played = [{pid: x.get(MINUTES, 0.0) for pid, x in sides[gid][team]} for _, gid in games]
        for i, (_, gid) in enumerate(games):
            if i < 5:
                continue
            recent = played[max(0, i - 10) : i]
            fresh = set().union(*played[max(0, i - 3) : i])
            sat = []
            for pid in fresh - set(played[i]):
                minutes = [p[pid] for p in recent if pid in p]
                if len(minutes) >= 5 and np.mean(minutes) >= REGULAR_MINUTES:
                    sat.append((pid, float(np.mean(minutes))))
            out[(gid, team)] = sat
    return out


def out_for(out: dict, roles: dict[int, int], pid: int, gid: str, team: int, opp: int) -> tuple[float, float, float]:
    """Minutes a game of same-role and other-role teammates who sat, and of the opponent's regulars who sat."""
    mine = roles.get(pid, 1)
    sat = out.get((gid, team), [])
    same = sum(m for q, m in sat if roles.get(q, 1) == mine)
    other = sum(m for q, m in sat if roles.get(q, 1) != mine)
    return same, other, sum(m for _, m in out.get((gid, opp), []))


def cases() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rows of context signals, the misses per category, and the line and actual per stat."""
    games = {s: season_games(s) for s in SEASONS}
    types, centres = big_types(games)
    print("types of big (rebounds, blocks, 3PA, steals, assists, FTA per 36):")
    for j, c in enumerate(centres):
        print(f"  {j}: {np.round(c, 1)}")
    rows, misses, preds, acts = [], [], [], []
    by_opp: dict[tuple[int, int], list] = defaultdict(list)
    by_big: dict[tuple[int, int], list] = defaultdict(list)
    own: dict[int, list] = defaultdict(list)
    past_teams: dict[int, set[int]] = defaultdict(set)
    for season in SEASONS:
        g = games[season]
        sides: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
        for pid, gl in g.items():
            for gid, (_day, team, line) in gl.items():
                sides[gid][team].append((pid, line))
        results: dict[int, list[tuple[int, bool, float]]] = defaultdict(list)
        lead_big: dict[tuple[str, int], int] = {}
        for gid, teams in sides.items():
            if len(teams) != 2:
                continue
            (a, pa), (b, pb) = teams.items()
            day = g[pa[0][0]][gid][0]
            sa, sb = sum(x.get(PTS, 0.0) for _, x in pa), sum(x.get(PTS, 0.0) for _, x in pb)
            results[a].append((day, sa > sb, sa + sb))
            results[b].append((day, sb > sa, sa + sb))
            for team, side in teams.items():
                bigs = [(x.get(MINUTES, 0.0), types[(season, pid)]) for pid, x in side if (season, pid) in types]
                if bigs:
                    lead_big[(gid, team)] = max(bigs)[1]
        out = regulars_out(g, sides)
        proj, last, names = projections(season), season_lines(season - 1), _names(season)
        past = [season_pool(season - k)[0] for k in range(1, PROVEN_SEASONS + 1)]
        todo = []
        for pid, pr in proj.items():
            gl = sorted(((d, gid, team, x) for gid, (d, team, x) in g.get(pid, {}).items()), key=lambda t: t[0])
            age = season_age(names.get(pid, ""), season)
            proven = all(p.get(pid, 0) >= PROVEN_GAMES for p in past)
            keys = sorted({k for *_, x in gl for k in x})
            total = dict.fromkeys(keys, 0.0)
            for i, (day, gid, team, x) in enumerate(gl):
                if i >= FIRST:
                    so_far = {k: v / i for k, v in total.items()} | {GAMES_PLAYED: i}
                    stats = [_entry(season, 1, pr)] + ([_entry(season - 1, 0, last[pid])] if pid in last else [])
                    line = _blend_stats({"stats": [*stats, _entry(season, 0, so_far)]}, season, age, proven=proven)[0]
                    m = line.get(MINUTES, 0.0)
                    if m > 0:
                        shift = 0.0
                        for w, days in zip(RECENT_MINUTES_WEIGHTS, (7, 15), strict=True):
                            inside = [y[MINUTES] for d, *_, y in gl[:i] if d >= day - days]
                            if inside:
                                shift += w * (float(np.mean(inside)) - m)
                        line = with_minutes(line, min(max(m + shift, 0.0), 40.0))
                        todo.append((day, pid, gid, team, line, x))
                for k in keys:
                    total[k] += x.get(k, 0.0)
        for day, pid, gid, team, line, x in sorted(todo, key=lambda t: (t[0], t[1])):
            opp = next((t for t in sides[gid] if t != team), None)
            if opp is None:
                continue
            pl, al = {s: line.get(s, 0.0) for s in STATS}, {s: x.get(s, 0.0) for s in STATS}
            miss = np.array(cats(al)) - np.array(cats(pl))
            ho, hown = by_opp[(pid, opp)], own[pid]
            bt = lead_big.get((gid, opp))
            hb = by_big[(pid, bt)] if bt is not None else []
            mine = [r for r in results[team] if r[0] < day]
            theirs = [r for r in results[opp] if r[0] < day]
            rows.append(
                [season, pl[PTS], len(ho), len(hb), len(hown), opp in past_teams[pid] and opp != team]
                + [(not mine[-1][1]) if mine else 0.0]
                + [(np.mean([r[1] for r in mine]) - 0.5) if len(mine) >= 5 else 0.0]
                + [(np.mean([r[1] for r in theirs]) - 0.5) if len(theirs) >= 5 else 0.0]
                + [(np.mean([r[2] for r in theirs]) - 225.0) if len(theirs) >= 5 else 0.0]
                + list(np.mean(ho, 0) if ho else np.zeros(len(CATS)))
                + list(np.mean(hb, 0) if hb else np.zeros(len(CATS)))
                + list(np.mean(hown, 0) if hown else np.zeros(len(CATS)))
                + list(out_for(out, season_roles(g, season), pid, gid, team, opp))
            )
            misses.append(miss)
            preds.append([pl[s] for s in STATS])
            acts.append([al[s] for s in STATS])
            ho.append(miss)
            hown.append(miss)
            if bt is not None:
                hb.append(miss)
        for pid, gl in g.items():
            past_teams[pid] |= {team for _, team, _ in gl.values()}
    return np.array(rows, float), np.array(misses), np.array(preds), np.array(acts)


def gain(y: np.ndarray, season: np.ndarray, mask: np.ndarray, base: list, extra: list) -> list[float]:
    """Per left-out season: % less squared miss with ``extra`` on top of ``base`` (lists of signal columns)."""
    out = []
    for t in SEASONS:
        tr, te = mask & (season != t), mask & (season == t)

        def err(cols: list, tr: np.ndarray = tr, te: np.ndarray = te) -> float:
            a = np.column_stack([np.ones(tr.sum())] + [c[tr] for c in cols])
            b = np.linalg.lstsq(a, y[tr], rcond=None)[0]
            return float(np.mean((y[te] - np.column_stack([np.ones(te.sum())] + [c[te] for c in cols]) @ b) ** 2))

        e0 = err(base)
        out.append((e0 - err(base + extra)) / e0 * 100)
    return out


def main() -> None:
    x, y, pred, act = cases()
    season, points = x[:, 0], x[:, 1]
    n_opp, n_big, n_own, former, lost, win, opp_win, pace = (x[:, i] for i in range(2, 10))
    k = len(CATS)
    vs_opp, vs_big, mine = x[:, 10 : 10 + k], x[:, 10 + k : 10 + 2 * k], x[:, 10 + 2 * k : 10 + 3 * k]
    for label, mask in (("every player", np.ones(len(x), bool)), ("stars", points >= STAR_POINTS)):
        print(f"\n{label} ({mask.sum()} games): signals better in all five seasons, % less squared miss")
        for c, name in enumerate(CATS):
            signals = {
                "vs this opponent": vs_opp[:, c] * n_opp / (n_opp + 6),
                "vs his type of center": vs_big[:, c] * n_big / (n_big + 15),
                "his own miss": mine[:, c] * n_own / (n_own + 20),
                "opponent win%": opp_win,
                "opponent pace": pace,
                "former team": former,
                "team lost last game": lost,
                "team win%": win,
            }
            passed = []
            for sname, sig in signals.items():
                g = gain(y[:, c], season, mask, [], [sig])
                if min(g) > 0:
                    passed.append(f"{sname} {min(g):.2f}-{max(g):.2f}")
            print(f"  {name:5s} " + ("; ".join(passed) if passed else "none"))
        print("  on top of his own miss (min..max over seasons):")
        for c, name in enumerate(CATS):
            o = mine[:, c] * n_own / (n_own + 20)
            parts = []
            for sname, sig in (
                ("vs this opponent", vs_opp[:, c] * n_opp / (n_opp + 6)),
                ("vs his type of center", vs_big[:, c] * n_big / (n_big + 15)),
            ):
                g = gain(y[:, c], season, mask, [o], [sig])
                parts.append(f"{sname} {min(g):+.2f}..{max(g):+.2f}")
            print(f"    {name:5s} " + " | ".join(parts))
    level(pred, act, season, points)
    k3 = 10 + 3 * k
    sat(pred, act, season, x[:, k3], x[:, k3 + 1], x[:, k3 + 2])
    print("\nmean miss per game: the line, then with a factor per stat for the opponent's win% (fitted on the others)")
    for j, stat in enumerate(STATS):
        miss0, miss1 = [], []
        for t in SEASONS:
            tr, te = season != t, season == t
            z = pred[tr, j] * opp_win[tr]
            f = np.dot(z, act[tr, j] - pred[tr, j]) / np.dot(z, z)
            miss0.append(np.abs(act[te, j] - pred[te, j]).mean())
            miss1.append(np.abs(act[te, j] - pred[te, j] * (1 + f * opp_win[te])).mean())
        print(f"  stat {stat:2d}: {np.mean(miss0):.3f} -> {np.mean(miss1):.3f}")


def sat(
    pred: np.ndarray, act: np.ndarray, season: np.ndarray, same: np.ndarray, other: np.ndarray, opp: np.ndarray
) -> None:
    """Teammates and opponents who sat: his minutes, then each stat per minute once the minutes are counted."""
    j = STATS.index(MINUTES)
    miss = act[:, j] - pred[:, j]
    print(
        "\nregulars who sat (fresh absences): minutes he gained per minute of theirs, fitted on the other four seasons"
    )
    for name, cols in (
        ("same role", [same]),
        ("same and other role", [same, other]),
        ("and the opponent's", [same, other, opp]),
    ):
        g = gain(miss, season, np.ones(len(miss), bool), [], cols)
        w = np.linalg.lstsq(np.column_stack([np.ones(len(miss))] + cols), miss, rcond=None)[0][1:]
        print(f"  {name:20s} {np.round(w, 3)}  % less squared miss {min(g):+.2f}..{max(g):+.2f}")
    print(f"  in use: same role {SAME_ROLE}, other role {OTHER_ROLE}")
    hit = same > 0
    shifted = miss - SAME_ROLE * same - OTHER_ROLE * other
    print(
        f"  games with a same-role regular out ({hit.mean():.1%}): mean miss"
        f" {np.abs(miss[hit]).mean():.2f} -> {np.abs(shifted[hit]).mean():.2f} min"
    )
    minutes = np.maximum(pred[:, j] + SAME_ROLE * same + OTHER_ROLE * other, 1.0)
    share_own, share_opp = (same + other) / 240, opp / 240
    print(
        "  then per minute (a usage bump): % less squared miss with his line x share of team minutes out,"
        " own | opponent's"
    )
    for i, stat in enumerate(STATS):
        if stat == MINUTES:
            continue
        line = pred[:, i] * minutes / np.maximum(pred[:, j], 1.0)
        y = act[:, i] - line
        g1 = gain(y, season, np.ones(len(y), bool), [], [line * share_own])
        g2 = gain(y, season, np.ones(len(y), bool), [], [line * share_opp])
        print(f"    stat {stat:2d}: {min(g1):+.2f}..{max(g1):+.2f} | {min(g2):+.2f}..{max(g2):+.2f}")


def level(pred: np.ndarray, act: np.ndarray, season: np.ndarray, points: np.ndarray) -> None:
    """The line's level: a factor per stat (what he did = factor x the line), fitted on the other seasons."""
    print("\nthe line's level: factor per stat fitted on the other four seasons, % less squared miss, mean miss")
    for j, stat in enumerate(STATS):
        factors, gains, stars, miss0, miss1 = [], [], [], [], []
        for t in SEASONS:
            tr, te = season != t, season == t
            f = np.dot(pred[tr, j], act[tr, j]) / np.dot(pred[tr, j], pred[tr, j])
            factors.append(f)
            for out, m in ((gains, te), (stars, te & (points >= STAR_POINTS))):
                e0 = np.mean((act[m, j] - pred[m, j]) ** 2)
                out.append((e0 - np.mean((act[m, j] - f * pred[m, j]) ** 2)) / e0 * 100)
            miss0.append(np.abs(act[te, j] - pred[te, j]).mean())
            miss1.append(np.abs(act[te, j] - f * pred[te, j]).mean())
        mark = "  better in all five" if min(gains) > 0 else ""
        print(
            f"  stat {stat:2d}: factor {np.mean(factors):.3f} ({min(factors):.3f}-{max(factors):.3f})"
            f"  gain {min(gains):+.2f}..{max(gains):+.2f}%  stars {min(stars):+.2f}..{max(stars):+.2f}%"
            f"  miss {np.mean(miss0):.3f} -> {np.mean(miss1):.3f}{mark}"
        )


if __name__ == "__main__":
    main()

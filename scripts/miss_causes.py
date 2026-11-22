"""Which parts of our game calls keep missing the same way: the hand-off from the why-we-lost log
(app/lostlog.py) to the small-signals check every three days (Can 2026-11-22: "each iteration would lead
us to better estimations").

    uv run python -m scripts.miss_causes                 # the season so far (2027)
    uv run python -m scripts.miss_causes 2026 --json     # any season; --json for the scheduled check

Every finished game of the season is called again the way the app calls it before tip-off
(gamebox.Box.call_parts, ratings from the games before that day; injury news at tip-off isn't kept
offline, so late news is left out), and each team's miss is split as lostlog.team_pieces splits it:
pace, threes, free throws and the rest. Three readings come out:

- shares: how much of the total's and the margin's miss each piece carries this season (its covariance
  with the miss over the miss's variance, so the shares add to 1), next to the range of 2021-22..2025-26.
  Threes and free throws are shooting luck over a season; a share leaving its usual range says the
  season plays differently, not that the model broke.
- league: the average of each piece over every team game. A piece that leans one way all season (pace
  over-called, say) is the kind of miss a weight can fix.
- teams: each team's average piece in its own points and in its opponents' points. One team that keeps
  missing the same way (its pace, its threes against the call's expectation) is named.

A lean is named a candidate only when it is at least REPEAT_T standard errors from zero over at least
REPEAT_GAMES games (a team's lean measured past the league's, so a league-wide lean is named once); with
30 teams and 7 pieces each, chance alone names one about every fourth run, which the blind test then
throws out. A candidate changes nothing: it goes to the signal search and is tested blind on five seasons
first (Can's rule: a signal earns weight only by beating the current estimate in every season).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict

from app import lostlog
from app.analysis import gamebox, gamescore
from app.espn.parse import PRO_TEAM_ABBREV
from scripts.durability_backtest import LOGS, game_logs
from scripts.next_game_matchup_backtest import schedule

PIECES = (lostlog.PACE, "three", "free_throws", lostlog.OTHER)
REPEAT_T = 3.5
REPEAT_GAMES = 8
# The shares over 5,225 games of 2021-22..2025-26 (low, high across the five seasons; the same replay).
PAST_SHARES = {
    "total": {
        lostlog.PACE: (0.25, 0.31),
        "three": (0.35, 0.39),
        "free_throws": (0.02, 0.03),
        lostlog.OTHER: (0.32, 0.38),
    },
    "margin": {"three": (0.50, 0.63)},
}
SHARE_SLACK = 0.05  # a share this far outside its past range is named; inside that it is a normal season
SHARE_GAMES = 400  # fewer games and a share swings more than that by chance (2027 at 231 games: threes 29%)


def season_rows(season: int) -> list[gamebox.GameRow]:
    game_logs(season)
    rows: list[gamebox.GameRow] = []
    for team in range(1, 31):
        rows += gamebox.parse_rows(json.loads((LOGS / f"{season}-{team}.json").read_text()), season)
    return rows


def season_games(season: int) -> list[dict]:
    """Every finished game of a season: day, teams, each team's call and its miss split in pieces."""
    team_games = gamescore.team_games(season_rows(season))
    start = gamescore.season_end(gamescore.team_games(season_rows(season - 1)))
    sched = gamebox.parse_schedule(schedule(season))
    box = gamebox.Box(season, {}, {}, {}, schedule=sched, team_games=team_games, score_start=start)
    by_game: dict[int, list[gamescore.TeamGame]] = defaultdict(list)
    for g in team_games:
        by_game[g.game].append(g)
    out = []
    for gid, pair in sorted(by_game.items()):
        if len(pair) != 2:
            continue
        day, one = pair[0].day, pair[0]
        game = next((x for x in sched.get(one.team, []) if x.day == day), None)
        if game is None:
            continue
        home, away = (one, pair[1]) if game.home else (pair[1], one)
        ap, hp = box.call_parts(day, away.team, home.team)
        split = lostlog.team_pieces(box, day, away.team, home.team, (sum(ap.values()), sum(hp.values())), set())
        if split is None:
            continue
        out.append({"game": gid, "day": day, "away": away.team, "home": home.team, "split": split})
    return out


def _mean_t(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    if n < 2:
        return (xs[0] if xs else 0.0), 0.0
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, (m / math.sqrt(var / n) if var > 0 else 0.0)


def _shares(misses: list[float], pieces: dict[str, list[float]]) -> dict[str, float]:
    n = len(misses)
    if n < 2:
        return {}
    mm = sum(misses) / n
    var = sum((x - mm) ** 2 for x in misses)
    if var <= 0:
        return {}
    out = {}
    for k, xs in pieces.items():
        mx = sum(xs) / n
        out[k] = sum((x - mx) * (y - mm) for x, y in zip(xs, misses, strict=True)) / var
    return out


def summary(games: list[dict], abbrev: dict[int, str] | None = None) -> dict:
    """The three readings (module doc) and the candidates among them."""
    abbrev = abbrev or {}
    total_miss: list[float] = []
    margin_miss: list[float] = []
    total_pieces: dict[str, list[float]] = {k: [] for k in PIECES}
    margin_pieces: dict[str, list[float]] = {k: [] for k in PIECES}
    league: dict[str, list[float]] = {k: [] for k in PIECES}
    own: dict[tuple[int, str], list[float]] = defaultdict(list)
    allowed: dict[tuple[int, str], list[float]] = defaultdict(list)
    for g in games:
        a, h = g["split"][g["away"]], g["split"][g["home"]]
        total_miss.append(a["actual"] - a["called"] + h["actual"] - h["called"])
        margin_miss.append((h["actual"] - h["called"]) - (a["actual"] - a["called"]))
        for k in PIECES:
            total_pieces[k].append(a["pieces"][k] + h["pieces"][k])
            margin_pieces[k].append(h["pieces"][k] - a["pieces"][k])
        for team, opp, side in ((g["away"], g["home"], a), (g["home"], g["away"], h)):
            for k in PIECES:
                league[k].append(side["pieces"][k])
                own[(team, k)].append(side["pieces"][k])
                if k != lostlog.PACE:  # pace is the game's: both teams' pace pieces lean together
                    allowed[(opp, k)].append(side["pieces"][k])

    shares = {"total": _shares(total_miss, total_pieces), "margin": _shares(margin_miss, margin_pieces)}
    candidates: list[dict] = []
    for kind, past in PAST_SHARES.items():
        for k, (lo, hi) in past.items():
            v = shares[kind].get(k)
            if v is not None and len(games) >= SHARE_GAMES and (v < lo - SHARE_SLACK or v > hi + SHARE_SLACK):
                candidates.append(
                    {"kind": "share", "what": f"{k} carries {v:.0%} of the {kind}'s miss (past {lo:.0%}-{hi:.0%})"}
                )
    lean = {}
    for k, xs in league.items():
        m, t = _mean_t(xs)
        lean[k] = {"mean": round(m, 2), "t": round(t, 1), "n": len(xs)}
        if len(xs) >= REPEAT_GAMES and abs(t) >= REPEAT_T:
            candidates.append({"kind": "league", "what": f"{k} {m:+.1f} points a team game over {len(xs)} team games"})
    teams = []
    for side, groups in (("own points", own), ("opponents' points", allowed)):
        for (team, k), xs in sorted(groups.items()):
            whose = "games" if k == lostlog.PACE else side
            m, t = _mean_t([x - lean[k]["mean"] for x in xs])
            if len(xs) < REPEAT_GAMES or abs(t) < REPEAT_T:
                continue
            row = {
                "team": abbrev.get(team, str(team)),
                "piece": k,
                "whose": whose,
                "mean": round(m, 2),
                "t": round(t, 1),
            }
            teams.append({**row, "n": len(xs)})
            candidates.append(
                {
                    "kind": "team",
                    "what": f"{row['team']} {k} in its {whose}: {m:+.1f} a game past the league over {len(xs)} games",
                }
            )
    return {
        "games": len(games),
        "shares": {kind: {k: round(v, 3) for k, v in s.items()} for kind, s in shares.items()},
        "league": lean,
        "teams": teams,
        "candidates": candidates,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("season", nargs="?", type=int, default=2027)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    out = {"season": args.season, **summary(season_games(args.season), PRO_TEAM_ABBREV)}
    if args.json:
        print(json.dumps(out))
        return
    print(f"{args.season}: {out['games']} games")
    for kind, s in out["shares"].items():
        print(f"  {kind:<6} share of miss: " + "  ".join(f"{k} {v:.0%}" for k, v in s.items()))
    print("  league lean: " + "  ".join(f"{k} {v['mean']:+.2f} (t {v['t']:+.1f})" for k, v in out["league"].items()))
    print(f"  candidates ({len(out['candidates'])}):")
    for c in out["candidates"]:
        print(f"    - {c['what']}")


if __name__ == "__main__":
    main()

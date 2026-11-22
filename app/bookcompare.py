"""iddaa's lines beside ours and what happened, for the admin comparison page.

Can 2026-11-21: "We're not trying to match the sportsbook, we're trying to match the actual score."
So a number is judged against the result, never by which side of iddaa's line the result fell:
- a number is right (✓) when the result lands within BANDS of it (points within 5, rebounds or
  assists within 2...), and iddaa's number is judged by the same band;
- "closer" says whose number was nearer the result (or a tie);
- the gap is ours less iddaa's: + when we expect more. The page lists the biggest + gaps first.
COMPARISON ONLY: nothing here feeds the model.

Inputs, all saved before tip-off: iddaa's record for the day (app/iddaa.py, with our call at the
same look), the line log's record (app/linelog.py: our player lines), and the day's games (Scores'
day body: final scores and everyone's line once played).
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Iterable

from app import iddaa, linelog
from app.analysis import rungs as rung_chances

# How near the result a number must land to count as right.
BANDS = {
    "pts": 5.0,
    "reb": 2.0,
    "ast": 2.0,
    "stl": 1.0,
    "blk": 1.0,
    "three_pm": 1.0,
    "team_pts": 8.0,
    "total": 12.0,
    "margin": 8.0,
}
PLAYER_STATS = ("three_pm", "reb", "ast", "stl", "blk", "pts")  # the site's stat order
# The day body's names for the player stats.
DAY_STAT = {"pts": "pts", "reb": "reb", "ast": "ast", "stl": "stl", "blk": "blk", "three_pm": "tpm"}
OURS_AT = {name: i for i, (name, _) in enumerate(linelog.STATS)}
LEVEL = 0.1  # misses this close read as level
MIN_MINUTES = 10.0  # estimates listed before the book's player bets: players we expect to play this much
RUNG_LEVEL = 0.005  # rung scores this close read as level
SUFFIX = re.compile(r"( (jr|sr|ii|iii|iv|v))+$")


def name_key(name: str) -> str:
    """A name for matching iddaa's spelling to ESPN's: folded, suffixes off, initials run together
    ("V.J. Edgecombe" and "VJ Edgecombe" both read "vj edgecombe")."""
    words = SUFFIX.sub("", iddaa.fold(name)).split()
    out: list[str] = []
    for w in words:
        if len(words) > 2 and len(w) == 1 and len(out) == 1 and len(out[0]) <= 2:
            out[0] += w
        else:
            out.append(w)
    return " ".join(out)


def match_names(book_names: Iterable[str], espn: dict[int, str]) -> dict[str, int]:
    """iddaa name -> ESPN player id, among the players of one game (espn: id -> name). First the
    whole name; failing that, a unique same last name with the same first initial."""
    keys: dict[str, list[int]] = {}
    for pid, name in espn.items():
        keys.setdefault(name_key(name), []).append(pid)
    out = {}
    for name in book_names:
        key = name_key(name)
        hit = keys.get(key)
        if hit and len(hit) == 1:
            out[name] = hit[0]
            continue
        words = key.split()
        if len(words) < 2:
            continue
        near = [
            pid
            for k, pids in keys.items()
            for pid in pids
            if k.split() and k.split()[-1] == words[-1] and k[0] == words[0][0]
        ]
        if len(near) == 1:
            out[name] = near[0]
    return out


def judge(ours: float | None, book: float | None, actual: float | None, band: float) -> dict:
    """One number pair against the result: the gap, each side's ✓ (None until there is a result,
    or when that side has no number) and who was closer."""
    gap = round(ours - book, 1) if ours is not None and book is not None else None
    out = {"ours": ours, "book": book, "actual": actual, "gap": gap, "ours_hit": None, "book_hit": None}
    out["closer"] = None
    if actual is None:
        return out
    if ours is not None:
        out["ours_hit"] = abs(actual - ours) <= band
    if book is not None:
        out["book_hit"] = abs(actual - book) <= band
    if ours is not None and book is not None:
        a, b = abs(actual - ours), abs(actual - book)
        out["closer"] = "tie" if abs(a - b) <= LEVEL else ("ours" if a < b else "book")
    return out


RUNG_SOURCE = "iddaa"  # the only bulletin whose rungs count (iddaa's own, not a dealer's)


def compare_day(book: dict | None, lines: dict | None, day: dict | None, names: dict[int, str]) -> dict:
    """The day's comparison: per game its match rows and player rows, and per stat a tally.

    book: iddaa's record (app/iddaa.py); lines: the line log's record; day: Scores' day body (None
    before the games); names: ESPN player id -> name, for players not yet in a box score."""
    games_played = {g["id"]: g for g in (day or {}).get("games", [])}
    ours_lines = (lines or {}).get("lines", {})
    out_games = []
    for gid, g in sorted(((book or {}).get("games") or {}).items(), key=lambda kv: (kv[1]["tip"], kv[0])):
        read = iddaa.readout(g)
        margin = iddaa.overround(g)
        # Can 2026-11-22: "No nesine." Only rungs read from iddaa's own bulletin are scored, so the
        # value picks, the rung scores and the record skip nights saved from nesine (no "source").
        scored = g.get("source") == RUNG_SOURCE
        played = games_played.get(int(gid)) or {}
        final = played.get("status") == "final"
        a_pts = (played.get("away") or {}).get("score") if final else None
        h_pts = (played.get("home") or {}).get("score") if final else None
        teams = {g["away"], g["home"]}

        # Players: everyone of these two teams we know a name for.
        box = {ln["player_id"]: ln for ln in played.get("lines", [])} if final else {}
        known = {pid: ln["name"] for pid, ln in box.items()}
        for pid, (team, *_) in ours_lines.items():
            if team in teams and int(pid) not in known:
                known[int(pid)] = names.get(int(pid), "")
        known = {pid: n for pid, n in known.items() if n}
        ids = match_names(read["players"], known)
        players = []
        for name, stats in read["players"].items():
            pid = ids.get(name)
            mine = ours_lines.get(str(pid)) if pid is not None else None
            got = box.get(pid) if pid is not None else None
            for stat in PLAYER_STATS:
                if stat not in stats:
                    continue
                ours = mine[1 + OURS_AT[stat]] if mine else None
                if final and pid is not None:
                    actual = float(got[DAY_STAT[stat]]) if got else None  # no line: he didn't play
                else:
                    actual = None
                steps = ladder_rungs(stat, ours, (g.get("players") or {}).get(name, {}).get(stat, []), margin, actual)
                players.append(
                    {
                        "name": known.get(pid, name) if pid else name,
                        "player_id": pid,
                        "team": (mine[0] if mine else (got or {}).get("pro_team_id")),
                        "stat": stat,
                        "played": None if not final or pid is None else got is not None,
                        **judge(ours, stats[stat], actual, BANDS[stat]),
                        "ladder": [[r["n"], r["book"]] for r in steps],
                        "rungs": steps if scored else [],
                        "best_value": max((r["value"] for r in steps if r["value"] is not None), default=None)
                        if scored
                        else None,
                        "rung_closer": rung_closer(steps) if scored else None,
                    }
                )

        # Our estimates for the players the book has no bet on yet (Can 2026-11-22: its player bets
        # mostly come after 19:00 Turkey time, so until then the night is our numbers alone).
        priced = {ids[n] for n in read["players"] if n in ids}
        estimates = []
        for pid_s, (team, *nums) in ours_lines.items():
            pid = int(pid_s)
            if team not in teams or pid in priced or not known.get(pid) or nums[OURS_AT["minutes"]] < MIN_MINUTES:
                continue
            got = box.get(pid)
            estimates.append(
                {
                    "name": known[pid],
                    "player_id": pid,
                    "team": team,
                    **{stat: nums[OURS_AT[stat]] for stat in ("minutes", *PLAYER_STATS)},
                    "played": None if not final else got is not None,
                    "actual": {stat: float(got[DAY_STAT[stat]]) for stat in PLAYER_STATS} if got else None,
                }
            )
        estimates.sort(key=lambda e: (-e["pts"], e["name"]))

        # The game: winner, margin, total, team points.
        markets = []
        ours_call = g.get("ours")
        chance_home = read["home_chance"]
        if ours_call:
            away_o, home_o, sure = ours_call
            pick_home = home_o > away_o
            pick_other = ("home", "away") if pick_home else ("away", "home")
            ours_home_chance = sure if pick_home else 1 - sure
            won_home = None if a_pts is None else h_pts > a_pts
            markets.append(
                {
                    "market": "winner",
                    "pick": g["home"] if pick_home else g["away"],
                    "book_pick": None if chance_home is None else (g["home"] if chance_home >= 0.5 else g["away"]),
                    "ours": round(sure, 3),
                    "book": None
                    if chance_home is None
                    else round(chance_home if pick_home else 1 - chance_home, 3),  # iddaa's chance for our pick
                    "gap": None
                    if chance_home is None
                    else round((ours_home_chance - chance_home) * (1 if pick_home else -1), 3),
                    "actual": None,
                    "won": None if won_home is None else (g["home"] if won_home else g["away"]),
                    "ours_hit": None if won_home is None else won_home == pick_home,
                    "book_hit": None if won_home is None or chance_home is None else won_home == (chance_home >= 0.5),
                    # Nearer: whoever gave the team that won the higher chance (level within 1 point).
                    "closer": None
                    if won_home is None or chance_home is None
                    else _winner_closer(ours_home_chance, chance_home, won_home),
                    "odds": _odds_of(g.get("win"), pick_other, offset=0),
                }
            )
            # Margin, from our pick's side: + he wins by that much.
            sign = 1 if pick_home else -1
            book_margin = None if read["margin"] is None else -read["margin"] * sign
            actual_margin = None if a_pts is None else (h_pts - a_pts) * sign
            markets.append(
                {
                    "market": "margin",
                    "pick": g["home"] if pick_home else g["away"],
                    **judge(float((home_o - away_o) * sign), book_margin, actual_margin, BANDS["margin"]),
                    "odds": _odds_of(g.get("margin"), pick_other, offset=1),
                }
            )
            markets.append(
                {
                    "market": "total",
                    **judge(
                        float(away_o + home_o),
                        read["total"],
                        None if a_pts is None else float(a_pts + h_pts),
                        BANDS["total"],
                    ),
                    "odds": _odds_of(g.get("total"), ("under", "over"), offset=1),
                }
            )
            for side, ours_pts, book_pts, got_pts in (
                ("away", away_o, read["away_pts"], a_pts),
                ("home", home_o, read["home_pts"], h_pts),
            ):
                markets.append(
                    {
                        "market": "team_pts",
                        "pick": g[side],
                        **judge(
                            float(ours_pts), book_pts, None if got_pts is None else float(got_pts), BANDS["team_pts"]
                        ),
                        "odds": _odds_of(g.get(f"{side}_pts"), ("under", "over"), offset=1),
                    }
                )
        out_games.append(
            {
                "game_id": int(gid),
                "away": g["away"],
                "home": g["home"],
                "tip": g["tip"],
                "checked": g["checked"],
                "source": g.get("source"),
                "status": played.get("status", "scheduled"),
                "away_score": a_pts,
                "home_score": h_pts,
                "markets": markets,
                "players": players,
                "estimates": estimates,
            }
        )
    return {"games": out_games, "tally": tally(out_games)}


def _odds_of(saved: list | None, sides: tuple[str, str], offset: int) -> list[dict]:
    """The book's two prices for a game market, as saved (app/iddaa.py): win is [home, away];
    margin [line, home, away]; totals [line, under, over]. sides names them in the row's own frame
    ("pick"/"other" from our pick's side, or "under"/"over"); prices iddaa doesn't offer are left out."""
    if not saved:
        return []
    if sides[0] in ("home", "away"):  # winner and margin: our pick first
        prices = {"home": saved[offset], "away": saved[offset + 1]}
        named = (("pick", prices[sides[0]]), ("other", prices[sides[1]]))
    else:
        named = ((sides[0], saved[offset]), (sides[1], saved[offset + 1]))
    # Only prices over MIN_ODD (Can 2026-11-21: at or under it the book is closing a near-certainty).
    return [{"side": side, "odd": float(odd)} for side, odd in named if odd and odd > iddaa.MIN_ODD]


def ladder_rungs(
    stat: str, ours: float | None, ladder: Iterable[tuple[int, float]], margin: float, actual: float | None
) -> list[dict]:
    """The book's "n+" rungs (odds over MIN_ODD) with both chances side by side.

    book: the book's chance, its game margin taken out (as the ladder line is read); ours: our
    chance from our line (analysis/rungs.py, five replayed seasons); value: what one unit returns on
    average at the book's odds if our chance is right (+ = the price pays more than our chance says
    it should); hit: did he get there (None until a result, or when he didn't play)."""
    out = []
    for n, odd in sorted(ladder):
        if not iddaa.priced(odd):
            continue
        p = rung_chances.chance(stat, ours, n)
        out.append(
            {
                "n": n,
                "odd": odd,
                "book": round(min(1.0, 1 / (odd * margin)), 3),
                "ours": p,
                "value": rung_chances.value(p, odd),
                "hit": None if actual is None else actual >= n,
            }
        )
    return out


def brier(steps: list[dict], side: str) -> float | None:
    """Mean squared miss of one side's chances over the settled rungs (0 is perfect)."""
    got = [(r[side] - (1.0 if r["hit"] else 0.0)) ** 2 for r in steps if r["hit"] is not None and r[side] is not None]
    return sum(got) / len(got) if got else None


def rung_closer(steps: list[dict]) -> str | None:
    """Whose chances read the result better, rung by rung (lower Brier score). Fairer than comparing
    one number with the ladder's middle: the book sells overs only and sets its rungs high."""
    a, b = brier(steps, "ours"), brier(steps, "book")
    if a is None or b is None:
        return None
    return "tie" if abs(a - b) <= RUNG_LEVEL else ("ours" if a < b else "book")


def _winner_closer(ours_home: float, book_home: float, won_home: bool) -> str:
    ours, book = (ours_home, book_home) if won_home else (1 - ours_home, 1 - book_home)
    return "tie" if abs(ours - book) <= 0.01 else ("ours" if ours > book else "book")


def _mean_miss(rows: list[dict], side: str) -> float | None:
    misses = [abs(r["actual"] - r[side]) for r in rows if r.get("actual") is not None and r.get(side) is not None]
    return round(sum(misses) / len(misses), 2) if misses else None


def tally(games: list[dict]) -> list[dict]:
    """Per stat and market: how many rows, the median gap, how many have ours above iddaa, and once
    results are in, how many each side got right and who was closer how often."""
    rows: dict[str, list[dict]] = {}
    for g in games:
        for p in g["players"]:
            rows.setdefault(p["stat"], []).append(p)
        for m in g["markets"]:
            rows.setdefault(m["market"], []).append(m)
    order = [*PLAYER_STATS, "winner", "margin", "total", "team_pts"]
    out = []
    for key in order:
        rs = rows.get(key)
        if not rs:
            continue
        gaps = [r["gap"] for r in rs if r["gap"] is not None]
        judged = [r for r in rs if r["ours_hit"] is not None or r["book_hit"] is not None]
        steps = [x for r in rs for x in r.get("rungs", ())]
        picks = [x for x in steps if x["value"] is not None and x["value"] > 0 and x["hit"] is not None]
        ours_b, book_b = brier(steps, "ours"), brier(steps, "book")
        out.append(
            {
                "stat": key,
                "lines": len(rs),
                "compared": len(gaps),
                "median_gap": round(statistics.median(gaps), 2) if gaps else None,
                "ours_above": sum(1 for x in gaps if x > 0),
                "judged": len(judged),
                "ours_hits": sum(1 for r in judged if r["ours_hit"]),
                "book_hits": sum(1 for r in judged if r["book_hit"]),
                "ours_closer": sum(1 for r in judged if r["closer"] == "ours"),
                "book_closer": sum(1 for r in judged if r["closer"] == "book"),
                "level": sum(1 for r in judged if r["closer"] == "tie"),
                "ours_miss": _mean_miss(judged, "ours"),
                "book_miss": _mean_miss(judged, "book"),
                "rungs": sum(1 for x in steps if x["hit"] is not None),
                "ours_rung_score": None if ours_b is None else round(ours_b, 4),
                "book_rung_score": None if book_b is None else round(book_b, 4),
                "ours_rung_closer": sum(1 for r in rs if r.get("rung_closer") == "ours"),
                "book_rung_closer": sum(1 for r in rs if r.get("rung_closer") == "book"),
                "value_picks": len(picks),
                "value_hits": sum(1 for x in picks if x["hit"]),
                "value_return": round(sum(x["odd"] - 1 if x["hit"] else -1.0 for x in picks), 2),
            }
        )
    return out


# The running record: odds bands for the value picks, chance buckets for calibration.
ODDS_BANDS = ((iddaa.MIN_ODD, 2.0), (2.0, 4.0), (4.0, 10.0), (10.0, None))
CHANCE_BUCKETS = ((0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01))


def record(nights: list[dict]) -> dict:
    """Many compared nights (compare_day outputs) as one record of the settled player rungs.

    picks: per stat and odds band, the rungs with value over 0 by our chance, how many he reached
    and the net units of one unit on each at the book's odds (+ = they paid). calibration: per
    side, chance buckets with what was said on average and how often it happened. score: both
    sides' Brier scores over the same rungs (lower is better). series: one line per night
    (night_line), oldest first, for a running balance."""
    steps = [
        (p["stat"], r)
        for night in nights
        for g in night["games"]
        for p in g["players"]
        for r in p.get("rungs", ())
        if r["hit"] is not None and r["ours"] is not None
    ]
    picks = []
    for stat in (*PLAYER_STATS, "all"):
        for lo, hi in ODDS_BANDS:
            got = [
                r
                for s, r in steps
                if (stat == "all" or s == stat) and r["value"] > 0 and r["odd"] >= lo and (hi is None or r["odd"] < hi)
            ]
            if not got:
                continue
            picks.append(
                {
                    "stat": stat,
                    "odds_from": lo,
                    "odds_to": hi,
                    "picks": len(got),
                    "hits": sum(1 for r in got if r["hit"]),
                    "said": round(sum(r["ours"] for r in got) / len(got), 3),
                    "net": round(sum(r["odd"] - 1 if r["hit"] else -1.0 for r in got), 2),
                }
            )
    calibration = []
    for side in ("ours", "book"):
        for lo, hi in CHANCE_BUCKETS:
            got = [r for _, r in steps if lo <= r[side] < hi]
            if got:
                calibration.append(
                    {
                        "side": side,
                        "chance_from": lo,
                        "chance_to": min(hi, 1.0),
                        "rungs": len(got),
                        "said": round(sum(r[side] for r in got) / len(got), 3),
                        "happened": round(sum(1 for r in got if r["hit"]) / len(got), 3),
                    }
                )
    rows = [r for _, r in steps]
    ours_b, book_b = brier(rows, "ours"), brier(rows, "book")
    return {
        "series": [night_line(n) for n in nights],
        "nights": len(nights),
        "rungs": len(rows),
        "ours_score": None if ours_b is None else round(ours_b, 4),
        "book_score": None if book_b is None else round(book_b, 4),
        "picks": picks,
        "calibration": calibration,
    }


def night_line(night: dict) -> dict:
    """One night of the record: its settled value picks, how many landed, net units at the book's
    odds, what our chances marked them at (sum of value), how many players they spread over and the
    odds of the biggest one that landed. settled: the night has results (False before tip-off)."""
    picks = [
        (p["name"], r)
        for g in night["games"]
        for p in g["players"]
        for r in p.get("rungs", ())
        if r["hit"] is not None and r["value"] is not None and r["value"] > 0
    ]
    landed = [r["odd"] for _, r in picks if r["hit"]]
    return {
        "day": night.get("day"),
        "settled": any(g.get("status") == "final" for g in night["games"]),
        "picks": len(picks),
        "hits": len(landed),
        "net": round(sum(r["odd"] - 1 if r["hit"] else -1.0 for _, r in picks), 2),
        "marked": round(sum(r["value"] for _, r in picks), 2),
        "players": len({name for name, _ in picks}),
        "top_hit": max(landed, default=None),
    }

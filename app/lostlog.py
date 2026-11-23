"""Why a settled pick lost, and which causes keep coming back (Can 2026-11-22: "these would be
golden if we can understand why we couldn't estimate it correctly"; "each iteration would lead us to
better estimations").

Every settled row on Our Estimates gets one reason: the biggest piece of the miss (the result less our
number) that points the way the miss went. The pieces:

- a game's points, per team (the same split as the Nov 21 replay, 5,225 games of 2021-22..2025-26):
    pace         possessions over the called pace, at our call's points a possession
    three        threes made over what the team was expected to hit on its attempts (its own and the
                 opponent's 3P% kept only as far as the call keeps them, gamescore's luck weights)
    free_throws  free throws made over the team's usual rate on its attempts
    late_news    how far our call moves once everyone ruled out at tip-off counts (app/pregame.py)
    other        the rest: twos, turnovers, rebounds; told by the player furthest from his line
  Over five seasons threes carry 35-39% of a total's miss and 50-63% of a margin's, pace 25-31% of
  a total's (about none of a margin's), free throws 2-3%, the rest 32-38%.
- a player's number: minutes (his minutes over our line's, at our line's rate) and the rest (shots
  taken, shot-making, rebounds or assists in the same minutes).

Each reason carries a cause: luck (threes, free throws, shot-making), pace, minutes, late_news or
other. The log keeps the bets the page takes (one per row: iddaa_compare.ts baremOf, mirrored by
``bet_of``), settles them, and counts the causes of the lost ones over the last nights. A cause that
is not luck and keeps coming back is a candidate for the signal search, tested blind on five seasons
before anything changes (Can's rule: a signal earns weight only by beating the current estimate in
every season). COMPARISON ONLY: nothing here feeds the model.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable

from app import iddaa
from app.analysis import gamescore
from app.analysis.gamebox import Box

LUCK, PACE, MINUTES, LATE, OTHER = "luck", "pace", "minutes", "late_news", "other"
CAUSES = (LUCK, PACE, MINUTES, LATE, OTHER)

# How far a game result lands from our call, one standard deviation, for a game bet's chance (the
# page's GAME_SPREAD: margin 13.6 over 5,225 games; total and team points from the call's misses).
SPREAD = {"margin": 13.6, "total": 17.5, "team_pts": 10.9}
LEAN = 0.05  # a game line we lean to by less than this takes no bet (the page's rule)
LATE_MIN = 1.0  # late news under a point is rounding: our saved call is whole points
BLOWOUT = 20  # a final margin this big: starters sit the end
STANDOUT = 6.0  # points off his line that make a player the story of an "other" miss
# What a cause has to reach to be named a repeat: this many lost bets of one kind over the nights
# read, and this share of that kind's losses.
REPEAT_MIN = 8
REPEAT_SHARE = 0.3
# A game bet's chance leans this much on the book's own price (Can 2026-11-23, "Blend: 90% book, 10%
# us"; the page's BOOK_WEIGHT). Book moneylines of 2021-22..2023-24 (3,346 games): when we disagree,
# the result lands on the book's number, and a blend fitted with each season left out puts 0.86-1.00
# on the book. Can: keep it "for now", but the engine has to earn its own edge, so every lost game
# bet where the book was closer is listed (``gems``) with what we missed, and ``book_check`` refits
# the weight on the settled nights: as our own chance gets better it should come down.
BOOK_WEIGHT = 0.9
BOOK_CUT = 1.235  # iddaa's usual cut on a two-way game line, for a side whose other price is missing
STAT_WORD = {
    "pts": "points",
    "reb": "rebounds",
    "ast": "assists",
    "stl": "steals",
    "blk": "blocks",
    "three_pm": "threes",
}
# The day body's player fields (app/scores.py) and the line log's (app/linelog.py) for each stat.
DAY_STAT = {"pts": "pts", "reb": "reb", "ast": "ast", "stl": "stl", "blk": "blk", "three_pm": "tpm"}


def _normal(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def fair_chance(odd: float, against: float | None) -> float:
    """The chance a book price reads as with its cut taken out: from the line's two prices, or its
    usual cut when only one is there (the page's fairChance)."""
    p = 1 / odd
    return p / (p + 1 / against) if against and against > 1 else min(p / BOOK_CUT, 0.99)


def _logit(p: float) -> float:
    q = min(max(p, 0.001), 0.999)
    return math.log(q / (1 - q))


def blend_chance(ours: float, fair: float | None, weight: float = BOOK_WEIGHT) -> float:
    """A game bet's chance: ours blended with the book's own in log-odds (the page's blendChance)."""
    if fair is None:
        return ours
    return 1 / (1 + math.exp(-((1 - weight) * _logit(ours) + weight * _logit(fair))))


def _num(v: float) -> str:
    return f"{v:g}"


# ---------------------------------------------------------------- game pieces


def team_pieces(
    box: Box, day: int, away: int, home: int, ours: tuple[float, float], out: set[int]
) -> dict[int, dict] | None:
    """Each team's miss against our saved call (``ours``: away, home) split into pieces, from the
    season's box (its team games: what happened; its scores before ``day``: what we expected).
    None when the game isn't in the box yet."""
    played = {g.team: g for g in box.team_games if g.day == day and g.team in (away, home)}
    if away not in played or home not in played or played[away].opponent != home:
        return None
    s = box.day_scores(day)
    a, h = s.team(away), s.team(home)
    lg_poss = s.rate("poss")
    lg3 = s.rate("tpm") / s.rate("tpa")
    called_poss = (a["poss"] / a["w"]) * (h["poss"] / h["w"]) / lg_poss
    poss = played[away].stats["poss"]
    with_out = None
    if out:
        ap, hp = box.call_parts(day, away, home, out)
        with_out = (sum(ap.values()), sum(hp.values()))
    out_pieces = {}
    for i, (team, me, them) in enumerate(((away, a, h), (home, h, a))):
        st = played[team].stats
        called = float(ours[i])
        own3, allowed3 = me["tpm"] / me["tpa"], them["tpm_a"] / them["tpa_a"]
        exp3 = lg3 + (1 + gamescore.OWN_THREE_LUCK) * (own3 - lg3) + (1 + gamescore.OPP_THREE_LUCK) * (allowed3 - lg3)
        ft_rate = me["ftm"] / me["fta"]
        pace = (poss - called_poss) * called / called_poss
        three = 3 * (st["tpm"] - st["tpa"] * exp3)
        ft = st["ftm"] - st["fta"] * ft_rate
        late = 0.0
        if with_out is not None and abs(with_out[i] - called) >= LATE_MIN:
            late = with_out[i] - called
        miss = st["pf"] - called
        out_pieces[team] = {
            "called": called,
            "actual": st["pf"],
            "poss": poss,
            "called_poss": called_poss,
            "tpm": st["tpm"],
            "tpa": st["tpa"],
            "exp3": exp3,
            "ftm": st["ftm"],
            "fta": st["fta"],
            "ft_rate": ft_rate,
            "pieces": {
                PACE: pace,
                "three": three,
                "free_throws": ft,
                LATE: late,
                OTHER: miss - pace - three - ft - late,
            },
        }
    return out_pieces


def _combine(sides: list[tuple[dict, float]]) -> dict[str, float]:
    """Pieces summed over teams, each with its sign (+1, or -1 for the side a margin counts against)."""
    out: dict[str, float] = {}
    for side, sign in sides:
        for k, v in side["pieces"].items():
            out[k] = out.get(k, 0.0) + sign * v
    return out


def _biggest(pieces: dict[str, float], miss: float) -> tuple[str, float] | None:
    """The piece that points the way the miss went and is biggest. None for a miss of 0."""
    if miss == 0:
        return None
    same = [(k, v) for k, v in pieces.items() if v * miss > 0]
    return max(same, key=lambda kv: abs(kv[1])) if same else None


def game_why(miss: float, sides: list[tuple[int, str, dict, float]], standouts: list[dict]) -> dict | None:
    """The reason a game row's result landed where it did against our number.

    miss: the result less ours (winner and margin: from our pick's side); sides: (team id, abbrev,
    team_pieces entry, sign) for the teams this row counts, sign -1 for the side a margin counts
    against; standouts: the game's players with our points line and what they scored (for "other")."""
    pieces = _combine([(p, sign) for _, _, p, sign in sides])
    got = _biggest(pieces, miss)
    if got is None:
        return None
    kind, points = got
    # The team whose piece of this kind did the most the miss's way, for the words.
    _, team, side, _ = max(sides, key=lambda t: t[3] * t[2]["pieces"][kind] * (1 if miss > 0 else -1))
    if kind == PACE:
        d = side["poss"] - side["called_poss"]
        text = (
            f"Game ran {abs(d):.0f} possessions {'faster' if d > 0 else 'slower'} than we called "
            f"({side['poss']:.0f} vs {side['called_poss']:.0f})"
        )
        cause = PACE
    elif kind == "three":
        text = f"{team} made {side['tpm']:.0f} of {side['tpa']:.0f} threes (we expected {side['exp3'] * 100:.0f}%)"
        cause = LUCK
    elif kind == "free_throws":
        text = f"{team} made {side['ftm']:.0f} of {side['fta']:.0f} free throws (usual {side['ft_rate'] * 100:.0f}%)"
        cause = LUCK
    elif kind == LATE:
        text = f"Injury news after our call: counting who was ruled out moves {team} {side['pieces'][LATE]:+.0f}"
        cause = LATE
    else:
        signs = {tid: sign for tid, _, _, sign in sides}
        abbrev_of = {tid: ab for tid, ab, _, _ in sides}
        best = None
        for s in standouts:
            sign = signs.get(s["team"])
            if sign is None:
                continue
            swing = sign * (s["pts"] - s["line"])
            if swing * miss > 0 and (best is None or abs(swing) > abs(best[1])):
                best = (s, swing)
        if best is not None and abs(best[1]) >= STANDOUT:
            s = best[0]
            who = s["name"] if len(sides) == 1 else f"{s['name']} ({abbrev_of.get(s['team'], '')})"
            text = f"{who} scored {s['pts']:.0f} (our line {s['line']:.1f})"
        else:
            text = f"{team} scored {side['pieces'][OTHER]:+.0f} beyond shooting and pace (twos, turnovers, boards)"
        cause = OTHER
    return {"cause": cause, "piece": kind, "points": round(points, 1), "text": text}


# ---------------------------------------------------------------- player pieces


def player_why(stat: str, ours: float, actual: float, line: list[float], got: dict, margin: int | None) -> dict | None:
    """The reason a player's number landed where it did against our line.

    line: the line log's numbers for him (linelog.STATS order: minutes first, fga at 9); got: his
    row in the day body; margin: his game's final margin (blowouts sit starters)."""
    miss = actual - ours
    m0, m1 = line[0], float(got.get("minutes") or 0)
    if miss == 0 or m0 <= 0:
        return None
    minutes = (m1 - m0) * ours / m0
    rest = miss - minutes
    word = STAT_WORD[stat]
    if minutes * miss > 0 and abs(minutes) >= abs(rest):
        blow = margin is not None and margin >= BLOWOUT
        text = f"Played {m1:.0f} minutes (our line {m0:.0f})" + (f" in a {margin}-point game" if blow else "")
        return {"cause": MINUTES, "piece": "minutes", "points": round(minutes, 1), "text": text}
    if stat == "pts":
        fga, fgm = float(got.get("fga") or 0), float(got.get("fgm") or 0)
        want = line[9] * m1 / m0 if len(line) > 9 and line[9] else None
        if want and miss < 0 and fga < 0.8 * want:
            return {
                "cause": OTHER,
                "piece": "shots",
                "points": round(rest, 1),
                "text": f"Took {fga:.0f} shots in {m1:.0f} minutes (about {want:.0f} expected)",
            }
        return {"cause": LUCK, "piece": "shooting", "points": round(rest, 1), "text": f"Shot {fgm:.0f} of {fga:.0f}"}
    if stat == "three_pm":
        return {
            "cause": LUCK,
            "piece": "shooting",
            "points": round(rest, 1),
            "text": f"Made {actual:.0f} threes in {m1:.0f} minutes (our line {ours:.1f})",
        }
    return {
        "cause": OTHER,
        "piece": "rate",
        "points": round(rest, 1),
        "text": f"{actual:.0f} {word} in {m1:.0f} minutes (our line {ours:.1f})",
    }


# ---------------------------------------------------------------- the night


def annotate(
    games: list[dict],
    box: Box | None,
    day: int,
    out: set[int],
    lines: dict | None,
    played: dict | None,
    abbrev: dict[int, str],
) -> None:
    """Writes a ``why`` into every settled row of ``games`` (bookcompare.compare_day's games), in
    place: None where the row is not final or can't be explained."""
    ours_lines = (lines or {}).get("lines", {})
    by_game = {g["id"]: g for g in (played or {}).get("games", [])}
    for g in games:
        final = g.get("status") == "final" and g.get("away_score") is not None
        day_game = by_game.get(g["game_id"]) or {}
        box_rows = {ln["player_id"]: ln for ln in day_game.get("lines", [])}
        margin = abs(g["home_score"] - g["away_score"]) if final else None
        standouts = []
        for pid, row in box_rows.items():
            mine = ours_lines.get(str(pid))
            if mine:
                standouts.append({"name": row["name"], "team": mine[0], "pts": row["pts"], "line": mine[2]})
        sides = None
        team_rows = {m["pick"]: m["ours"] for m in g["markets"] if m["market"] == "team_pts"}
        a, h = g["away"], g["home"]
        if final and box is not None and a in team_rows and h in team_rows:
            sides = team_pieces(box, day, a, h, (team_rows[a], team_rows[h]), out)
        for m in g["markets"]:
            m["why"] = None
            if sides is None:
                continue
            if m["market"] == "total":
                miss = (g["away_score"] + g["home_score"]) - m["ours"]
                counted = [(a, 1.0), (h, 1.0)]
            elif m["market"] == "team_pts":
                t = m["pick"]
                miss = (g["away_score"] if t == a else g["home_score"]) - m["ours"]
                counted = [(t, 1.0)]
            else:  # winner, margin: from our pick's side
                pick = m["pick"]
                other = h if pick == a else a
                got = (g["away_score"] - g["home_score"]) * (1 if pick == a else -1)
                miss = got - (sides[pick]["called"] - sides[other]["called"])
                counted = [(pick, 1.0), (other, -1.0)]
            m["why"] = game_why(miss, [(t, abbrev.get(t, ""), sides[t], sign) for t, sign in counted], standouts)
        for p in g["players"]:
            p["why"] = None
            pid = p.get("player_id")
            if not final or pid is None or p.get("ours") is None or p.get("actual") is None:
                continue
            row, mine = box_rows.get(pid), ours_lines.get(str(pid))
            if row is None or not mine:
                continue
            p["why"] = player_why(p["stat"], p["ours"], p["actual"], mine[1:], row, margin)


def bet_of(row: dict, kind: str) -> dict | None:
    """The one bet the page takes on a row (src/lib/iddaaCompare.ts baremOf), settled once there is a
    result: its words, the book's odds, our chance and won (True/False, None for a push or no result).
    None when the row takes no bet."""
    if kind == "player":
        ours = row.get("ours")
        if ours is None:
            return None
        priced = [r for r in row.get("rungs") or [] if iddaa.priced(r["odd"]) and r["n"] <= ours]
        if not priced:
            return None
        top = max(priced, key=lambda r: r["n"])
        won = None if row.get("played") is not True or row.get("actual") is None else row["actual"] >= top["n"]
        return {"bet": f"{top['n']}+", "odd": top["odd"], "chance": top["ours"], "won": won}
    odds = {o["side"]: o["odd"] for o in row.get("odds") or []}
    if row["market"] == "winner":
        odd = odds.get("pick")
        if odd is None or not iddaa.priced(odd) or row.get("ours") is None:
            return None
        fair = fair_chance(odd, odds.get("other"))
        return _game_bet("to win", None, odd, row["ours"], fair, row.get("ours_hit"))
    ours, book = row.get("ours"), row.get("book")
    if ours is None or book is None or abs(ours - book) < LEAN:
        return None
    over = ours > book
    if row["market"] == "margin":
        side, facing = ("pick", "other") if over else ("other", "pick")
    else:
        side, facing = ("over", "under") if over else ("under", "over")
    odd = odds.get(side)
    if odd is None or not iddaa.priced(odd):
        return None
    alone = _normal(abs(ours - book) / SPREAD[row["market"]])
    actual = row.get("actual")
    won = None if actual is None or actual == book else (actual > book) == over
    return _game_bet("over" if over else "under", book, odd, alone, fair_chance(odd, odds.get(facing)), won)


def _game_bet(bet: str, line: float | None, odd: float, alone: float, fair: float, won: bool | None) -> dict:
    out = {"bet": bet, "odd": odd, "chance": round(blend_chance(alone, fair), 3), "won": won}
    out.update({"alone": round(alone, 3), "fair": round(fair, 3)})
    if line is not None:
        out["line"] = line
    return out


def _signed(v: float) -> str:
    return f"+{_num(v)}" if v > 0 else _num(v) if v < 0 else "0"


def _words(row: dict, b: dict, game: dict, abbrev: dict[int, str]) -> str:
    """A game bet as the page writes it: "SA to win", "NO +19.5", "Over 227.5", "MEM over 111.5"."""
    m = row["market"]
    if m == "winner":
        return f"{abbrev.get(row['pick'], '')} to win"
    over = b["bet"] == "over"
    if m == "margin":
        pick = row["pick"]
        other = game["home"] if pick == game["away"] else game["away"]
        return (
            f"{abbrev.get(pick, '')} {_signed(-b['line'])}" if over else f"{abbrev.get(other, '')} {_signed(b['line'])}"
        )
    if m == "team_pts":
        return f"{abbrev.get(row['pick'], '')} {'over' if over else 'under'} {_num(b['line'])}"
    return f"{'Over' if over else 'Under'} {_num(b['line'])}"


def night(games: list[dict], day: int, abbrev: dict[int, str]) -> dict:
    """One night's settled bets (after ``annotate``): how many, won, our expected wins, and every lost
    bet with its reason."""
    settled, lost, game_bets = [], [], []
    for g in games:
        label = f"{abbrev.get(g['away'], '')} @ {abbrev.get(g['home'], '')}"
        rows = [("game", m) for m in g["markets"]] + [("player", p) for p in g["players"]]
        for kind, row in rows:
            b = bet_of(row, kind)
            if b is None or b["won"] is None:
                continue
            settled.append(b)
            if kind == "game":
                game_bets.append([b["alone"], b["fair"], 1 if b["won"] else 0])
            if b["won"]:
                continue
            why = row.get("why") or {}
            if kind == "player":
                what = f"{row['name']} {b['bet']} {STAT_WORD[row['stat']]}"
                group = row["stat"]
            else:
                what, group = _words(row, b, g, abbrev), row["market"]
            lost.append(
                {
                    "game": label,
                    "group": group,
                    "bet": what,
                    "odd": b["odd"],
                    "chance": b["chance"],
                    "ours": row.get("ours"),
                    "book": row.get("book"),
                    "actual": row.get("actual"),
                    "cause": why.get("cause"),
                    "piece": why.get("piece"),
                    "points": why.get("points"),
                    "text": why.get("text", ""),
                    "alone": b.get("alone"),
                    "fair": b.get("fair"),
                    "book_closer": _book_closer(row) if kind == "game" else None,
                }
            )
    return {
        "day": day,
        "bets": len(settled),
        "won": sum(1 for b in settled if b["won"]),
        "expected": round(sum(b["chance"] or 0 for b in settled), 1),
        "lost": lost,
        "game_bets": game_bets,
    }


def _book_closer(row: dict) -> bool | None:
    """Whether the book's number sat nearer the result than ours (a winner: the book favoured the
    other team)."""
    ours, book, actual = row.get("ours"), row.get("book"), row.get("actual")
    if row["market"] == "winner":
        return None if book is None else book < 0.5
    if ours is None or book is None or actual is None:
        return None
    return abs(actual - book) < abs(actual - ours)


def book_check(nights: Iterable[dict]) -> dict:
    """How the settled game bets read on our chance alone, the book's alone and the blend (log loss,
    lower is better), and the book weight that would have read them best: as our own chance gets
    better that weight should come down (it changes the page only after a blind test)."""
    bets = [b for n in nights for b in n.get("game_bets", [])]

    def loss(w: float) -> float:
        out = 0.0
        for alone, fair, won in bets:
            p = min(max(blend_chance(alone, fair, w), 1e-6), 1 - 1e-6)
            out -= math.log(p if won else 1 - p)
        return out / len(bets)

    if not bets:
        return {"bets": 0, "ours": None, "book": None, "blend": None, "best_weight": None}
    grid = [i / 20 for i in range(21)]
    best = min(grid, key=loss)
    return {
        "bets": len(bets),
        "ours": round(loss(0.0), 4),
        "book": round(loss(1.0), 4),
        "blend": round(loss(BOOK_WEIGHT), 4),
        "best_weight": best,
    }


def repeats(nights: Iterable[dict]) -> dict:
    """The causes of the lost bets over ``nights``: a count per cause, and per (cause, kind of bet)
    the ones that are not luck and came back REPEAT_MIN+ times at REPEAT_SHARE+ of that kind's
    losses: candidates for the signal search (each still tested blind on five seasons)."""
    lost = [x for n in nights for x in n["lost"]]
    causes = Counter(x["cause"] or "unexplained" for x in lost)
    by_group = Counter(x["group"] for x in lost)
    pairs = Counter((x["cause"], x["group"], x["piece"]) for x in lost if x["cause"] and x["cause"] != LUCK)
    flagged = []
    for (cause, group, piece), n in pairs.most_common():
        share = n / by_group[group]
        flagged.append(
            {
                "cause": cause,
                "group": group,
                "piece": piece,
                "count": n,
                "share": round(share, 2),
                "candidate": n >= REPEAT_MIN and share >= REPEAT_SHARE,
            }
        )
    # the gems (Can 2026-11-23): game bets where the book was nearer the result than we were, by what
    # we missed; a cause that keeps coming back is the engine's to fix (tested blind first)
    gems = [x for x in lost if x.get("book_closer")]
    gem_pairs = Counter((x["cause"] or "unexplained", x["group"], x["piece"]) for x in gems)
    gem_groups = Counter(x["group"] for x in gems)
    return {
        "lost": len(lost),
        "causes": [{"cause": c, "count": n} for c, n in causes.most_common()],
        "repeats": flagged,
        "gems": [
            {
                "cause": cause,
                "group": group,
                "piece": piece,
                "count": n,
                "share": round(n / gem_groups[group], 2),
                "candidate": cause != LUCK and n >= REPEAT_MIN and n / gem_groups[group] >= REPEAT_SHARE,
            }
            for (cause, group, piece), n in gem_pairs.most_common()
        ],
        "book": book_check(nights),
    }

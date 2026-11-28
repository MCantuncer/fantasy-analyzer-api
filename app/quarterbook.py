"""iddaa's quarter and first-half bets beside our quarter numbers, judged by what happened.

Can 2026-11-24: "Why we don't have quarter base analysis for the matches on Our Estimates?" and "we
need engine, not just averages. Need edges, and correct edges." Our quarters are the pre-game box's
(app/quarters.py: Our call cut into quarters by the favourite's fade and both teams' quarter shapes),
from the call kept at the book's last look. Each bet's chance is read from how far real quarters
landed from those numbers over 6,150 games (2021-22 to 2025-26, cdn.nba.com play-by-play, the
pre-game quarter fitted with the season left out; scripts/quarter_backtest.py): one standard
deviation, normal (68.5% within one, 95.4% within two). Quarter winners priced this way were
calibrated (said 52% won 52%, 57% won 60%, 64% won 63%).

Can picked (2026-11-24) "two steps": these bets are shown and kept on record first; they go into
Kupons only once the record shows they beat the price. COMPARISON ONLY: nothing here feeds the model.

iddaa market types (its event answer, saved raw in each game's "board" since 2026-11-22):
    776 quarter winner, three-way (line = the quarter; outcomes "1" home, "0" tie, "2" away)
    777 quarter total (line "quarter|points"; "Alt" under, "Üst" over)
    116 first-half total, 115 first-half handicap for the home team ("1" home, "2" away),
    746 / 747 first-half home / away team points
A market with several lines keeps its main one (its two prices closest), as the game lines do.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

from app import iddaa

# The trial (Can picked "two steps" on 2026-11-24): quarter picks are shown and recorded, kept out of
# Kupons, and judged after about two weeks on whether the value picks beat the price.
TRIAL_FROM = date(2026, 11, 24)
TRIAL_DAYS = 14
# Price bands for the record (Can 2026-11-28: "most of the correct ones are the easiest ones"):
# whether the short prices win often enough to pay. Lower bound in, upper bound out.
ODDS_BANDS = [("under 1.50", 0.0, 1.5), ("1.50-1.79", 1.5, 1.8), ("1.80-2.19", 1.8, 2.2), ("2.20+", 2.2, math.inf)]

QUARTER_TOTAL_SD = 8.4  # a quarter's points, both teams, against ours
QUARTER_MARGIN_SD = 8.2  # a quarter's margin against ours
HALF_TOTAL_SD = 12.2
HALF_MARGIN_SD = 11.1
HALF_TEAM_SD = 8.2

Q_WINNER, Q_TOTAL = 776, 777
H1_MARGIN, H1_TOTAL, H1_HOME, H1_AWAY = 115, 116, 746, 747
UNDER, OVER = "Alt", "Üst"


def _cdf(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def over_chance(ours: float, line: float, sd: float) -> float:
    """Chance the result goes over ``line`` (a whole line: past it; level is a push, left out)."""
    edge = line + 0.5 if float(line).is_integer() else line
    return 1 - _cdf((edge - ours) / sd)


def three_way(margin: float, sd: float) -> tuple[float, float, float]:
    """Home wins, tie, away wins, from our home margin: a tie is a margin within half a point of 0."""
    away = _cdf((-0.5 - margin) / sd)
    home = 1 - _cdf((0.5 - margin) / sd)
    return home, 1 - home - away, away


def _line(text: str) -> float | None:
    try:
        return float(str(text).replace(",", "."))
    except ValueError:
        return None


def _priced(outcomes: list) -> dict[str, float]:
    out = {}
    for name, odd in outcomes:
        try:
            out[str(name)] = float(odd)
        except (TypeError, ValueError):
            continue
    return out


def _main(rows: list[tuple[float, dict[str, float]]], first: str, second: str):
    """The main line of a two-way market: of its open lines, the one whose two prices are closest."""
    best = None
    for line, odds in rows:
        a, b = odds.get(first, 0.0), odds.get(second, 0.0)
        if a <= iddaa.NOT_OFFERED or b <= iddaa.NOT_OFFERED:
            continue
        if best is None or abs(a - b) < best[0]:
            best = (abs(a - b), line, a, b)
    return best[1:] if best else None


def markets(board: list | None) -> dict:
    """The quarter and first-half markets in a saved board, each at its main line:
    {("q_total", k): (line, under, over), ("q_winner", k): {"1": .., "0": .., "2": ..},
     ("h1_total", 0): (line, under, over), ("h1_margin", 0): (home line, home, away),
     ("h1_home", 0) / ("h1_away", 0): (line, under, over)}."""
    two: dict[tuple[str, int], list] = {}
    out: dict = {}
    for st, sov, outcomes in board or []:
        odds = _priced(outcomes)
        if st == Q_WINNER:
            k = _line(sov)
            if k and 1 <= k <= 4 and all(odds.get(s, 0) > iddaa.NOT_OFFERED for s in ("1", "2")):
                out[("q_winner", int(k))] = odds
        elif st == Q_TOTAL:
            q, _, pts = str(sov).partition("|")
            k, line = _line(q), _line(pts)
            if k and 1 <= k <= 4 and line is not None:
                two.setdefault(("q_total", int(k)), []).append((line, odds))
        elif st in (H1_TOTAL, H1_HOME, H1_AWAY, H1_MARGIN):
            line = _line(sov)
            key = {H1_TOTAL: "h1_total", H1_HOME: "h1_home", H1_AWAY: "h1_away", H1_MARGIN: "h1_margin"}[st]
            if line is not None:
                two.setdefault((key, 0), []).append((line, odds))
    for key, rows in two.items():
        got = _main(rows, "1", "2") if key[0] == "h1_margin" else _main(rows, UNDER, OVER)
        if got:
            out[key] = got
    return out


def _side(side: str, odd: float, ours: float, fair: float) -> dict:
    return {"side": side, "odd": odd, "ours": round(ours, 3), "book": round(fair, 3), "value": round(ours * odd - 1, 3)}


def _fair(odds: list[float]) -> list[float]:
    inv = [1 / o for o in odds]
    return [x / sum(inv) for x in inv]


def bets(
    board: list | None,
    away: int,
    home: int,
    ours: tuple[list[float], list[float]] | None,
    actual: tuple[list[float], list[float]] | None,
) -> list[dict]:
    """One game's quarter and first-half bets: iddaa's line and prices, our number and our chance on
    each side, the side we lean to, and once played the result. ours / actual: (away Q1-Q4, home Q1-Q4)."""
    if not ours:
        return []
    oa, oh = ours
    done = actual is not None and len(actual[0]) >= 4 and len(actual[1]) >= 4
    aa, ah = (actual[0][:4], actual[1][:4]) if done else ([], [])
    out = []
    for (kind, k), m in sorted(markets(board).items(), key=lambda kv: (kv[0][1] or 9, kv[0][0])):
        q = slice(k - 1, k) if k else slice(0, 2)
        if kind in ("q_total", "h1_total", "h1_home", "h1_away"):
            line, under, over = m
            if kind == "h1_home":
                number, sd, got = sum(oh[q]), HALF_TEAM_SD, sum(ah[q]) if done else None
            elif kind == "h1_away":
                number, sd, got = sum(oa[q]), HALF_TEAM_SD, sum(aa[q]) if done else None
            else:
                number = sum(oa[q]) + sum(oh[q])
                sd = QUARTER_TOTAL_SD if kind == "q_total" else HALF_TOTAL_SD
                got = sum(aa[q]) + sum(ah[q]) if done else None
            p = over_chance(number, line, sd)
            f_under, f_over = _fair([under, over])
            sides = [_side("under", under, 1 - p, f_under), _side("over", over, p, f_over)]
            lean = "over" if p >= 0.5 else "under"
            result = None if got is None else ("push" if got == line else "over" if got > line else "under")
        elif kind == "h1_margin":
            line, home_odd, away_odd = m  # home handicap: home + line beats away
            number = sum(oh[q]) - sum(oa[q])
            p = over_chance(number, -line, HALF_MARGIN_SD)  # home covers: margin past -line
            f_home, f_away = _fair([home_odd, away_odd])
            sides = [_side("home", home_odd, p, f_home), _side("away", away_odd, 1 - p, f_away)]
            lean = "home" if p >= 0.5 else "away"
            got = sum(ah[q]) - sum(aa[q]) if done else None
            result = None if got is None else ("push" if got == -line else "home" if got > -line else "away")
        else:  # q_winner, three-way
            number = oh[k - 1] - oa[k - 1]
            ph, pt, pa = three_way(number, QUARTER_MARGIN_SD)
            names = [s for s in ("1", "0", "2") if m.get(s, 0) > iddaa.NOT_OFFERED]
            fair = dict(zip(names, _fair([m[s] for s in names]), strict=True))
            chance = {"1": ph, "0": pt, "2": pa}
            label = {"1": "home", "0": "tie", "2": "away"}
            sides = [_side(label[s], m[s], chance[s], fair[s]) for s in names]
            lean = "home" if ph >= pa else "away"
            line = None
            got = ah[k - 1] - aa[k - 1] if done else None
            result = None if got is None else ("home" if got > 0 else "away" if got < 0 else "tie")
        pick = next((s for s in sides if s["side"] == lean), None)
        # Only a price over MIN_ODD is the book's number (Can 2026-11-21).
        if pick is not None and pick["odd"] <= iddaa.MIN_ODD:
            pick = None
        out.append(
            {
                "market": kind,
                "period": k,
                "line": line,
                "ours": round(number, 1),
                "sides": sides,
                "pick": None if pick is None else pick["side"],
                "edge": None if pick is None else pick["value"],
                "actual": None if got is None else round(got, 1),
                "result": result,
                "hit": None if result is None or pick is None or result == "push" else result == pick["side"],
            }
        )
    return out


def record(games: list[dict]) -> dict:
    """Every settled pick across the nights given (each game's "quarters" from bets): how many, how
    many won, our average chance and the book's for them, what one unit on each returned at the
    book's odds, and need: how often they had to win to break even at those odds (picks over the sum
    of their odds). All picks, and the edge picks alone (our chance x odds over 1). By market and by
    price band too."""

    def sums(rows: list[tuple[dict, dict]]) -> dict:
        n = len(rows)
        won = sum(1 for b, _ in rows if b["hit"])
        ret = sum((s["odd"] if b["hit"] else 0.0) - 1 for b, s in rows)
        return {
            "bets": n,
            "won": won,
            "ours": round(sum(s["ours"] for _, s in rows) / n, 3) if n else None,
            "book": round(sum(s["book"] for _, s in rows) / n, 3) if n else None,
            "returned": round(ret, 2),
            "need": round(n / sum(s["odd"] for _, s in rows), 3) if n else None,
        }

    settled = []
    for g in games:
        for b in g.get("quarters") or []:
            if b["hit"] is None or b["pick"] is None:
                continue
            side = next(s for s in b["sides"] if s["side"] == b["pick"])
            settled.append((b, side))
    edge = [(b, s) for b, s in settled if s["value"] > 0]
    kinds = sorted({b["market"] for b, _ in settled})
    return {
        "all": sums(settled),
        "edge": sums(edge),
        "by_market": {
            k: {
                "all": sums([x for x in settled if x[0]["market"] == k]),
                "edge": sums([x for x in edge if x[0]["market"] == k]),
            }
            for k in kinds
        },
        "by_odds": {
            band: {
                "all": sums([x for x in settled if lo <= x[1]["odd"] < hi]),
                "edge": sums([x for x in edge if lo <= x[1]["odd"] < hi]),
            }
            for band, lo, hi in ODDS_BANDS
            if any(lo <= s["odd"] < hi for _, s in settled)
        },
    }


def trial(nights: list[dict]) -> dict:
    """The trial's dates and its record night by night (oldest first), for the page's water line.
    nights: compared nights, each with its "date" (ISO, or None when the schedule didn't read) and
    its games' "quarters". Every night with a settled pick joins the series, the ones before the
    trial's first day too (the boards were saved from Nov 22, so the record counts them), and the
    start moves back to the first of them: the line, its units and the win rates beside it then
    cover the same picks (Can 2026-11-28: the line read −7.68 beside the record's 49% / 60%).
    edge_returned is what the value picks made that night."""
    series = []
    for n in nights:
        when = n.get("date")
        if not when:
            continue
        got = record(n.get("games") or [])
        if got["all"]["bets"] == 0:
            continue
        series.append(
            {
                "date": when,
                "bets": got["all"]["bets"],
                "won": got["all"]["won"],
                "returned": got["all"]["returned"],
                "edge_bets": got["edge"]["bets"],
                "edge_won": got["edge"]["won"],
                "edge_returned": got["edge"]["returned"],
            }
        )
    series.sort(key=lambda x: x["date"])
    start = min([TRIAL_FROM, *(date.fromisoformat(x["date"]) for x in series)])
    return {
        "start": start.isoformat(),
        "review": (TRIAL_FROM + timedelta(days=TRIAL_DAYS)).isoformat(),
        "series": series,
    }

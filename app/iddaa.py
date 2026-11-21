"""iddaa's NBA lines as they stood at tip-off, saved night by night, to set beside ours later.

Can 2026-11-21: an admin page that compares iddaa (the Turkish sportsbook) with our own numbers:
player points, rebounds, assists and the rest, the winner, the margin and the game total, and after
the games who landed nearer what happened. COMPARISON ONLY: nothing here feeds our model.

iddaa lists a game only until it starts, so a past night can't be read back: a watch reads iddaa's
lines in the hours before tip-off and, for each game, keeps its last look before the game starts (as
app/linelog.py does for our lines). They are read through nesine, an iddaa dealer whose bulletin is
one call for every game (Can's pick). What is saved is nearly raw (the odds), so the way a line is
read from them can change later without losing a night. A day's record:

    {"games": {espn game id: {
        "event": iddaa id, "away": team, "home": team, "tip": iso, "checked": iso,
        "win": [home odd, away odd],                      0 where iddaa doesn't offer that side
        "margin" | "total" | "home_pts" | "away_pts": [line, first odd, second odd],
        "players": {name: {stat: [[n, odd], ...]}},       "Name n+" ladders, stats as STATS
        "ours": [away, home, chance]}}}                   our call at the same look (app/main.py)

The margin line is the home team's handicap (-5.5: home by 6 or more), its odds home then away; a
total's odds are under then over. Only each market's main line (its two odds closest) is kept.
"""

from __future__ import annotations

import re
import statistics
import unicodedata
from datetime import UTC, datetime, timedelta

import httpx

from app.scores import Schedule, ScheduledGame

# nesine's full pre-match bulletin (Can 2026-11-21: "Go with nesine"). iddaa's own API
# (sportsbookv2.iddaa.com/sportsbook/events, /event/<id>) has the same games in the shape read here.
BULLETIN = "https://cdnbulten.nesine.com/api/bulten/getprebultenfull"
USER_AGENT = "Mozilla/5.0 (compatible; fantasy-analyzer/1.0)"
TIMEOUT = 20.0  # the bulletin is about 1.6 MB
NBA = 138  # iddaa's (and nesine's) league id for the NBA
AHEAD = timedelta(hours=6)  # games starting this soon are read
TIP_SLACK = timedelta(hours=3)  # an iddaa game and an ESPN game this far apart in start time can't match

# iddaa market sub-types. Outcome "1" is the home team (iddaa's "hn").
WINNER = 2  # overtime included
MAIN_MARKETS = {113: "margin", 114: "total", 619: "home_pts", 620: "away_pts"}
STATS = {831: "pts", 832: "reb", 833: "ast", 834: "stl", 835: "blk", 836: "three_pm"}
NOT_OFFERED = 1.0  # iddaa shows odd 1.00 for a closed outcome
DEFAULT_MARGIN = 1.21  # iddaa's usual two-way overround, when a game has no two-way market to measure
RUNG = re.compile(r"^(?P<name>.+?)\s+(?P<n>\d+)\+$")


def fold(text: str) -> str:
    """Lower case, accents off, punctuation as spaces, "la" spelled out: for matching names."""
    plain = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    words = re.sub(r"[^a-z0-9]+", " ", plain).split()
    return " ".join("los angeles" if w == "la" else w for w in words)


def team_matches(iddaa_name: str, espn_name: str) -> bool:
    """iddaa names a team by its city ("Philadelphia") or city and name ("LA Clippers"); ESPN's is
    "Location Name". A match when every word of iddaa's is in ESPN's."""
    ours = set(fold(espn_name).split())
    return bool(iddaa_name.strip()) and set(fold(iddaa_name).split()) <= ours


def match_game(event: dict, schedule: Schedule) -> ScheduledGame | None:
    """The ESPN game an iddaa event is: same two teams, home and away, starting within TIP_SLACK.
    None when no game, or more than one, fits."""
    try:
        tip = datetime.fromtimestamp(int(event["d"]), UTC)
    except (KeyError, TypeError, ValueError):
        return None
    home, away = str(event.get("hn") or ""), str(event.get("an") or "")
    fits = []
    for games in schedule.games.values():
        for g in games:
            if abs(g.start - tip) > TIP_SLACK:
                continue
            h = schedule.teams.get(g.home, ("", ""))[1]
            a = schedule.teams.get(g.away, ("", ""))[1]
            if team_matches(home, h) and team_matches(away, a):
                fits.append(g)
    return fits[0] if len(fits) == 1 else None


def _odds(market: dict) -> dict[str, float]:
    out = {}
    for o in market.get("o") or []:
        try:
            out[str(o["n"])] = float(o["odd"])
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _main_line(markets: list[dict], first: str, second: str) -> list[float] | None:
    """[line, first odd, second odd] of the market's main line: of its open lines, the one whose two
    odds are closest."""
    best = None
    for m in markets:
        odds = _odds(m)
        a, b = odds.get(first, 0.0), odds.get(second, 0.0)
        if a <= NOT_OFFERED or b <= NOT_OFFERED:
            continue
        try:
            line = float(str(m.get("sov")).replace(",", "."))
        except ValueError:
            continue
        if best is None or abs(a - b) < best[0]:
            best = (abs(a - b), [line, a, b])
    return best[1] if best else None


def parse_event(data: dict) -> dict:
    """One game's markets from iddaa's event answer, as saved (see the module's record)."""
    by_type: dict[int, list[dict]] = {}
    for m in data.get("m") or []:
        by_type.setdefault(m.get("st"), []).append(m)
    out: dict = {}
    winner = by_type.get(WINNER)
    if winner:
        odds = _odds(winner[0])
        home, away = odds.get("1", 0.0), odds.get("2", 0.0)
        if home > NOT_OFFERED or away > NOT_OFFERED:
            out["win"] = [home if home > NOT_OFFERED else 0.0, away if away > NOT_OFFERED else 0.0]
    for st, key in MAIN_MARKETS.items():
        first, second = ("1", "2") if key == "margin" else ("Alt", "Üst")
        got = _main_line(by_type.get(st, []), first, second)
        if got:
            out[key] = got
    players: dict[str, dict[str, list[list[float]]]] = {}
    for st, stat in STATS.items():
        for m in by_type.get(st, []):
            for o in m.get("o") or []:
                hit = RUNG.match(str(o.get("n") or "").strip())
                try:
                    odd = float(o.get("odd"))
                except (TypeError, ValueError):
                    continue
                if not hit or odd <= NOT_OFFERED:
                    continue
                players.setdefault(hit["name"], {}).setdefault(stat, []).append([int(hit["n"]), odd])
    for stats in players.values():
        for rungs in stats.values():
            rungs.sort()
    out["players"] = players
    return out


def overround(game: dict) -> float:
    """iddaa's margin on this game: how far its two-way markets' implied chances add up past 1."""
    sums = [1 / v[1] + 1 / v[2] for k in MAIN_MARKETS.values() if (v := game.get(k))]
    return statistics.median(sums) if sums else DEFAULT_MARGIN


def ladder_line(rungs: list[list[float]], margin: float) -> float | None:
    """The line an "n+" ladder implies: the number he goes past half the time, margin taken out.
    Rung n reads as a chance he ends above n - 0.5; between two rungs the line is drawn straight.
    None when the ladder doesn't cross a half (every rung above it or below it)."""
    points = sorted((n - 0.5, min(1.0, 1 / (odd * margin))) for n, odd in rungs if odd > NOT_OFFERED)
    for (t0, p0), (t1, p1) in zip(points, points[1:], strict=False):
        if p0 >= 0.5 > p1:
            return round(t0 + (p0 - 0.5) / (p0 - p1) * (t1 - t0), 1)
    return None


def home_chance(game: dict) -> float | None:
    """iddaa's chance the home team wins, margin taken out. With one side closed, from the other."""
    win = game.get("win")
    if not win:
        return None
    home, away = win
    if home and away:
        return round((1 / home) / (1 / home + 1 / away), 3)
    m = overround(game)
    return round(1 / (home * m), 3) if home else round(1 - 1 / (away * m), 3)


def readout(game: dict) -> dict:
    """A saved game as lines: home win chance, home handicap, totals and each player's line per stat."""
    m = overround(game)
    players = {}
    for name, stats in (game.get("players") or {}).items():
        lines = {stat: ladder_line(rungs, m) for stat, rungs in stats.items()}
        players[name] = {k: v for k, v in lines.items() if v is not None}
    return {
        "home_chance": home_chance(game),
        **{k: (game[k][0] if k in game else None) for k in MAIN_MARKETS.values()},
        "players": {n: v for n, v in players.items() if v},
    }


def from_nesine(event: dict) -> dict:
    """A game from nesine's bulletin in iddaa's own shape (the one parse_event reads). nesine is an
    iddaa dealer and keeps iddaa's market codes; it numbers outcomes ("1" home, "2" away; for totals
    1 under, 2 over), names players "Last, First n+", and marks an open market MS 1."""
    markets = []
    for m in event.get("MA") or []:
        st = m.get("MST")
        if m.get("MS") != 1 or st not in (WINNER, *MAIN_MARKETS, *STATS):
            continue
        outcomes = []
        for o in m.get("OCA") or []:
            if st in STATS:
                hit = RUNG.match(str(o.get("ON") or "").strip())
                if not hit:
                    continue
                last, _, first = hit["name"].partition(",")
                name = f"{first.strip()} {last.strip()}".strip()
                outcomes.append({"n": f"{name} {hit['n']}+", "odd": o.get("O")})
            else:
                n = {1: "1", 2: "2"}.get(o.get("N"))
                if n and st in MAIN_MARKETS and st != 113:
                    n = {"1": "Alt", "2": "Üst"}[n]
                if n:
                    outcomes.append({"n": n, "odd": o.get("O")})
        markets.append({"st": st, "sov": str(m.get("SOV")), "o": outcomes})
    try:
        tip = int(event.get("ESD") or 0) // 1000
    except (TypeError, ValueError):
        tip = 0
    return {"i": event.get("C"), "hn": event.get("HN"), "an": event.get("AN"), "ci": NBA, "d": tip, "m": markets}


async def fetch(
    transport: httpx.AsyncBaseTransport | None = None,
) -> list[tuple[dict, dict]]:
    """iddaa's NBA games still to start, read from nesine's bulletin (one call for every sport):
    (the game, its markets), both in iddaa's shape. Raises httpx.HTTPError (or ValueError on a bad
    answer) when the bulletin can't be read."""
    headers = {"User-Agent": USER_AGENT, "Referer": "https://www.nesine.com/"}
    async with httpx.AsyncClient(transport=transport, timeout=TIMEOUT, headers=headers) as http:
        resp = await http.get(BULLETIN)
        resp.raise_for_status()
        events = ((resp.json() or {}).get("sg") or {}).get("EA") or []
    games = [from_nesine(e) for e in events if e.get("LC") == NBA and e.get("TYPE") == 2]
    return [(g, g) for g in games]


def update(
    records: dict[int, dict], schedule: Schedule, looks: list[tuple[dict, dict]], now: datetime
) -> dict[int, dict]:
    """The days' records with this look written in, for each game not yet started that tips off
    within AHEAD: its markets replace the last look's. A started game keeps what it had.
    records: day -> record (missing days start empty). Returns only the days that changed."""
    changed: dict[int, dict] = {}
    stamp = now.isoformat(timespec="seconds")
    for event, data in looks:
        game = match_game(event, schedule)
        if game is None or not (now < game.start <= now + AHEAD):
            continue
        rec = changed.get(game.day) or {"games": dict((records.get(game.day) or {}).get("games", {}))}
        rec["games"][str(game.id)] = {
            "event": int(event.get("i") or 0),
            "away": game.away,
            "home": game.home,
            "tip": game.start.isoformat(timespec="seconds"),
            "checked": stamp,
            **parse_event(data),
        }
        changed[game.day] = rec
    return changed

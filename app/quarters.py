"""Points per quarter before tip-off: Our call cut into the game's four quarters.

Each side's quarters start from its full-game call (analysis/gamescore.py, with tonight's lineup), so
pace, the opponent, players out, home court and a back-to-back are already in them. On top of that,
three pieces that each beat a plain league split in every one of five seasons (2021-22 to 2025-26,
6,150 games of NBA play-by-play, fitted with each season left out and again on earlier seasons only;
scripts/quarter_backtest.py):

  * the league's shape: a game's points fall about 25.2%, 25.0%, 25.2% and 24.0% in the four quarters
    (the rest is overtime), so the fourth runs about a point lower;
  * the favourite starts fast and fades: for each point of called margin, about 0.04 more in the first
    quarter and 0.03 fewer in the fourth, when leads empty the benches;
  * the team's own shape: the share of its points it scores in each quarter against the league's, and the
    share the opponent allows in each, this season so far with last season's (kept halfway) worth
    PRIOR_GAMES games. Teams' scoring shapes carry from one season to the next (r about 0.3), but they
    are small: about 0.3 points a quarter, 1-2 for the strongest.

Who wins a quarter was not made more predictable by any of this (57% right, against 66% for the game),
and a team's quarter is off by about 4.7 points on average.

This season's quarters come from ESPN's public scoreboard, one read per finished day, kept. When the
scoreboard can't be read, a team's shape is last season's alone.

Once a game is on (live_quarters, Can 2026-11-23): quarters played are the real ones, and each quarter
still to come is the pre-game number plus GIVE_BACK x how far the team's margin so far runs past the
margin we called, per quarter played. A team ahead of our call gives back about 6.5% of that extra
margin in each quarter left. On the same five seasons, at the end of each quarter, it lowered the
squared miss of every quarter to come in every season (0.3-0.8%, season left out and earlier seasons
only; scripts/quarter_backtest.py). Tested and not kept: the night's scoring pace (both teams'
points so far against the call; fails 2024-25), cut-offs for leads or deficits past 10 (each fails a
season) and a weight per quarter to come (fails 2022-23).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date

import httpx

from app.gameclock import SCOREBOARD, TIMEOUT, USER_AGENT

QUARTERS = 4
# Share of a team's full-game points in each quarter (five seasons; about 0.75% goes to overtime).
LEAGUE_SHARE = (0.2520, 0.2503, 0.2521, 0.2404)
# Points a quarter for each point of called margin (+ the favourite, - the underdog).
FAVOURITE = (0.0386, 0.0129, 0.0117, -0.0292)
LEVEL = (-0.035, -0.036, -0.035, -0.036)
OWN_SHAPE = 0.433  # x the call x the team's share of its points in a quarter above the league's
OPP_SHAPE = 0.213  # x the call x the opponent's share of points allowed in a quarter above the league's
PRIOR_KEEP = 0.5  # of last season's shape kept into this one
PRIOR_GAMES = 30.0  # games of it a team starts the season with
GIVE_BACK = -0.0648  # x (margin so far - called margin so far) per quarter played, in each quarter to come
QUARTER_SECONDS = 720.0

Four = tuple[float, float, float, float]

# Last season's regulation points by quarter, scored and allowed, by ESPN pro team id: the start of
# each team's shape. 2025-26 (82 games each) for the 2027 season; refresh each autumn from the
# season just ended (scripts/quarter_backtest.py prints this table).
LAST_SEASON: dict[int, dict[int, tuple[Four, Four]]] = {
    2027: {
        1: ((2505, 2471, 2392, 2310), (2417, 2394, 2425, 2253)),  # ATL
        2: ((2432, 2417, 2341, 2202), (2250, 2154, 2234, 2127)),  # BOS
        3: ((2365, 2381, 2376, 2300), (2541, 2424, 2406, 2405)),  # NO
        4: ((2360, 2273, 2496, 2376), (2511, 2439, 2600, 2385)),  # CHI
        5: ((2481, 2290, 2552, 2463), (2314, 2393, 2427, 2301)),  # CLE
        6: ((2382, 2289, 2392, 2238), (2606, 2358, 2478, 2292)),  # DAL
        7: ((2535, 2398, 2570, 2389), (2378, 2326, 2449, 2311)),  # DEN
        8: ((2439, 2513, 2370, 2296), (2333, 2245, 2224, 2145)),  # DET
        9: ((2327, 2272, 2391, 2355), (2358, 2326, 2412, 2289)),  # GS
        10: ((2354, 2344, 2408, 2247), (2233, 2187, 2207, 2282)),  # HOU
        11: ((2359, 2279, 2348, 2198), (2470, 2534, 2555, 2277)),  # IND
        12: ((2394, 2227, 2415, 2246), (2356, 2222, 2343, 2285)),  # LAC
        13: ((2487, 2348, 2394, 2302), (2459, 2374, 2385, 2171)),  # LAL
        14: ((2601, 2462, 2456, 2380), (2405, 2473, 2471, 2361)),  # MIA
        15: ((2247, 2331, 2245, 2210), (2423, 2457, 2419, 2251)),  # MIL
        16: ((2385, 2367, 2433, 2422), (2333, 2268, 2404, 2327)),  # MIN
        17: ((2171, 2186, 2172, 2125), (2474, 2393, 2338, 2263)),  # BKN
        18: ((2424, 2398, 2375, 2316), (2353, 2260, 2307, 2078)),  # NY
        19: ((2366, 2423, 2400, 2243), (2369, 2402, 2409, 2200)),  # ORL
        20: ((2489, 2451, 2242, 2235), (2458, 2395, 2449, 2143)),  # PHI
        21: ((2371, 2388, 2334, 2112), (2266, 2408, 2212, 2198)),  # PHX
        22: ((2428, 2454, 2322, 2224), (2506, 2341, 2305, 2291)),  # POR
        23: ((2325, 2271, 2292, 2167), (2513, 2583, 2432, 2352)),  # SAC
        24: ((2614, 2488, 2434, 2267), (2283, 2276, 2381, 2184)),  # SA
        25: ((2503, 2388, 2488, 2284), (2179, 2179, 2265, 2147)),  # OKC
        26: ((2380, 2346, 2469, 2379), (2626, 2498, 2679, 2471)),  # UTAH
        27: ((2262, 2336, 2406, 2229), (2627, 2566, 2561, 2450)),  # WSH
        28: ((2339, 2334, 2521, 2163), (2274, 2325, 2352, 2178)),  # TOR
        29: ((2410, 2352, 2375, 2247), (2496, 2458, 2483, 2442)),  # MEM
        30: ((2502, 2345, 2517, 2122), (2426, 2164, 2314, 2188)),  # CHA
    }
}


@dataclass(frozen=True)
class QuarterGame:
    """A finished game's regulation points by quarter, each side."""

    game_id: int
    away: int  # pro team id
    home: int
    away_points: Four
    home_points: Four


def parse_quarters(raw: dict) -> list[QuarterGame]:
    """Finished regular-season (and NBA Cup) games on one scoreboard day, with four quarters each side."""
    out: list[QuarterGame] = []
    for event in raw.get("events") or []:
        try:
            if int((event.get("season") or {}).get("type") or 2) != 2:
                continue  # preseason, play-in, playoffs
            comp = (event.get("competitions") or [{}])[0]
            if ((comp.get("status") or event.get("status") or {}).get("type") or {}).get("state") != "post":
                continue
            sides: dict[str, tuple[int, Four]] = {}
            for c in comp.get("competitors") or []:
                lines = [float(x["value"]) for x in c.get("linescores") or []]
                if len(lines) < QUARTERS:
                    raise ValueError("no quarters")
                sides[c["homeAway"]] = (int(c["team"]["id"]), tuple(lines[:QUARTERS]))
            (away, ap), (home, hp) = sides["away"], sides["home"]
            out.append(QuarterGame(int(event["id"]), away, home, ap, hp))
        except (KeyError, TypeError, ValueError):
            continue
    return out


async def fetch_quarters(day: date, transport: httpx.AsyncBaseTransport | None = None) -> list[QuarterGame] | None:
    """That day's finished games by quarter, or None when the scoreboard can't be read."""
    async with httpx.AsyncClient(transport=transport, timeout=TIMEOUT, headers={"User-Agent": USER_AGENT}) as http:
        try:
            resp = await http.get(SCOREBOARD, params={"dates": day.strftime("%Y%m%d")})
            if resp.status_code != 200:
                return None
            return parse_quarters(resp.json())
        except (httpx.HTTPError, ValueError):
            return None


def _add(a: list[float], b: Iterable[float]) -> None:
    for i, x in enumerate(b):
        a[i] += x


@dataclass
class Shapes:
    """Each team's regulation points by quarter this season, scored and allowed, and the league's."""

    scored: dict[int, list[float]] = field(default_factory=dict)
    allowed: dict[int, list[float]] = field(default_factory=dict)
    games: dict[int, int] = field(default_factory=dict)
    league: list[float] = field(default_factory=lambda: [0.0] * QUARTERS)

    def add(self, g: QuarterGame) -> None:
        for team, mine, theirs in ((g.away, g.away_points, g.home_points), (g.home, g.home_points, g.away_points)):
            _add(self.scored.setdefault(team, [0.0] * QUARTERS), mine)
            _add(self.allowed.setdefault(team, [0.0] * QUARTERS), theirs)
            self.games[team] = self.games.get(team, 0) + 1
            _add(self.league, mine)


def shapes(games: Iterable[QuarterGame]) -> Shapes:
    s = Shapes()
    seen: set[int] = set()
    for g in games:
        if g.game_id not in seen:
            seen.add(g.game_id)
            s.add(g)
    return s


def _share(points: Iterable[float]) -> list[float] | None:
    p = list(points)
    total = sum(p)
    return [x / total for x in p] if total > 0 else None


def _last_league(season: int) -> list[float] | None:
    table = LAST_SEASON.get(season)
    if not table:
        return None
    sums = [0.0] * QUARTERS
    for scored, _ in table.values():
        _add(sums, scored)
    return _share(sums)


def team_shape(season: int, s: Shapes, team: int, allowed: bool) -> list[float]:
    """A team's share of points (scored, or ``allowed``) in each quarter above the league's: this season
    so far, with last season's, kept PRIOR_KEEP, worth PRIOR_GAMES games."""
    league = _share(s.league) or _last_league(season) or [0.25] * QUARTERS
    now = _share((s.allowed if allowed else s.scored).get(team, ()))
    own = [x - lg for x, lg in zip(now, league, strict=True)] if now else [0.0] * QUARTERS
    n = s.games.get(team, 0) if now else 0
    prior = [0.0] * QUARTERS
    last, last_league = LAST_SEASON.get(season, {}).get(team), _last_league(season)
    if last and last_league:
        then = _share(last[1] if allowed else last[0])
        if then:
            prior = [PRIOR_KEEP * (x - lg) for x, lg in zip(then, last_league, strict=True)]
    w = n + PRIOR_GAMES
    return [(n * a + PRIOR_GAMES * b) / w for a, b in zip(own, prior, strict=True)]


@dataclass(frozen=True)
class QuarterLine:
    """One side's points by quarter, whole (adding up to the rounded sum of ``raw``), and the pieces."""

    points: tuple[int, ...]
    raw: tuple[float, ...]
    league: tuple[float, ...]  # the call x the league's share
    favourite: tuple[float, ...]  # the favourite's fast start and fourth-quarter fade (+ level)
    shape: tuple[float, ...]  # the team's shape and the opponent's allowed shape
    games: int  # this season's games in its shape


def quarter_line(season: int, s: Shapes, team: int, opponent: int, call: float, opponent_call: float) -> QuarterLine:
    """``team``'s points by quarter against ``opponent``, from both sides' calls (raw, before rounding)."""
    margin = call - opponent_call
    own = team_shape(season, s, team, allowed=False)
    opp = team_shape(season, s, opponent, allowed=True)
    league = tuple(call * sh for sh in LEAGUE_SHARE)
    fav = tuple(FAVOURITE[k] * margin + LEVEL[k] for k in range(QUARTERS))
    shape = tuple(call * (OWN_SHAPE * own[k] + OPP_SHAPE * opp[k]) for k in range(QUARTERS))
    raw = tuple(league[k] + fav[k] + shape[k] for k in range(QUARTERS))
    return QuarterLine(
        points=tuple(whole(list(raw))),
        raw=tuple(round(x, 2) for x in raw),
        league=tuple(round(x, 2) for x in league),
        favourite=tuple(round(x, 2) for x in fav),
        shape=tuple(round(x, 2) for x in shape),
        games=s.games.get(team, 0),
    )


def whole(values: list[float]) -> list[int]:
    """Whole numbers close to ``values`` that add up to their rounded sum (largest remainder)."""
    target = round(sum(values))
    floors = [int(v // 1) for v in values]
    left = target - sum(floors)
    order = sorted(range(len(values)), key=lambda i: values[i] - floors[i], reverse=True)
    for i in range(left):
        floors[order[i % len(order)]] += 1
    return floors


def live_quarters(
    pre: list[float],
    pre_opponent: list[float],
    real: list[float],
    real_opponent: list[float],
    period: int,
    seconds: float | None,
) -> list[float]:
    """A side's four quarters during a game: the real ones played, the one in progress as its real points
    so far plus the rest of it, and the pre-game number (``pre``, raw) for the rest, each moved by
    GIVE_BACK x the side's margin past the called one per quarter played. ``real`` holds the points of
    each period started (the one in progress last); ``seconds`` are left in it. In overtime, or past it,
    the four are the real ones."""
    real = [*real, *([0.0] * QUARTERS)][:QUARTERS]
    real_opponent = [*real_opponent, *([0.0] * QUARTERS)][:QUARTERS]
    if period > QUARTERS or period <= 0:
        return [float(x) for x in real] if period > QUARTERS else list(pre)
    cur = period - 1
    left = min(1.0, max(0.0, (seconds if seconds is not None else QUARTER_SECONDS) / QUARTER_SECONDS))
    played = cur + 1 - left  # quarters played, in quarters
    if played <= 0:
        return list(pre)
    had = sum(pre[:cur]) + pre[cur] * (1 - left)
    had_opponent = sum(pre_opponent[:cur]) + pre_opponent[cur] * (1 - left)
    past = (sum(real[: cur + 1]) - sum(real_opponent[: cur + 1])) - (had - had_opponent)
    move = GIVE_BACK * past / played
    out = [float(x) for x in real[:cur]]
    out.append(real[cur] + (pre[cur] + move) * left)
    out += [pre[k] + move for k in range(cur + 1, QUARTERS)]
    return out

"""Other ways to call each game, recorded before tip-off and scored once it is final.

Can 2026-11-26: "Running engine with 3-4 different parameters and weights for next game and save them
to check which one was better might be helpful?" Some tunings will make "crazy estimations" that turn
out right, so each one is kept and settled night by night, to tell a lead from luck over time.

RECORDED ONLY: nothing here changes Our call, Our Estimates or the Kupons. A tuning that keeps beating
Live earns a blind five-season test before it can move the engine (Can's rule: a signal gets weight
only by beating the current estimate in every season).

The tunings (TUNINGS):
    live        what the site shows: Our call with ESPN's OUT tags, its chance blended with the book's
                (90% book, the page's blend; ours alone when the book has no price)
    ours_alone  Our call with nobody's help: no book, nba.com's inactive list as well as ESPN's OUT
                tags, and the starting-five minutes term (STARTER_MINUTES) on tonight's player lines
    book        iddaa's own numbers, margin taken out: the price everyone else is betting into
    rivalry     the book plus the "rivalry stack" (STACK_EDGE) for a fallen contender in a slump
                against a recent playoff foe; recorded only on games where the stack fires
    crazy_one   Crazy One, a persona that believes what fans believe and the data backs on average but
                not in every season (STORY): Our call (ESPN's OUT tags, no book) moved by the rivalry
                stack, a favourite bouncing back after a losing streak, a fallen contender in a slump,
                late-season tankers, and two "one-man shows" meeting (the total, not the winner).
                Shown only in its own world, never as Our call

A day's record (JSON keys as strings):

    {"games": {espn game id: {
        "away": team, "home": team, "tip": iso, "checked": iso,     checked: the last look before tip-off
        "fives": [away status, home status],                         nba.com's lineup status, None unseen
        "out": {"espn": [player], "nba": [player], "unmatched": [name]},
        "book": [home chance, home margin, total] | None,            fair chance; margin = home minus away
        "stack": {"team": team, "edge": +wins per game, "star_out": bool} | None,
        "calls": {tuning: [away points, home points, home chance]},
        "beliefs": [[belief, team code, points toward that team]],   what moved Crazy One's call; team
                                                                     code "" for the game's total
        "players": {player: [team, starts, start share, *PLAYER_STATS]}}}}     ours_alone's moved lines

Live's player lines are the line log's (app/linelog.py); only the lines ours_alone moves are kept here.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import datetime
from statistics import NormalDist

from app import iddaa
from app.analysis import gamescore
from app.analysis.gamebox import Box
from app.espn.parse import PRO_TEAM_ABBREV
from app.linelog import STATS as LINE_STATS
from app.lostlog import BOOK_WEIGHT, blend_chance
from app.scores import ScheduledGame

LIVE, OURS_ALONE, BOOK, RIVALRY, CRAZY_ONE = "live", "ours_alone", "book", "rivalry", "crazy_one"
TUNINGS = (LIVE, OURS_ALONE, BOOK, RIVALRY, CRAZY_ONE)

# Starting-five minutes (area map dig A, 2026-11-25): a player who starts tonight but usually comes off
# the bench plays about this many minutes more than his line, times (starts tonight - his start share
# lately); a usual starter on the bench tonight, as many fewer. Blind: minutes miss -3.4..-5.5% every
# season. His other stats move with his minutes.
STARTER_MINUTES = 4.6
MOVED_MINUTES = 0.5  # lines moved less than this stay Live's
# The line log's stats a moved line keeps, in this order (app/linelog.py STATS names).
PLAYER_STATS = ("minutes", "pts", "reb", "ast", "three_pm")

# The rivalry stack (motivation report, 2026-11-25; odds 2011-2025): a team that won .600+ last season,
# is under .600 now and won 1+ fewer of its last 5 than expected, against a team it met in the playoffs
# in the last 2 seasons, won 10.4 more games per 100 than the book said (13 of 15 seasons, n 122;
# +12.0 with the foe's star playing, +3 ±11 with him out). Found after about 30 cuts: a hypothesis.
STACK_EDGE = 0.12
STACK_EDGE_STAR_OUT = 0.03
STRONG_LAST = 0.600
SLUMP_GAMES = 5
SLUMP_WINS = 1.0
STAR_PPG = 20.0  # points a game this season that make the foe's player "his star"
STAR_GAMES = 3  # a star out for most of the season (Edwards, 4 games by Nov 25) still counts

# Last season's records (2025-26, final standings) and the playoff series of the last two seasons, in
# ESPN's team codes: what the stack reads for the 2027 season. Add a season's line each summer.
LAST_SEASON: dict[int, dict[str, tuple[int, int]]] = {
    2027: {
        "OKC": (64, 18), "SA": (62, 20), "DET": (60, 22), "BOS": (56, 26), "DEN": (54, 28), "NY": (53, 29),
        "LAL": (53, 29), "CLE": (52, 30), "HOU": (52, 30), "MIN": (49, 33), "TOR": (46, 36), "ATL": (46, 36),
        "PHI": (45, 37), "ORL": (45, 37), "PHX": (45, 37), "CHA": (44, 38), "MIA": (43, 39), "POR": (42, 40),
        "LAC": (42, 40), "GS": (37, 45), "MIL": (32, 50), "CHI": (31, 51), "NO": (26, 56), "DAL": (26, 56),
        "MEM": (25, 57), "SAC": (22, 60), "UTAH": (22, 60), "BKN": (20, 62), "IND": (19, 63), "WSH": (17, 65),
    },
}  # fmt: skip
PLAYOFF_SERIES: dict[int, tuple[tuple[str, str], ...]] = {
    2025: (
        ("CLE", "MIA"), ("IND", "MIL"), ("NY", "DET"), ("BOS", "ORL"), ("OKC", "MEM"), ("DEN", "LAC"),
        ("LAL", "MIN"), ("HOU", "GS"), ("CLE", "IND"), ("NY", "BOS"), ("OKC", "DEN"), ("MIN", "GS"),
        ("IND", "NY"), ("OKC", "MIN"), ("OKC", "IND"),
    ),
    2026: (
        ("DET", "ORL"), ("CLE", "TOR"), ("NY", "ATL"), ("BOS", "PHI"), ("OKC", "PHX"), ("LAL", "HOU"),
        ("DEN", "MIN"), ("SA", "POR"), ("DET", "CLE"), ("NY", "PHI"), ("OKC", "LAL"), ("MIN", "SA"),
        ("CLE", "NY"), ("OKC", "SA"), ("NY", "SA"),
    ),
}  # fmt: skip
PLAYOFF_SEASONS = 2

# Crazy One's beliefs, the "Story believer" (Can 2026-11-26). Each makes sense to a fan and paid on
# average over past seasons, but none held in every season, so the engine doesn't use them. Picked over
# two other candidates (absences and lineup talent doubled; high totals shrunk) on a replay of 2021-22..
# 2024-25 against Our call, confirmed on 2025-26 and 2026-27 (/mnt/project-files/crazy-one/replay):
# winners' log loss +0.08 +0.10 -0.22 +0.10 | +0.21 -0.20 % better; the stack's 27 games 17 right vs 14.
STACK = "rivalry"  # the rivalry stack (STACK_EDGE), on Our call's chance
BOUNCE = "bounce"  # a favourite that lost BOUNCE_STREAK+ in a row: +0.65 margin, 6 of 8 seasons
BOUNCE_STREAK = 3
BOUNCE_MARGIN = 0.65
BOUNCE_BIG_FAVOURITE = 6.0  # favoured by this much: BOUNCE_BIG_MARGIN instead
BOUNCE_BIG_MARGIN = 2.1
SLUMP = "slump"  # a fallen contender in a slump, against anyone: plays nearer its true level
SLUMP_MARGIN = 0.5
TANK = "tank"  # under TANK_PCT after TANK_GAMES games: ran under the book in all of 2021-2025
TANK_GAMES = 55
TANK_PCT = 0.35
TANK_MARGIN = -1.5
# Two "one-man shows" (possession map, Can 2026-11-26, /mnt/project-files/possession-map): when both teams
# run their finishing (FGA + 0.44 FTA + TO) through few players, both in the league's ONE_MAN_TOP that day,
# the game ran under Our call's total in 4 of 5 seasons 2022-2026 (about -1 to -2.7 more than other
# games; over in 2021-22), and the third quarter under the quarter estimate in all six 2021-2026
# (-0.1 to -1.8): neither the stars nor their teammates get the usual lift after halftime, as if the
# defenses load up on the one man. Not every season on the total, so the engine doesn't use it.
ONE_MAN = "one_man"
ONE_MAN_TOP = 10  # of the 30 teams, by finishing concentration this season (gamebox.finishing_spread)
ONE_MAN_GAMES = 10  # games this season before a team's concentration counts
ONE_MAN_HALF_LIFE = 20.0  # games, as the engine's team ratings
ONE_MAN_TOTAL = -1.0  # points off the game's total
TOTAL = ""  # a belief's team code when it moves the game's total, not the winner
STORY = (STACK, BOUNCE, SLUMP, TANK, ONE_MAN)

_NORMAL = NormalDist()


@dataclass(frozen=True)
class TeamForm:
    """What the stack reads about one team before tonight."""

    code: str  # ESPN's team code
    won: int  # games won this season
    played: int
    recent: tuple[tuple[bool, float], ...] = ()  # its last games: (won, the chance it was given), oldest first
    stars: frozenset[int] = frozenset()  # its STAR_PPG+ players this season
    one_man: bool = False  # among the league's ONE_MAN_TOP one-man shows (one_man_shows)


def playoff_foes(season: int, a: str, b: str) -> bool:
    """Did these two teams meet in the playoffs in one of the PLAYOFF_SEASONS before ``season``?"""
    pair = {a, b}
    return any(set(p) == pair for s in range(season - PLAYOFF_SEASONS, season) for p in PLAYOFF_SERIES.get(s, ()))


def fallen_and_slumping(season: int, t: TeamForm) -> bool:
    """A .600+ team last season, under .600 now, that won SLUMP_WINS+ fewer of its last SLUMP_GAMES
    games than its chances said."""
    last = LAST_SEASON.get(season, {}).get(t.code)
    if last is None or sum(last) == 0 or last[0] / sum(last) < STRONG_LAST:
        return False
    if t.played == 0 or t.won / t.played >= STRONG_LAST:
        return False
    recent = t.recent[-SLUMP_GAMES:]
    if len(recent) < SLUMP_GAMES:
        return False
    return sum(won for won, _ in recent) - sum(p for _, p in recent) <= -SLUMP_WINS


def stack(season: int, away: TeamForm, home: TeamForm, out: set[int]) -> dict | None:
    """The rivalry stack for this game: which side it backs and by how much, or None when it doesn't
    fire. When both sides qualify they cancel out."""
    if not playoff_foes(season, away.code, home.code):
        return None
    sides = [(t, foe) for t, foe in ((away, home), (home, away)) if fallen_and_slumping(season, t)]
    if len(sides) != 1:
        return None
    team, foe = sides[0]
    star_out = bool(foe.stars & out)
    return {"team": team.code, "edge": STACK_EDGE_STAR_OUT if star_out else STACK_EDGE, "star_out": star_out}


def losing_streak(t: TeamForm) -> int:
    """Games lost in a row going into tonight (up to the recent games kept)."""
    n = 0
    for won, _ in reversed(t.recent):
        if won:
            break
        n += 1
    return n


def beliefs(season: int, away: TeamForm, home: TeamForm, margin: float, fired: dict | None) -> list[list]:
    """What Crazy One believes about this game: [belief, team code, points toward that team], from
    Our call's home ``margin`` (ESPN's OUT tags, no book) and the rivalry stack if it fired; a belief
    about the game's total has team code TOTAL and its points move the total."""
    out: list[list] = []
    if fired is not None:
        side_home = fired["team"] == home.code
        p = gamescore.win_chance(margin)
        p_home = p if margin >= 0 else 1 - p
        p_side = p_home if side_home else 1 - p_home
        out.append([STACK, fired["team"], round(margin_of(p_side + fired["edge"]) - margin_of(p_side), 2)])
    for t, fav_by in ((away, -margin), (home, margin)):
        if fav_by > 0 and losing_streak(t) >= BOUNCE_STREAK:
            out.append([BOUNCE, t.code, BOUNCE_BIG_MARGIN if fav_by >= BOUNCE_BIG_FAVOURITE else BOUNCE_MARGIN])
        if fallen_and_slumping(season, t):
            out.append([SLUMP, t.code, SLUMP_MARGIN])
        if t.played >= TANK_GAMES and t.won / t.played < TANK_PCT:
            out.append([TANK, t.code, TANK_MARGIN])
    if away.one_man and home.one_man:
        out.append([ONE_MAN, TOTAL, ONE_MAN_TOTAL])
    return out


def one_man_shows(box: Box, day: int) -> set[int]:
    """The league's ONE_MAN_TOP teams by how concentrated their finishing has been this season before
    ``day`` (gamebox.finishing_spread, recent games counting most), among teams with ONE_MAN_GAMES+."""
    decay = 0.5 ** (1 / ONE_MAN_HALF_LIFE)
    level: dict[int, float] = {}
    for team, games in box.finishing.items():
        before = [v for d, v in games if d < day]
        if len(before) < ONE_MAN_GAMES:
            continue
        w = [decay ** (len(before) - 1 - i) for i in range(len(before))]
        level[team] = sum(x * y for x, y in zip(before, w, strict=True)) / sum(w)
    return set(sorted(level, key=lambda t: (-level[t], t))[:ONE_MAN_TOP])


def margin_of(chance: float) -> float:
    """The home margin a home win chance stands for, at our call's spread (gamescore.WIN_SPREAD)."""
    p = min(max(chance, 0.001), 0.999)
    return _NORMAL.inv_cdf(p) * gamescore.WIN_SPREAD


def home_chance(away: float, home: float) -> float:
    """Our call's home win chance from its two scores."""
    c = gamescore.win_chance(home - away)
    return c if home >= away else 1 - c


def book_numbers(game: dict | None) -> list[float] | None:
    """[home chance, home margin, total] from a saved iddaa game, or None without a priced winner.
    iddaa's margin line is the home team's handicap (-5.5: home by 6), so the margin is minus it."""
    if not game:
        return None
    p = iddaa.home_chance(game)
    if p is None:
        return None
    line = game.get("margin")
    total = game.get("total")
    margin = -float(line[0]) if line and line[0] not in ("", None) else margin_of(p)
    return [p, round(margin, 1), float(total[0]) if total and total[0] not in ("", None) else None]


def _points(margin: float, total: float | None, fallback_total: float) -> tuple[float, float]:
    t = fallback_total if total is None else total
    return round((t - margin) / 2, 1), round((t + margin) / 2, 1)


@dataclass(frozen=True)
class GameInputs:
    """Everything the tunings read for one game, gathered by the watch (app/main.py)."""

    away: int
    home: int
    tip: datetime
    live_call: tuple[float, float]  # Our call's points, ESPN's OUT tags
    alone_call: tuple[float, float]  # Our call's points, ESPN's OUT tags and nba.com's inactive list
    out_espn: frozenset[int] = frozenset()
    out_nba: frozenset[int] = frozenset()
    unmatched: tuple[str, ...] = ()  # nba.com inactive names we couldn't find
    fives: tuple[str | None, str | None] = (None, None)  # nba.com's lineup status, away and home
    starters: frozenset[int] = frozenset()  # tonight's starters (both teams), when nba.com lists them
    book: dict | None = None  # the saved iddaa game
    stack: dict | None = None
    lines: dict[int, tuple[int, dict[str, float]]] | None = None  # player -> (team, Live's line by stat)
    share: dict[int, float] | None = None  # player -> his start share lately
    beliefs: tuple[list, ...] = ()  # Crazy One's (beliefs()), [belief, team code, points toward it]


def moved_lines(g: GameInputs) -> dict[str, list]:
    """ours_alone's player lines that differ from Live's: the starting five where nba.com lists it,
    and nba.com's inactive players at zero."""
    out: dict[str, list] = {}
    listed = {g.away: g.fives[0] is not None, g.home: g.fives[1] is not None}
    for pid, (team, line) in (g.lines or {}).items():
        if team not in listed:
            continue
        share = (g.share or {}).get(pid)
        starts = pid in g.starters
        if pid in g.out_nba and pid not in g.out_espn:
            out[str(pid)] = [team, 0, None if share is None else round(share, 2), *(0.0 for _ in PLAYER_STATS)]
            continue
        if not listed[team] or share is None:
            continue
        delta = STARTER_MINUTES * (float(starts) - share)
        minutes = line.get("minutes") or 0.0
        if abs(delta) < MOVED_MINUTES or minutes <= 0:
            continue
        scale = max(minutes + delta, 0.0) / minutes
        tuned = (round(line.get(s, 0.0) * scale, 1) for s in PLAYER_STATS)
        out[str(pid)] = [team, int(starts), round(share, 2), *tuned]
    return out


def calls(g: GameInputs) -> tuple[dict[str, list[float]], list[float] | None]:
    """Each tuning's [away points, home points, home chance] for this game, and the book's numbers."""
    la, lh = g.live_call
    oa, oh = g.alone_call
    book = book_numbers(g.book)
    ours_live = home_chance(la, lh)
    home_code = PRO_TEAM_ABBREV.get(g.home)
    lean = sum(pts if code == home_code else -pts for _, code, pts in g.beliefs if code != TOTAL)
    total = sum(pts for _, code, pts in g.beliefs if code == TOTAL)
    ca, ch = la - lean / 2 + total / 2, lh + lean / 2 + total / 2
    out = {
        LIVE: [round(la, 1), round(lh, 1), round(blend_chance(ours_live, book[0] if book else None, BOOK_WEIGHT), 3)],
        OURS_ALONE: [round(oa, 1), round(oh, 1), round(home_chance(oa, oh), 3)],
        CRAZY_ONE: [round(ca, 1), round(ch, 1), round(home_chance(ca, ch), 3)],
    }
    if book is not None:
        p, margin, total = book
        out[BOOK] = [*_points(margin, total, la + lh), round(p, 3)]
        if g.stack is not None:
            backed_home = g.stack["team"] == PRO_TEAM_ABBREV.get(g.home)
            q = min(max(p + (g.stack["edge"] if backed_home else -g.stack["edge"]), 0.01), 0.99)
            out[RIVALRY] = [*_points(margin_of(q), total, la + lh), round(q, 3)]
    return out, book


def update(record: dict | None, day_games: Iterable[tuple[int, GameInputs]], now: datetime) -> dict:
    """The day's record with this look written in for each game that hasn't tipped off. A game that
    has started keeps what it had: its last look before tip-off."""
    out = {"games": dict((record or {}).get("games") or {})}
    stamp = now.isoformat(timespec="seconds")
    for gid, g in day_games:
        if now >= g.tip:
            continue
        made, book = calls(g)
        out["games"][str(gid)] = {
            "away": g.away,
            "home": g.home,
            "tip": g.tip.isoformat(timespec="seconds"),
            "checked": stamp,
            "fives": list(g.fives),
            "out": {"espn": sorted(g.out_espn), "nba": sorted(g.out_nba), "unmatched": list(g.unmatched)},
            "book": book,
            "stack": g.stack,
            "calls": made,
            "beliefs": [list(b) for b in g.beliefs],
            "players": moved_lines(g),
        }
    return out


# ---------------------------------------------------------------- scoring


def _loss(p: float, won: bool) -> float:
    q = min(max(p if won else 1 - p, 0.001), 0.999)
    return -math.log(q)


def settle_game(game: dict, final: tuple[float, float]) -> dict[str, dict]:
    """Each tuning's score on one final game: winner right, log loss, Brier, margin and total miss."""
    away, home = final
    home_won = home > away
    out = {}
    for name, (a, h, p) in (game.get("calls") or {}).items():
        out[name] = {
            "hit": (p >= 0.5) == home_won,
            "loss": _loss(p, home_won),
            "brier": (p - float(home_won)) ** 2,
            "margin_miss": abs((home - away) - (h - a)),
            "total_miss": abs((home + away) - (h + a)),
            "chance": p,
        }
    return out


def settle_players(game: dict, live_lines: dict[str, list], results: dict[str, list]) -> list[dict]:
    """ours_alone's moved lines against Live's for players with a result: |miss| in each PLAYER_STATS.
    live_lines and results as the line log keeps them ([team, *linelog.STATS] and [*linelog.STATS])."""
    index = {name: i for i, (name, _) in enumerate(LINE_STATS)}
    rows = []
    for pid, (_team, starts, _share, *tuned) in (game.get("players") or {}).items():
        got, live = results.get(pid), live_lines.get(pid)
        if got is None or live is None:
            continue
        actual = [got[index[s]] if got else 0.0 for s in PLAYER_STATS]
        was = [live[1 + index[s]] for s in PLAYER_STATS]
        rows.append(
            {
                "player": int(pid),
                "starts": bool(starts),
                "tuned": [abs(x - y) for x, y in zip(tuned, actual, strict=True)],
                "live": [abs(x - y) for x, y in zip(was, actual, strict=True)],
            }
        )
    return rows


def edge_range(diffs: list[float]) -> tuple[float, float, float] | None:
    """Mean of paired differences and its 95% range; None under 2 games."""
    n = len(diffs)
    if n < 2:
        return None
    mean = sum(diffs) / n
    sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (n - 1))
    half = 1.96 * sd / math.sqrt(n)
    return mean, mean - half, mean + half


def summary(scored: list[dict[str, dict]], players: list[dict]) -> list[dict]:
    """Per tuning over the settled games: games, winners right, average log loss, Brier, margin and
    total miss, and its edge over Live in log loss (Live's less its own: + it was better) with a 95%
    range over the games both have. ``leading`` only when the whole range is above zero, so a few
    lucky nights don't count. ours_alone also gets its moved player lines against Live's."""
    out = []
    for name in TUNINGS:
        rows = [s[name] for s in scored if name in s]
        n = len(rows)
        item: dict = {"tuning": name, "games": n}
        if n:
            item |= {
                "hits": sum(r["hit"] for r in rows),
                "loss": round(sum(r["loss"] for r in rows) / n, 4),
                "brier": round(sum(r["brier"] for r in rows) / n, 4),
                "margin_miss": round(sum(r["margin_miss"] for r in rows) / n, 2),
                "total_miss": round(sum(r["total_miss"] for r in rows) / n, 2),
            }
        if name != LIVE:
            got = edge_range([s[LIVE]["loss"] - s[name]["loss"] for s in scored if name in s and LIVE in s])
            if got is not None:
                item |= {"edge": round(got[0], 4), "edge_low": round(got[1], 4), "edge_high": round(got[2], 4)}
            item["leading"] = got is not None and got[1] > 0
        if name == OURS_ALONE and players:
            k = len(players)
            item["players"] = {
                "lines": k,
                "tuned_miss": [round(sum(p["tuned"][i] for p in players) / k, 2) for i in range(len(PLAYER_STATS))],
                "live_miss": [round(sum(p["live"][i] for p in players) / k, 2) for i in range(len(PLAYER_STATS))],
            }
        out.append(item)
    return out


# ---------------------------------------------------------------- gathering (the watch)


def finals(box: Box) -> dict[int, dict[int, float]]:
    """Each finished game's points by team, from the box's team games."""
    out: dict[int, dict[int, float]] = {}
    for g in box.team_games:
        out.setdefault(g.game, {})[g.team] = g.stats["pf"]
    return out


def name_index(players: Iterable[tuple[int, int, str]]) -> dict[str, list[tuple[int, int]]]:
    """Folded name -> [(player, team)] for matching nba.com's names to ESPN's players."""
    out: dict[str, list[tuple[int, int]]] = {}
    for pid, team, name in players:
        out.setdefault(iddaa.fold(name), []).append((pid, team))
    return out


def match_player(name: str, team: int, names: dict[str, list[tuple[int, int]]]) -> int | None:
    """ESPN's player for an nba.com name: the one on this team, else the only one with that name."""
    found = names.get(iddaa.fold(name)) or []
    same = [pid for pid, t in found if t == team]
    if same:
        return same[0]
    return found[0][0] if len(found) == 1 else None


def _last_look(looks: dict | None, team: int) -> dict | None:
    entry = ((looks or {}).get("teams") or {}).get(str(team)) or {}
    seen = entry.get("looks") or []
    return seen[-1] if seen else None


def team_form(box: Box, team: int, day: int, past_books: dict[int, dict]) -> TeamForm:
    """Record, recent games with the chance each gave it (the book's when saved, else Our call's) and
    its STAR_PPG+ players, from the games before ``day``."""
    games = sorted((g for g in box.team_games if g.team == team and g.day < day), key=lambda g: g.day)
    won = [g.stats["pf"] > g.stats["pa"] for g in games]
    recent = []
    for g, w in zip(games[-SLUMP_GAMES:], won[-SLUMP_GAMES:], strict=True):
        sched = next((x for x in box.schedule.get(team, []) if x.day == g.day), None)
        home = sched.home if sched is not None else True
        away_team, home_team = (g.opponent, team) if home else (team, g.opponent)
        saved = ((past_books.get(g.day) or {}).get("games") or {}).get(str(g.game))
        p_home = iddaa.home_chance(saved) if saved else None
        if p_home is None:
            p_home = home_chance(*_call(box, g.day, away_team, home_team))
        recent.append((w, p_home if home else 1 - p_home))
    stars: set[int] = set()
    if box.lineups is not None:
        for pid, rows in box.lineups.games.get(team, {}).items():
            before = [pts for d, _, pts in rows if d < day]
            if len(before) >= STAR_GAMES and sum(before) / len(before) >= STAR_PPG:
                stars.add(pid)
    return TeamForm(
        code=PRO_TEAM_ABBREV.get(team, ""),
        won=sum(won),
        played=len(won),
        recent=tuple(recent),
        stars=frozenset(stars),
    )


def gather(
    box: Box,
    season: int,
    games: Iterable[ScheduledGame],
    out_espn: set[int],
    looks: dict | None,
    book: dict | None,
    lines: dict | None,
    names: dict[str, list[tuple[int, int]]],
    past_books: dict[int, dict],
) -> list[tuple[int, GameInputs]]:
    """Each game's inputs from what the other watches saved: ESPN's OUT tags (``out_espn``), nba.com's
    fives (``looks``, app/startlog.py), iddaa (``book``, app/iddaa.py), Live's player lines (``lines``,
    app/linelog.py) and the league's players by name (``names``)."""
    index = {name: i for i, (name, _) in enumerate(LINE_STATS)}
    live_lines = {
        int(pid): (row[0], {s: row[1 + index[s]] for s in PLAYER_STATS})
        for pid, row in ((lines or {}).get("lines") or {}).items()
    }
    out = []
    shows: dict[int, set[int]] = {}
    for g in games:
        teams = (g.away, g.home)
        if g.day not in shows:
            shows[g.day] = one_man_shows(box, g.day)
        fives: list[str | None] = []
        starters: set[int] = set()
        out_nba: set[int] = set()
        unmatched: list[str] = []
        for team in teams:
            look = _last_look(looks, team)
            fives.append(look.get("status") if look else None)
            if not look:
                continue
            for _nba, name, *_ in look.get("starters") or []:
                pid = match_player(name, team, names)
                if pid is not None:
                    starters.add(pid)
            for _nba, name, *_ in look.get("inactive") or []:
                pid = match_player(name, team, names)
                if pid is None:
                    unmatched.append(name)
                else:
                    out_nba.add(pid)
        live = _call(box, g.day, g.away, g.home, out_espn)
        alone = _call(box, g.day, g.away, g.home, out_espn | out_nba)
        away_form = replace(team_form(box, g.away, g.day, past_books), one_man=g.away in shows[g.day])
        home_form = replace(team_form(box, g.home, g.day, past_books), one_man=g.home in shows[g.day])
        fired = stack(season, away_form, home_form, out_espn | out_nba)
        believed = beliefs(season, away_form, home_form, live[1] - live[0], fired)
        share = {
            pid: f.start_share
            for pid, (team, _) in live_lines.items()
            if team in teams and (f := box.form.get(pid)) is not None and f.team == team and f.start_share is not None
        }
        out.append(
            (
                g.id,
                GameInputs(
                    away=g.away,
                    home=g.home,
                    tip=g.start,
                    live_call=live,
                    alone_call=alone,
                    out_espn=frozenset(p for p in out_espn if _on(box, p, teams)),
                    out_nba=frozenset(out_nba),
                    unmatched=tuple(unmatched),
                    fives=(fives[0], fives[1]),
                    starters=frozenset(starters),
                    book=((book or {}).get("games") or {}).get(str(g.id)),
                    stack=fired,
                    lines={p: v for p, v in live_lines.items() if v[0] in teams},
                    share=share,
                    beliefs=tuple(believed),
                ),
            )
        )
    return out


def _call(box: Box, day: int, away: int, home: int, out: set[int] | None = None) -> tuple[float, float]:
    a, h = box.call_parts(day, away, home, out)
    return sum(a.values()), sum(h.values())


def _on(box: Box, player: int, teams: tuple[int, int]) -> bool:
    """Is this ruled-out player on one of tonight's two teams (his last team in the box)?"""
    f = box.form.get(player)
    return f is not None and f.team in teams


# ---------------------------------------------------------------- Crazy One's world

BOLD = 0.10  # win chance apart from Our call that makes a call "bold"


def ours_of(game: dict) -> list[float] | None:
    """Our call as Scores shows it (ESPN's OUT tags, no book): [away, home, home chance]."""
    live = (game.get("calls") or {}).get(LIVE)
    if not live:
        return None
    return [live[0], live[1], round(home_chance(live[0], live[1]), 3)]


def crazy_record(games: list[tuple[dict, tuple[float, float]]]) -> dict:
    """Crazy One against Our call on the final games it called: winners right, its edge in log loss a
    game (Our call's less its own: + it was better) with a 95% range, ``leading`` only when the whole
    range is above zero, and its bold calls (BOLD+ apart) with how each went."""
    diffs, bold = [], []
    crazy_hits = ours_hits = 0
    for g, (away, home) in games:
        crazy, ours = (g.get("calls") or {}).get(CRAZY_ONE), ours_of(g)
        if not crazy or not ours:
            continue
        home_won = home > away
        crazy_hits += (crazy[2] >= 0.5) == home_won
        ours_hits += (ours[2] >= 0.5) == home_won
        diffs.append(_loss(ours[2], home_won) - _loss(crazy[2], home_won))
        if abs(crazy[2] - ours[2]) >= BOLD:
            bold.append((g, crazy[2] - ours[2], (crazy[2] >= 0.5) == home_won))
    got = edge_range(diffs)
    return {
        "games": len(diffs),
        "crazy_hits": crazy_hits,
        "ours_hits": ours_hits,
        "edge": None if got is None else round(got[0], 4),
        "edge_low": None if got is None else round(got[1], 4),
        "edge_high": None if got is None else round(got[2], 4),
        "leading": got is not None and got[1] > 0,
        "bold": bold,
    }

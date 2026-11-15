"""Final scores and garbage-time stats for finished games, from ESPN's play-by-play.

ESPN's game summary lists every play with the clock, both scores and the players in it,
and its box score marks the starters. That is enough to tell, play by play, whether the
game was already decided, using the rule Cleaning the Glass made common: in the 4th
quarter, one team leads by 25+ with 12:00 to 9:00 left, by 20+ with 9:00 to 6:00 left or
by 10+ in the last 6:00, and two or fewer starters (both teams together) are on the floor.

Each play is judged on the score and the lineup just before it. A player's line in that
time is his garbage-time line. It is kept only when his whole line, rebuilt from the
plays, matches the box score; when it doesn't, the game shows its score and nothing else,
so a play ESPN logged in an odd way can never put a wrong number on the card.

A finished game never changes, so each summary is read once and the result kept.
"""

from __future__ import annotations

import asyncio
import re
from collections import OrderedDict
from dataclasses import dataclass, field

import httpx

SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"
# ESPN's site API refuses (403) a user agent that carries a link.
USER_AGENT = "Mozilla/5.0 (compatible; fantasy-analyzer/1.0)"
TIMEOUT = 8.0
KEEP = 3000  # finished games kept in memory; a game is ~30 small lines

STATS = ("minutes", "pts", "reb", "ast", "stl", "blk", "tpm", "to", "fgm", "fga", "ftm", "fta")
# Box score columns a rebuilt line must match before its garbage-time part is trusted.
CHECKED = ("pts", "reb", "ast", "stl", "blk", "tpm", "to", "fgm", "fga", "ftm", "fta")
# Turnovers ESPN logs without the word in the play's type.
TURNOVERS = ("Traveling",)
# Plays that put a name in the log without him being on the floor.
OFF_FLOOR = re.compile(r"technical|ejection|ejected|delay", re.I)


@dataclass(frozen=True)
class GameNotes:
    """One finished game: each team's final score, and each player's garbage-time line."""

    scores: dict[int, int]  # ESPN team id -> final score
    team_of: dict[int, int]  # athlete id -> ESPN team id
    garbage: dict[int, dict[str, float]] = field(default_factory=dict)  # only players whose line checked out

    def score_for(self, athlete_id: int) -> tuple[int, int] | None:
        """(his team's score, the other team's) or None when he isn't in this game."""
        team = self.team_of.get(athlete_id)
        if team is None or len(self.scores) != 2 or team not in self.scores:
            return None
        opp = next(s for t, s in self.scores.items() if t != team)
        return self.scores[team], opp


def threshold(seconds_left: float) -> int:
    """The lead that decides a 4th quarter game with this much time left."""
    if seconds_left > 540:
        return 25
    if seconds_left > 360:
        return 20
    return 10


def clock_seconds(display: str) -> float:
    """ESPN's clock, '11:39' or '26.0', in seconds left in the period."""
    display = (display or "").strip()
    try:
        if ":" in display:
            m, s = display.split(":", 1)
            return int(m) * 60 + float(s)
        return float(display or 0)
    except ValueError:
        return 0.0


def _box(raw: dict) -> tuple[dict[int, int], set[int], dict[int, dict[str, float]]]:
    """Each athlete's team, the starters, and each athlete's box-score line."""
    team_of: dict[int, int] = {}
    starters: set[int] = set()
    lines: dict[int, dict[str, float]] = {}
    for team in (raw.get("boxscore") or {}).get("players") or []:
        tid = int((team.get("team") or {}).get("id") or 0)
        for block in team.get("statistics") or []:
            keys = block.get("keys") or []
            for a in block.get("athletes") or []:
                aid = int((a.get("athlete") or {}).get("id") or 0)
                if not aid:
                    continue
                team_of[aid] = tid
                if a.get("starter"):
                    starters.add(aid)
                values = dict(zip(keys, a.get("stats") or [], strict=False))
                line = _box_line(values)
                if line is not None:
                    lines[aid] = line
    return team_of, starters, lines


def _pair(value: str) -> tuple[float, float]:
    made, _, att = (value or "").partition("-")
    return float(made or 0), float(att or 0)


def _box_line(v: dict[str, str]) -> dict[str, float] | None:
    try:
        fgm, fga = _pair(v["fieldGoalsMade-fieldGoalsAttempted"])
        tpm, _ = _pair(v["threePointFieldGoalsMade-threePointFieldGoalsAttempted"])
        ftm, fta = _pair(v["freeThrowsMade-freeThrowsAttempted"])
        return {
            "pts": float(v["points"]),
            "reb": float(v["rebounds"]),
            "ast": float(v["assists"]),
            "stl": float(v["steals"]),
            "blk": float(v["blocks"]),
            "to": float(v["turnovers"]),
            "tpm": tpm,
            "fgm": fgm,
            "fga": fga,
            "ftm": ftm,
            "fta": fta,
        }
    except (KeyError, ValueError):
        return None  # a player who didn't play has no numbers


def _people(play: dict) -> list[int]:
    out = []
    for p in play.get("participants") or []:
        try:
            out.append(int((p.get("athlete") or {}).get("id") or 0))
        except (TypeError, ValueError):
            out.append(0)
    return out


def _kind(play: dict) -> str:
    return str((play.get("type") or {}).get("text") or "")


def _period(play: dict) -> int:
    try:
        return int((play.get("period") or {}).get("number") or 0)
    except (TypeError, ValueError):
        return 0


def _starting_fives(plays: list[dict]) -> dict[int, set[int]]:
    """Who was on the floor when each period began.

    The log names the five only through what they do, so a player counts as starting the
    period when he is in a play (or leaves the game) before he has come in during it.
    """
    out: dict[int, set[int]] = {}
    came_in: dict[int, set[int]] = {}
    for play in plays:
        per = _period(play)
        on = out.setdefault(per, set())
        seen = came_in.setdefault(per, set())
        people = [p for p in _people(play) if p]
        if _kind(play) == "Substitution":
            if len(people) >= 2:
                entering, leaving = people[0], people[1]
                if leaving not in seen:
                    on.add(leaving)
                seen.add(entering)
            continue
        if OFF_FLOOR.search(_kind(play)) or OFF_FLOOR.search(str(play.get("text") or "")):
            continue
        on.update(p for p in people if p not in seen)
    return out


def parse_summary(raw: dict) -> GameNotes | None:
    """Scores and garbage-time lines from one finished game's summary; None when it has no final score."""
    scores: dict[int, int] = {}
    try:
        for c in ((raw.get("header") or {}).get("competitions") or [{}])[0].get("competitors") or []:
            scores[int(c["id"])] = int(c["score"])
    except (KeyError, TypeError, ValueError, IndexError):
        return None
    if len(scores) != 2:
        return None
    team_of, starters, box = _box(raw)
    plays = raw.get("plays") or []
    if not plays or not starters:
        return GameNotes(scores=scores, team_of=team_of)
    total, garbage = rebuild(plays, starters)
    trusted: dict[int, dict[str, float]] = {}
    for aid, line in box.items():
        built = total.get(aid, dict.fromkeys(STATS, 0.0))
        if all(abs(built[k] - line[k]) < 0.5 for k in CHECKED):
            g = garbage.get(aid, dict.fromkeys(STATS, 0.0))
            trusted[aid] = {k: round(v, 1) for k, v in g.items()}
    return GameNotes(scores=scores, team_of=team_of, garbage=trusted)


def rebuild(plays: list[dict], starters: set[int]) -> tuple[dict[int, dict[str, float]], dict[int, dict[str, float]]]:
    """Every player's whole line and his garbage-time line, rebuilt from the plays."""
    fives = _starting_fives(plays)
    total: dict[int, dict[str, float]] = {}
    garbage: dict[int, dict[str, float]] = {}

    def add(aid: int, key: str, n: float, decided: bool) -> None:
        if not aid:
            return
        total.setdefault(aid, dict.fromkeys(STATS, 0.0))[key] += n
        if decided:
            garbage.setdefault(aid, dict.fromkeys(STATS, 0.0))[key] += n

    on: set[int] = set()
    period = None
    last_clock = 0.0
    home = away = 0
    for play in plays:
        per = _period(play)
        left = clock_seconds(str((play.get("clock") or {}).get("displayValue") or ""))
        if per != period:
            period = per
            on = set(fives.get(per, set()))
            last_clock = 720.0 if per <= 4 else 300.0
        # Decided or not, on the score and the five on the floor since the last play. That
        # state covers the floor time up to this play and the play itself.
        decided = per == 4 and abs(home - away) >= threshold(left) and len(on & starters) <= 2
        gone = max(0.0, last_clock - left)
        if gone:
            for aid in on:
                add(aid, "minutes", gone / 60, decided)
        last_clock = left

        kind = _kind(play)
        text = str(play.get("text") or "")
        people = _people(play)
        first = people[0] if people else 0
        second = people[1] if len(people) > 1 else 0
        if kind == "Substitution":
            if len(people) >= 2:
                on.discard(second)
                on.add(first)
        elif "Free Throw" in kind:
            add(first, "fta", 1, decided)
            if play.get("scoringPlay"):
                add(first, "ftm", 1, decided)
                add(first, "pts", 1, decided)
        elif play.get("shootingPlay") and "Heave" not in kind:  # box scores leave end-of-period heaves out
            add(first, "fga", 1, decided)
            three = "three point" in text.lower()
            if play.get("scoringPlay"):
                value = float(play.get("scoreValue") or (3 if three else 2))
                add(first, "fgm", 1, decided)
                add(first, "pts", value, decided)
                if value == 3:
                    add(first, "tpm", 1, decided)
                if "assist" in text.lower():
                    add(second, "ast", 1, decided)
            elif "blocks" in text.lower():
                add(second, "blk", 1, decided)
        elif "Rebound" in kind:
            add(first, "reb", 1, decided)
        elif "Turnover" in kind or kind in TURNOVERS:
            add(first, "to", 1, decided)
            if "steals" in text.lower():
                add(second, "stl", 1, decided)
        try:
            home = int(play.get("homeScore") or home)
            away = int(play.get("awayScore") or away)
        except (TypeError, ValueError):
            pass

    return total, garbage


class SummaryBook:
    """Finished games' notes, each summary read once and kept (newest kept first when full)."""

    def __init__(self, keep: int = KEEP) -> None:
        self._keep = keep
        self._notes: OrderedDict[int, GameNotes] = OrderedDict()
        self._lock = asyncio.Lock()

    def get(self, game_id: int) -> GameNotes | None:
        return self._notes.get(game_id)

    def _put(self, game_id: int, notes: GameNotes) -> None:
        self._notes[game_id] = notes
        self._notes.move_to_end(game_id)
        while len(self._notes) > self._keep:
            self._notes.popitem(last=False)

    async def notes(
        self, game_ids: list[int], transport: httpx.AsyncBaseTransport | None = None
    ) -> dict[int, GameNotes]:
        """Notes for these finished games; any ESPN can't give right now are left out (and tried next time)."""
        out = {gid: n for gid in game_ids if (n := self._notes.get(gid)) is not None}
        missing = [gid for gid in game_ids if gid not in out]
        if not missing:
            return out
        async with httpx.AsyncClient(transport=transport, timeout=TIMEOUT, headers={"User-Agent": USER_AGENT}) as http:

            async def one(gid: int) -> tuple[int, GameNotes | None]:
                try:
                    resp = await http.get(SUMMARY, params={"event": gid})
                    if resp.status_code != 200:
                        return gid, None
                    return gid, parse_summary(resp.json())
                except (httpx.HTTPError, ValueError):
                    return gid, None

            pairs = await asyncio.gather(*(one(g) for g in missing))
        for gid, notes in pairs:
            if notes is not None:
                self._put(gid, notes)
                out[gid] = notes
        return out

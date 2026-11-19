"""The chance a player with an injury tag plays his next game, and whether a player listed out is coming back.

ESPN's fantasy data says only DAY_TO_DAY or OUT. The NBA's own injury report says more: probable,
questionable, doubtful, out, available, and every team files it several times a day. Five seasons
of those reports (2021-22 to 2025-26, every version), matched to who actually played, give how often
each tag ended in a game played (``PLAY``). Two things move it besides the tag: how long before
tip-off the tag was read (players still questionable an hour out are often game-time calls that go
their way) and the player's role (a bench player listed questionable sits more often, coach's
decisions included). ``scripts/play_chance_backtest.py`` builds the table and tests it blind, each
season against a table made from the other four: the tag with the hours beats a flat day-to-day
rate in every season, and the role beats the tag and hours alone in every season and in 2026-27 so
far. Injury type (soreness, sprain, ...), a missed last game and back-to-backs were tried and
didn't hold in every season, so they don't count.

Live, the tag comes from the comment in ESPN's injury feed ("Thompson (knee) is questionable for
Wednesday's game"), which uses the NBA report's words. A comment must be about the game in question:
ESPN keeps re-dating old comments ("ruled out for Tuesday's game" stamped Wednesday morning), so a
comment that names a weekday counts only for a game on that weekday (New York time), and one that
names none only when written after his team's last game. With no comment about the game, a
day-to-day player gets the rate of all probable, questionable and doubtful players together.

A player listed out (``COMEBACK``): the chance he plays the next game falls with the games he has
missed in a row (regulars of 20+ minutes, same order in all five seasons, and better than one flat
rate in each). Once the news upgrades him for the next game (questionable, probable, available),
the tag's chance takes over: that upgrade is the clearest sign he's coming back.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import httpx

from app.gameclock import TIMEOUT, USER_AGENT

# ESPN's site injury feed. The second host serves the same feed when the first refuses a caller.
FEED_URLS = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries",
    "https://site.web.api.espn.com/apis/site/v2/sports/basketball/nba/injuries",
)

PROBABLE, QUESTIONABLE, DOUBTFUL, AVAILABLE, OUT = "Probable", "Questionable", "Doubtful", "Available", "Out"
DAY_TO_DAY = "Day-to-day"  # ESPN's tag with no fresher word from the team
NEW_YORK = ZoneInfo("America/New_York")
WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
WEEKDAY = re.compile(r"\b(" + "|".join(WEEKDAYS) + r")\b", re.I)

# Share who played, by the tag read at least this many hours before tip-off (the largest that fits)
# and his role. Fitted on 2021-22 to 2025-26 NBA injury reports by scripts/play_chance_backtest.py.
HOURS = (24.0, 8.0, 5.0, 3.0, 1.5, 0.5)
ROLES = ("bench", "rotation", "starter")  # under 15, 15 to 25, 25+ minutes a game
PLAY: dict[str, tuple[tuple[float, float, float], ...]] = {
    #               bench  rotation starter
    PROBABLE: (
        (0.71, 0.90, 0.93),  # 24h+
        (0.71, 0.90, 0.94),  # 8h
        (0.75, 0.93, 0.96),  # 5h
        (0.76, 0.93, 0.97),  # 3h
        (0.76, 0.95, 0.97),  # 1.5h
        (0.81, 0.95, 0.98),  # 30 min
    ),
    QUESTIONABLE: (
        (0.48, 0.54, 0.56),
        (0.46, 0.52, 0.56),
        (0.43, 0.52, 0.55),
        (0.43, 0.52, 0.55),
        (0.44, 0.53, 0.58),
        (0.45, 0.57, 0.69),
    ),
    DOUBTFUL: (
        (0.05, 0.06, 0.07),
        (0.04, 0.05, 0.05),
        (0.03, 0.03, 0.03),
        (0.01, 0.02, 0.02),
        (0.02, 0.02, 0.01),
        (0.00, 0.03, 0.02),
    ),
    AVAILABLE: (
        (0.51, 0.90, 0.97),
        (0.47, 0.90, 0.97),
        (0.54, 0.90, 0.99),
        (0.55, 0.90, 0.99),
        (0.59, 0.91, 0.99),
        (0.65, 0.93, 0.99),
    ),
    # Probable, questionable and doubtful together: a day-to-day tag with no word on which.
    DAY_TO_DAY: (
        (0.47, 0.57, 0.62),
        (0.46, 0.56, 0.62),
        (0.45, 0.56, 0.62),
        (0.45, 0.55, 0.63),
        (0.46, 0.57, 0.65),
        (0.47, 0.62, 0.74),
    ),
}

# A player listed out: chance he plays the next game, by team games missed in a row so far
# (from that number up to the next). 2021-22 to 2025-26, regulars of 20+ minutes a game.
COMEBACK = ((1, 0.47), (2, 0.20), (4, 0.14), (7, 0.10))

# The comment's words for each tag. The last one named wins: "upgraded from questionable to probable".
WORDS = (
    (re.compile(r"\bprobable\b", re.I), PROBABLE),
    (re.compile(r"\bquestionable\b|\bgame-time decision\b", re.I), QUESTIONABLE),
    (re.compile(r"\bdoubtful\b", re.I), DOUBTFUL),
    (
        re.compile(
            r"\bruled out\b|\bwon't play\b|\bwill not play\b|\bwill miss\b|\bwon't suit up\b|\bout for\b|\binactive\b",
            re.I,
        ),
        OUT,
    ),
    (
        re.compile(
            r"\b(?:is|been|as|be) available\b|\bwill play\b|\bcleared to play\b|\bexpected to play\b|\bis active\b"
            r"|\bin the starting lineup\b|\bwill start\b|\boff the injury report\b",
            re.I,
        ),
        AVAILABLE,
    ),
)


@dataclass(frozen=True)
class FeedNote:
    """One player's entry in ESPN's injury feed."""

    player_id: int
    status: str  # "Out", "Day-To-Day"
    comment: str  # ESPN's one-line note
    at: datetime | None  # when ESPN wrote it
    return_date: str | None  # ESPN's expected return, YYYY-MM-DD, if it gives one

    def tag(self) -> str | None:
        return tag_of(self.comment)

    def about(self, tip: datetime, last_tip: datetime | None) -> bool:
        """Whether the comment is about the game that tips at ``tip``."""
        if self.at is None or self.at > tip:
            return False
        named = {m.lower() for m in WEEKDAY.findall(self.comment)}
        if named:
            local = tip.astimezone(NEW_YORK)
            return WEEKDAYS[local.weekday()] in named and self.at >= tip - timedelta(days=6)
        return last_tip is None or self.at > last_tip


@dataclass(frozen=True)
class Chance:
    chance: float  # 0 to 1
    tag: str  # PROBABLE, QUESTIONABLE, DOUBTFUL, AVAILABLE, OUT, DAY_TO_DAY
    basis: str  # what the number comes from, in words
    upgraded: bool = False  # listed out by ESPN, but the latest news has him in play
    missed_in_a_row: int | None = None  # for a player listed out


def tag_of(comment: str) -> str | None:
    """The NBA report's tag a comment names; the last one named when it names more than one."""
    best: tuple[int, str] | None = None
    for pattern, tag in WORDS:
        for m in pattern.finditer(comment or ""):
            if best is None or m.start() > best[0]:
                best = (m.start(), tag)
    return best[1] if best else None


def role_of(minutes: float | None) -> int:
    if minutes is None:
        return 1
    return 0 if minutes < 15 else 1 if minutes < 25 else 2


def hours_row(hours: float) -> int:
    """The table row for a tag read this many hours before tip-off."""
    for i, h in enumerate(HOURS):
        if hours >= h:
            return i
    return len(HOURS) - 1


def play_chance(tag: str, hours: float, minutes: float | None) -> float:
    return PLAY[tag][hours_row(hours)][role_of(minutes)]


def comeback_chance(missed: int) -> float:
    """Chance a player listed out plays his next game, having missed ``missed`` team games in a row."""
    out = COMEBACK[0][1]
    for k, p in COMEBACK:
        if missed >= k:
            out = p
    return out


def _when(hours: float) -> str:
    if hours >= 24:
        return "a day or more before tip-off"
    if hours >= 1.5:
        return f"about {int(HOURS[hours_row(hours)])} hours before tip-off"
    return "within the last hour or so before tip-off" if hours >= 0.5 else "right before tip-off"


def _role_words(minutes: float | None) -> str:
    return ("bench players (under 15 minutes)", "rotation players (15 to 25 minutes)", "players of 25+ minutes")[
        role_of(minutes)
    ]


def estimate(
    espn_status: str,
    note: FeedNote | None,
    tip: datetime,
    last_tip: datetime | None,
    now: datetime,
    minutes: float | None,
    missed: int | None = None,
) -> Chance | None:
    """His chance to play the game that tips at ``tip``, or None when there's nothing to say.

    ``espn_status`` is ESPN's fantasy tag, ``note`` his entry in ESPN's injury feed, ``last_tip``
    when his team's previous game started (a note from before it is about that game), ``minutes``
    his minutes a game this season and ``missed`` the team games he has missed in a row, for a
    player listed out.
    """
    if now >= tip:
        return None
    tag = note.tag() if note is not None and note.about(tip, last_tip) else None
    hours = (tip - now).total_seconds() / 3600
    out = espn_status in ("OUT", "INJURY_RESERVE", "SUSPENSION")
    if tag in (PROBABLE, QUESTIONABLE, DOUBTFUL, AVAILABLE):
        p = play_chance(tag, hours, minutes)
        basis = (
            f"Listed {tag.lower()} {_when(hours)}: {round(p * 100)}% of {_role_words(minutes)} "
            "with that tag played (NBA injury reports, 2021-22 to 2025-26)."
        )
        return Chance(p, tag, basis, upgraded=out, missed_in_a_row=missed if out else None)
    if tag == OUT:
        return Chance(0.0, OUT, "Ruled out for this game.", missed_in_a_row=missed)
    if out:
        if espn_status == "SUSPENSION" or missed is None or missed <= 0:
            return None  # just hurt (or suspended): no history to go on yet
        p = comeback_chance(missed)
        games = "game" if missed == 1 else "games"
        basis = (
            f"Out, {missed} {games} missed in a row and no word yet on this one: "
            f"{round(p * 100)}% of regulars out that long played their next game (2021-22 to 2025-26)."
        )
        return Chance(p, OUT, basis, missed_in_a_row=missed)
    if espn_status == "DAY_TO_DAY":
        p = play_chance(DAY_TO_DAY, hours, minutes)
        basis = (
            f"Day-to-day with no word yet on which tag, {_when(hours)}: {round(p * 100)}% of "
            f"{_role_words(minutes)} listed probable, questionable or doubtful played."
        )
        return Chance(p, DAY_TO_DAY, basis)
    return None


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def parse_feed(raw: dict) -> dict[int, FeedNote]:
    """Every player in ESPN's injury feed, by ESPN player id (the same id the fantasy data uses)."""
    out: dict[int, FeedNote] = {}
    for team in raw.get("injuries") or []:
        for item in team.get("injuries") or []:
            try:
                pid = int((item.get("athlete") or {}).get("id") or 0)
            except (TypeError, ValueError):
                continue
            if not pid:
                # The feed leaves out the id field; the athlete's page links carry it.
                links = " ".join(str(x.get("href") or "") for x in (item.get("athlete") or {}).get("links") or [])
                m = re.search(r"/id/(\d+)", links)
                if not m:
                    continue
                pid = int(m.group(1))
            details = item.get("details") or {}
            out[pid] = FeedNote(
                player_id=pid,
                status=str(item.get("status") or ""),
                comment=str(item.get("shortComment") or ""),
                at=_parse_time(item.get("date")),
                return_date=(str(details["returnDate"])[:10] if details.get("returnDate") else None),
            )
    return out


async def fetch_feed(transport: httpx.AsyncBaseTransport | None = None) -> dict[int, FeedNote] | None:
    """ESPN's injury feed, or None when neither host answers."""
    async with httpx.AsyncClient(transport=transport, timeout=TIMEOUT, headers={"User-Agent": USER_AGENT}) as http:
        for url in FEED_URLS:
            try:
                resp = await http.get(url)
                if resp.status_code == 200:
                    return parse_feed(resp.json())
            except (httpx.HTTPError, ValueError):
                continue
    return None

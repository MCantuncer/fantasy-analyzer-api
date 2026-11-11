"""The public sample league: masked team names and which team a visitor scouts first.

A visitor who hasn't signed in sees one real public league, analysed once a day. Its fantasy
team names are swapped for basketball play calls, so the page shows real players and real
numbers without naming the people who run those teams. This module is pure: it takes finished
reports and returns masked copies, and picks the team with the clearest move of the day.
"""

import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from app.schemas import AnalysisReport, LeagueSummary

SAMPLE_LEAGUE_NAME = "Sample league"

# Fixed order: a team keeps its play call for the whole season (teams are named by sorted id).
PLAY_CALLS: tuple[tuple[str, str], ...] = (
    ("Horns", "HRN"),
    ("Hammer", "HAM"),
    ("Spain", "SPN"),
    ("Zipper", "ZIP"),
    ("Floppy", "FLP"),
    ("Stagger", "STG"),
    ("Pistol", "PST"),
    ("Elevator", "ELV"),
    ("Ghost", "GST"),
    ("Chin", "CHN"),
    ("Iverson", "IVR"),
    ("Flex", "FLX"),
    ("Shuffle", "SHF"),
    ("Triangle", "TRI"),
    ("Motion", "MOT"),
    ("Delay", "DLY"),
    ("Loop", "LOP"),
    ("Thumbs", "THM"),
    ("Fist", "FST"),
    ("Veer", "VER"),
)

# Keys whose whole value is a fantasy team's name.
_NAME_KEYS = frozenset({"team_name", "partner_team_name", "opponent_name"})

# A gain counts as a clear move from this many playoff-odds points (odds are 0-1 fractions).
CLEAR_GAIN = 0.02
SMALL_GAIN = 0.01


@dataclass(frozen=True)
class MaskedTeam:
    name: str
    abbrev: str


def masked_names(team_ids: Iterable[int]) -> dict[int, MaskedTeam]:
    """A play call for every team, by sorted team id, so the same league always gets the same names."""
    ids = sorted(set(team_ids))
    if len(ids) > len(PLAY_CALLS):
        raise ValueError(f"the sample supports up to {len(PLAY_CALLS)} teams, got {len(ids)}")
    return {team_id: MaskedTeam(*PLAY_CALLS[i]) for i, team_id in enumerate(ids)}


class _Masker:
    def __init__(self, league: LeagueSummary, names: Mapping[int, MaskedTeam]):
        missing = {t.id for t in league.teams} - set(names)
        if missing:
            raise ValueError(f"no masked name for teams {sorted(missing)}")
        self.by_name = {t.name: names[t.id].name for t in league.teams}
        self.by_abbrev = {t.abbrev: names[t.id].abbrev for t in league.teams}
        self.names = names
        # Longest first, so "Lakers B" is replaced before "Lakers".
        ordered = [n for n in sorted(self.by_name, key=len, reverse=True) if n.strip()]
        self.in_text = re.compile("|".join(re.escape(n) for n in ordered)) if ordered else None

    def text(self, value: str) -> str:
        if self.in_text is None:
            return value
        return self.in_text.sub(lambda m: self.by_name[m.group(0)], value)

    def walk(self, value: Any, key: str | None = None) -> Any:
        if isinstance(value, dict):
            return {k: self.walk(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [self.walk(v, key) for v in value]
        if isinstance(value, str):
            if key in _NAME_KEYS and value in self.by_name:
                return self.by_name[value]
            return self.text(value)
        return value

    def league(self, data: dict) -> dict:
        teams = [{**t, "name": self.names[t["id"]].name, "abbrev": self.names[t["id"]].abbrev} for t in data["teams"]]
        return {**data, "id": 0, "name": SAMPLE_LEAGUE_NAME, "teams": teams}


def mask_report(report: AnalysisReport, names: Mapping[int, MaskedTeam]) -> AnalysisReport:
    """The same report with every fantasy team name, abbreviation and the league's name masked.

    Player names, NBA teams and numbers stay as they are. Free text (pitches, reasons, notes)
    has team names replaced wherever they appear.
    """
    masker = _Masker(report.league, names)
    data = report.model_dump(mode="json")
    league = masker.league(data.pop("league"))
    masked = masker.walk(data)
    masked["league"] = league
    return AnalysisReport.model_validate(masked)


Kind = Literal["pickup", "trade"]


@dataclass(frozen=True)
class Move:
    kind: Kind
    gain: float  # playoff-odds change, 0-1
    odds_before: float


def best_move(report: AnalysisReport) -> Move | None:
    """The move the Best move card leads with (mirrors the web's bestMoveOf): top trade or top
    straight add/drop, whichever is worth more, a pickup winning a tie. None means hold.

    A trade needs a yes, so it is weighed by its expected gain (its gain times the chance they
    accept, to one decimal of a point, as its row prints); a pickup needs no one and counts in
    full. ``Move.gain`` stays the full odds change, the number the card shows.

    A trade only counts when the partner's odds rise too, so the sample never shows a deal
    the other team gains nothing from.
    """
    trade = report.trades[0] if report.trades else None
    if trade is not None and not _partner_gains(trade):
        trade = None
    pickup = next((p for p in report.pickups if not p.open_spot_by_trade), None)
    trade_gain = trade.playoff_pct_after - trade.playoff_pct_before if trade else -1.0
    pickup_gain = pickup.playoff_pct_after - pickup.playoff_pct_before if pickup else -1.0
    expected = trade.acceptance * trade_gain if trade else -1.0
    if expected <= 0 and pickup_gain <= 0:
        return None
    if pickup is not None and pickup_gain > 0 and (expected <= 0 or _tenths(pickup_gain) >= _tenths(expected)):
        return Move("pickup", pickup_gain, pickup.playoff_pct_before)
    assert trade is not None
    return Move("trade", trade_gain, trade.playoff_pct_before)


def _tenths(gain: float) -> int:
    """An odds change (0-1) in tenths of a point, rounded half up like the web's Math.round."""
    return math.floor(gain * 1000 + 0.5)


def _partner_gains(trade) -> bool:
    before, after = trade.partner_playoff_pct_before, trade.partner_playoff_pct_after
    return before is not None and after is not None and after > before


def first_team(reports: Mapping[int, AnalysisReport]) -> int | None:
    """The team a visitor scouts first: the clearest move of the day.

    Teams whose best move gains at least 2 points (else at least 1), then the largest gain,
    then a pickup over a trade (no one has to agree), then odds nearest a coin flip, then the
    lowest team id so every visitor gets the same team. None when no team has such a move.
    """
    moves = {team_id: m for team_id, report in reports.items() if (m := best_move(report)) is not None}
    for floor in (CLEAR_GAIN, SMALL_GAIN):
        # Round so float noise (0.51 - 0.47) doesn't drop a move that is exactly on the floor.
        found = [(team_id, m) for team_id, m in moves.items() if round(m.gain, 6) >= floor]
        if found:
            found.sort(
                key=lambda tm: (
                    -round(tm[1].gain, 6),
                    tm[1].kind != "pickup",
                    abs(tm[1].odds_before - 0.5),
                    tm[0],
                )
            )
            return found[0][0]
    return None

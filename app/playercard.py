"""One player's card: his recent games, his season line and this league's moves of him.

ESPN's ``kona_playercard`` view returns all of it in one league call: a line for each of
his latest games (``statSplitTypeId`` 5), his season totals and averages (split 0) and
every executed transaction that touched him in this league (draft, waiver claims,
free-agent adds and drops, trades). Opponents and dates come from the NBA calendar.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.scores import Schedule, ScheduledGame

GAME_SPLIT = 5
SEASON_SPLIT = 0
GP = 42
# ESPN stat ids for the nine categories, their makes and attempts, and minutes.
STATS = {
    "minutes": 40,
    "pts": 0,
    "reb": 6,
    "ast": 3,
    "stl": 2,
    "blk": 1,
    "tpm": 17,
    "to": 11,
    "fgm": 13,
    "fga": 14,
    "ftm": 15,
    "fta": 16,
}
MOVE_KINDS = {"DRAFT": "draft", "ADD": "add", "DROP": "drop", "TRADE": "trade"}
# How the player moved on an add: off waivers (a claim) or straight from free agency.
VIA = {"WAIVER": "waiver", "FREEAGENT": "free_agent", "DRAFT": "draft"}
TRADE_TYPES = {"TRADE_ACCEPT", "TRADE_UPHOLD"}


@dataclass
class GameRow:
    game_id: int
    season: int
    day: int
    start_ms: int | None  # tip-off, epoch milliseconds; None when the calendar has no such game
    opponent_id: int  # pro team id, 0 when unknown
    home: bool | None
    stats: dict[str, float]


@dataclass
class SeasonLine:
    season: int
    games: int
    per_game: dict[str, float]
    fg_pct: float | None
    ft_pct: float | None


@dataclass
class CardMove:
    id: str
    kind: str  # draft / add / drop / trade
    via: str  # draft / waiver / free_agent / trade
    at_ms: int
    from_team_id: int  # 0 = free agency
    to_team_id: int
    bid: int | None = None  # FAAB on a waiver claim, price in an auction draft
    pick: int | None = None  # overall pick, for a draft


@dataclass
class Card:
    id: int
    name: str
    pro_team_id: int
    position_id: int
    injury_status: str
    on_team_id: int  # 0 = free agent
    status: str  # ONTEAM / FREEAGENT / WAIVERS
    percent_owned: float
    games: list[GameRow] = field(default_factory=list)
    season_line: SeasonLine | None = None
    moves: list[CardMove] = field(default_factory=list)


def _line(raw: dict) -> dict[str, float]:
    return {k: float(raw.get(str(sid), 0) or 0) for k, sid in STATS.items()}


def parse_card(entry: dict, season: int, schedules: dict[int, Schedule]) -> Card:
    """Turn one ``kona_playercard`` entry into a card.

    Only games he played (minutes > 0) are kept, newest first. ``schedules`` holds the
    NBA calendar by season, for opponents and tip-off times; a season without one still
    lists its games, without them.
    """
    p = entry.get("player") or {}
    pid = int(entry.get("id") or p.get("id") or 0)
    by_game: dict[int, ScheduledGame] = {
        g.id: g for sched in schedules.values() for gs in sched.games.values() for g in gs
    }

    games: list[GameRow] = []
    line: SeasonLine | None = None
    for st in p.get("stats") or []:
        split, source = st.get("statSplitTypeId"), st.get("statSourceId")
        if source:  # projections
            continue
        if split == GAME_SPLIT and st.get("externalId"):
            stats = _line(st.get("stats") or {})
            if stats["minutes"] <= 0:
                continue
            gid = int(st["externalId"])
            team = int(st.get("proTeamId") or p.get("proTeamId") or 0)
            game = by_game.get(gid)
            opp, home, start = 0, None, None
            if game is not None:
                home = game.home == team
                opp = game.away if home else game.home
                start = int(game.start.timestamp() * 1000)
            games.append(
                GameRow(
                    game_id=gid,
                    season=int(st.get("seasonId") or season),
                    day=int(st.get("scoringPeriodId") or 0),
                    start_ms=start,
                    opponent_id=opp,
                    home=home,
                    stats=stats,
                )
            )
        elif split == SEASON_SPLIT and st.get("seasonId") == season:
            total = st.get("stats") or {}
            avg = st.get("averageStats") or {}
            gp = int(total.get(str(GP), 0) or 0)
            if gp > 0:
                fga, fta = float(total.get("14", 0) or 0), float(total.get("16", 0) or 0)
                line = SeasonLine(
                    season=season,
                    games=gp,
                    per_game=_line(avg),
                    fg_pct=float(total.get("13", 0) or 0) / fga if fga else None,
                    ft_pct=float(total.get("15", 0) or 0) / fta if fta else None,
                )
    games.sort(key=lambda g: (g.season, g.day, g.start_ms or 0), reverse=True)

    own = p.get("ownership") or {}
    return Card(
        id=pid,
        name=str(p.get("fullName") or ""),
        pro_team_id=int(p.get("proTeamId") or 0),
        position_id=int(p.get("defaultPositionId") or 0),
        injury_status=str(p.get("injuryStatus") or "ACTIVE"),
        on_team_id=int(entry.get("onTeamId") or 0),
        status=str(entry.get("status") or ""),
        percent_owned=round(float(own.get("percentOwned") or 0), 1),
        games=games,
        season_line=line,
        moves=parse_moves(entry.get("transactions") or [], pid),
    )


def _bid(kind: str, via: str, bid) -> int | None:
    """FAAB paid on a waiver claim, or the price in an auction draft (ESPN puts 0 on a snake pick)."""
    if bid is None:
        return None
    if kind == "add" and via == "waiver":
        return int(bid)
    if kind == "draft" and bid:
        return int(bid)
    return None


def parse_moves(raw: list[dict], player_id: int) -> list[CardMove]:
    """Executed moves of this player in the league, newest first. Pending claims and lineup changes are left out."""
    out: list[CardMove] = []
    seen: set[str] = set()
    for t in raw:
        if t.get("status") != "EXECUTED" or t.get("id") in seen:
            continue
        ttype = str(t.get("type") or "")
        for i, item in enumerate(t.get("items") or []):
            if int(item.get("playerId") or 0) != player_id:
                continue
            kind = MOVE_KINDS.get(str(item.get("type") or ""))
            if kind is None:
                continue
            if kind == "trade" and ttype not in TRADE_TYPES:
                continue  # a proposal, not a done deal
            seen.add(str(t.get("id")))
            via = "trade" if kind == "trade" else VIA.get(ttype, "free_agent")
            bid = t.get("bidAmount")
            out.append(
                CardMove(
                    id=f"{t.get('id')}:{i}",
                    kind=kind,
                    via=via,
                    at_ms=int(t.get("processDate") or t.get("proposedDate") or 0),
                    from_team_id=int(item.get("fromTeamId") or 0),
                    to_team_id=int(item.get("toTeamId") or 0),
                    bid=_bid(kind, via, bid),
                    pick=int(item["overallPickNumber"]) if kind == "draft" and item.get("overallPickNumber") else None,
                )
            )
    out.sort(key=lambda m: m.at_ms, reverse=True)
    return out

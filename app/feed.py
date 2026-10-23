"""The league feed: ESPN's transaction log turned into moves people can react to.

ESPN returns transactions one scoring period (one day) at a time, including lineup
shuffles nobody cares about and the waiver bids that lost. This module keeps the
moves that change a roster (waiver claims, free-agent adds and drops, trades) and
attaches each losing bid to the claim that beat it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

ROSTER_MOVES = {"WAIVER": "waiver", "FREEAGENT": "free_agent"}
TRADE_TYPES = {"TRADE_ACCEPT", "TRADE_UPHOLD"}
# A losing bid is processed in the same waiver run as the winning one.
SAME_RUN_MS = 5 * 60 * 1000
# ESPN defaultPositionId, the position shown next to a name.
DEFAULT_POSITIONS = {1: "PG", 2: "SG", 3: "SF", 4: "PF", 5: "C"}


@dataclass
class Move:
    player_id: int
    kind: str  # add / drop / trade
    from_team_id: int  # 0 = free agency
    to_team_id: int


@dataclass
class LosingBid:
    team_id: int
    bid: int


@dataclass
class Transaction:
    id: str
    kind: str  # waiver / free_agent / trade
    team_id: int
    date_ms: int
    scoring_period: int
    bid: int | None
    moves: list[Move]
    losing_bids: list[LosingBid] = field(default_factory=list)

    @property
    def team_ids(self) -> list[int]:
        ids = [self.team_id]
        for m in self.moves:
            for t in (m.from_team_id, m.to_team_id):
                if t and t not in ids:
                    ids.append(t)
        return ids

    @property
    def player_ids(self) -> list[int]:
        return [m.player_id for m in self.moves]


def _date(t: dict) -> int:
    return int(t.get("processDate") or t.get("proposedDate") or 0)


def parse_transactions(raw: list[dict]) -> list[Transaction]:
    """Roster-changing transactions, newest first. ``raw`` may hold the same item twice."""
    by_id: dict[str, dict] = {}
    for t in raw:
        by_id.setdefault(t["id"], t)

    out: list[Transaction] = []
    failed: list[dict] = []
    seen_trades: set[str] = set()
    for t in by_id.values():
        kind, status = t.get("type"), t.get("status")
        items = t.get("items") or []
        if kind == "WAIVER" and status and status.startswith("FAILED"):
            failed.append(t)
            continue
        if status != "EXECUTED":
            continue
        if kind in ROSTER_MOVES:
            moves = [
                Move(int(i["playerId"]), i["type"].lower(), int(i.get("fromTeamId", 0)), int(i.get("toTeamId", 0)))
                for i in items
                if i.get("type") in ("ADD", "DROP")
            ]
            if not moves:
                continue
            bid = int(t.get("bidAmount", 0)) if kind == "WAIVER" else None
            out.append(
                Transaction(
                    t["id"], ROSTER_MOVES[kind], int(t["teamId"]), _date(t), int(t["scoringPeriodId"]), bid, moves
                )
            )
        elif kind in TRADE_TYPES:
            # An accepted trade can show up again when the league upholds it.
            key = t.get("relatedTransactionId") or t["id"]
            moves = [
                Move(int(i["playerId"]), "trade", int(i.get("fromTeamId", 0)), int(i.get("toTeamId", 0)))
                for i in items
                if i.get("type") == "TRADE"
            ]
            if key in seen_trades or not moves:
                continue
            seen_trades.add(key)
            out.append(Transaction(key, "trade", int(t["teamId"]), _date(t), int(t["scoringPeriodId"]), None, moves))

    claims = [tx for tx in out if tx.kind == "waiver"]
    for t in failed:
        added = {int(i["playerId"]) for i in t.get("items") or [] if i.get("type") == "ADD"}
        for tx in claims:
            won = {m.player_id for m in tx.moves if m.kind == "add"}
            if added & won and abs(tx.date_ms - _date(t)) <= SAME_RUN_MS and tx.team_id != int(t["teamId"]):
                tx.losing_bids.append(LosingBid(int(t["teamId"]), int(t.get("bidAmount", 0))))
                break
    for tx in claims:
        tx.losing_bids.sort(key=lambda b: -b.bid)

    out.sort(key=lambda tx: tx.date_ms, reverse=True)
    return out


@dataclass
class PlayerName:
    id: int
    name: str
    position: str
    pro_team_id: int


def parse_players(raw: list[dict]) -> dict[int, PlayerName]:
    out: dict[int, PlayerName] = {}
    for entry in raw:
        p = entry.get("player") or entry
        position = DEFAULT_POSITIONS.get(p.get("defaultPositionId", 0), "")
        out[int(p["id"])] = PlayerName(
            int(p["id"]), p.get("fullName", f"Player {p['id']}"), position, p.get("proTeamId", 0)
        )
    return out

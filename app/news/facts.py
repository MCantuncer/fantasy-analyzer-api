"""Hard facts from ESPN's player data: injury status and NBA team.

Each refresh takes a snapshot of every relevant player and compares it with the
last one. A change is a fact event: a status change ("OUT" to "ACTIVE") or a team
change (a trade, a signing, a release). Facts confirm or contradict what outlets
report. The first snapshot only sets the baseline, so a fresh server does not
announce every injured player as news.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from app.news.extract import Kind

FREE_AGENT_TEAM = 0
HEALTHY = {"ACTIVE", "NORMAL", ""}
STATUS_SEVERITY = {
    "ACTIVE": 0,
    "NORMAL": 0,
    "PROBABLE": 0,
    "DAY_TO_DAY": 1,
    "QUESTIONABLE": 1,
    "DOUBTFUL": 2,
    "OUT": 2,
    "SUSPENSION": 2,
    "INJURY_RESERVE": 3,
}
STATUS_LABEL = {
    "DAY_TO_DAY": "day-to-day",
    "QUESTIONABLE": "questionable",
    "DOUBTFUL": "doubtful",
    "OUT": "out",
    "SUSPENSION": "suspended",
    "INJURY_RESERVE": "out long term",
    "PROBABLE": "probable",
}


@dataclass
class PlayerState:
    id: int
    name: str
    pro_team_id: int
    status: str

    @property
    def injured(self) -> bool:
        return self.status not in HEALTHY and STATUS_SEVERITY.get(self.status, 1) > 0


def parse_states(players: list[dict]) -> dict[int, PlayerState]:
    out = {}
    for entry in players:
        p = entry.get("player") or entry
        if "id" not in p:
            continue
        out[int(p["id"])] = PlayerState(
            id=int(p["id"]),
            name=p.get("fullName", ""),
            pro_team_id=int(p.get("proTeamId") or 0),
            status=str(p.get("injuryStatus") or "ACTIVE"),
        )
    return out


@dataclass
class FactEvent:
    id: str
    player_id: int
    kind: Kind
    at: datetime
    old: str
    new: str
    severity: int = 0


def diff(before: dict[int, PlayerState], after: dict[int, PlayerState], now: datetime) -> list[FactEvent]:
    stamp = now.strftime("%Y%m%d%H%M")
    out = []
    for pid, new in after.items():
        old = before.get(pid)
        if old is None:
            continue  # new to the list: no baseline to compare with
        if old.pro_team_id != new.pro_team_id:
            if old.pro_team_id == FREE_AGENT_TEAM:
                kind: Kind = "signing"
            elif new.pro_team_id == FREE_AGENT_TEAM:
                kind = "waive"
            else:
                kind = "trade"
            out.append(
                FactEvent(f"fact:{pid}:team:{stamp}", pid, kind, now, str(old.pro_team_id), str(new.pro_team_id))
            )
        if old.status != new.status:
            if new.injured:
                kind = "injury"
            elif old.injured:
                kind = "return"
            else:
                continue
            out.append(
                FactEvent(
                    f"fact:{pid}:status:{stamp}",
                    pid,
                    kind,
                    now,
                    old.status,
                    new.status,
                    STATUS_SEVERITY.get(new.status, 1),
                )
            )
    return out

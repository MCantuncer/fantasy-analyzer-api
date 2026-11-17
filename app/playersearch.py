"""Every NBA player, small enough to search on the device as each key is pressed.

ESPN's ``players_wl`` season view lists every player this season (about 1,100) with only his
name, NBA team, position and how widely he is rostered across ESPN: about 260 KB for all of
them, against megabytes from the league's ``kona_player_info``. It is the same for every
league, so it is kept for hours. Who has him in a league comes from that league's rosters, and
injury tags from the news service's last look at the most-owned players.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from app.espn.parse import PRO_TEAM_ABBREV
from app.feed import DEFAULT_POSITIONS


@dataclass(frozen=True)
class IndexPlayer:
    id: int
    name: str
    pro_team: str  # "" when he is on no NBA team
    position: str
    percent_owned: float
    news_at: datetime | None = None  # ESPN's last news on him


# ESPN keeps long-gone players on their old NBA team (Memphis lists 39, Jarnell Stokes among them).
# A player on a team today has had news within a few months: the 30 teams then list 15 to 19.
CURRENT_NEWS = timedelta(days=120)


def on_roster(p: IndexPlayer, now: datetime, owned: bool = False) -> bool:
    """On his NBA team's roster today: rostered in the league, or in ESPN's news lately."""
    if not p.pro_team:
        return False
    return owned or (p.news_at is not None and now - p.news_at <= CURRENT_NEWS)


def _news_at(ms: object) -> datetime | None:
    return datetime.fromtimestamp(ms / 1000, UTC) if isinstance(ms, int | float) and ms > 0 else None


def parse_index(raw: list[dict]) -> list[IndexPlayer]:
    """Players from ``players_wl``, most rostered first; nameless rows and repeats are left out."""
    out: dict[int, IndexPlayer] = {}
    for p in raw:
        pid, name = p.get("id"), (p.get("fullName") or "").strip()
        if not isinstance(pid, int) or not name or pid in out:
            continue
        out[pid] = IndexPlayer(
            id=pid,
            name=name,
            pro_team=PRO_TEAM_ABBREV.get(p.get("proTeamId") or 0, ""),
            position=DEFAULT_POSITIONS.get(p.get("defaultPositionId") or 0, ""),
            percent_owned=round(float((p.get("ownership") or {}).get("percentOwned") or 0.0), 1),
            news_at=_news_at(p.get("lastNewsDate")),
        )
    return sorted(out.values(), key=lambda x: (-x.percent_owned, x.name))


HEALTHY = {"", "ACTIVE", "NORMAL"}


def injury_tag(status: str | None) -> str | None:
    """His ESPN injury status, or None when he has no tag (ESPN writes healthy as ACTIVE or NORMAL)."""
    return None if (status or "") in HEALTHY else status

"""Turns tagged items from every source into checked stories.

Items about the same player and the same kind of news (health, a move, a role)
within two days become one story. A story's status:

- ``confirmed``: ESPN's player data agrees (an injury status, a team change), or
  outlets from two different groups report it.
- ``reported``: one outlet group only.
- ``held``: only tips (Reddit). Hidden unless asked for.
- ``didnt_happen``: a trade reported as done, but the player's team has not
  changed three days later.

When sources disagree on how bad an injury is, the story says so and shows
each version; its severity is the worst one reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal

from app.news.extract import SEVERITY_LABEL, newsworthy, normalize
from app.news.facts import STATUS_LABEL, STATUS_SEVERITY, PlayerState
from app.news.sources import BY_ID

Status = Literal["confirmed", "reported", "held", "didnt_happen"]
FAMILY = {
    "injury": "health",
    "return": "health",
    "trade": "move",
    "rumor": "move",
    "signing": "move",
    "waive": "move",
    "role": "role",
    "other": "other",
}
JOIN_WINDOW = timedelta(hours=48)
TITLE_WINDOW = timedelta(hours=24)
TRADE_PROOF_WINDOW = timedelta(hours=72)
SAME_TITLE = 0.5  # word overlap for two headlines about nobody in particular to count as one story
FACT_SOURCE = "espn_players"
MOVES = ("trade", "signing", "waive")


@dataclass
class Item:
    """One stored item: a feed entry or a fact event, already tagged."""

    id: str
    source_id: str
    title: str
    url: str
    published: datetime
    summary: str
    kind: str
    player_ids: list[int]
    severity: int = 0
    body_part: str | None = None
    old: str = ""  # fact events: the value before
    new: str = ""  # and after

    @property
    def source_kind(self) -> str:
        return BY_ID[self.source_id].kind if self.source_id in BY_ID else "tip"


@dataclass
class Story:
    id: str
    kind: str
    status: Status
    headline: str
    summary: str
    published: datetime
    updated: datetime
    player_ids: list[int]
    items: list[Item]
    severity: int = 0
    conflict: str | None = None


@dataclass
class _Cluster:
    family: str
    items: list[Item] = field(default_factory=list)
    players: list[int] = field(default_factory=list)

    @property
    def last(self) -> datetime:
        return self.items[-1].published


def _words(title: str) -> set[str]:
    return {w for w in normalize(title).split() if len(w) > 2}


def _overlap(a: str, b: str) -> float:
    wa, wb = _words(a), _words(b)
    return len(wa & wb) / len(wa | wb) if wa and wb else 0.0


def cluster(items: list[Item]) -> list[_Cluster]:
    out: list[_Cluster] = []
    for item in sorted(items, key=lambda i: (i.published, i.id)):
        family = FAMILY.get(item.kind, "other")
        home = None
        for c in reversed(out):
            if c.family != family:
                continue
            if item.player_ids:
                if set(item.player_ids) & set(c.players) and item.published - c.last <= JOIN_WINDOW:
                    home = c
                    break
            elif not c.players and item.published - c.last <= TITLE_WINDOW:
                if _overlap(item.title, c.items[0].title) >= SAME_TITLE:
                    home = c
                    break
        if home is None:
            home = _Cluster(family)
            out.append(home)
        home.items.append(item)
        home.players.extend(p for p in item.player_ids if p not in home.players)
    return out


def describe_fact(item: Item, names: dict[int, str], teams: dict[int, str]) -> str:
    name = names.get(item.player_ids[0], "A player") if item.player_ids else "A player"

    def team(raw: str) -> str:
        return teams.get(int(raw), "a new team") if raw.isdigit() else raw

    if item.kind == "injury":
        return f"ESPN lists {name} as {STATUS_LABEL.get(item.new, item.new.lower().replace('_', ' '))}"
    if item.kind == "return":
        return f"ESPN lists {name} as active again"
    if item.kind == "trade":
        return f"{name} moves from {team(item.old)} to {team(item.new)} in ESPN's player data"
    if item.kind == "signing":
        return f"{name} joins {team(item.new)} in ESPN's player data"
    if item.kind == "waive":
        return f"{name} is no longer on {team(item.old)} in ESPN's player data"
    return f"ESPN updated {name}"


def _story_kind(c: _Cluster, trusted: list[Item]) -> str:
    pool = trusted or c.items
    facts = [i for i in pool if i.source_id == FACT_SOURCE]
    if c.family == "move":
        moves = [i for i in facts if i.kind in MOVES]
        if moves:
            return moves[-1].kind
        claimed = [i.kind for i in pool if i.kind in MOVES]
        return claimed[-1] if claimed else "rumor"
    return pool[-1].kind


def _injury_views(c: _Cluster, states: dict[int, PlayerState]) -> list[tuple[str, int, str | None]]:
    """(source name, severity, body part) for every source that says how bad an injury is."""
    views = []
    for i in c.items:
        if i.kind != "injury" or i.source_kind == "tip":
            continue
        if i.source_id == FACT_SOURCE:
            views.append(("ESPN status", i.severity, None))
        elif i.severity > 0:
            views.append((BY_ID[i.source_id].name, i.severity, i.body_part))
    if not any(v[0] == "ESPN status" for v in views):
        for pid in c.players[:1]:
            st = states.get(pid)
            if st is not None and st.injured:
                views.append(("ESPN status", STATUS_SEVERITY.get(st.status, 1), None))
    return views


def _conflict(views: list[tuple[str, int, str | None]], states: dict[int, PlayerState], pid: int | None) -> str | None:
    if len(views) < 2:
        return None
    sevs = [v[1] for v in views]
    parts = {v[2] for v in views if v[2]}
    if max(sevs) - min(sevs) < 2 and len(parts) < 2:
        return None
    seen: dict[str, str] = {}
    for name, sev, part in views:
        if name == "ESPN status" and pid is not None and pid in states:
            label = STATUS_LABEL.get(states[pid].status, SEVERITY_LABEL[sev])
        else:
            label = SEVERITY_LABEL[sev] + (f" ({part})" if part else "")
        seen.setdefault(name, label)
    return "Sources disagree: " + "; ".join(f"{k} says {v}" for k, v in seen.items()) + "."


def build_stories(
    items: list[Item], states: dict[int, PlayerState], teams: dict[int, str], now: datetime
) -> list[Story]:
    names = {pid: st.name for pid, st in states.items()}
    stories = []
    # Only NBA news makes a story: no game guides, betting, recaps, rankings or other leagues.
    items = [i for i in items if i.source_id == FACT_SOURCE or newsworthy(i.title)]
    for c in cluster(items):
        trusted = [i for i in c.items if i.source_kind != "tip"]
        outlets = [i for i in trusted if i.source_kind == "outlet"]
        facts = [i for i in trusted if i.source_id == FACT_SOURCE]
        groups = {BY_ID[i.source_id].group for i in outlets}
        kind = _story_kind(c, trusted)
        pid = c.players[0] if c.players else None

        fact_agrees = any(f.kind == kind for f in facts)
        if kind == "injury" and pid is not None and pid in states and states[pid].injured:
            fact_agrees = True

        if fact_agrees or len(groups) >= 2 or (facts and not outlets):
            status: Status = "confirmed"
        elif groups:
            status = "reported"
        else:
            status = "held"
        if (
            kind == "trade"
            and not any(f.kind in MOVES for f in facts)
            and pid in states
            and now - c.items[0].published >= TRADE_PROOF_WINDOW
        ):
            status = "didnt_happen"

        views = _injury_views(c, states) if c.family == "health" else []
        lead = outlets[-1] if outlets else (facts[-1] if facts else c.items[-1])
        headline = describe_fact(lead, names, teams) if lead.source_id == FACT_SOURCE else lead.title
        stories.append(
            Story(
                id=c.items[0].id,
                kind=kind,
                status=status,
                headline=headline,
                summary=lead.summary,
                published=c.items[0].published,
                updated=c.last,
                player_ids=c.players[:3],
                items=c.items,
                severity=max((v[1] for v in views), default=0),
                conflict=_conflict(views, states, pid) if kind == "injury" else None,
            )
        )
    stories.sort(key=lambda s: (s.updated, s.id), reverse=True)
    return stories

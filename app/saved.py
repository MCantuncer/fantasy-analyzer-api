"""Saved analysis reports: show the last one at once while it is still right, else say what changed.

A report is only as good as the rosters and injury news it was built on. Each saved report
keeps a small fingerprint of the league at the time: every rostered player, his injury
status, whether he sits in an IR slot and how many games he has played. When the league's
fingerprint still matches (and the report isn't too old) the saved report is served as it
is. When it doesn't, a fresh run starts and the changes are listed in plain words, so urgent
news ("Tatum is out") shows up right away instead of after the next run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from app import schemas
from app.espn.parse import IR_SLOT, League

# A saved report older than this is rebuilt even when nothing in the league moved: the
# days left in the matchup and the season shift under it.
SAVED_FRESH = timedelta(hours=6)

STATUS_WORDS = {
    "OUT": "is out",
    "DAY_TO_DAY": "is day-to-day",
    "INJURY_RESERVE": "is on injured reserve",
    "SUSPENSION": "is suspended",
}
HEALTHY = {"ACTIVE", "NORMAL"}


def report_key(key: tuple) -> str:
    """The analysis cache key (league, season, team, keep, not for sale, cookies hash) as text."""
    return json.dumps([list(k) if isinstance(k, tuple) else k for k in key])


def key_from_text(text: str) -> tuple:
    """``report_key`` back to the analysis cache key."""
    league_id, season, team_id, keep, not_for_sale, creds = json.loads(text)
    return (league_id, season, team_id, tuple(keep), tuple(not_for_sale), creds)


def touches(print_: dict, player_ids: set[int]) -> bool:
    """Whether any of these players is on a roster in this fingerprint."""
    wanted = {str(p) for p in player_ids}
    return any(wanted & t["players"].keys() for t in print_.get("teams", {}).values())


def fingerprint(league: League) -> dict:
    """What a report depends on that can change between runs, small enough to keep beside it."""
    return {
        "teams": {
            str(t.id): {
                "name": t.name,
                "players": {
                    str(p.id): [p.name, p.injury_status, p.lineup_slot == IR_SLOT, p.games_played] for p in t.roster
                },
            }
            for t in league.teams
        }
    }


@dataclass
class Saved:
    report: schemas.AnalysisReport
    fingerprint: dict
    built_at: datetime

    def fresh(self, now: datetime) -> bool:
        return now - self.built_at < SAVED_FRESH


def read_saved(row: dict | None) -> Saved | None:
    """A stored row back as a report, or None when it is missing or from an older report shape."""
    if not row:
        return None
    try:
        return Saved(
            report=schemas.AnalysisReport.model_validate(row["body"]),
            fingerprint=row["fingerprint"],
            built_at=datetime.fromisoformat(row["built_at"]).astimezone(UTC),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _status_text(name: str, old: str, new: str) -> str | None:
    if new in STATUS_WORDS:
        return f"{name} {STATUS_WORDS[new]}"
    if new in HEALTHY and old not in HEALTHY:
        return f"{name} is healthy again"
    return None


def changes(old: dict, new: dict, team_id: int) -> list[schemas.ReportChange]:
    """What moved in the league between two fingerprints, your team's news first."""
    mine = str(team_id)
    where_old = {pid: tid for tid, t in old.get("teams", {}).items() for pid in t["players"]}
    where_new = {pid: tid for tid, t in new.get("teams", {}).items() for pid in t["players"]}
    team_name = {tid: t["name"] for tid, t in {**old.get("teams", {}), **new.get("teams", {})}.items()}
    out: list[tuple[int, schemas.ReportChange]] = []
    games = 0

    def add(rank: int, kind: str, pid: str, text: str, *teams: str | None) -> None:
        ids = [t for t in teams if t is not None]
        out.append(
            (
                rank,
                schemas.ReportChange(
                    kind=kind,
                    text=text,
                    player_id=int(pid),
                    team_id=int(ids[-1]),
                    mine=mine in ids,
                ),
            )
        )

    for pid in sorted(set(where_old) | set(where_new), key=int):
        was, now = where_old.get(pid), where_new.get(pid)
        before = old["teams"][was]["players"][pid] if was else None
        after = new["teams"][now]["players"][pid] if now else None
        name = (after or before)[0]
        if was != now:
            if was and now:
                add(2, "moved", pid, f"{name} went from {team_name[was]} to {team_name[now]}", was, now)
            elif now:
                add(2, "added", pid, f"{team_name[now]} added {name}", now)
            else:
                add(2, "dropped", pid, f"{team_name[was]} dropped {name}", was)
        if before and after:
            games += max(0, after[3] - before[3])
            text = _status_text(name, before[1], after[1]) if before[1] != after[1] else None
            if text is None and before[2] != after[2]:
                text = f"{name} moved {'to' if after[2] else 'off'} IR"
            if text:
                add(1, "status", pid, text, now)
    ordered = [c for _, c in sorted(out, key=lambda rc: (not rc[1].mine, rc[0]))]
    if games:
        ordered.append(
            schemas.ReportChange(
                kind="games",
                text=f"New stats from {games} player game{'s' if games != 1 else ''}",
                player_id=None,
                team_id=None,
                mine=False,
            )
        )
    return ordered


def saved_row(report: schemas.AnalysisReport, league: League, built_at: datetime) -> dict:
    return {
        "body": report.model_dump(mode="json"),
        "fingerprint": fingerprint(league),
        "built_at": built_at.astimezone(UTC).isoformat(),
    }

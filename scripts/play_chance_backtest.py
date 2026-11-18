"""Builds and checks the play-chance tables in app/playchance.py from five seasons of NBA injury reports.

    uv run --with pandas --with pyarrow python -m scripts.play_chance_backtest

The NBA's official injury report (every version a team filed: three a day until Oct 2021, hourly
after, every 15 minutes since Dec 2025) comes from sportsdataverse's public archive, one parquet
file per season. Who played comes from ESPN's game lines (minutes > 0). The two are joined by the
player's name and the game's date (New York time). Downloads are cached in .cache/playchance.

For each listed player and game it reads his tag as it stood 24, 8, 5, 3, 1.5 and 0.5 hours before
tip-off (a player dropped from his team's latest report counts as cleared) and his minutes a game
over his last ten games. Each candidate table is then tested blind: every season against a table
made from the other four, by log loss. A signal counts only if it beats the simpler table in every
season (Can's rule). It prints the tables app/playchance.py holds.
"""

from __future__ import annotations

import json
import re
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import numpy as np
import pandas as pd

from app.analysis.injury import parse_gamelogs

SEASONS = (2022, 2023, 2024, 2025, 2026)  # ESPN's season ids: 2021-22 to 2025-26
FORWARD = 2027  # the season under way, a forward check
HOURS = (24.0, 8.0, 5.0, 3.0, 1.5, 0.5)
CACHE = Path(".cache/playchance")
REPORTS = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/nba_stats_injury_report/injury_report_{season}.parquet"
ESPN = "https://lm-api-reads.fantasy.espn.com/apis/v3/games/fba/seasons/{season}"
NEW_YORK = ZoneInfo("America/New_York")
PRIOR = 20  # pseudo-games a small cell borrows from the tag and hours alone


def _get(url: str, path: Path, **kw) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        resp = httpx.get(url, timeout=120, follow_redirects=True, **kw)
        resp.raise_for_status()
        path.write_bytes(resp.content)
    return path


def norm(name: str) -> str:
    if "," in name:
        last, first = name.split(",", 1)
        name = f"{first.strip()} {last.strip()}"
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    name = re.sub(r"[.'’\-]", " ", name)
    return " ".join(t for t in name.split() if t not in ("jr", "sr", "ii", "iii", "iv", "v"))


def tip_minutes(text: str | None) -> int:
    """Minutes after midnight of a "7:30 (ET)" start; the report leaves out am/pm, and games run noon to 10:30 pm."""
    h, m = (int(x) for x in (text or "7:00").split(" ")[0].split(":"))
    return (h + 12 if h <= 10 else h) * 60 + m


def season_rows(season: int) -> pd.DataFrame:
    """Every report line of one season, with the player's ESPN id, the day, whether he played,
    his minutes over his last ten games and the team games he had missed in a row."""
    sched = json.loads(
        _get(ESPN.format(season=season) + "?view=proTeamSchedules_wl", CACHE / f"sched-{season}.json").read_text()
    )
    date_day, team_days, team_ids = {}, {}, {}
    for t in sched["settings"]["proTeams"]:
        team_ids[f"{t['location']} {t['name']}".lower().replace("la clippers", "los angeles clippers")] = t["id"]
        team_days[t["id"]] = sorted(int(d) for d in t.get("proGamesByScoringPeriod") or {})
        for d, games in (t.get("proGamesByScoringPeriod") or {}).items():
            for g in games:
                date_day[datetime.fromtimestamp(g["date"] / 1000, UTC).astimezone(NEW_YORK).date().isoformat()] = int(d)
    minutes: dict[tuple[int, int], float] = {}
    names: dict[str, int] = {}
    twice: set[str] = set()
    for team in range(1, 31):
        flt = {
            "filterProTeamIds": {"value": [team]},
            "filterStatsForSplitTypeIds": {"value": [5]},
            "limit": 60,
            "sortPercOwned": {"sortPriority": 1, "sortAsc": False},
        }
        path = _get(
            ESPN.format(season=season) + "/players?view=kona_player_info&scoringPeriodId=0",
            CACHE / f"logs-{season}-{team}.json",
            headers={"X-Fantasy-Filter": json.dumps(flt)},
        )
        lines, info = parse_gamelogs(json.loads(path.read_text()), season=season)
        for ln in lines:
            minutes[(ln.player_id, ln.period)] = ln.minutes
        for pid, who in info.items():
            key = norm(who.name)
            if names.get(key, pid) != pid:
                twice.add(key)
            names[key] = pid
    last_day = max(d for _, d in minutes)

    df = pd.read_parquet(_get(REPORTS.format(season=season), CACHE / f"reports-{season}.parquet"))
    df = df[df.report_player_name.notna() & ~df.reason.fillna("").str.contains("G League|Two-Way", case=False)]
    df["espn"] = df.report_player_name.map(lambda n: None if norm(n) in twice else names.get(norm(n)))
    df["day"] = df.game_date.astype(str).map(date_day)
    df = df[df.espn.notna() & df.day.notna() & (df.day <= last_day)].copy()
    df["espn"], df["day"] = df.espn.astype(int), df.day.astype(int)
    df["played"] = [(e, d) in minutes for e, d in zip(df.espn, df.day, strict=True)]
    df["tip"] = pd.to_datetime(df.game_date.astype(str)) + pd.to_timedelta(df.game_time.map(tip_minutes), unit="m")
    df = df[df.report_datetime < df.tip]
    df["hours"] = (df.tip - df.report_datetime).dt.total_seconds() / 3600
    df["season"] = season

    keys = df[["espn", "day", "team"]].drop_duplicates()
    avg, run = [], []
    for e, d, team in zip(keys.espn, keys.day, keys.team, strict=True):
        before = [x for x in team_days.get(team_ids.get(team.lower(), 0), []) if x < d]
        got = [minutes[(e, x)] for x in before[-10:] if (e, x) in minutes]
        avg.append(float(np.mean(got)) if got else 0.0)
        n = 0
        for x in reversed(before):
            if (e, x) in minutes:
                break
            n += 1
        nxt = [x for x in team_days.get(team_ids.get(team.lower(), 0), []) if x > d][:1]
        run.append((n, (e, nxt[0]) in minutes if nxt else np.nan))
    keys["minutes"] = avg
    keys["missed"] = [r[0] for r in run]
    keys["next_played"] = [r[1] for r in run]
    return df.merge(keys, on=["espn", "day", "team"])


def snapshots(df: pd.DataFrame) -> pd.DataFrame:
    """Each player's tag at each of HOURS before tip: his line in the latest report filed by then,
    or "Cleared" when his team filed a later report without him."""
    out = []
    for h in HOURS:
        sub = df[df.hours >= h]
        team_last = sub.groupby(["game_id", "team"]).report_datetime.max().rename("team_last")
        last = sub.sort_values("report_datetime").groupby(["season", "game_id", "espn"]).tail(1)
        last = last.join(team_last, on=["game_id", "team"])
        last["tag"] = np.where(last.report_datetime < last.team_last, "Cleared", last.current_status)
        last["h"] = h
        out.append(last)
    snap = pd.concat(out)
    snap["role"] = pd.cut(snap.minutes, [-1, 15, 25, 999], labels=["bench", "rotation", "starter"]).astype(str)
    return snap


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 0.01, 0.99)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def table(train: pd.DataFrame, test: pd.DataFrame, keys: list[str], base_keys: list[str], target: str) -> np.ndarray:
    base = train.groupby(base_keys)[target].mean().rename("base")
    cells = train.groupby(keys)[target].agg(["sum", "size"])
    t = test.join(base, on=base_keys).join(cells, on=keys)
    t["base"] = t["base"].fillna(train[target].mean())
    return ((t["sum"].fillna(0) + PRIOR * t["base"]) / (t["size"].fillna(0) + PRIOR)).to_numpy()


def blind(rows: pd.DataFrame, target: str, candidates: dict[str, list[str]], base_keys: list[str]) -> pd.DataFrame:
    out = []
    for season in SEASONS:
        train, test = rows[rows.season != season], rows[rows.season == season]
        y = test[target].to_numpy(dtype=float)
        r = {"season": season, "n": len(test), "flat": log_loss(np.full(len(test), train[target].mean()), y)}
        for name, keys in candidates.items():
            r[name] = log_loss(table(train, test, keys, base_keys, target), y)
        out.append(r)
    return pd.DataFrame(out).round(4)


def main() -> None:
    rows = pd.concat(season_rows(s) for s in (*SEASONS, FORWARD))
    snap = snapshots(rows)
    past = snap[snap.season.isin(SEASONS)]

    tagged = past[past.tag.isin(["Probable", "Questionable", "Doubtful"])]
    tagged = tagged.assign(played=tagged.played.astype(float), missed_last=tagged.missed > 0)
    print("Probable / questionable / doubtful: who played, blind by season (log loss, lower is better)")
    candidates = {
        "tag+hours": ["tag", "h"],
        "+role": ["tag", "h", "role"],
        "+missed last game": ["tag", "h", "missed_last"],
    }
    print(blind(tagged, "played", candidates, ["tag", "h"]).to_string())
    now = snap[(snap.season == FORWARD) & snap.tag.isin(["Probable", "Questionable", "Doubtful"])]
    y = now.played.to_numpy(dtype=float)
    print(
        f"\n{FORWARD - 1}-{FORWARD % 100} so far ({len(now)} listings): tag+hours "
        f"{log_loss(table(tagged, now, ['tag', 'h'], ['tag', 'h'], 'played'), y):.4f}, +role "
        f"{log_loss(table(tagged, now, ['tag', 'h', 'role'], ['tag', 'h'], 'played'), y):.4f}"
    )

    print("\nPLAY tables (share who played), rows by hours before tip, columns bench / rotation / starter:")
    for tag in ("Probable", "Questionable", "Doubtful", "Available", "Cleared"):
        t = past[past.tag == tag].pivot_table(index="h", columns="role", values="played", aggfunc="mean")
        print(tag, t[["bench", "rotation", "starter"]].sort_index(ascending=False).round(2).to_string(), sep="\n")
    t = tagged.pivot_table(index="h", columns="role", values="played", aggfunc="mean")
    print(
        "Day-to-day (any of the three)",
        t[["bench", "rotation", "starter"]].sort_index(ascending=False).round(2).to_string(),
        sep="\n",
    )

    out = past[(past.h == 1.5) & (past.tag == "Out") & (past.minutes >= 20) & past.next_played.notna()].copy()
    out["next_played"] = out.next_played.astype(float)
    out["streak"] = pd.cut(out.missed + 1, [0, 1, 3, 6, 999], labels=["1", "2-3", "4-6", "7+"]).astype(str)
    print("\nOut, 20+ minutes: plays the next game, by games missed in a row (blind by season)")
    print(blind(out, "next_played", {"streak": ["streak"]}, ["streak"]).to_string())
    print(out.pivot_table(index="streak", columns="season", values="next_played", aggfunc="mean").round(2).to_string())
    print(out.groupby("streak").next_played.agg(["mean", "size"]).round(3).to_string())


if __name__ == "__main__":
    main()

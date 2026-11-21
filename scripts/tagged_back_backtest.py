"""Replays tagged comebacks (tonight.came_back_tagged) the way the app sees them.

    uv run --with pandas --with pyarrow python -m scripts.tagged_back_backtest [2027]

For every case of scripts/next_game_matchup_backtest.py whose team has a regular back from an absence
(15+ minutes a game, missed its last game or more) listed probable or questionable on the NBA injury
report 3 hours before tip-off (scripts/play_chance_backtest.py), whether he then played or not: his
teammates' lines without him as back (the app before) and with him as back by his day-to-day chance to
play, minutes only (gamebox.Teammate.chance). Printed per season: % less squared miss in each category,
in those games, and split by whether he played.
"""

from __future__ import annotations

import sys

import numpy as np

from app.analysis import gamebox
from app.analysis.tonight import TAGGED_BACK_HOURS
from app.playchance import DAY_TO_DAY, play_chance
from scripts.next_game_backtest import season_games
from scripts.next_game_matchup_backtest import CATS, SEASONS, cases, cats
from scripts.play_chance_backtest import season_rows

TAGS = ("Probable", "Questionable")


def tags_before(season: int, hours: float) -> dict[tuple[int, int], str]:
    """(player, day) -> his tag in the last report filed ``hours`` or more before tip-off ("Cleared" when
    his team filed a later one without him)."""
    df = season_rows(season)
    sub = df[df.hours >= hours].sort_values("report_datetime")
    team_last = sub.groupby(["game_id", "team"]).report_datetime.max().rename("team_last")
    last = sub.groupby(["game_id", "espn"]).tail(1).join(team_last, on=["game_id", "team"])
    return {
        (int(r.espn), int(r.day)): "Cleared" if r.report_datetime < r.team_last else r.current_status
        for r in last.itertuples(index=False)
    }


def season_lines(season: int) -> dict[str, np.ndarray]:
    tags = tags_before(season, TAGGED_BACK_HOURS)
    went: dict[tuple[int, int], set[int]] = {}
    for pid, gl in season_games(season).items():
        for _gid, (day, team, _) in gl.items():
            went.setdefault((day, team), set()).add(pid)
    out: dict[str, list] = {"before": [], "after": [], "actual": [], "played": []}
    for box, c in cases(season):
        playing = went.get((c.day, c.team), set())
        back, tagged, played = [], [], False
        for q, f in box.form.items():
            if q == c.player or f.team != c.team or f.minutes_long < gamebox.BACK_MINUTES:
                continue
            missed = box.games_missed(q, c.team, c.day)
            if missed < 1:
                continue
            mate = gamebox.Teammate("", f.minutes_long, f.usage, f.rate.get(gamebox.PTS, 0.0) * f.minutes_long, missed)
            if tags.get((q, c.day)) in TAGS:
                chance = play_chance(DAY_TO_DAY, TAGGED_BACK_HOURS, f.minutes_long)
                tagged.append(gamebox.Teammate(mate.name, mate.minutes, mate.usage, mate.points, missed, chance=chance))
                played = played or q in playing
            elif q in playing:
                back.append(mate)
        if not tagged:
            continue
        out["before"].append(cats(gamebox.apply(c.line, c.context, c.mates, c.gained, back, c.absent)[0]))
        out["after"].append(cats(gamebox.apply(c.line, c.context, c.mates, c.gained, back + tagged, c.absent)[0]))
        out["actual"].append(cats(c.actual))
        out["played"].append(played)
    return {k: np.array(v) for k, v in out.items()}


def gain(d: dict[str, np.ndarray], c: int, keep: np.ndarray) -> float:
    e0 = np.mean((d["actual"][keep, c] - d["before"][keep, c]) ** 2)
    e1 = np.mean((d["actual"][keep, c] - d["after"][keep, c]) ** 2)
    return (e0 - e1) / e0 * 100


def main() -> None:
    seasons = [*SEASONS, *(int(a) for a in sys.argv[1:])]
    per = {s: season_lines(s) for s in seasons}
    print("games with a tagged comeback: " + ", ".join(f"{s}: {len(d['actual'])}" for s, d in per.items()))
    print("% less squared miss, by season")
    for c, name in enumerate(CATS):
        print(f"  {name:4s} " + " ".join(f"{gain(d, c, np.ones(len(d['actual']), bool)):+.2f}%" for d in per.values()))
    for label, played in (("he played", True), ("he sat", False)):
        print(f"  MIN, {label}: " + " ".join(f"{gain(d, 0, d['played'] == played):+.2f}%" for d in per.values()))


if __name__ == "__main__":
    main()

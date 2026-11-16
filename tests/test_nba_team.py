from datetime import UTC, datetime, timedelta

from app.nbateam import played_days, result_of, team_id_of, team_season
from app.scores import Schedule, ScheduledGame

DEN, LAL, BOS = 7, 13, 2
TIP = datetime(2026, 11, 1, 0, 0, tzinfo=UTC)


def _schedule() -> Schedule:
    teams = {DEN: ("DEN", "Denver Nuggets"), LAL: ("LAL", "Los Angeles Lakers"), BOS: ("BOS", "Boston Celtics")}
    games = {
        d: [ScheduledGame(100 + d, d, TIP + timedelta(days=d), DEN if d % 2 else LAL, BOS if d % 2 else DEN, True)]
        for d in range(1, 6)
    }
    games[6] = [ScheduledGame(106, 6, TIP + timedelta(days=6), DEN, LAL, False)]
    return Schedule(teams=teams, games=games)


def _line(pid: int, team: int, pts: int, reb: int = 0, ast: int = 0) -> dict:
    return {"player_id": pid, "name": f"P{pid}", "pro_team_id": team, "pts": pts, "reb": reb, "ast": ast}


def _box(sched: Schedule, day: int, away_pts: int, home_pts: int) -> dict:
    g = sched.games[day][0]
    side = {g.away: away_pts, g.home: home_pts}
    return {
        "day": day,
        "date": f"2026-11-0{day}",
        "games": [
            {
                "id": g.id,
                "start": g.start.isoformat(),
                "status": "final",
                "away": {"pro_team_id": g.away, "abbrev": sched.teams[g.away][0], "score": away_pts},
                "home": {"pro_team_id": g.home, "abbrev": sched.teams[g.home][0], "score": home_pts},
                "lines": [_line(t * 10, t, pts - 80, 5) for t, pts in side.items()]
                + [_line(t * 10 + 1, t, pts - 90, 9, 9) for t, pts in side.items()],
            }
        ],
    }


def test_record_streak_and_next_game_from_day_boxes():
    sched = _schedule()
    now = TIP + timedelta(days=5, hours=4)
    assert team_id_of(sched, "den") == DEN and team_id_of(sched, "XYZ") is None
    assert played_days(sched, DEN, now) == [1, 2, 3, 4, 5]
    # DEN: away at BOS on odd days, home to LAL on even days.
    boxes = {1: _box(sched, 1, 110, 100), 2: _box(sched, 2, 120, 99), 3: _box(sched, 3, 101, 104)}
    boxes |= {4: _box(sched, 4, 98, 112), 5: _box(sched, 5, 115, 108)}
    team = team_season(sched, DEN, boxes, now)
    assert [r.won for r in team.results] == [True, False, False, True, True]
    assert (team.wins, team.losses, team.streak) == (3, 2, ("W", 2))
    last = team.results[-1]
    assert (last.opponent, last.home, last.points, last.opponent_points) == ("BOS", False, 115, 108)
    assert last.top is not None and last.top.player_id == DEN * 10 and last.top.pts == 35
    assert team.next_game is not None and team.next_game.id == 106


def test_games_without_a_box_or_not_over_do_not_count():
    sched = _schedule()
    box = _box(sched, 1, 110, 100)
    assert result_of({**box["games"][0], "status": "live"}, DEN, 1, None) is None
    blank = {**box["games"][0], "away": {**box["games"][0]["away"], "score": 0}}
    blank["home"] = {**blank["home"], "score": 0}
    assert result_of(blank, DEN, 1, None) is None
    assert result_of(box["games"][0], LAL, 1, None) is None  # not in it
    assert team_season(sched, DEN, {}, TIP).streak is None

"""The backtest replays a season without peeking at the future."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from app.backtest.history import History
from app.backtest.replay import Replay

DAYS = 21  # three Monday-to-Sunday matchup weeks
OPENING = datetime(2024, 10, 22, 2, 0, tzinfo=UTC)  # a Monday-night game, in UTC
CATS = {"0": "PTS", "6": "REB", "19": "FG%"}
# Two fantasy teams of three players; pro team = player's own NBA team, all play every day.
PLAYERS = {101: (1, 22.0), 102: (2, 10.0), 103: (3, 6.0), 201: (4, 18.0), 202: (5, 12.0), 203: (6, 8.0)}


def _line(pid: int, day: int) -> dict:
    team, pts = PLAYERS[pid]
    wobble = 1.0 + 0.2 * np.sin(pid + day)
    fga = round(pts / 1.2 * wobble)
    return {
        "pid": pid,
        "team": team,
        "stats": {"0": round(pts * wobble), "6": 5.0, "13": round(fga * 0.5), "14": fga, "42": 1.0, "40": 30.0},
    }


def _result(mine: float, theirs: float) -> str:
    return "WIN" if mine > theirs else "LOSS" if mine < theirs else "TIE"


@pytest.fixture
def history() -> History:
    # Player 203 gets hurt after day 9 and never plays again.
    box = {d: [_line(pid, d) for pid in PLAYERS if not (pid == 203 and d > 9)] for d in range(1, DAYS + 1)}
    schedule = []
    for week in range(1, 4):
        days = range(7 * week - 6, 7 * week + 1)
        side = {}
        for tid, pids in ((1, (101, 102, 103)), (2, (201, 202, 203))):
            lines = [line for d in days for line in box[d] if line["pid"] in pids]
            fgm, fga = sum(x["stats"]["13"] for x in lines), sum(x["stats"]["14"] for x in lines)
            side[tid] = {"0": sum(x["stats"]["0"] for x in lines), "6": sum(x["stats"]["6"] for x in lines)}
            side[tid]["19"] = fgm / fga
        sbs = {
            tid: {k: {"score": v, "result": _result(v, side[3 - tid][k])} for k, v in side[tid].items()}
            for tid in (1, 2)
        }
        winner = "HOME" if sum(v["result"] == "WIN" for v in sbs[1].values()) > 1 else "AWAY"
        schedule.append(
            {
                "matchupPeriodId": week,
                "home": {"teamId": 1, "cumulativeScore": {"scoreByStat": sbs[1]}},
                "away": {"teamId": 2, "cumulativeScore": {"scoreByStat": sbs[2]}},
                "winner": winner,
            }
        )
    pro_teams = [
        {
            "id": team,
            "proGamesByScoringPeriod": {
                str(d): [{"date": int((OPENING + timedelta(days=d - 1)).timestamp() * 1000)}]
                for d in range(1, DAYS + 1)
            },
        }
        for team, _ in PLAYERS.values()
    ]
    league = {
        "id": 1,
        "seasonId": 2025,
        "settings": {
            "name": "Test",
            "scoringSettings": {
                "scoringType": "H2H_CATEGORY",
                "scoringItems": [{"statId": int(k), "isReverseItem": False} for k in CATS],
            },
            "scheduleSettings": {"matchupPeriodCount": 3, "playoffTeamCount": 1},
            "rosterSettings": {"lineupSlotCounts": {"11": 2, "12": 1, "13": 0}},
        },
        "status": {"finalScoringPeriod": DAYS, "latestScoringPeriod": DAYS},
        "teams": [
            {"id": 1, "abbrev": "ONE", "playoffSeed": 1, "record": {"overall": {"percentage": 0.6}}},
            {"id": 2, "abbrev": "TWO", "playoffSeed": 2, "record": {"overall": {"percentage": 0.4}}},
        ],
        "schedule": schedule,
    }
    rosters = {d: {1: [[101, 11], [102, 11], [103, 12]], 2: [[201, 11], [202, 11], [203, 12]]} for d in box}
    return History(
        league_id=1,
        season=2025,
        league=league,
        pro_schedule={"settings": {"proTeams": pro_teams}},
        players={
            pid: {"fullName": f"P{pid}", "eligibleSlots": [11, 12], "proTeamId": t} for pid, (t, _) in PLAYERS.items()
        },
        prior={},
        box=box,
        rosters=rosters,
    )


def test_snapshot_knows_only_the_past(history):
    replay = Replay(history)
    lg = replay.snapshot(2)
    assert lg.current_period == 2 and lg.calendar.today == 8
    # Week 1 is played, weeks 2 and 3 are not.
    assert {m.period: m.winner != "UNDECIDED" for m in lg.schedule} == {1: True, 2: False, 3: False}
    one = lg.team(1)
    assert one.wins + one.losses + one.ties == 3  # each category of week 1 counts
    # Stats are averages of the 7 games before day 8 only.
    p = next(p for p in one.roster if p.id == 101)
    assert p.games_played == 7


def test_player_who_stopped_playing_counts_as_out(history):
    replay = Replay(history)
    out = next(p for p in replay.snapshot(3).team(2).roster if p.id == 203)
    assert out.injury_status == "OUT" and not out.available
    healthy = next(p for p in replay.snapshot(2).team(2).roster if p.id == 203)
    assert healthy.available  # on day 8 he had not missed a game yet


def test_scores_are_well_formed(history):
    replay = Replay(history)
    weekly = replay.weekly().summary()
    assert weekly["n"] == 2 * 2 * len(CATS)  # weeks 2-3, both sides, every category
    assert 0.0 <= weekly["brier"] <= 1.0 and weekly["log_loss"] > 0
    assert set(weekly["categories"]) == set(CATS.values())
    season = replay.season(cuts=(2, 3), n_sims=200).summary()
    assert season["n"] == 4 and set(season["by_cut"]) == {2, 3}
    assert 0.0 <= season["brier"] <= 1.0

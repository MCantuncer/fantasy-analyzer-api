"""The trade offer dataset: outcomes rebuilt correctly, and features that never peek past the offer."""

import asyncio
import time
from datetime import UTC, datetime, timedelta

import pytest

from app.backtest.history import History
from app.tradedata.features import Snapshot, offer_row
from app.tradedata.fetch import Throttle, has_trades
from app.tradedata.offers import ACCEPTED, ACCEPTED_NOT_PROCESSED, WITHDRAWN, build_offers, roster_moves

DAYS = 14  # two Monday-to-Sunday weeks
OPENING = datetime(2024, 10, 21, 23, 0, tzinfo=UTC)
TEAM_A = list(range(101, 107))
TEAM_B = list(range(201, 207))
INJURED = 206  # stops playing after day 4


def _line(pid: int, day: int) -> dict:
    pts = float(pid % 100) * 3  # 3, 6, ..., 18: higher id, better player
    if pid == 103 and day == 7:
        pts = 60.0  # a huge game on day 7
    return {"pid": pid, "team": pid, "stats": {"0": pts, "6": 4.0, "13": 4.0, "14": 9.0, "42": 1.0, "40": 30.0}}


def _rosters() -> dict[int, dict[int, list[list[int]]]]:
    out = {}
    for d in range(1, DAYS + 1):
        a, b = list(TEAM_A), list(TEAM_B)
        if d >= 9:  # 103 for 203, accepted on day 8, players show on the new team from day 9
            a[a.index(103)], b[b.index(203)] = 203, 103
        out[d] = {1: [[p, 0] for p in a], 2: [[p, 0] for p in b]}
    return out


@pytest.fixture
def history() -> History:
    box = {d: [_line(p, d) for p in TEAM_A + TEAM_B if not (p == INJURED and d > 4)] for d in range(1, DAYS + 1)}
    pro_teams = [
        {
            "id": pid,
            "proGamesByScoringPeriod": {
                str(d): [{"date": int((OPENING + timedelta(days=d - 1)).timestamp() * 1000)}]
                for d in range(1, DAYS + 1)
            },
        }
        for pid in TEAM_A + TEAM_B
    ]
    week1 = {
        "matchupPeriodId": 1,
        "home": {"teamId": 1, "cumulativeScore": {"scoreByStat": {"0": {"result": "WIN"}, "6": {"result": "TIE"}}}},
        "away": {"teamId": 2, "cumulativeScore": {"scoreByStat": {"0": {"result": "LOSS"}, "6": {"result": "TIE"}}}},
        "winner": "HOME",
    }
    week2 = {"matchupPeriodId": 2, "home": {"teamId": 1}, "away": {"teamId": 2}, "winner": "UNDECIDED"}
    league = {
        "settings": {
            "name": "Test",
            "scoringSettings": {"scoringType": "H2H_CATEGORY", "scoringItems": [{"statId": 0}, {"statId": 6}]},
            "scheduleSettings": {"matchupPeriodCount": 2, "playoffTeamCount": 1},
            "tradeSettings": {"vetoVotesRequired": 2},
        },
        "status": {"finalScoringPeriod": DAYS},
        "teams": [{"id": 1}, {"id": 2}],
        "schedule": [week1, week2],
        "draftDetail": {"picks": [{"playerId": 206, "overallPickNumber": 1, "bidAmount": 0, "teamId": 2}]},
    }
    return History(
        league_id=9,
        season=2025,
        league=league,
        pro_schedule={"settings": {"proTeams": pro_teams}},
        players={p: {"fullName": f"P{p}", "eligibleSlots": [0], "proTeamId": p} for p in TEAM_A + TEAM_B},
        prior={},
        box=box,
        rosters=_rosters(),
    )


def _tx(type_: str, day: int, team: int, items=(), status="EXECUTED", related="r", tid=None) -> dict:
    return {
        "id": tid or f"{type_}-{day}-{team}",
        "type": type_,
        "status": status,
        "scoringPeriodId": day,
        "proposedDate": day * 86_400_000 + team,
        "teamId": team,
        "relatedTransactionId": related,
        "items": [
            {"type": t, "playerId": p, "fromTeamId": f, "toTeamId": to, "fromLineupSlotId": 0} for t, p, f, to in items
        ],
    }


@pytest.fixture
def transactions() -> dict[int, list[dict]]:
    return {
        3: [_tx("TRADE_PROPOSAL", 3, 1, [("TRADE", 105, 1, 2), ("TRADE", 206, 2, 1)], status="CANCELED")],
        5: [_tx("TRADE_DECLINE", 5, 2)],
        8: [_tx("TRADE_ACCEPT", 8, 2, related="p1"), _tx("TRADE_ACCEPT", 8, 1, related="p1b")],
        11: [_tx("TRADE_ACCEPT", 11, 1, related="p2"), _tx("TRADE_VETO", 11, 2, related="p2")],
        12: [_tx("FREEAGENT", 12, 1, [("ADD", 999, 0, 1)])],
    }


def test_roster_moves_are_overnight_team_changes():
    assert roster_moves(_rosters()) == {9: [(103, 1, 2), (203, 2, 1)]}


def test_offers_and_outcomes(history, transactions):
    offers, stats = build_offers(9, 2025, transactions, history.rosters)
    by = {o.outcome: o for o in offers}
    assert sorted(o.outcome for o in offers) == [ACCEPTED, ACCEPTED_NOT_PROCESSED, WITHDRAWN]

    withdrawn = by[WITHDRAWN]
    assert withdrawn.proposer == 1 and withdrawn.receivers == [2] and withdrawn.day == 3
    assert set(withdrawn.moves) == {(105, 1, 2), (206, 2, 1)}

    # The accept by team 2 pairs with the move seen on day 9; team 1's accept is the same trade.
    trade = by[ACCEPTED]
    assert trade.proposer == 1 and trade.receivers == [2] and trade.day == 8
    assert set(trade.moves) == {(103, 1, 2), (203, 2, 1)}
    assert "proposer ambiguous: both sides accepted" in trade.notes
    assert trade.prior_offers == 1 and not trade.repeat
    assert stats["duplicate_accepts"] == 1

    # Accepted on day 11 but nobody moved: vetoed, with the vote counted.
    assert by[ACCEPTED_NOT_PROCESSED].vetoes == 1
    assert stats["declines"] == 1 and stats["declines_by_team"] == {2: 1}


def test_features_only_see_the_days_before(history, transactions):
    snap = Snapshot(history, {})
    gp, tot = snap.totals(103, 7)
    assert gp == 6 and tot[0] == 6 * 9.0  # the 60-point game on day 7 is not visible on day 7
    gp, tot = snap.totals(103, 8)
    assert gp == 7 and tot[0] == 6 * 9.0 + 60.0
    assert snap.last_game(103, 8)["PTS"] == 60.0
    assert snap.totals(103, 8, window=3)[0] == 3


def test_offer_row(history, transactions):
    offers, _ = build_offers(9, 2025, transactions, history.rosters)
    snap = Snapshot(history, {})
    trade = next(o for o in offers if o.outcome == ACCEPTED)
    row = offer_row(snap, trade, [])
    hot = next(p for p in row["players"] if p["pid"] == 103)
    assert hot["fantasy_team"] == 1 and hot["to_team"] == 2
    assert hot["last_game"]["PTS"] == 60.0 and hot["season"]["gp"] == 7
    # Sides mirror each other.
    a, b = row["sides"][1], row["sides"][2]
    assert a["gives"] == [103] and a["gets"] == [203]
    assert a["net_season"] == pytest.approx(-b["net_season"])
    # Only week 1 (days 1-7) is finished on day 8; each category counts.
    one = next(t for t in row["teams"] if t["team_id"] == 1)
    assert (one["wins"], one["losses"], one["ties"], one["periods_done"]) == (1, 0, 1, 1)

    withdrawn = next(o for o in offers if o.outcome == WITHDRAWN)
    row = offer_row(snap, withdrawn, [])
    hurt = next(p for p in row["players"] if p["pid"] == INJURED)
    assert not hurt["availability"]["likely_out"]  # on day 3 he was still playing
    assert hurt["league_draft"]["pick"] == 1
    later = snap.player(INJURED, 10, 2)
    assert later["availability"]["likely_out"] and later["availability"]["missed_in_a_row"] == 5


def test_has_trades(transactions):
    assert has_trades(transactions)
    assert not has_trades({12: transactions[12]})


def test_throttle_spaces_requests():
    async def burst():
        th = Throttle(rate=50)
        t0 = time.monotonic()
        await asyncio.gather(*(th.wait() for _ in range(6)))
        return time.monotonic() - t0

    assert asyncio.run(burst()) >= 5 / 50 * 0.9


def test_preseason_trade_and_pickups():
    draft = {1: [[103, -1], [104, -1]], 2: [[203, -1]]}
    day1 = {1: [[203, 0], [104, 0]], 2: [[103, 0]]}  # 103 for 203 before opening night
    day2 = {1: [[203, 0]], 2: [[103, 0], [104, 0]]}  # 104 dropped by 1, picked up by 2: no trade
    tx = {
        0: [_tx("TRADE_ACCEPT", 0, 2, related="p0")],
        2: [_tx("FREEAGENT", 2, 2, [("ADD", 104, 0, 2)])],
    }
    offers, stats = build_offers(9, 2025, tx, {1: day1, 2: day2}, day0=draft)
    assert [(o.outcome, sorted(o.moves)) for o in offers] == [(ACCEPTED, [(103, 1, 2), (203, 2, 1)])]
    assert stats["trades_without_accept"] == 0


def test_vetoed_accept_just_before_the_real_one(history):
    tx = {
        7: [_tx("TRADE_ACCEPT", 7, 2, related="vetoed", tid="a7"), _tx("TRADE_VETO", 7, 1, related="vetoed")],
        8: [_tx("TRADE_ACCEPT", 8, 2, related="real", tid="a8")],
    }
    offers, _ = build_offers(9, 2025, tx, history.rosters)
    got = {o.offer_id: (o.outcome, o.vetoes) for o in offers}
    assert got == {"a7": (ACCEPTED_NOT_PROCESSED, 1), "a8": (ACCEPTED, 0)}


def test_off_season_accepts_are_left_out(history):
    tx = {0: [_tx("TRADE_ACCEPT", 0, 2)]}
    offers, stats = build_offers(9, 2025, tx, history.rosters, draft_ms=10**12)
    assert offers == [] and stats["pre_draft_accepts"] == 1

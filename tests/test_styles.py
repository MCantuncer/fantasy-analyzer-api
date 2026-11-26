from app import styles

SLOTS = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 11: 3}


def player(pid, slots, day=None, played=True, avg=None, gp=10):
    stats = []
    if day is not None and played:
        stats.append({"scoringPeriodId": day, "statSplitTypeId": 5, "statSourceId": 0, "stats": {"42": 1, "0": 10}})
    if avg is not None:
        stats.append(
            {"seasonId": 2027, "statSplitTypeId": 0, "statSourceId": 0, "averageStats": avg, "stats": {"42": gp}}
        )
    return {"id": pid, "eligibleSlots": slots, "stats": stats}


def entry(p, slot):
    return {"playerId": p["id"], "lineupSlotId": slot, "playerPoolEntry": {"player": p}}


def test_max_starts_respects_positions():
    # Two centres and one slot that takes a centre (C) plus three UTIL: both fit.
    assert styles._max_starts([{4, 11}, {4, 11}], {4: 1, 11: 3}) == 2
    # Two point guards for one PG slot and nothing else: only one starts.
    assert styles._max_starts([{0}, {0}], {0: 1, 4: 1}) == 1


def test_summarize_day_counts_starts_left_on_the_bench():
    day = 9
    benched = player(2, [0, 5, 11, 12], day)
    raw = {
        "teams": [
            {
                "id": 1,
                "roster": {
                    "entries": [
                        entry(player(1, [0, 5, 11, 12], day), 0),
                        entry(benched, 12),  # played but sat on the bench with UTIL open
                        entry(player(3, [4, 11, 12], day, played=False), 4),
                    ]
                },
            }
        ]
    }
    s = styles.summarize_day(raw, day, SLOTS, 2027)
    assert s.lineups[1].starts == 1
    assert s.lineups[1].best == 2
    assert s.rosters[1] == [1, 2, 3]


def tx(kind, team, day, items, status="EXECUTED", bid=0, tid=None):
    return {
        "id": tid or f"{kind}-{team}-{day}-{items[0][1] if items else 0}",
        "type": kind,
        "status": status,
        "teamId": team,
        "scoringPeriodId": day,
        "bidAmount": bid,
        "items": [{"type": t, "playerId": p} for t, p in items],
    }


def test_read_moves_counts_adds_drops_lost_bids_and_declines():
    adds, drops, lost, trades = styles.read_moves(
        [
            tx("WAIVER", 1, 3, [("ADD", 10), ("DROP", 11)], bid=4),
            tx("WAIVER", 1, 3, [("ADD", 10), ("DROP", 11)], bid=4),  # the same item twice
            tx("WAIVER", 2, 3, [("ADD", 10)], status="FAILED_INVALIDPLAYERSOURCE"),
            tx("FREEAGENT", 2, 5, [("ADD", 12)]),
            tx("TRADE_DECLINE", 3, 6, [], tid="d1"),
            tx("TRADE_DECLINE", 3, 9, [], tid="d2"),
        ]
    )
    assert [(a.player_id, a.bid) for a in adds[1]] == [(10, 4)]
    assert [d.player_id for d in drops[1]] == [11]
    assert lost[2] == 1 and [a.player_id for a in adds[2]] == [12]
    assert trades[3].declined == 2


def week(period, a, b, results):
    """A finished week: results maps stat id -> 'a' / 'b' / 'tie'."""

    def side(team, me):
        by = {
            sid: {"result": "TIE" if r == "tie" else ("WIN" if r == me else "LOSS"), "score": 1}
            for sid, r in results.items()
        }
        return {"teamId": team, "cumulativeScore": {"scoreByStat": by}}

    return {"matchupPeriodId": period, "winner": "HOME", "home": side(a, "a"), "away": side(b, "b")}


SETTINGS = {
    "rosterSettings": {"lineupSlotCounts": {"0": 1, "4": 1, "11": 1, "12": 2, "13": 1}},
    "scoringSettings": {"scoringItems": [{"statId": i} for i in (19, 20, 17, 6, 3, 2, 1, 11, 0)]},
}


def test_category_records_and_win_path():
    # Team 1 always wins FG%/BLK, wins AST/TO/FT% 2 of 3, never PTS.
    weeks = []
    for p in (1, 2, 3):
        r = {"19": "a", "1": "a", "0": "b", "6": "b", "17": "b", "2": "b"}
        for sid in ("3", "11", "20"):
            r[sid] = "a" if p != 3 else "b"
        weeks.append(week(p, 1, 2, r))
    weeks.append({"matchupPeriodId": 4, "winner": "UNDECIDED", "home": {"teamId": 1}, "away": {"teamId": 2}})
    cats = styles.scoring_cats(SETTINGS)
    recs = styles.category_records(weeks, cats, through_period=4)
    share = {r.cat: round(r.share, 2) for r in recs[1]}
    assert share["FG%"] == 1.0 and share["PTS"] == 0.0 and share["AST"] == 0.67
    style = styles.ManagerStyle(
        1, "", "", "unknown", 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, recs[1], [], [], ["STL", "TO"]
    )
    path = styles.win_path(style)
    assert set(path.lock) == {"FG%", "BLK"}
    assert set(path.swing) == {"AST", "TO", "FT%"}
    assert "PTS" in path.let_go
    assert path.adds_should_bring == [c for c in path.swing if c != "TO"]


def _style(team, shares, chases=()):
    recs = [styles.CategoryRecord(c, s * 5, 5) for c, s in shares.items()]
    return styles.ManagerStyle(team, "", "", "unknown", 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, recs, [], [], list(chases))


def test_matchup_calls():
    me = _style(
        1, {"FG%": 1, "BLK": 1, "AST": 0.6, "TO": 0.6, "FT%": 0.6, "REB": 0.2, "STL": 0.2, "3PM": 0.2, "PTS": 0}
    )
    them = _style(
        2, {"FG%": 1, "BLK": 0, "AST": 0.6, "TO": 0, "FT%": 0.8, "REB": 0.2, "STL": 0.4, "3PM": 0.8, "PTS": 0.6}
    )
    live = {
        "BLK": (9, 9),
        "TO": (15, 30),
        "FG%": (0.562, 0.483),
        "PTS": (198, 192),
        "AST": (36, 48),
        "FT%": (0.760, 0.774),
        "REB": (54, 67),
    }
    calls = {c.cat: c for c in styles.matchup_calls(me, them, live)}
    assert calls["BLK"].call == "flip"  # tied, and they never win it
    assert calls["TO"].call == "hold" and calls["TO"].status == "ahead"  # fewer turnovers lead
    assert calls["FG%"].call == "hold"
    assert calls["PTS"].call == "bonus"  # ahead in a category the path lets go
    assert calls["AST"].call == "fight"
    assert calls["FT%"].call == "fight"  # their stronghold, but close and on my path
    assert calls["REB"].call == "let_go"
    assert calls["STL"].status == "not_started"


def test_labels():
    med = styles.LeagueMedians(adds_per_week=1.2, starts_per_week=38.8, lineup_moves_per_week=18.1, start_share=0.99)

    def row(**kw):
        base = dict(
            team_id=1,
            activity="",
            lineup="",
            door="unknown",
            adds=10,
            adds_per_week=1.0,
            quick_drops=0,
            faab_spent=0,
            bids_lost=0,
            starts_per_week=39,
            start_share=1.0,
            missed_starts=0,
            lineup_moves_per_week=18,
            ir_moves=0,
            trades=0,
            declined=0,
            categories=[],
            wins=[],
            gives_away=[],
            chases=[],
        )
        base.update(kw)
        return styles.ManagerStyle(**base)

    assert styles.activity_label(row(adds_per_week=3.5, adds=18, quick_drops=5), med, 2.9) == "waiver_hawk"
    assert styles.activity_label(row(adds_per_week=2.9, adds=15, quick_drops=9), med, 2.9) == "streamer"
    assert styles.activity_label(row(adds_per_week=0.4, lineup_moves_per_week=9.3), med, 2.9) == "set_and_forget"
    assert styles.activity_label(row(adds_per_week=0.4), med, 2.9) == "draft_loyalist"
    assert styles.activity_label(row(adds_per_week=1.4), med, 2.9) == "steady_hands"
    assert styles.lineup_label(row(missed_starts=8), med, 5) == "leaks_starts"
    assert styles.lineup_label(row(starts_per_week=31.9), med, 5) == "low_volume"
    assert styles.lineup_label(row(start_share=0.991), med, 5) == "sharp"
    assert styles.lineup_label(row(start_share=0.98), med, 5) == "steady"


def test_manager_styles_endpoint(client, fake_espn):
    r = client.post("/api/manager-styles", json={"league_id": 744636428, "season": 2027})
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["teams"]) == 10
    mine = [t for t in body["teams"] if t["mine"]]
    assert len(mine) == 1 and mine[0]["team_id"] == body["my_team_id"] == 3
    assert body["win_path"] is not None
    assert {t["activity"] for t in body["teams"]} <= {
        "waiver_hawk",
        "streamer",
        "steady_hands",
        "draft_loyalist",
        "set_and_forget",
    }
    calls_before = fake_espn.calls
    again = client.post("/api/manager-styles", json={"league_id": 744636428, "season": 2027})
    assert again.json() == body
    assert fake_espn.calls == calls_before  # the built report is reused


def test_manager_styles_needs_sign_in(fake_espn):
    import httpx
    from fastapi.testclient import TestClient

    from app.accounts import MemoryAccountStore
    from app.espn.client import EspnClient
    from app.main import create_app
    from app.store import MemoryStore

    app = create_app(
        EspnClient(transport=httpx.MockTransport(fake_espn.handler)), store=MemoryStore(), accounts=MemoryAccountStore()
    )
    assert TestClient(app).post("/api/manager-styles", json={"league_id": 744636428}).status_code == 401

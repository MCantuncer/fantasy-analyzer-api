from app.gameclock import apply_on_court, clock_label, parse_on_court, parse_scoreboard


def test_clock_labels():
    assert clock_label(4, "2:31", "2:31 - 4th") == "Q4 2:31"
    assert clock_label(2, "0.0", "Halftime") == "Half"
    assert clock_label(3, "0.0", "End of 3rd") == "End Q3"
    assert clock_label(5, "0:48", "0:48 - OT") == "OT 0:48"
    assert clock_label(6, "3:10", "3:10 - 2OT") == "2OT 3:10"
    assert clock_label(0, "12:00", "") is None


def test_parse_scoreboard_skips_broken_events():
    raw = {
        "events": [
            {
                "id": "401",
                "status": {"period": 1, "displayClock": "7:02", "type": {"state": "in", "shortDetail": "7:02 - 1st"}},
                "competitions": [
                    {"competitors": [{"team": {"abbreviation": "bos"}}, {"team": {"abbreviation": "ORL"}}]}
                ],
            },
            {"id": "402", "status": {"type": {"state": "pre"}}, "competitions": [{}]},
            {"nope": True},
        ]
    }
    clocks = parse_scoreboard(raw)
    assert clocks[401].label == "Q1 7:02" and clocks[401].teams == frozenset({"BOS", "ORL"})
    assert clocks[402].state == "pre" and clocks[402].label is None
    assert len(clocks) == 2


def _box(*teams: list[tuple[int, bool]]) -> dict:
    return {
        "boxscore": {
            "players": [
                {"statistics": [{"athletes": [{"athlete": {"id": str(i)}, "active": on} for i, on in team]}]}
                for team in teams
            ]
        }
    }


def test_on_court_is_the_five_espn_marks_active_on_each_side():
    home = [(i, i <= 5) for i in range(1, 11)]
    away = [(i, i <= 105) for i in range(101, 111)]
    assert parse_on_court(_box(home, away)) == frozenset({1, 2, 3, 4, 5, 101, 102, 103, 104, 105})


def test_on_court_unknown_unless_both_sides_show_five():
    five = [(i, i <= 5) for i in range(1, 11)]
    assert parse_on_court(_box(five, [(i, False) for i in range(101, 111)])) is None  # nobody flagged
    assert parse_on_court(_box(five, [(i, True) for i in range(101, 111)])) is None  # everybody flagged
    assert parse_on_court(_box(five)) is None
    assert parse_on_court({}) is None


def test_lines_of_games_in_progress_say_who_is_on_the_floor():
    line = lambda pid: {"player_id": pid, "pts": 0}  # noqa: E731
    body = {
        "games": [
            {"id": 1, "status": "live", "lines": [line(10), line(11)]},
            {"id": 2, "status": "live", "lines": [line(20)]},  # ESPN couldn't tell
            {"id": 3, "status": "final", "lines": [line(30)]},
        ]
    }
    out = apply_on_court(body, {1: frozenset({10, 99}), 3: frozenset({30})})
    assert [ln.get("on_court") for g in out["games"] for ln in g["lines"]] == [True, False, None, None]
    assert "on_court" not in body["games"][0]["lines"][0]  # the cached day is left as it was

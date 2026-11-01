from app.gameclock import clock_label, parse_scoreboard


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

import copy

import pytest

from app.espn.parse import (
    ACTUAL_WEIGHT_GAMES,
    GAMES_PLAYED,
    MINUTES,
    MINUTES_WEIGHT_GAMES,
    parse_league,
    parse_player,
)


def test_league_settings(league):
    assert league.name == "Chill Primera Division"
    assert league.scoring_type == "H2H_CATEGORY"
    assert sorted(c.name for c in league.categories) == sorted(
        ["PTS", "REB", "AST", "STL", "BLK", "3PM", "FG%", "FT%", "TO"]
    )
    assert next(c for c in league.categories if c.name == "TO").reverse
    assert len(league.teams) == 10
    assert league.starter_slots == 10
    assert league.playoff_teams == 6
    assert league.regular_season_periods == 18


def test_rosters_and_free_agents(league):
    me = league.team(3)
    assert me.name == "Kucukesat Aspavas"
    assert len(me.roster) == 13
    assert me.faab_remaining is not None
    names = {p.name for p in me.roster}
    assert "Giannis Antetokounmpo" in names
    assert league.free_agents and all(p.on_team_id == 0 for p in league.free_agents)
    giannis = next(p for p in me.roster if p.name == "Giannis Antetokounmpo")
    assert giannis.per_game[0] > 20  # points per game
    assert giannis.per_game[14] > giannis.per_game[13]  # FGA > FGM


def test_schedule_covers_regular_season(league):
    periods = {m.period for m in league.schedule}
    assert periods == set(range(1, 19))
    assert all(len([m for m in league.schedule if m.period == p]) == 5 for p in periods)


def test_positions_follow_espn_slot_ids(league):
    roster = {p.name: p for p in league.team(3).roster}
    assert roster["Jarrett Allen"].positions == ["C"]
    assert "PG" in roster["Payton Pritchard"].positions and "C" not in roster["Payton Pritchard"].positions


def test_waiver_priority_only_without_faab(recorded):
    league = parse_league(recorded["league"], recorded["free_agents"])
    assert all(t.waiver_priority is None for t in league.teams)  # FAAB league: bids decide claims

    raw = copy.deepcopy(recorded["league"])
    raw["settings"]["acquisitionSettings"]["isUsingAcquisitionBudget"] = False
    rolling = parse_league(raw, recorded["free_agents"])
    ranks = {t["id"]: t["waiverRank"] for t in raw["teams"]}
    assert {t.id: t.waiver_priority for t in rolling.teams} == ranks
    assert all(t.faab_remaining is None for t in rolling.teams)


def test_season_stats_take_over_faster_for_points_than_for_free_throws():
    def stats(source, values):
        return {"seasonId": 2027, "statSourceId": source, "statSplitTypeId": 0, "averageStats": values}

    projected = {"0": 20.0, "6": 8.0, "15": 4.0, "16": 5.0, "42": 70.0}
    hot = {"0": 30.0, "6": 12.0, "15": 5.0, "16": 5.0, "42": 10.0}
    p = parse_player({"player": {"id": 1, "fullName": "X", "stats": [stats(1, projected), stats(0, hot)]}}, 2027)
    share = {
        k: (p.per_game[k] - float(projected[str(k)])) / (float(hot[str(k)]) - float(projected[str(k)]))
        for k in (0, 6, 15)
    }
    assert share[6] == pytest.approx(10 / (10 + ACTUAL_WEIGHT_GAMES))
    assert share[0] > share[6] > share[15]


def _line(season, source, values):
    return {"seasonId": season, "statSourceId": source, "statSplitTypeId": 0, "averageStats": values}


# Brook Lopez on 2 Nov 2026: ESPN projected a bigger role than he had last season or has now.
LOPEZ_PROJECTED = {"0": 10.62, "1": 1.69, "17": 1.61, "13": 3.94, "14": 8.31, "40": 26.2, "42": 72.0}
LOPEZ_LAST = {"0": 8.45, "1": 1.16, "17": 1.52, "13": 3.09, "14": 7.23, "40": 21.8, "42": 75.0}
LOPEZ_NOW = {"0": 12.33, "1": 1.0, "17": 1.33, "13": 4.67, "14": 8.5, "40": 20.17, "42": 6.0}


def _lopez(*lines):
    return parse_player({"player": {"id": 3448, "fullName": "Brook Lopez", "stats": list(lines)}}, 2027)


def test_projected_minutes_lean_toward_last_season():
    p = _lopez(_line(2027, 1, LOPEZ_PROJECTED), _line(2026, 0, LOPEZ_LAST))
    minutes = 0.75 * 26.2 + 0.25 * 21.8
    assert p.per_game[MINUTES] == pytest.approx(minutes)
    scale = minutes / 26.2
    assert p.per_game[1] == pytest.approx(1.69 * scale)  # blocks follow the minutes
    assert p.per_game[13] / p.per_game[14] == pytest.approx(3.94 / 8.31)  # shooting rates don't change
    assert p.per_game[GAMES_PLAYED] == 72.0


def test_minutes_trust_this_season_sooner_than_the_stats():
    p = _lopez(_line(2027, 1, LOPEZ_PROJECTED), _line(2026, 0, LOPEZ_LAST), _line(2027, 0, LOPEZ_NOW))
    prior = 0.75 * 26.2 + 0.25 * 21.8
    w = 6 / (6 + MINUTES_WEIGHT_GAMES)
    assert p.per_game[MINUTES] == pytest.approx((1 - w) * prior + w * 20.17)
    # Before the fix he kept ESPN's 26-minute role: 1.53 blocks a game after six games at 1.0.
    assert p.per_game[1] < 1.45


def test_no_last_season_keeps_espn_minutes():
    p = _lopez(_line(2027, 1, LOPEZ_PROJECTED))
    assert p.per_game[MINUTES] == 26.2 and p.per_game[1] == 1.69
    short = {**LOPEZ_LAST, "42": 4.0}  # a few games last season say little about his role
    assert _lopez(_line(2027, 1, LOPEZ_PROJECTED), _line(2026, 0, short)).per_game[MINUTES] == 26.2


def test_lines_without_minutes_are_left_alone():
    projected = {k: v for k, v in LOPEZ_PROJECTED.items() if k != "40"}
    now = {k: v for k, v in LOPEZ_NOW.items() if k != "40"}
    p = _lopez(_line(2027, 1, projected), _line(2026, 0, LOPEZ_LAST), _line(2027, 0, now))
    w = 6 / (6 + ACTUAL_WEIGHT_GAMES)
    assert p.per_game[1] == pytest.approx((1 - w) * 1.69 + w * 1.0)

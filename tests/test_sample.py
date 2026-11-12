import pytest

from app.analysis.report import Analyzer
from app.sample import (
    PLAY_CALLS,
    SAMPLE_LEAGUE_NAME,
    best_move,
    first_team,
    mask_report,
    masked_names,
)
from app.schemas import AnalysisReport

from .conftest import LEAGUE_ID, MY_TEAM


@pytest.fixture(scope="module")
def report(league) -> AnalysisReport:
    return Analyzer(league, MY_TEAM).report()


@pytest.fixture(scope="module")
def names(report):
    return masked_names(t.id for t in report.league.teams)


def test_play_calls_are_unique():
    assert len({n for n, _ in PLAY_CALLS}) == len(PLAY_CALLS)
    assert len({a for _, a in PLAY_CALLS}) == len(PLAY_CALLS)


def test_masked_names_follow_sorted_team_ids():
    names = masked_names([11, 3, 1, 3])
    assert [names[i].name for i in (1, 3, 11)] == ["Horns", "Hammer", "Spain"]
    assert names[3].abbrev == "HAM"
    assert masked_names([1, 3, 11]) == names  # same league, same names


def test_masked_names_has_a_limit():
    with pytest.raises(ValueError):
        masked_names(range(len(PLAY_CALLS) + 1))


def test_no_real_team_name_or_league_name_survives(report, names):
    masked = mask_report(report, names)
    text = masked.model_dump_json()
    for team in report.league.teams:
        assert team.name not in text
    assert report.league.name not in text
    assert str(LEAGUE_ID) not in text
    assert masked.league.name == SAMPLE_LEAGUE_NAME
    assert masked.league.id == 0


def test_masking_keeps_ids_players_and_numbers(report, names):
    masked = mask_report(report, names)
    assert masked.team_id == report.team_id
    assert masked.team_name == names[MY_TEAM].name
    assert [t.id for t in masked.league.teams] == [t.id for t in report.league.teams]
    assert [t.abbrev for t in masked.league.teams] == [names[t.id].abbrev for t in report.league.teams]
    assert [p.name for p in masked.roster] == [p.name for p in report.roster]
    assert [p.pro_team for p in masked.roster] == [p.pro_team for p in report.roster]
    assert [r.playoff_pct for r in masked.season] == [r.playoff_pct for r in report.season]
    assert [r.team_name for r in masked.season] == [names[r.team_id].name for r in report.season]
    for before, after in zip(report.trades, masked.trades, strict=True):
        assert after.partner_team_name == names[before.partner_team_id].name
        assert after.playoff_pct_after == before.playoff_pct_after
    if report.week is not None:
        assert masked.week is not None
        assert masked.week.opponent_name == names[report.week.opponent_team_id].name


def test_team_names_in_free_text_are_replaced(report, names):
    real = report.league.teams[0]
    note = f"{real.name} punts FT%; {real.name} again"
    changed = report.model_copy(update={"assumptions": report.assumptions.model_copy(update={"notes": [note]})})
    masked = mask_report(changed, names)
    assert masked.assumptions.notes == [f"{names[real.id].name} punts FT%; {names[real.id].name} again"]


def test_masking_needs_a_name_for_every_team(report):
    with pytest.raises(ValueError):
        mask_report(report, masked_names([report.league.teams[0].id]))


# --- the team a visitor scouts first ---


def _pickup(before: float, after: float, by_trade: bool = False) -> dict:
    return {
        "open_spot_by_trade": by_trade,
        "playoff_pct_before": before,
        "playoff_pct_after": after,
    }


def _trade(
    before: float,
    after: float,
    partner: tuple[float | None, float | None] = (0.3, 0.35),
    acceptance: float = 1.0,
) -> dict:
    return {
        "acceptance": acceptance,
        "playoff_pct_before": before,
        "playoff_pct_after": after,
        "partner_playoff_pct_before": partner[0],
        "partner_playoff_pct_after": partner[1],
    }


def _report(pickups: list[dict] = (), trades: list[dict] = ()) -> AnalysisReport:
    from app.schemas import PickupIdea, TradeIdea

    return AnalysisReport.model_construct(
        pickups=[PickupIdea.model_construct(**p) for p in pickups],
        trades=[TradeIdea.model_construct(**t) for t in trades],
    )


def test_best_move_mirrors_the_card():
    assert best_move(_report()) is None
    assert best_move(_report(pickups=[_pickup(0.5, 0.49)])) is None
    move = best_move(_report(pickups=[_pickup(0.47, 0.51)], trades=[_trade(0.47, 0.51)]))
    assert move is not None and move.kind == "pickup"  # a pickup wins a tie
    move = best_move(_report(pickups=[_pickup(0.47, 0.49)], trades=[_trade(0.47, 0.55)]))
    assert move is not None and move.kind == "trade"
    # a pickup that needs a trade first doesn't count as a straight add/drop
    assert best_move(_report(pickups=[_pickup(0.4, 0.6, by_trade=True)])) is None


def test_best_move_weighs_the_chance_a_trade_is_accepted():
    # +8 at 27% is worth +2.2: the sure +4 pickup leads
    move = best_move(_report(pickups=[_pickup(0.48, 0.52)], trades=[_trade(0.48, 0.56, acceptance=0.27)]))
    assert move is not None and move.kind == "pickup"
    # +12 at 50% is worth +6: the trade leads, and its gain is the full change the card shows
    move = best_move(_report(pickups=[_pickup(0.48, 0.52)], trades=[_trade(0.48, 0.60, acceptance=0.5)]))
    assert move is not None and move.kind == "trade"
    assert move.gain == pytest.approx(0.12)
    # +8 at 50% is worth +4, the same as the pickup: the pickup wins the tie
    move = best_move(_report(pickups=[_pickup(0.47, 0.51)], trades=[_trade(0.47, 0.55, acceptance=0.5)]))
    assert move is not None and move.kind == "pickup"
    # a pickup that lowers the odds never leads
    move = best_move(_report(pickups=[_pickup(0.48, 0.4798)], trades=[_trade(0.48, 0.50, acceptance=0.01)]))
    assert move is not None and move.kind == "trade"


def test_best_move_skips_trades_the_partner_gains_nothing_from():
    flat = _trade(0.4, 0.6, partner=(0.3, 0.3))
    unknown = _trade(0.4, 0.6, partner=(None, None))
    assert best_move(_report(trades=[flat])) is None
    assert best_move(_report(trades=[unknown])) is None
    move = best_move(_report(pickups=[_pickup(0.4, 0.42)], trades=[flat]))
    assert move is not None and move.kind == "pickup"


def test_first_team_is_the_clearest_move():
    reports = {
        1: _report(pickups=[_pickup(0.47, 0.51)]),
        2: _report(pickups=[_pickup(0.20, 0.27)]),
        3: _report(),
    }
    assert first_team(reports) == 2


def test_first_team_prefers_a_pickup_then_a_coin_flip_then_the_lowest_id():
    assert first_team({1: _report(trades=[_trade(0.4, 0.44)]), 2: _report(pickups=[_pickup(0.1, 0.14)])}) == 2
    assert first_team({1: _report(pickups=[_pickup(0.1, 0.14)]), 2: _report(pickups=[_pickup(0.47, 0.51)])}) == 2
    assert first_team({4: _report(pickups=[_pickup(0.47, 0.51)]), 2: _report(pickups=[_pickup(0.47, 0.51)])}) == 2


def test_first_team_falls_back_to_a_small_move_then_none():
    assert first_team({1: _report(pickups=[_pickup(0.5, 0.51)]), 2: _report(pickups=[_pickup(0.5, 0.505)])}) == 1
    assert first_team({1: _report(pickups=[_pickup(0.5, 0.505)]), 2: _report()}) is None
    assert first_team({}) is None


def test_first_team_on_a_real_report(report):
    move = best_move(report)
    expected = MY_TEAM if move is not None and round(move.gain, 6) >= 0.01 else None
    assert first_team({MY_TEAM: report}) == expected


# --- the day's index and which day is shown ---


def test_sample_index_ranks_names_and_flags_moves(report, names):
    from app.sample import Move, sample_index

    ids = sorted(t.id for t in report.league.teams)
    ranks = {team_id: float(len(ids) - i) for i, team_id in enumerate(ids)}  # the highest id projects first
    moves = {ids[0]: Move("pickup", 0.03, 0.5), ids[1]: Move("pickup", 0.005, 0.5)}
    index = sample_index("2026-11-12", 2027, report.league, moves, ranks, names)
    assert index["default_team_id"] == ids[0]
    teams = {t["id"]: t for t in index["teams"]}
    assert teams[ids[-1]]["projected_rank"] == 1 and teams[ids[0]]["projected_rank"] == len(ids)
    assert teams[ids[0]]["has_move"] and not teams[ids[1]]["has_move"] and not teams[ids[2]]["has_move"]
    assert teams[ids[0]]["name"] == names[ids[0]].name
    assert index["league"]["name"] == SAMPLE_LEAGUE_NAME and index["league"]["size"] == len(ids)
    assert report.league.name not in str(index)


def test_sample_index_with_no_move_starts_on_the_lowest_id(report, names):
    from app.sample import sample_index

    ids = sorted(t.id for t in report.league.teams)
    index = sample_index("2026-11-12", 2027, report.league, {}, {i: 1.0 for i in ids}, names)
    assert index["default_team_id"] == ids[0]
    assert not any(t["has_move"] for t in index["teams"])


def test_serving_day_falls_back_to_a_recent_day_with_a_move():
    from datetime import date

    from app.sample import serving_day

    today = date(2026, 11, 12)
    days = ["2026-11-12", "2026-11-11", "2026-11-10", "2026-11-07"]
    assert serving_day([], lambda d: True, today) is None
    assert serving_day(days, lambda d: True, today) == "2026-11-12"
    assert serving_day(days, lambda d: d == "2026-11-10", today) == "2026-11-10"
    # 5 days old is past the fallback: today's hold cards are shown
    assert serving_day(days, lambda d: d == "2026-11-07", today) == "2026-11-12"
    assert serving_day(days, lambda d: False, today) == "2026-11-12"

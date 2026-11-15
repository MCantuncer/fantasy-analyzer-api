import pytest

from app.analysis.form import (
    FGA,
    FGM,
    FTA,
    FTM,
    GP,
    KEEP,
    LAST_15_DAYS,
    MINUTES,
    PTS,
    TPA,
    TPM,
    hot_run,
    hot_run_ref,
    split_run,
    value,
)
from app.playercard import parse_card

REB, AST, STL, BLK, TO = 6, 3, 2, 1, 11
# A bench guard's usual per-game line.
USUAL = {MINUTES: 22.0, PTS: 9.0, REB: 3.0, AST: 2.5, STL: 0.7, BLK: 0.2, TPM: 1.2, TPA: 3.5, TO: 1.2}
USUAL |= {FGM: 3.4, FGA: 8.5, FTM: 1.0, FTA: 1.3}


def _scaled(line: dict[int, float], k: float) -> dict[int, float]:
    return {s: v * k for s, v in line.items()}


def _shot(line: dict[int, float], fg: float) -> dict[int, float]:
    """The line with his field goals going in at ``fg`` on the same attempts, points to match."""
    out = dict(line)
    out[FGM] = fg * line[FGA]
    out[PTS] = line[PTS] + 2 * (out[FGM] - line[FGM])
    return out


def _entry(season_line, season_gp, recent, recent_gp, last=None, last_gp=0, season=2027) -> list[dict]:
    def split(s, split_id, line, gp):
        return {
            "seasonId": s,
            "statSourceId": 0,
            "statSplitTypeId": split_id,
            "averageStats": {str(k): v for k, v in {**line, GP: gp}.items()},
        }

    out = [split(season, 0, season_line, season_gp), split(season, LAST_15_DAYS, recent, recent_gp)]
    if last is not None:
        out.append(split(season - 1, 0, last, last_gp))
    return out


def _season(before: dict, n_before: int, recent: dict, n_recent: int) -> dict:
    n = n_before + n_recent
    return {k: (before.get(k, 0.0) * n_before + recent.get(k, 0.0) * n_recent) / n for k in set(before) | set(recent)}


def test_the_parts_add_up_to_the_gap():
    recent = _shot(_scaled(USUAL, 1.4), 0.55)
    parts = split_run(USUAL, recent)
    assert sum(parts) == pytest.approx(value(recent) - value(USUAL))


def test_a_minutes_jump_is_a_bigger_role():
    recent = _scaled(USUAL, 33.0 / 22.0)  # same per-minute line and shooting, 33 minutes
    run = hot_run(_entry(_season(USUAL, 8, recent, 6), 14, recent, 6, USUAL, 70), 2027)
    assert run is not None and run.verdict == "bigger_role"
    assert run.minutes == pytest.approx(run.gap)
    assert run.likely_kept == pytest.approx(KEEP["minutes"])


def test_hot_shooting_on_the_same_minutes_is_a_straw_fire():
    recent = _shot(USUAL, 0.62)  # 62% on the same shots, same minutes
    run = hot_run(_entry(_season(USUAL, 8, recent, 6), 14, recent, 6, USUAL, 70), 2027)
    assert run is not None and run.verdict == "straw_fire"
    assert run.minutes == pytest.approx(0.0, abs=1e-9)
    assert run.likely_kept == pytest.approx(KEEP["shooting"], abs=0.02)
    ref = hot_run_ref(run)
    assert ref.fg_recent == pytest.approx(0.62, abs=1e-3) and ref.fg_usual == pytest.approx(0.4, abs=1e-3)
    assert ref.minutes_recent == ref.minutes_usual == 22.0


def test_no_run_without_a_real_gap_or_enough_games():
    small = _scaled(USUAL, 1.05)
    assert hot_run(_entry(_season(USUAL, 8, small, 6), 14, small, 6, USUAL, 70), 2027) is None
    hot = _shot(_scaled(USUAL, 1.5), 0.6)
    assert hot_run(_entry(_season(USUAL, 12, hot, 2), 14, hot, 2, USUAL, 70), 2027) is None  # 2 games
    # No last season and only 3 games before the run: no usual level to compare with.
    assert hot_run(_entry(_season(USUAL, 3, hot, 6), 9, hot, 6), 2027) is None


def test_the_card_carries_his_hot_run():
    recent = _scaled(USUAL, 33.0 / 22.0)
    entry = {
        "id": 7,
        "player": {
            "id": 7,
            "fullName": "Bench Guard",
            "stats": _entry(_season(USUAL, 8, recent, 6), 14, recent, 6, USUAL, 70),
        },
    }
    card = parse_card(entry, 2027, {})
    assert card.hot_run is not None and card.hot_run.verdict == "bigger_role"

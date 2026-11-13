import numpy as np
import pytest

from scripts.acceptance_fit import decline_factors, decline_weights


def _rows(league: int, season: int, accepted: int, withdrawn: int) -> list[dict]:
    return [{"league": league, "season": season, "accepted": k < accepted} for k in range(accepted + withdrawn)]


ROWS = _rows(1, 2024, accepted=1, withdrawn=3) + _rows(2, 2025, accepted=1, withdrawn=1)
SEASONS = [
    {"league_id": 1, "season": 2024, "outcomes": {"accepted": 2, "withdrawn": 10}, "declines": 30},
    {"league_id": 1, "season": 2023, "outcomes": {"withdrawn": 5}, "declines": 500},  # no replayed offers
    {"league_id": 2, "season": 2025, "outcomes": {"accepted": 3, "withdrawn": 4}, "declines": 0},
    {"league_id": 9, "season": 2025, "outcomes": {"withdrawn": 7}, "declines": 70},  # another league
]


def test_a_withdrawn_offer_stands_for_its_leagues_declines():
    # League 1: 10 withdrawn and 30 declined in the season it has replayed offers in.
    assert decline_factors(ROWS, SEASONS) == {1: 4.0, 2: 1.0}


def test_weights_count_declines_and_keep_leagues_equal():
    w = decline_weights(ROWS, decline_factors(ROWS, SEASONS))
    y = np.array([r["accepted"] for r in ROWS], float)
    one = np.array([r["league"] == 1 for r in ROWS])
    assert w.sum() == pytest.approx(len(ROWS))
    assert w[one].sum() == pytest.approx(w[~one].sum())
    # 1 accepted against 3 withdrawn that each stand for 4 offers: 1 in 13, not 1 in 4.
    assert np.average(y[one], weights=w[one]) == pytest.approx(1 / 13)
    assert np.average(y[~one], weights=w[~one]) == pytest.approx(1 / 2)

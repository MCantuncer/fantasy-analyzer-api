import pytest

import app.analysis.perceived as perceived
import scripts.offer_replay as replay
from app.analysis.model import CategoryModel
from app.analysis.perceived import PerceivedValue


def test_set_takes_only_numbers_trade_value_has():
    assert replay.weight_overrides(["W_PROJECTION=0", "DRAFT_FADE_GAMES=1e9"]) == {
        "W_PROJECTION": 0.0,
        "DRAFT_FADE_GAMES": 1e9,
    }
    for bad in ("W_NOPE=1", "np=1", "PerceivedValue=1", "W_SEASON=lots", "W_SEASON"):
        with pytest.raises(SystemExit):
            replay.weight_overrides([bad])


def test_overridden_weights_reach_trade_value(league, monkeypatch):
    model = CategoryModel(league)
    roster = league.team(3).roster

    def projection_weights():
        pv = PerceivedValue(league, model)
        return [w for p in roster for name, w, _ in pv.signals(p) if name == "projection"]

    assert projection_weights() and all(w == perceived.W_PROJECTION for w in projection_weights())
    monkeypatch.setattr(perceived, "W_PROJECTION", perceived.W_PROJECTION)  # put back after the test
    replay.use_weights({"W_PROJECTION": 0.0})
    assert projection_weights() and not any(projection_weights())


def test_a_season_without_a_draft_market_is_skipped(monkeypatch):
    flat = {pid: {"adp": 140.0} for pid in range(300)}  # what ESPN returns for some finished seasons
    real = {pid: {"adp": pid + 1.0, "auction_avg": 5.0} for pid in range(300)}
    monkeypatch.setattr(replay, "_meta", lambda league, season: {2026: flat, 2025: real}.get(season, {}))
    assert replay.has_draft_market(1, 2025)
    assert not replay.has_draft_market(1, 2026)
    assert not replay.has_draft_market(1, 2024)  # no pool file at all


def _offers(league: int, n: int, accepted_high: bool, first_id: int = 0) -> list[dict]:
    """n offers with ratios 1..n; the top quarter is accepted, or the bottom quarter."""
    top = range(n - n // 4, n) if accepted_high else range(n // 4)
    return [{"id": first_id + k, "league": league, "ratio": k + 1.0, "accepted": k in top} for k in range(n)]


def test_compare_scores_each_large_league_on_the_offers_both_replays_have():
    n = replay.LARGE_LEAGUE
    baseline = _offers(1, n, accepted_high=True) + _offers(2, 10, accepted_high=True, first_id=1000)
    flipped = [{**r, "ratio": -r["ratio"]} for r in baseline]
    only_here = [{"id": 5000 + k, "league": 3, "ratio": 1.0 + k, "accepted": k < 5} for k in range(n)]
    assert replay.compare(flipped + only_here, baseline) == {1: (1.0, 0.0)}
    assert replay.compare(baseline, baseline) == {1: (1.0, 1.0)}

"""Heavy tests (parameter sweeps). Run with ``-m slow``."""

import pytest

pytestmark = pytest.mark.slow

STRATEGY = "SMACrossoverStrategy"
SMALL_RANGES = {
    "fast_length": {"min": 10, "max": 12, "step": 2},
    "slow_length": {"min": 30, "max": 32, "step": 2},
}


def test_heatmap_grid_best_and_robustness(api_call, cached_asset):
    asset, interval = cached_asset
    code, envelope = api_call("heatmap", {
        "asset": asset, "interval": interval, "strategy": STRATEGY,
        "start_date": "2024-01-01", "end_date": "2024-03-01",
        "param_ranges": SMALL_RANGES,
        "workers": 1, "max_combos": 20,
    })
    assert code == 0 and envelope["ok"] is True, envelope.get("error")
    data = envelope["data"]
    assert len(data["grid"]) == 4
    assert data["x_param"] == "fast_length"
    assert data["y_param"] == "slow_length"
    assert set(data["best"]) >= {"by_profit", "by_net_profit", "by_sharpe", "by_drawdown"}
    assert len(data["robustness"]) == 4
    assert envelope["artifacts"]


def test_heatmap_rejects_too_many_combinations(api_call, cached_asset):
    asset, interval = cached_asset
    code, envelope = api_call("heatmap", {
        "asset": asset, "interval": interval, "strategy": STRATEGY,
        "start_date": "2024-01-01", "end_date": "2024-01-15",
        "param_ranges": {"fast_length": {"min": 5, "max": 50, "step": 1},
                         "slow_length": {"min": 20, "max": 60, "step": 1}},
        "max_combos": 10,
    })
    assert code == 1
    assert envelope["ok"] is False
    assert "max_combos" in envelope["error"]

"""Tests for the heatmap structured result builder (grid/best/robustness)."""

import numpy as np
import pandas as pd

from heatmap import _build_structured_results


def _frame():
    rows = []
    for x in [1, 2]:
        for y in [10, 20]:
            rows.append({
                "x": x,
                "y": y,
                "profit": float(x * y),
                "net_profit": float(x * y * 10),
                "num_trades": 3,
                "win_rate": 50.0,
                "profit_factor": 2.0,
                "drawdown": 1.0,
                "drawdown_pct": 1.0,
                "avg_trade_profit": 1.0,
                "avg_trade_profit_pct": 1.0,
                "avg_trade_duration": 1.0,
                "sharpe_ratio": 0.5,
                "sortino_ratio": 0.6,
                "volatility": 0.1,
                "pnl_image": "",
            })
    return pd.DataFrame(rows)


def test_structured_keys_and_axes():
    result = _build_structured_results(_frame(), {"a": np.array([1, 2]), "b": np.array([10, 20])})
    assert set(result) >= {"grid", "best", "robustness", "param_ranges", "x_param", "y_param"}
    assert result["x_param"] == "a"
    assert result["y_param"] == "b"
    assert len(result["grid"]) == 4


def test_best_contains_highest_profit():
    result = _build_structured_results(_frame(), {"a": [1, 2], "b": [10, 20]})
    top = result["best"]["by_profit"][0]
    assert top["profit"] == max(cell["profit"] for cell in result["grid"])


def test_robustness_uses_neighbours():
    result = _build_structured_results(_frame(), {"a": [1, 2], "b": [10, 20]})
    assert len(result["robustness"]) == 4
    # 2x2 grid: every cell sees all four cells as neighbours
    assert all(cell["neighbor_count"] == 4 for cell in result["robustness"])
    assert all(cell["neighbor_mean"] is not None for cell in result["robustness"])

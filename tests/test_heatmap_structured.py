"""Tests for the heatmap structured result builder (grid/best/robustness)."""

import json

import numpy as np
import pandas as pd

from heatmap import _build_structured_results, _write_sidecar


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


def test_plateaus_attached_and_consistent():
    result = _build_structured_results(_frame(), {"a": [1, 2], "b": [10, 20]})
    assert "plateaus" in result
    plateaus = result["plateaus"]
    assert plateaus is not None and "error" not in plateaus
    assert plateaus["x_param"] == "a" and plateaus["y_param"] == "b"
    assert plateaus["score"]["total_cells"] == 4


def test_plateau_config_is_passed_through():
    result = _build_structured_results(
        _frame(), {"a": [1, 2], "b": [10, 20]},
        plateau_config={"min_trades": 9999})
    assert result["plateaus"]["score"]["min_trades"] == 9999
    assert result["plateaus"]["score"]["valid_cells"] == 0


def test_write_sidecar_is_json_safe_and_typed(tmp_path):
    html = tmp_path / "heatmap.html"
    html.write_text("<html></html>")
    structured = _build_structured_results(_frame(), {"a": [1, 2], "b": [10, 20]})
    structured["grid"][0]["x"] = np.int64(structured["grid"][0]["x"])

    sidecar = _write_sidecar(str(html), structured, {"asset": "TEST"})
    assert sidecar == str(tmp_path / "heatmap.json")

    payload = json.loads((tmp_path / "heatmap.json").read_text())
    assert payload["meta"]["asset"] == "TEST"
    assert isinstance(payload["grid"][0]["x"], (int, float))
    assert payload["plateaus"]["x_param"] == "a"

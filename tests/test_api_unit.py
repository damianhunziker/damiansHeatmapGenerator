"""Unit tests for the core.api helpers (no data / no subprocess)."""

import numpy as np
import pandas as pd
import pytest

from core import api


class _StrategyWithRanges:
    @staticmethod
    def get_parameter_ranges():
        return {
            "a": np.arange(0, 3),
            "b": np.arange(0, 2),
            "c": np.arange(0, 2),
        }


class _StrategyWithParams:
    @staticmethod
    def get_parameters():
        return {"p": (1, "a parameter")}

    @staticmethod
    def get_parameter_ranges():
        return {}


def test_schema_exposes_all_tools():
    schema = api.get_schema()
    for tool in ["pnl", "chart_analysis", "heatmap", "automator", "fetcher",
                 "series", "plateaus", "validate_plateau", "plot_plateau"]:
        assert tool in schema["tools"]
    assert schema["intervals"]
    assert isinstance(schema["strategies"], dict)


def _plateau_grid():
    rows = []
    for x in range(6):
        for y in range(6):
            good = 2 <= x <= 4 and 2 <= y <= 4
            rows.append({
                "x": x, "y": y,
                "sharpe_ratio": 2.0 if good else 0.5,
                "profit_factor": 2.0 if good else 1.0,
                "drawdown_pct": 10.0 if good else 30.0,
                "num_trades": 100 if good else 50,
                "profit": 30.0 if good else 5.0,
            })
    return rows


def test_plateaus_tool_runs_on_inline_grid():
    envelope = api.run_tool("plateaus", {
        "grid": _plateau_grid(),
        "param_ranges": {"x": list(range(6)), "y": list(range(6))},
    })
    assert envelope["ok"] is True
    data = envelope["data"]
    assert data["source"] == "inline-grid"
    assert data["x_param"] == "x" and data["y_param"] == "y"
    assert data["region_count"] >= 1


def test_plateaus_tool_derives_ranges_from_grid():
    envelope = api.run_tool("plateaus", {"grid": _plateau_grid()})
    assert envelope["ok"] is True
    assert envelope["data"]["x_param"] == "x"


def test_plateaus_tool_reads_sidecar(tmp_path):
    import json

    sidecar = tmp_path / "hm.json"
    sidecar.write_text(json.dumps({
        "grid": _plateau_grid(),
        "param_ranges": {"x": list(range(6)), "y": list(range(6))},
    }))
    envelope = api.run_tool("plateaus", {"grid_path": str(sidecar)})
    assert envelope["ok"] is True
    assert envelope["data"]["source"] == str(sidecar)


def test_normalize_region_accepts_best_region_block():
    block = {"x": {"param": "a", "min": 1, "max": 3},
             "y": {"param": "b", "min": 4, "max": 6}}
    assert api._normalize_region(block) == {
        "a": {"min": 1, "max": 3}, "b": {"min": 4, "max": 6}}
    plain = {"a": {"min": 1, "max": 3}}
    assert api._normalize_region(plain) is plain


def test_validate_plateau_requires_region():
    envelope = api.run_tool("validate_plateau", {"start_date": "2024-01-01",
                                                 "end_date": "2024-02-01"})
    assert envelope["ok"] is False
    assert "region" in envelope["error"]


def test_validate_plateau_splits_windows(monkeypatch):
    calls = []

    def fake_heatmap(params, options):
        calls.append((params["start_date"], params["end_date"], params["param_ranges"]))
        return {"plateaus": {
            "region_count": 1,
            "best_region": {"region_score": 1.5},
            "representative": {"x": 1, "y": 2, "method": "medoid"},
        }}, []

    monkeypatch.setattr(api, "_run_heatmap", fake_heatmap)
    envelope = api.run_tool("validate_plateau", {
        "asset": "BTCUSDT", "strategy": "S",
        "start_date": "2024-01-01", "end_date": "2024-03-31", "windows": 3,
        "region": {"x": {"param": "a", "min": 1, "max": 3},
                   "y": {"param": "b", "min": 4, "max": 6}},
    })
    assert envelope["ok"] is True
    data = envelope["data"]
    assert len(calls) == 3
    assert set(calls[0][2].keys()) == {"a", "b"}
    assert data["summary"]["windows_with_plateau"] == 3
    assert data["summary"]["consistency"] == 1.0
    assert data["summary"]["representative_mean"] == {"x": 1.0, "y": 2.0}


def test_validate_plateau_reports_failed_windows(monkeypatch):
    def flaky_heatmap(params, options):
        raise RuntimeError("boom")

    monkeypatch.setattr(api, "_run_heatmap", flaky_heatmap)
    envelope = api.run_tool("validate_plateau", {
        "asset": "BTCUSDT", "strategy": "S",
        "start_date": "2024-01-01", "end_date": "2024-02-01", "windows": 2,
        "region": {"a": {"min": 1, "max": 3}},
    })
    assert envelope["ok"] is True
    data = envelope["data"]
    assert data["summary"]["consistency"] == 0.0
    assert all(not entry["ok"] for entry in data["results"])


def test_plot_plateau_writes_artifact(tmp_path):
    out = tmp_path / "overlay.png"
    envelope = api.run_tool("plot_plateau", {
        "grid": _plateau_grid(),
        "param_ranges": {"x": list(range(6)), "y": list(range(6))},
        "output": str(out),
    })
    assert envelope["ok"] is True
    assert out.exists()
    assert envelope["artifacts"][0]["path"] == str(out)


def test_parse_ranges_from_list():
    ranges = api._parse_param_ranges({"x": [1, 2, 3]}, _StrategyWithRanges, 100)
    assert list(ranges["x"]) == [1, 2, 3]


def test_parse_ranges_from_step_spec():
    ranges = api._parse_param_ranges({"x": {"min": 0, "max": 1, "step": 0.5}}, _StrategyWithRanges, 100)
    assert list(ranges["x"]) == [0.0, 0.5, 1.0]


def test_parse_ranges_from_count_spec():
    ranges = api._parse_param_ranges({"x": {"min": 0, "max": 1, "count": 3}}, _StrategyWithRanges, 100)
    assert list(ranges["x"]) == [0.0, 0.5, 1.0]


def test_parse_ranges_from_scalar():
    ranges = api._parse_param_ranges({"x": 7}, _StrategyWithRanges, 100)
    assert list(ranges["x"]) == [7]


def test_parse_ranges_enforces_max_combos():
    with pytest.raises(ValueError, match="max_combos"):
        api._parse_param_ranges({"x": {"min": 0, "max": 10, "step": 1}}, _StrategyWithRanges, 5)


def test_parse_ranges_default_uses_first_two_strategy_ranges():
    ranges = api._parse_param_ranges(None, _StrategyWithRanges, 100)
    assert set(ranges.keys()) == {"a", "b"}


def test_parse_ranges_requires_min_and_max():
    with pytest.raises(ValueError, match="min"):
        api._parse_param_ranges({"x": {"min": 0}}, _StrategyWithRanges, 100)


def test_build_strategy_params_merges_defaults_and_common():
    params = api._build_strategy_params(
        _StrategyWithParams,
        {"asset": "BTCUSDT", "initial_equity": 500, "p": 9, "trade_direction": "long"},
    )
    assert params["p"] == 9
    assert params["initial_equity"] == 500
    assert params["fee_pct"] == 0.04
    assert params["asset"] == "BTCUSDT"
    assert params["trade_direction"] == "long"


def test_lookback_candles_counts_in_range():
    index = pd.date_range("2024-01-01", periods=10, freq="D")
    data = pd.DataFrame({"price_close": range(10)}, index=index)
    lookback, end_lookback = api._lookback_candles(data, "2024-01-03", "2024-01-06")
    assert (lookback, end_lookback) == (4, 6)


def test_lookback_candles_none_when_no_dates():
    data = pd.DataFrame(index=pd.date_range("2024-01-01", periods=3))
    assert api._lookback_candles(data, None, None) == (None, None)


def test_clean_range_rounds_floats():
    assert api._clean_range(np.arange(0.0, 0.3, 0.1)) == [0.0, 0.1, 0.2]


def test_run_tool_unknown_returns_error_envelope():
    envelope = api.run_tool("does-not-exist", {})
    assert envelope["ok"] is False
    assert "Unknown tool" in envelope["error"]


def test_artifact_builds_path_and_url(tmp_path):
    target = tmp_path / "report.html"
    target.write_text("hi")
    artifact = api._artifact(str(target))
    assert artifact["path"] == str(target)
    assert artifact["url"].startswith("http")


def test_dumps_serializes_non_finite_and_numpy():
    text = api.dumps({"x": float("inf"), "y": np.int64(2)})
    assert '"x": null' in text
    assert '"y": 2' in text

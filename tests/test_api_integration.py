"""End-to-end tests of the ``--api`` JSON contract (subprocess, cached data)."""

import pytest

pytestmark = pytest.mark.integration

STRATEGY = "SMACrossoverStrategy"
START = "2024-01-01"
END = "2024-02-01"
SMALL_RANGES = {
    "fast_length": {"min": 10, "max": 12, "step": 2},
    "slow_length": {"min": 30, "max": 32, "step": 2},
}


def _common(asset, interval):
    return {"asset": asset, "interval": interval, "strategy": STRATEGY,
            "start_date": START, "end_date": END}


def test_schema_api(api_call):
    code, envelope = api_call("schema", {})
    assert code == 0
    assert envelope["ok"] is True
    assert "pnl" in envelope["data"]["tools"]


def test_unknown_tool_fails_with_nonzero_exit(api_call):
    code, envelope = api_call("does-not-exist", {})
    assert code == 1
    assert envelope["ok"] is False
    assert "Unknown tool" in envelope["error"]


def test_fetcher_returns_cache_range(api_call, cached_asset):
    asset, interval = cached_asset
    code, envelope = api_call("fetcher", {"asset": asset, "interval": interval})
    assert code == 0 and envelope["ok"] is True
    assert envelope["data"]["rows"] > 0
    assert envelope["data"]["start"] and envelope["data"]["end"]


def test_pnl_returns_all_directions(api_call, cached_asset):
    asset, interval = cached_asset
    code, envelope = api_call("pnl", {**_common(asset, interval), "direction": "both"})
    assert code == 0 and envelope["ok"] is True, envelope.get("error")
    directions = envelope["data"]["directions"]
    assert set(directions) == {"both", "long", "short"}
    both = directions["both"]
    assert "metrics" in both and isinstance(both["trades"], list)
    assert envelope["artifacts"] and envelope["artifacts"][0]["url"]


def test_chart_analysis_returns_summary(api_call, cached_asset):
    asset, interval = cached_asset
    code, envelope = api_call("chart_analysis", _common(asset, interval))
    assert code == 0 and envelope["ok"] is True, envelope.get("error")
    data = envelope["data"]
    assert {"summary", "trades", "divergence_indicators"} <= set(data)
    assert "win_rate" in data["summary"]
    assert envelope["artifacts"]


def test_series_inline_rows(api_call, cached_asset):
    asset, interval = cached_asset
    code, envelope = api_call("series", {
        **_common(asset, interval),
        "end_date": "2024-01-15",
        "columns": "price_close,long_entry",
        "tail": 3,
    })
    assert code == 0 and envelope["ok"] is True, envelope.get("error")
    data = envelope["data"]
    assert data["count"] == 3
    assert data["columns"] == ["price_close", "long_entry"]
    assert len(data["rows"]) == 3


def test_series_writes_file_for_parquet(api_call, cached_asset):
    asset, interval = cached_asset
    code, envelope = api_call("series", {
        **_common(asset, interval),
        "columns": "price_close",
        "format": "parquet",
    })
    assert code == 0 and envelope["ok"] is True, envelope.get("error")
    assert envelope["data"]["path"].endswith((".parquet", ".csv"))


def test_heatmap_derived_axis(api_call, cached_asset):
    asset, interval = cached_asset
    code, envelope = api_call("heatmap", {
        "asset": asset, "interval": interval, "strategy": STRATEGY,
        "start_date": "2024-01-01", "end_date": "2024-03-01",
        "param_ranges": {"kama_scale": {"min": 1, "max": 2, "step": 1}, "fast_length": [10]},
        "derived_params": {"slow_length": {"scale": "kama_scale", "base": 30, "kind": "window"}},
        "workers": 1,
    })
    assert code == 0 and envelope["ok"] is True, envelope.get("error")
    data = envelope["data"]
    assert data["derived_params"]["slow_length"]["kind"] == "window"
    assert len(data["grid"]) == 2


def test_heatmap_derived_warmup_warning(api_call, cached_asset):
    asset, interval = cached_asset
    code, envelope = api_call("heatmap", {
        "asset": asset, "interval": interval, "strategy": STRATEGY,
        "start_date": "2024-01-01", "end_date": "2024-03-01",
        "param_ranges": {"kama_scale": {"min": 1, "max": 2, "step": 1}, "fast_length": [10]},
        "derived_params": {"kama_slow": {"scale": "kama_scale", "base": 5000, "kind": "alpha"}},
        "workers": 1,
    })
    assert code == 0 and envelope["ok"] is True, envelope.get("error")
    assert envelope["warnings"], "expected a KAMA warmup warning"
    assert "warmup" in envelope["warnings"][0].lower()


def test_heatmap_invalid_derived_scale_is_rejected(api_call, cached_asset):
    asset, interval = cached_asset
    code, envelope = api_call("heatmap", {
        "asset": asset, "interval": interval, "strategy": STRATEGY,
        "start_date": "2024-01-01", "end_date": "2024-03-01",
        "param_ranges": {"kama_scale": [1, 2], "fast_length": [10]},
        "derived_params": {"slow_length": {"scale": "does_not_exist", "base": 30}},
        "workers": 1,
    })
    assert code == 1 and envelope["ok"] is False
    assert "not a swept parameter" in envelope["error"]

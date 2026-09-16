"""Tests for the generic parameter resolution suite (core.params)."""

import numpy as np
import pytest

from core.params import (
    DEFAULT_CONFIG_PATH,
    ResolutionError,
    load_resolver_config,
    normalize_config,
    register_op,
    resolve_params,
    resolver_max_period,
)
from core.params.ops import eval_node, validate_node


# ---------------------------------------------------------------------------
# ops
# ---------------------------------------------------------------------------

def test_eval_scale_ops():
    scope = {"k": 2}
    assert eval_node({"op": "scale_window", "base": 16, "k": {"$input": True}}, scope, "k", {}) == 32
    assert eval_node({"op": "scale_alpha", "base": 4, "k": {"$input": True}}, scope, "k", {}) == 9
    assert eval_node({"op": "scale_alpha", "base": 24, "k": {"$input": True}}, scope, "k", {}) == 49
    assert eval_node({"op": "scale_naive", "base": 4, "k": {"$input": True}}, scope, "k", {}) == 8


def test_eval_arithmetic_and_bounds():
    scope = {"risk_pct": 5, "k": 4}
    node = {"op": "sub", "args": [1.0, {"op": "div", "args": [{"$param": "risk_pct"}, 100]}]}
    assert eval_node(node, scope, "k", {}) == pytest.approx(0.95)
    assert eval_node({"op": "clamp", "args": [{"$input": True}, 1, 3]}, scope, "k", {}) == 3
    assert eval_node({"op": "max", "args": [2, 3, 1]}, scope, "k", {}) == 3


def test_eval_snippet_binds_args():
    snippets = {"scaled": {"op": "mul", "args": [{"$param": "base"}, {"$param": "k"}]}}
    node = {"$snippet": "scaled", "args": {"base": 16, "k": {"$input": True}}}
    assert eval_node(node, {"k": 3}, "k", snippets) == 48


def test_eval_unknown_param_raises():
    with pytest.raises(ResolutionError, match="unknown parameter"):
        eval_node({"$param": "nope"}, {}, None, {})


def test_validate_rejects_unknown_op_and_snippet():
    with pytest.raises(ResolutionError, match="unknown op"):
        validate_node({"op": "drop_table"}, {}, "x")
    with pytest.raises(ResolutionError, match="unknown snippet"):
        validate_node({"$snippet": "nope"}, {}, "x")


def test_register_custom_op():
    register_op("triple", lambda value: 3 * value)
    assert eval_node({"op": "triple", "value": 4}, {}, None, {}) == 12


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

def _minimal_config(**overrides):
    cfg = {
        "version": 1,
        "resolvers": [{
            "id": "r1",
            "input": "kama_normalized_length",
            "outputs": {"entry_kama_length": {"op": "scale_window", "base": 16, "k": {"$input": True}}},
        }],
    }
    cfg.update(overrides)
    return cfg


def test_normalize_config_ok():
    out = normalize_config(_minimal_config())
    assert out["resolvers"][0]["id"] == "r1"
    assert out["resolvers"][0]["input"] == "kama_normalized_length"


def test_normalize_config_rejects_duplicate_id():
    cfg = _minimal_config(resolvers=[_minimal_config()["resolvers"][0]] * 2)
    with pytest.raises(ResolutionError, match="duplicate resolver id"):
        normalize_config(cfg)


def test_normalize_config_requires_outputs():
    cfg = _minimal_config(resolvers=[{"id": "x", "input": "k", "outputs": {}}])
    with pytest.raises(ResolutionError, match="non-empty 'outputs'"):
        normalize_config(cfg)


def test_normalize_config_rejects_bad_version():
    with pytest.raises(ResolutionError, match="version"):
        normalize_config(_minimal_config(version=99))


def test_default_config_exists_and_loads():
    import os
    assert os.path.exists(DEFAULT_CONFIG_PATH)
    cfg = load_resolver_config(use_default=True)
    ids = {r["id"] for r in cfg["resolvers"]}
    assert "kama_normalized_length" in ids
    assert "slow_kama_normalized_length" in ids
    assert "all_exact_window" in cfg["snippets"]


def test_inline_overrides_default_by_id():
    inline = {
        "resolvers": [{
            "id": "kama_normalized_length",
            "input": "kama_normalized_length",
            "outputs": {"entry_kama_length": {"op": "scale_naive", "base": 16, "k": {"$input": True}}},
        }]
    }
    cfg = load_resolver_config(inline=inline)
    # only one resolver with that id remains, and it is the inline one
    matches = [r for r in cfg["resolvers"] if r["id"] == "kama_normalized_length"]
    assert len(matches) == 1
    assert matches[0]["outputs"]["entry_kama_length"]["op"] == "scale_naive"


# ---------------------------------------------------------------------------
# suite / resolve
# ---------------------------------------------------------------------------

def test_resolve_all_exact_for_live_kama():
    cfg = load_resolver_config()
    params = {"kama_normalized_length": 2, "entry_filter": 0.7}
    out = resolve_params(params, cfg["resolvers"], strategy="LiveKAMASSLStrategy",
                         snippets=cfg["snippets"])
    assert out["entry_kama_length"] == 32
    assert out["entry_kama_fast"] == 9
    assert out["entry_kama_slow"] == 49
    assert out["kama2_length"] == 30
    assert out["exit_kama_slow"] == 41
    assert out["entry_filter"] == 0.7  # untouched


def test_resolve_for_dmx():
    cfg = load_resolver_config()
    out = resolve_params({"kama_normalized_length": 2}, cfg["resolvers"],
                         strategy="DMXStrategy", snippets=cfg["snippets"])
    assert out["slow_kama_length"] == 1000
    assert out["slow_kama_fast"] == 49
    assert out["slow_kama_slow"] == 721


def test_resolve_is_strategy_scoped():
    cfg = load_resolver_config()
    out = resolve_params({"kama_normalized_length": 2}, cfg["resolvers"],
                         strategy="SMACrossoverStrategy", snippets=cfg["snippets"])
    assert "entry_kama_length" not in out
    assert "slow_kama_length" not in out


def test_resolve_noop_without_input():
    cfg = load_resolver_config()
    params = {"entry_filter": 0.7}
    out = resolve_params(params, cfg["resolvers"], strategy="LiveKAMASSLStrategy",
                         snippets=cfg["snippets"])
    assert out == params


def test_resolve_records():
    cfg = load_resolver_config()
    records = []
    resolve_params({"kama_normalized_length": 3}, cfg["resolvers"],
                   strategy="LiveKAMASSLStrategy", snippets=cfg["snippets"], records=records)
    assert records and records[0]["id"] == "kama_normalized_length"
    assert records[0]["value"] == 3
    assert "entry_kama_length" in records[0]["resolved"]


def test_resolver_fires_when_swept_by_id_alias():
    """Sweeping the resolver id (e.g. slow_kama_normalized_length) also works."""
    cfg = load_resolver_config()
    records = []
    out = resolve_params({"slow_kama_normalized_length": 2}, cfg["resolvers"],
                         strategy="DMXStrategy", snippets=cfg["snippets"], records=records)
    assert out["slow_kama_length"] == 1000
    assert out["slow_kama_fast"] == 49
    assert out["slow_kama_slow"] == 721
    assert records[0]["swept"] == "slow_kama_normalized_length"
    assert records[0]["input"] == "kama_normalized_length"


def test_id_alias_is_strategy_scoped():
    cfg = load_resolver_config()
    out = resolve_params({"slow_kama_normalized_length": 2}, cfg["resolvers"],
                         strategy="LiveKAMASSLStrategy", snippets=cfg["snippets"])
    # the DMX resolver must not fire for LiveKAMA
    assert "slow_kama_length" not in out


def test_resolver_max_period_with_id_alias():
    cfg = load_resolver_config()
    ranges = {"slow_kama_normalized_length": np.array([2])}
    assert resolver_max_period(ranges, cfg["resolvers"], snippets=cfg["snippets"]) == 721


# ---------------------------------------------------------------------------
# clean conversion at the API/MCP boundary (prepare_heatmap_params)
# ---------------------------------------------------------------------------

def test_prepare_heatmap_params_converts_alias_to_input():
    from core.api import prepare_heatmap_params
    from classes.strategies_dmx.dmx_strategy import DMXStrategy

    prep = prepare_heatmap_params(
        {"param_ranges": {"slow_kama_normalized_length": [1, 2], "squeeze_limit": [150]}},
        DMXStrategy, 100)
    assert "kama_normalized_length" in prep["ranges"]
    assert "slow_kama_normalized_length" not in prep["ranges"]
    assert prep["conversions"] == {"slow_kama_normalized_length": "kama_normalized_length"}


def test_prepare_heatmap_params_normalizes_derived_alias():
    from core.api import prepare_heatmap_params
    from classes.strategies_dmx.dmx_strategy import DMXStrategy

    prep = prepare_heatmap_params({
        "param_ranges": {"slow_kama_normalized_length": [1, 2], "squeeze_limit": [150]},
        "derived_params": {
            "slow_kama_normalized_length": {
                "base": 16, "kind": "window", "scale": "slow_kama_normalized_length"}},
    }, DMXStrategy, 100)
    assert "kama_normalized_length" in prep["derived"]
    assert prep["derived"]["kama_normalized_length"]["scale"] == "kama_normalized_length"


def test_prepare_heatmap_params_rejects_virtual_without_resolver():
    from core.api import prepare_heatmap_params
    from classes.strategies.macd_strategy import SMACrossoverStrategy

    with pytest.raises(ValueError, match="virtual parameter"):
        prepare_heatmap_params(
            {"param_ranges": {"kama_normalized_length": [1, 2], "fast_length": [10]}},
            SMACrossoverStrategy, 100)


def test_prepare_heatmap_params_rejects_alias_and_input_together():
    from core.api import prepare_heatmap_params
    from classes.strategies_dmx.dmx_strategy import DMXStrategy

    with pytest.raises(ValueError, match="alias"):
        prepare_heatmap_params(
            {"param_ranges": {"slow_kama_normalized_length": [1, 2],
                              "kama_normalized_length": [1, 2], "squeeze_limit": [150]}},
            DMXStrategy, 100)


def test_prepare_heatmap_params_uses_config_range_token():
    from core.api import prepare_heatmap_params
    from classes.strategies_dmx.dmx_strategy import DMXStrategy

    prep = prepare_heatmap_params(
        {"param_ranges": {"kama_normalized_length": "config", "squeeze_limit": [150]}},
        DMXStrategy, 100)
    assert prep["ranges"]["kama_normalized_length"].tolist() == [1, 2, 3, 4, 5, 6, 7, 8]




def test_hook_and_middleware_run():
    from core.params import hook, middleware

    @hook("pre_resolve", priority=-100)
    def _pre(params, ctx):
        if params.get("__marker__"):
            params = dict(params)
            params["hooked"] = True
        return params

    @middleware(priority=-100)
    def _mw(nxt, params, ctx):
        out = nxt(params, ctx)
        if params.get("__marker__"):
            out["middlewared"] = True
        return out

    cfg = load_resolver_config()
    out = resolve_params({"kama_normalized_length": 2, "__marker__": True},
                         cfg["resolvers"], strategy="LiveKAMASSLStrategy",
                         snippets=cfg["snippets"])
    assert out.get("hooked") is True
    assert out.get("middlewared") is True


def test_resolver_max_period():
    cfg = load_resolver_config()
    ranges = {"kama_normalized_length": np.array([1, 4])}
    # k=4 -> slow = 4*(360+1)-1 = 1443 for DMX, 4*(24+1)-1=99 for LiveKAMA
    assert resolver_max_period(ranges, cfg["resolvers"], snippets=cfg["snippets"]) == 1443


def test_resolver_max_period_name_filter():
    cfg = load_resolver_config()
    ranges = {"kama_normalized_length": np.array([4])}
    only_lengths = resolver_max_period(
        ranges, cfg["resolvers"], snippets=cfg["snippets"],
        name_filter=lambda n: "length" in n.lower())
    # slow_kama_length = 4*500 = 2000
    assert only_lengths == 2000


# ---------------------------------------------------------------------------
# virtual parameter ranges (min/max/step)
# ---------------------------------------------------------------------------

def test_default_config_declares_virtual_range():
    cfg = load_resolver_config()
    assert cfg["virtual_params"]["kama_normalized_length"]["range"] == {
        "min": 1, "max": 8, "step": 1}


def test_resolver_level_range_is_merged():
    cfg = normalize_config({
        "version": 1,
        "resolvers": [{
            "id": "r", "input": "v", "range": {"min": 1, "max": 3, "step": 1},
            "outputs": {"x": {"op": "scale_window", "base": 2, "k": {"$input": True}}},
        }],
    })
    assert cfg["virtual_params"]["v"]["range"] == {"min": 1, "max": 3, "step": 1}


def test_top_level_virtual_range_wins_over_resolver():
    cfg = normalize_config({
        "version": 1,
        "virtual_params": {"v": {"range": {"min": 5, "max": 6, "step": 1}}},
        "resolvers": [{
            "id": "r", "input": "v", "range": {"min": 1, "max": 2, "step": 1},
            "outputs": {"x": {"op": "scale_window", "base": 2, "k": {"$input": True}}},
        }],
    })
    assert cfg["virtual_params"]["v"]["range"] == {"min": 5, "max": 6, "step": 1}


def test_bad_virtual_range_rejected():
    with pytest.raises(ResolutionError, match="needs 'min' and 'max'"):
        normalize_config({"version": 1, "virtual_params": {"v": {"range": {"min": 1}}}})


def test_parse_param_ranges_accepts_min_max_step_for_virtual():
    from core.api import _parse_param_ranges
    ranges = _parse_param_ranges(
        {"kama_normalized_length": {"min": 1, "max": 4, "step": 1}, "entry_filter": [0.7]},
        None, 400)
    assert ranges["kama_normalized_length"].tolist() == [1, 2, 3, 4]


def test_parse_param_ranges_config_token():
    from core.api import _parse_param_ranges
    defaults = {"kama_normalized_length": {"range": {"min": 1, "max": 4, "step": 1}}}
    ranges = _parse_param_ranges(
        {"kama_normalized_length": "config", "entry_filter": [0.7]},
        None, 400, virtual_defaults=defaults)
    assert ranges["kama_normalized_length"].tolist() == [1, 2, 3, 4]


def test_parse_param_ranges_use_config_ranges_injects_missing():
    from core.api import _parse_param_ranges
    defaults = {"kama_normalized_length": {"range": {"min": 1, "max": 3, "step": 1}}}
    ranges = _parse_param_ranges(
        {"entry_filter": [0.7]}, None, 400,
        virtual_defaults=defaults, use_config_ranges=True)
    assert ranges["kama_normalized_length"].tolist() == [1, 2, 3]


def test_parse_param_ranges_config_token_without_range_raises():
    from core.api import _parse_param_ranges
    with pytest.raises(ValueError, match="defines none"):
        _parse_param_ranges({"v": "config"}, None, 400, virtual_defaults={})


def test_parse_param_ranges_count_spec():
    from core.api import _parse_param_ranges
    ranges = _parse_param_ranges(
        {"kama_normalized_length": {"min": 1, "max": 8, "count": 4}, "entry_filter": [0.7]},
        None, 400)
    assert ranges["kama_normalized_length"].tolist() == [1.0, 3.3333333333333335, 5.666666666666667, 8.0]


def test_describe_resolvers_includes_virtual_params():
    from core.params import describe_resolvers
    cfg = load_resolver_config()
    desc = describe_resolvers(cfg["resolvers"], cfg["snippets"], cfg["virtual_params"])
    assert "kama_normalized_length" in desc["virtual_params"]
    assert desc["virtual_params"]["kama_normalized_length"]["range"]["max"] == 8


def test_applicable_virtual_params_filters_by_strategy():
    from core.params import applicable_virtual_params
    cfg = load_resolver_config()
    live = applicable_virtual_params(
        cfg["virtual_params"], cfg["resolvers"], "LiveKAMASSLStrategy")
    sma = applicable_virtual_params(
        cfg["virtual_params"], cfg["resolvers"], "SMACrossoverStrategy")
    assert "kama_normalized_length" in live
    assert "kama_normalized_length" not in sma



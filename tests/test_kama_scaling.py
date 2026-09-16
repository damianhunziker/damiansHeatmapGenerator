"""Tests for exact KAMA parameter scaling (core.kama_scaling)."""

import numpy as np
import pytest

from core.kama_scaling import (
    expand_derived,
    max_effective_period,
    normalize_derived,
    scale_alpha_period,
    scale_window,
)


def test_window_scaling():
    assert scale_window(16, 1) == 16
    assert scale_window(16, 6) == 96
    assert scale_window(16, 0.5) == 8
    assert scale_window(0.2, 1) >= 1


def test_alpha_scaling_is_exact():
    # exact rule: P' = k * (P + 1) - 1
    assert scale_alpha_period(2, 6) == 17
    assert scale_alpha_period(4, 6) == 29
    assert scale_alpha_period(24, 6) == 149
    # deliberately different from the naive k*P approximation
    assert scale_alpha_period(4, 6) != 4 * 6
    assert scale_alpha_period(2, 1) == 2


def test_alpha_period_has_minimum_two():
    assert scale_alpha_period(2, 0.1) == 2


def test_expand_derived_applies_exact_scaling():
    specs = {
        "entry_kama_length": {"scale": "kama_scale", "base": 16, "kind": "window"},
        "entry_kama_fast": {"scale": "kama_scale", "base": 4, "kind": "alpha"},
    }
    out = expand_derived({"kama_scale": 6, "entry_filter": 0.7}, specs)
    assert out["entry_kama_length"] == 96
    assert out["entry_kama_fast"] == 29
    assert out["entry_filter"] == 0.7  # untouched


def test_expand_derived_ignores_missing_scale():
    specs = {"x": {"scale": "kama_scale", "base": 1, "kind": "window"}}
    assert "x" not in expand_derived({"other": 1}, specs)


def test_normalize_derived_rejects_unknown_scale():
    ranges = {"kama_scale": np.array([1, 2])}
    with pytest.raises(ValueError, match="not a swept parameter"):
        normalize_derived({"x": {"scale": "nope", "base": 1}}, ranges)


def test_normalize_derived_requires_base():
    ranges = {"kama_scale": np.array([1, 2])}
    with pytest.raises(ValueError, match="base"):
        normalize_derived({"x": {"scale": "kama_scale"}}, ranges)


def test_normalize_derived_rejects_bad_kind():
    ranges = {"kama_scale": np.array([1, 2])}
    with pytest.raises(ValueError, match="kind"):
        normalize_derived({"x": {"scale": "kama_scale", "base": 1, "kind": "bogus"}}, ranges)


def test_max_effective_period():
    ranges = {"kama_scale": np.array([1, 6])}
    specs = {"entry_kama_slow": {"scale": "kama_scale", "base": 24, "kind": "alpha"}}
    assert max_effective_period(ranges, specs) == 149

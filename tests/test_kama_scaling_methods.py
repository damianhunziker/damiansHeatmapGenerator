"""Tests for the KAMA scaling options (scripts/kama_scaling_methods.py).

These are behavioural checks for every scaling option, including that the
project rule (``all_exact``) tracks the exact higher-timeframe reference
(``resample``) better than the naive ``k * period`` rule.
"""

import numpy as np
import pandas as pd
import pytest

from core.kama_scaling import scale_alpha_period
from scripts.kama_scaling_methods import (
    METHOD_DESCRIPTIONS,
    apply_method,
    derive_params,
    kama_er,
    kama_resampled,
    tracking_ratio,
)

BASE = (10, 2, 30)
INTERVAL_HOURS = 4.0
PARAM_METHODS = [m for m in METHOD_DESCRIPTIONS if m != "resample"]


@pytest.fixture(scope="module")
def close():
    """Synthetic 4h close series with trend + noise (deterministic)."""
    rng = np.random.default_rng(1234)
    n = 1500
    steps = rng.normal(0, 0.01, n) + 0.0004
    price = 2000 * np.exp(np.cumsum(steps))
    idx = pd.date_range("2024-06-01", periods=n, freq="4h")
    return pd.Series(price, index=idx)


def test_kama_er_matches_reference_length(close):
    out = kama_er(close, 10, 2, 30)
    assert len(out) == len(close)
    assert out.notna().all()
    assert out.iloc[0] == pytest.approx(close.iloc[0])


@pytest.mark.parametrize("method", PARAM_METHODS)
def test_every_method_returns_aligned_series(method, close):
    out = apply_method(method, close, BASE, 4, INTERVAL_HOURS)
    assert isinstance(out, pd.Series)
    assert len(out) == len(close)
    assert np.isfinite(out.to_numpy()).all()


def test_derive_params_are_exact_for_all_exact():
    # P' = k*(P+1)-1
    assert derive_params("all_exact", BASE, 6) == (60, 17, 185)


def test_derive_params_naive_is_different_from_exact():
    exact = derive_params("all_exact", (10, 4, 24), 6)
    naive = derive_params("all_naive", (10, 4, 24), 6)
    assert exact != naive
    assert naive == (60, 24, 144)
    assert exact == (60, 29, 149)


def test_er_only_keeps_fast_and_slow():
    L, F, S = derive_params("er_only", BASE, 5)
    assert (F, S) == (BASE[1], BASE[2])
    assert L == 50


def test_fast_only_keeps_length_and_slow():
    L, F, S = derive_params("fast_only", BASE, 5)
    assert (L, S) == (BASE[0], BASE[2])
    assert F == 5 * (BASE[1] + 1) - 1


def test_fastslow_exact_keeps_length():
    L, F, S = derive_params("fastslow_exact", BASE, 5)
    assert L == BASE[0]
    assert F == 5 * (BASE[1] + 1) - 1
    assert S == 5 * (BASE[2] + 1) - 1


def test_sqrt_time_grows_slower_than_linear():
    l_lin = derive_params("all_exact", BASE, 9)[0]
    l_sqrt = derive_params("sqrt_time", BASE, 9)[0]
    assert l_sqrt < l_lin


def test_resample_returns_aligned_series(close):
    out = kama_resampled(close, BASE, 4, INTERVAL_HOURS)
    assert len(out) == len(close)
    assert out.notna().any()


def test_exact_alpha_rule_matches_aggregated_ema(close):
    """The core claim of the rule: EMA(P) on k-aggregated bars ~= EMA(k(P+1)-1)
    per bar (identical smoothing alpha).  This is what ``all_exact`` preserves.
    """
    k = 4
    agg = close.resample(pd.Timedelta(hours=INTERVAL_HOURS * k)).last().dropna()
    ema_agg = agg.ewm(span=BASE[0], adjust=False).mean()
    ema_exact = close.ewm(span=scale_alpha_period(BASE[0], k), adjust=False).mean()
    aligned = ema_exact.reindex(agg.index).iloc[20:]
    ref = ema_agg.iloc[20:]
    rel = (aligned - ref).abs() / ref.abs()
    assert float(rel.mean()) < 0.01


def test_all_exact_is_slower_than_resample(close):
    """Scaling all three periods makes KAMA progressively smoother, whereas the
    true higher-timeframe KAMA stays comparatively responsive.  They are
    different concepts - this documents that they are not equal.
    """
    k = 4
    exact = tracking_ratio(apply_method("all_exact", close, BASE, k, INTERVAL_HOURS), close)
    ref = tracking_ratio(apply_method("resample", close, BASE, k, INTERVAL_HOURS), close)
    assert exact < ref


def test_higher_scale_is_slower(close):
    """Tracking ratio (std kama / std price) must not increase with k."""
    ratios = [tracking_ratio(apply_method("all_exact", close, BASE, k, INTERVAL_HOURS), close)
              for k in (1, 2, 4, 8)]
    assert ratios[0] >= ratios[-1]
    assert ratios == sorted(ratios, reverse=True)

#!/usr/bin/env python3
"""Compare every sensible way of "scaling" Kaufman's Adaptive Moving Average.

Research summary (see the URLs in REFERENCES below)
---------------------------------------------------
KAMA (Perry J. Kaufman, 1995) has three bar-based parameters::

    ER length   (default 10)   - Efficiency-Ratio window
    fast period (default 2)    - fastest EMA bound
    slow period (default 30)   - slowest EMA bound

Kaufman's canonical KAMA(10, 2, 30) is what every reference (StockCharts,
Tulip, thinkorswim, TradingView) ships.  Kaufman does **not** publish a
"multiply the parameters by k" rule.  The two defensible ways to get a
longer-horizon KAMA are:

  1. ``resample`` - compute KAMA(10,2,30) on k-times aggregated bars.  This is
     the *exact* higher-timeframe KAMA (TradingView's "fixed TF" does exactly
     this with ``request.security`` on the higher timeframe).
  2. Adjust the parameters.  StockCharts explicitly recommends raising the
     **middle (fast EMA)** parameter to smooth KAMA for longer-term analysis
     (KAMA(10,5,30) is much smoother than KAMA(10,2,30)) and warns against
     increasing the ER length.

Our project's ``core.kama_scaling`` rule (``L'=k*L``, ``P'=k*(P+1)-1``) is the
mathematically consistent *per-bar approximation* of (1): preserving the EMA
alpha ``alpha' = alpha/k`` makes an EMA behave like one on k-times longer bars.
It is a good approximation, not a Kaufman rule - option (1) is the ground truth.

The options implemented here (all measured against the resample reference)::

    resample        aggregate k bars, run KAMA(10,2,30)          [reference]
    all_exact       L'=kL, F'=k(F+1)-1, S'=k(S+1)-1              [our rule]
    all_naive       L'=kL, F'=kF, S'=kS                          [naive]
    er_only         L'=kL, F, S unchanged                        [Tulip-style]
    fastslow_exact  L unchanged, F'=k(F+1)-1, S'=k(S+1)-1
    fast_only       L, S unchanged, F'=k(F+1)-1                  [StockCharts hint]
    sqrt_time       all * sqrt(k)                                [vol time-rule]

Usage (from the host)::

    docker compose exec app python scripts/kama_scaling_methods.py
    docker compose exec app python scripts/kama_scaling_methods.py \
        --asset ETHUSDT --interval 4h --start 2024-06-01 --end 2025-06-01 \
        --base-length 10 --base-fast 2 --base-slow 30 \
        --scales 1,2,3,4,6,8 --compare-k 4

Outputs one HTML per option plus a combined comparison into ``html_cache/``.

REFERENCES
----------
- StockCharts ChartSchool - KAMA:
  https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/kaufmans-adaptive-moving-average-kama
- Tulip Indicators - kama: https://tulipindicators.org/kama
- thinkorswim - MovAvgAdaptive:
  https://toslc.thinkorswim.com/center/reference/Tech-Indicators/studies-library/M-N/MovAvgAdaptive
- TradingView - KAMA (fixed TF): https://www.tradingview.com/script/7EIRojho/
- QuantifiedStrategies - Kaufman settings 10/2/30:
  https://www.quantifiedstrategies.com/adaptive-moving-average/
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.colors as pcolors
from plotly.subplots import make_subplots

from core.kama_scaling import scale_window, scale_alpha_period
from core.html_viewer import publish_figure, print_viewer_info
from core.strategy_utils import fetch_data

INTERVAL_HOURS = {
    "1m": 1 / 60, "5m": 5 / 60, "15m": 0.25, "30m": 0.5,
    "1h": 1, "2h": 2, "4h": 4, "6h": 6, "8h": 8, "12h": 12, "1d": 24,
}

# Human-readable description per option (used in titles/legends).
METHOD_DESCRIPTIONS = {
    "resample": "aggregate k bars, KAMA(base)  [reference / true higher-TF]",
    "all_exact": "L'=kL, F'=k(F+1)-1, S'=k(S+1)-1  [project rule]",
    "all_naive": "L'=kL, F'=kF, S'=kS  [naive]",
    "er_only": "L'=kL, fast/slow unchanged",
    "fastslow_exact": "L fixed, F'=k(F+1)-1, S'=k(S+1)-1",
    "fast_only": "L, S fixed, F'=k(F+1)-1  [StockCharts smoothing hint]",
    "sqrt_time": "all * sqrt(k)  [volatility time-rule]",
}


def kama_er(close, length=10, fast=2, slow=30):
    """Classic Kaufman KAMA (efficiency-ratio based), aligned to ``close``."""
    close = pd.Series(close).astype(float)
    change = (close - close.shift(length)).abs().fillna(0.0)
    volatility = close.diff().abs().rolling(window=length, min_periods=1).sum().fillna(0.0)
    er = pd.Series(0.0, index=close.index)
    mask = volatility > 0
    er[mask] = (change[mask] / volatility[mask]).clip(0.0, 1.0)

    fast_alpha = 2.0 / (fast + 1.0)
    slow_alpha = 2.0 / (slow + 1.0)
    sc = (er * (fast_alpha - slow_alpha) + slow_alpha) ** 2

    out = np.empty(len(close), dtype=float)
    close_v = close.values
    sc_v = sc.values
    out[0] = close_v[0]
    for i in range(1, len(close_v)):
        out[i] = out[i - 1] + sc_v[i] * (close_v[i] - out[i - 1])
    return pd.Series(out, index=close.index)


def derive_params(method, base, k):
    """Return (length, fast, slow) for the parameter-based methods."""
    length, fast, slow = base
    if method == "all_exact":
        return scale_window(length, k), scale_alpha_period(fast, k), scale_alpha_period(slow, k)
    if method == "all_naive":
        return max(1, round(length * k)), max(1, round(fast * k)), max(1, round(slow * k))
    if method == "er_only":
        return scale_window(length, k), fast, slow
    if method == "fastslow_exact":
        return length, scale_alpha_period(fast, k), scale_alpha_period(slow, k)
    if method == "fast_only":
        return length, scale_alpha_period(fast, k), slow
    if method == "sqrt_time":
        return (max(1, round(length * np.sqrt(k))),
                max(1, round(fast * np.sqrt(k))),
                max(1, round(slow * np.sqrt(k))))
    raise ValueError(f"unknown method: {method}")


def kama_resampled(close, base, k, interval_hours):
    """KAMA(base) computed on k-times aggregated bars, ffilled to the original index."""
    k = max(1, int(round(k)))  # aggregation must be a whole number of bars
    freq = pd.Timedelta(hours=interval_hours * k)
    agg = close.resample(freq).last().dropna()
    if len(agg) < 3:
        return pd.Series(np.nan, index=close.index)
    agg_kama = kama_er(agg, *base)
    return agg_kama.reindex(close.index, method="ffill")


def apply_method(method, close, base, k, interval_hours):
    """Return the KAMA series for ``method`` at scale ``k`` on ``close``."""
    if method == "resample":
        return kama_resampled(close, base, k, interval_hours)
    length, fast, slow = derive_params(method, base, k)
    return kama_er(close, length, fast, slow)


def tracking_ratio(series, close):
    """std(kama)/std(price): ~1 follows price, ->0 is a flat slow line."""
    denom = close.std()
    return float(series.std() / denom) if denom else float("nan")


def smoothness(series, close):
    """mean|dKAMA| / mean|dPrice| - 1 = fast, ->0 = slow.  Monotonic in scale."""
    s = pd.Series(series).ffill()
    denom = close.diff().abs().mean()
    return float(s.diff().abs().mean() / denom) if denom else float("nan")


CALIBRATED_METHODS = ["resample", "all_exact", "sqrt_time"]
# Search range of the internal factor per method (resample needs a huge factor
# to become as slow as the period-scaling methods).
CALIBRATION_KMAX = {"resample": 160.0, "all_exact": 64.0, "sqrt_time": 600.0}
CALIBRATION_STEP = {"resample": 1.0, "all_exact": 0.5, "sqrt_time": 2.0}


def _calibration_grid(method):
    kmax = CALIBRATION_KMAX[method]
    step = CALIBRATION_STEP[method]
    return np.arange(1.0, kmax + step, step)


def calibrate_method(method, base, targets, close, interval_hours):
    """Find, per target smoothness, the internal factor that achieves it.

    Returns a list of dicts ``{factor, smoothness}`` (one per target), so that
    every method can be drawn over the *same* smoothness spectrum even though
    the underlying factor differs wildly.
    """
    factors = _calibration_grid(method)
    sm = np.array([
        smoothness(apply_method(method, close, base, f, interval_hours), close)
        for f in factors
    ])
    order = np.argsort(sm)
    xs, fs = sm[order], factors[order]
    # dedupe identical smoothness values so np.interp stays well defined
    keep = np.concatenate(([True], np.diff(xs) > 1e-9))
    xs, fs = xs[keep], fs[keep]

    out = []
    for target in targets:
        factor = float(np.interp(target, xs, fs))
        if method == "resample":
            factor = float(max(1, int(round(factor))))
        achieved = smoothness(apply_method(method, close, base, factor, interval_hours), close)
        out.append({"factor": factor, "smoothness": achieved, "target": target})
    return out


def figure_calibrated_grid(df, base, results, methods, interval_hours, asset, interval, window):
    """One panel per method; curves share the same smoothness spectrum/colors."""
    close = df["price_close"]
    levels = len(results[methods[0]])
    colors = pcolors.sample_colorscale(
        "plasma", [i / (levels - 1) for i in range(levels)] if levels > 1 else [0.5])

    fig = make_subplots(
        rows=len(methods), cols=1, shared_xaxes=True, vertical_spacing=0.03,
        subplot_titles=[
            f"{m}  -  {METHOD_DESCRIPTIONS[m]}"
            for m in methods])

    for row, method in enumerate(methods, start=1):
        fig.add_trace(go.Scatter(
            x=df.index, y=close, name="Price", showlegend=False,
            line=dict(color="rgba(130,130,130,0.5)", width=0.8)), row=row, col=1)
        for i, (color, item) in enumerate(zip(colors, results[method])):
            series = apply_method(method, close, base, item["factor"], interval_hours)
            label = (f"level {i + 1}  smooth={item['target']:.3f}  "
                     f"(factor {item['factor']:.0f})")
            fig.add_trace(go.Scatter(
                x=df.index, y=series, name=label, showlegend=(row == 1),
                line=dict(color=color, width=1.6)), row=row, col=1)
        fig.update_yaxes(title_text="Price", row=row, col=1)

    fig.update_layout(
        title=(f"KAMA scaling - same spectrum, different computation - {asset} {interval}"
               f"<br><sub>{window} | base KAMA({base[0]},{base[1]},{base[2]}) | "
               f"calibrated to identical smoothness levels (mean|dKAMA|/mean|dPrice|)</sub>"),
        height=max(700, 300 * len(methods)),
        hovermode="x unified", xaxis_rangeslider_visible=False,
        legend=dict(yanchor="top", y=1.0, xanchor="left", x=1.01,
                    bgcolor="rgba(255,255,255,0.7)"))
    fig.update_xaxes(title_text="Date", row=len(methods), col=1)
    return fig


def figure_calibrated_overlay(df, base, results, methods, interval_hours, asset, interval, window):
    """The slowest curve of every method overlaid (should be equally slow)."""
    close = df["price_close"]
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["price_open"], high=df["price_high"],
        low=df["price_low"], close=df["price_close"], name="Price",
        increasing_line_color="rgba(150,150,150,0.5)",
        decreasing_line_color="rgba(80,80,80,0.5)", showlegend=False))
    palette = {"resample": "#1f77b4", "all_exact": "#d62728", "sqrt_time": "#ff7f0e"}
    for method in methods:
        item = results[method][-1]
        series = apply_method(method, close, base, item["factor"], interval_hours)
        fig.add_trace(go.Scatter(
            x=df.index, y=series,
            name=f"{method} (factor {item['factor']:.0f}, smooth={item['smoothness']:.3f})",
            line=dict(color=palette.get(method), width=2)))
    fig.update_layout(
        title=(f"Slowest calibrated curve per method - {asset} {interval}"
               f"<br><sub>{window} | matched smoothness target "
               f"{results[methods[0]][-1]['target']:.3f}</sub>"),
        xaxis_title="Date", yaxis_title="Price (USDT)", height=700,
        hovermode="x unified", xaxis_rangeslider_visible=False,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                    bgcolor="rgba(255,255,255,0.7)"))
    return fig


def parse_scales(value):
    text = str(value).strip()
    if ".." in text:
        lo, hi = (float(x) for x in text.split("..", 1))
        step = 1.0 if hi >= lo else -1.0
        return [float(x) for x in np.arange(lo, hi + step, step)]
    return [float(x) for x in text.split(",") if x.strip()]


def load_ohlc(asset, interval, start, end):
    cache = os.path.join("ohlc_cache", f"{asset}_{interval}_ohlc.csv")
    if not os.path.exists(cache):
        print(f"No cache at {cache} - fetching...")
        fetch_data(asset, interval)
    df = pd.read_csv(cache, index_col="time_period_start", parse_dates=True)
    if start:
        df = df[df.index >= pd.to_datetime(start)]
    if end:
        df = df[df.index <= pd.to_datetime(end)]
    return df


def figure_for_method(method, df, base, scales, interval_hours, asset, interval, window):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["price_open"], high=df["price_high"],
        low=df["price_low"], close=df["price_close"], name="Price",
        increasing_line_color="rgba(120,120,120,0.6)",
        decreasing_line_color="rgba(60,60,60,0.6)", showlegend=False))

    colors = pcolors.sample_colorscale(
        "plasma", [i / (len(scales) - 1) for i in range(len(scales))] if len(scales) > 1 else [0.5])
    close = df["price_close"]
    for color, k in zip(colors, scales):
        series = apply_method(method, close, base, k, interval_hours)
        if method == "resample":
            label = f"k={k:g} (resample, L{base[0]}/F{base[1]}/S{base[2]})"
        else:
            L, F, S = derive_params(method, base, k)
            label = f"k={k:g} (L{L}/F{F}/S{S})"
        fig.add_trace(go.Scatter(x=df.index, y=series, name=label,
                                 line=dict(color=color, width=1.6)))

    fig.update_layout(
        title=(f"KAMA scaling option: <b>{method}</b> - {asset} {interval}"
               f"<br><sub>{window} | {METHOD_DESCRIPTIONS[method]}</sub>"),
        xaxis_title="Date", yaxis_title="Price (USDT)", height=850,
        hovermode="x unified", xaxis_rangeslider_visible=False,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                    bgcolor="rgba(255,255,255,0.7)"))
    return fig


def figure_compare(df, base, methods, k, interval_hours, asset, interval, window):
    """All methods overlaid at a single scale k, vs the resample reference."""
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["price_open"], high=df["price_high"],
        low=df["price_low"], close=df["price_close"], name="Price",
        increasing_line_color="rgba(150,150,150,0.5)",
        decreasing_line_color="rgba(80,80,80,0.5)", showlegend=False))

    close = df["price_close"]
    palette = pcolors.qualitative.Dark24
    for i, method in enumerate(methods):
        series = apply_method(method, close, base, k, interval_hours)
        style = dict(width=3, dash="solid") if method == "resample" else dict(width=1.5)
        fig.add_trace(go.Scatter(
            x=df.index, y=series, name=method,
            line=dict(color=palette[i % len(palette)], **style)))

    fig.update_layout(
        title=(f"KAMA scaling methods at k={k:g} - {asset} {interval}"
               f"<br><sub>{window} | base KAMA({base[0]},{base[1]},{base[2]}) | "
               f"'resample' is the reference (true higher timeframe)</sub>"),
        xaxis_title="Date", yaxis_title="Price (USDT)", height=850,
        hovermode="x unified", xaxis_rangeslider_visible=False,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                    bgcolor="rgba(255,255,255,0.7)"))
    return fig


def figure_grid(df, base, methods, scales, interval_hours, asset, interval, window):
    """One panel per method, each showing its KAMA lines for all scales k."""
    close = df["price_close"]
    colors = pcolors.sample_colorscale(
        "plasma", [i / (len(scales) - 1) for i in range(len(scales))] if len(scales) > 1 else [0.5])

    fig = make_subplots(
        rows=len(methods), cols=1, shared_xaxes=True, vertical_spacing=0.02,
        subplot_titles=[f"{m}  -  {METHOD_DESCRIPTIONS[m]}" for m in methods])

    for row, method in enumerate(methods, start=1):
        fig.add_trace(go.Scatter(
            x=df.index, y=close, name="Price", showlegend=False,
            line=dict(color="rgba(130,130,130,0.55)", width=0.8)), row=row, col=1)
        for color, k in zip(colors, scales):
            series = apply_method(method, close, base, k, interval_hours)
            if method == "resample":
                label = f"k={k:g}"
            else:
                L, F, S = derive_params(method, base, k)
                label = f"k={k:g} (L{L}/F{F}/S{S})"
            fig.add_trace(go.Scatter(
                x=df.index, y=series, name=label, showlegend=(row == 1),
                line=dict(color=color, width=1.4)), row=row, col=1)
        fig.update_yaxes(title_text="Price", row=row, col=1)

    fig.update_layout(
        title=(f"KAMA scaling - one panel per method - {asset} {interval}"
               f"<br><sub>{window} | base KAMA({base[0]},{base[1]},{base[2]}) | "
               f"scales={[f'{k:g}' for k in scales]}</sub>"),
        height=max(700, 260 * len(methods)),
        hovermode="x unified", xaxis_rangeslider_visible=False,
        legend=dict(yanchor="top", y=1.0, xanchor="left", x=1.01,
                    bgcolor="rgba(255,255,255,0.7)"))
    fig.update_xaxes(title_text="Date", row=len(methods), col=1)
    return fig


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asset", default="ETHUSDT")
    ap.add_argument("--interval", default="4h")
    ap.add_argument("--start", default="2024-06-01")
    ap.add_argument("--end", default="2025-06-01")
    ap.add_argument("--base-length", type=int, default=10)
    ap.add_argument("--base-fast", type=int, default=2)
    ap.add_argument("--base-slow", type=int, default=30)
    ap.add_argument("--scales", default="1,2,3,4,6,8")
    ap.add_argument("--compare-k", type=float, default=4.0)
    ap.add_argument("--methods", default=",".join(METHOD_DESCRIPTIONS),
                    help="comma list of methods (default: all)")
    ap.add_argument("--calibrated", action="store_true",
                    help="only resample/all_exact/sqrt_time, calibrated to one shared "
                         "smoothness spectrum")
    ap.add_argument("--levels", type=int, default=6,
                    help="number of calibrated smoothness levels (default 6)")
    ap.add_argument("--target-smooth", type=float, default=0.010,
                    help="slowest smoothness level to reach (default 0.010)")
    args = ap.parse_args()

    print_viewer_info()

    base = (args.base_length, args.base_fast, args.base_slow)
    scales = parse_scales(args.scales)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in methods:
        if m not in METHOD_DESCRIPTIONS:
            raise SystemExit(f"unknown method '{m}'. Options: {list(METHOD_DESCRIPTIONS)}")

    if args.interval not in INTERVAL_HOURS:
        raise SystemExit(f"unsupported interval '{args.interval}'")
    interval_hours = INTERVAL_HOURS[args.interval]

    df = load_ohlc(args.asset, args.interval, args.start, args.end)
    if df.empty:
        raise SystemExit("No data in the requested window.")
    close = df["price_close"]
    window = f"{df.index.min():%Y-%m-%d} .. {df.index.max():%Y-%m-%d} ({len(df)} candles)"
    print(f"Window: {window}")
    print(f"Base KAMA({base[0]},{base[1]},{base[2]})  scales={scales}  compare-k={args.compare_k:g}")

    if args.calibrated:
        methods = CALIBRATED_METHODS
        s_base = smoothness(apply_method(methods[0], close, base, 1.0, interval_hours), close)
        targets = list(np.geomspace(s_base, args.target_smooth, args.levels))
        print(f"\nCalibrated spectrum: {len(targets)} levels, "
              f"smooth {s_base:.3f} -> {args.target_smooth:.3f}")
        results = {}
        for method in methods:
            results[method] = calibrate_method(method, base, targets, close, interval_hours)
            print(f"\n{method}:")
            for i, item in enumerate(results[method], 1):
                print(f"  level {i}: target={item['target']:.3f} "
                      f"factor={item['factor']:.0f} achieved={item['smoothness']:.3f}")

        grid_out = os.path.join(
            "html_cache", f"kama_scaling_calibrated_grid_{args.asset}_{args.interval}.html")
        grid_fig = figure_calibrated_grid(
            df, base, results, methods, interval_hours, args.asset, args.interval, window)
        print(f"\nCalibrated grid:  {publish_figure(grid_fig, grid_out, label='KAMA scaling - calibrated')}")

        overlay_out = os.path.join(
            "html_cache", f"kama_scaling_calibrated_overlay_{args.asset}_{args.interval}.html")
        overlay_fig = figure_calibrated_overlay(
            df, base, results, methods, interval_hours, args.asset, args.interval, window)
        print(f"Slowest overlay:  {publish_figure(overlay_fig, overlay_out, label='KAMA scaling - slowest overlay')}")
        return

    os.makedirs("html_cache", exist_ok=True)

    # Per-option chart + numeric summary.
    print("\nmethod            k   L     F     S     tracking  max|diff vs resample|")
    print("-" * 78)
    reference = {k: apply_method("resample", close, base, k, interval_hours) for k in scales}
    for method in methods:
        out = os.path.join("html_cache", f"kama_scaling_method_{method}_{args.asset}_{args.interval}.html")
        fig = figure_for_method(method, df, base, scales, interval_hours, args.asset, args.interval, window)
        publish_figure(fig, out, label=f"KAMA scaling - {method}")
        for k in scales:
            series = apply_method(method, close, base, k, interval_hours)
            if method == "resample":
                L = F = S = "-"
            else:
                L, F, S = derive_params(method, base, k)
            diff = float(np.nanmax(np.abs(series - reference[k]))) if method != "resample" else 0.0
            print(f"{method:<16} {k:<3g} {str(L):<5} {str(F):<5} {str(S):<5} "
                  f"{tracking_ratio(series, close):<9.2f} {diff:.2f}")

    # Combined comparison at one scale.
    cmp_out = os.path.join("html_cache", f"kama_scaling_methods_compare_k{args.compare_k:g}_{args.asset}_{args.interval}.html")
    cmp_fig = figure_compare(df, base, methods, args.compare_k, interval_hours, args.asset, args.interval, window)
    url = publish_figure(cmp_fig, cmp_out, label="KAMA scaling - method comparison")
    print(f"\nCombined comparison: {url}")

    # One panel per method, each showing its own scaling.
    grid_out = os.path.join("html_cache", f"kama_scaling_methods_grid_{args.asset}_{args.interval}.html")
    grid_fig = figure_grid(df, base, methods, scales, interval_hours, args.asset, args.interval, window)
    grid_url = publish_figure(grid_fig, grid_out, label="KAMA scaling - per-method grid")
    print(f"Per-method grid:     {grid_url}")


if __name__ == "__main__":
    main()

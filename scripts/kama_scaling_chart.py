#!/usr/bin/env python3
"""Visual proof of exact KAMA parameter scaling on a price chart.

Plots one price chart with N KAMAs, each at a different scale ``k``.  The three
bar-based KAMA parameters are derived from a single ``k`` with the exact rules
from ``core.kama_scaling``::

    length' = k * length
    period' = k * (period + 1) - 1        # exact, for fast / slow

The KAMA formula is DMX's range-based ``_ama`` (the one used for the DMX
``slow_kama``), so the picture matches what the strategy actually computes.

Example (from the host)::

    docker compose exec app python scripts/kama_scaling_chart.py

    # custom window / scales
    docker compose exec app python scripts/kama_scaling_chart.py \
        --asset ETHUSDT --interval 4h \
        --start 2024-06-01 --end 2025-06-01 \
        --base-length 16 --base-fast 4 --base-slow 24 \
        --scales 1,2,3,4,5,6,7,8,9,10
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.colors as pcolors

from core.kama_scaling import scale_window, scale_alpha_period
from core.html_viewer import publish_figure, print_viewer_info
from core.strategy_utils import fetch_data
from classes.strategies_dmx.dmx_strategy import DMXStrategy


def parse_scales(value):
    """Parse a comma list like ``1,2,3`` or ``1..10`` into a list of floats."""
    text = str(value).strip()
    if ".." in text:
        lo, hi = text.split("..", 1)
        lo, hi = float(lo), float(hi)
        step = 1.0 if hi >= lo else -1.0
        return [float(x) for x in np.arange(lo, hi + step, step)]
    return [float(x) for x in text.split(",") if x.strip()]


def load_ohlc(asset, interval, start, end):
    """Load cached OHLC (fetching once if missing) and slice to [start, end]."""
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


def build_figure(df, kamas, asset, interval, window_label):
    """One price chart (candles) plus one KAMA line per scale."""
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["price_open"], high=df["price_high"],
        low=df["price_low"], close=df["price_close"],
        name="Price",
        increasing_line_color="rgba(120,120,120,0.7)",
        decreasing_line_color="rgba(60,60,60,0.7)",
        showlegend=False,
    ))

    n = len(kamas)
    colors = pcolors.sample_colorscale(
        "plasma", [i / (n - 1) for i in range(n)] if n > 1 else [0.5]
    )

    for color, item in zip(colors, kamas):
        k = item["k"]
        label = f"k={k:g}  (L{item['length']}/F{item['fast']}/S{item['slow']})"
        fig.add_trace(go.Scatter(
            x=df.index, y=item["kama"],
            name=label, line=dict(color=color, width=1.6),
        ))

    fig.update_layout(
        title=(
            f"KAMA scaling (DMX range-KAMA) - {asset} {interval}"
            f"<br><sub>{window_label} | base "
            f"L{kamas[0]['base'][0]}/F{kamas[0]['base'][1]}/S{kamas[0]['base'][2]}, "
            f"exact scale L'=k*L, P'=k*(P+1)-1</sub>"
        ),
        xaxis_title="Date",
        yaxis_title="Price (USDT)",
        height=900,
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                    bgcolor="rgba(255,255,255,0.7)"),
        xaxis_rangeslider_visible=False,
    )
    return fig


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asset", default="ETHUSDT")
    ap.add_argument("--interval", default="4h")
    ap.add_argument("--start", default="2024-06-01")
    ap.add_argument("--end", default="2025-06-01")
    ap.add_argument("--base-length", type=int, default=16)
    ap.add_argument("--base-fast", type=int, default=4)
    ap.add_argument("--base-slow", type=int, default=24)
    ap.add_argument("--scales", default="1..10",
                    help="comma list (1,2,3) or range (1..10)")
    ap.add_argument("--output", default=None,
                    help="HTML path (default html_cache/kama_scaling_<asset>_<interval>.html)")
    ap.add_argument("--png", action="store_true",
                    help="also write a PNG next to the HTML (needs kaleido)")
    args = ap.parse_args()

    print_viewer_info()

    scales = parse_scales(args.scales)
    df = load_ohlc(args.asset, args.interval, args.start, args.end)
    if df.empty:
        raise SystemExit("No data in the requested window.")
    window_label = f"{df.index.min():%Y-%m-%d} .. {df.index.max():%Y-%m-%d} ({len(df)} candles)"
    print(f"Window: {window_label}")

    strategy = DMXStrategy()
    close = df["price_close"]

    kamas = []
    print(f"Base KAMA: L={args.base_length} F={args.base_fast} S={args.base_slow}")
    for k in scales:
        length = scale_window(args.base_length, k)
        fast = scale_alpha_period(args.base_fast, k)
        slow = scale_alpha_period(args.base_slow, k)
        kama = strategy._ama(close, length=length, fast=fast, slow=slow)
        kamas.append({
            "k": k, "length": length, "fast": fast, "slow": slow,
            "kama": kama, "base": (args.base_length, args.base_fast, args.base_slow),
        })
        print(f"  k={k:<5g} -> L={length:<5d} F={fast:<5d} S={slow:<5d} "
              f"last={kama.iloc[-1]:.2f}")

    fig = build_figure(df, kamas, args.asset, args.interval, window_label)

    out = args.output or os.path.join(
        "html_cache", f"kama_scaling_{args.asset}_{args.interval}.html")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    url = publish_figure(fig, out, label="KAMA Scaling Chart")
    print(f"\nChart: {url}")

    if args.png:
        png = os.path.splitext(out)[0] + ".png"
        try:
            fig.write_image(png, width=1600, height=900, scale=2)
            print(f"PNG:   {os.path.abspath(png)}")
        except Exception as exc:  # pragma: no cover - optional dependency
            print(f"PNG skipped ({exc.__class__.__name__}: {exc})")


if __name__ == "__main__":
    main()

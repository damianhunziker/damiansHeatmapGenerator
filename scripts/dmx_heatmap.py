#!/usr/bin/env python3
"""DMX 2D-Heatmap: Squeeze-Sensibilitaet x Exit-KAMA-Geschwindigkeit (long).

Sweept genau zwei Parameter und laesst alle anderen DMX-Parameter auf Default:
  x = squeeze_limit       (Squeeze-Recog Sensibilitaet; kleiner = sensibler)
  y = slow_kama_length    (Exit-KAMA long; kuerzer = schneller)

Beispiel (im Container):
  docker compose exec app python scripts/dmx_heatmap.py \
      --asset IBKR_AAPL --interval 4h --years 20 \
      --squeeze-limits 100,150,200,250,300,350,400 \
      --kama-lengths 100,200,300,400,500,600,700
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from core.strategy_utils import fetch_data, ensure_timestamp_column, print_logo
from core.html_viewer import print_viewer_info
from classes.strategies_dmx.dmx_strategy import DMXStrategy
from heatmap import create_heatmap


def parse_int_list(value):
    return [int(x) for x in str(value).split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asset", default="IBKR_AAPL")
    ap.add_argument("--interval", default="4h")
    ap.add_argument("--years", type=float, default=20.0, help="Analysefenster (Jahre, 0 = alles)")
    ap.add_argument("--squeeze-limits", default="100,150,200,250,300,350,400",
                    help="x-Achse: squeeze_limit Werte (komma-getrennt)")
    ap.add_argument("--kama-lengths", default="100,200,300,400,500,600,700",
                    help="y-Achse: slow_kama_length Werte (komma-getrennt)")
    ap.add_argument("--equity", type=float, default=10000.0)
    ap.add_argument("--fee", type=float, default=0.04)
    args = ap.parse_args()

    print_logo()
    print_viewer_info()

    print(f"DMX-Heatmap: {args.asset} {args.interval}, letzte {args.years} Jahre")

    data = fetch_data(args.asset, args.interval)
    data = ensure_timestamp_column(data)

    if args.years and args.years > 0:
        cutoff = data.index.max() - pd.Timedelta(days=int(round(args.years * 365.25)))
        data = data[data.index >= cutoff].copy()

    print(f"Analysefenster: {data.index.min()} .. {data.index.max()} ({len(data)} Candles)")

    timeframe_data = {"primary": {"interval": args.interval, "data": data}}

    param_ranges = {
        "squeeze_limit": np.array(parse_int_list(args.squeeze_limits)),
        "slow_kama_length": np.array(parse_int_list(args.kama_lengths)),
    }
    n_combos = len(param_ranges["squeeze_limit"]) * len(param_ranges["slow_kama_length"])
    print(f"Sweep: squeeze_limit={list(param_ranges['squeeze_limit'])} x "
          f"slow_kama_length={list(param_ranges['slow_kama_length'])} = {n_combos} Kombinationen")

    create_heatmap(
        timeframe_data=timeframe_data,
        strategy_class=DMXStrategy,
        param_ranges=param_ranges,
        initial_equity=args.equity,
        fee_pct=args.fee,
        last_n_candles_analyze=len(data),
        last_n_candles_display=len(data),
        interval=args.interval,
        asset=args.asset,
        strategy_name="DMXStrategy",
        start_date=str(data.index.min().date()),
        end_date=str(data.index.max().date()),
    )


if __name__ == "__main__":
    main()

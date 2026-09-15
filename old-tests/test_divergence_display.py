#!/usr/bin/env python3

import pandas as pd
import numpy as np
from classes.strategies.live_kama_ssl_strategy import LiveKAMASSLStrategy
import os

def test_divergence_display():
    """Test if divergences are being displayed in indicator subplots"""
    
    # Load cached data
    cache_file = "ohlc_cache/BTCUSDT_4h_ohlc.csv"
    if not os.path.exists(cache_file):
        print(f"Cache file {cache_file} not found. Please run the fetcher first.")
        return
    
    # Load data
    print("Loading data...")
    data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
    
    # Filter data to a smaller range for testing
    start_date = "2024-01-01"
    end_date = "2024-02-01"
    data = data[start_date:end_date]
    
    print(f"Data shape: {data.shape}")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    
    # Initialize strategy with debug mode
    strategy = LiveKAMASSLStrategy(
        debug_mode=True,
        trade_direction='both',
        calculate_divergences=True,
        show_pivot_points=True,
        start_date=start_date,
        end_date=end_date,
        asset='BTCUSDT'
    )
    
    print("\n" + "="*60)
    print("TESTING DIVERGENCE DISPLAY")
    print("="*60)
    
    # Prepare signals (this will calculate divergences)
    print("\nPreparing signals...")
    signals_data = strategy.prepare_signals(data)
    
    print(f"\nSignals data shape: {signals_data.shape}")
    print(f"Columns: {signals_data.columns.tolist()}")
    
    # Check if divergences were calculated
    if hasattr(strategy, 'divergence_results'):
        print(f"\nLong profile divergences found:")
        for ind, divs in strategy.divergence_results.items():
            print(f"  {ind}: {len(divs)} divergences")
    
    if hasattr(strategy, 'short_divergence_results'):
        print(f"\nShort profile divergences found:")
        for ind, divs in strategy.short_divergence_results.items():
            print(f"  {ind}: {len(divs)} divergences")
    
    # Create figure (this should draw divergences)
    print("\nCreating figure...")
    fig = strategy.create_figure(signals_data)
    
    print(f"\nFigure created with {len(fig.data)} traces")
    
    # Save the figure
    output_file = "test_divergence_display.html"
    fig.write_html(output_file)
    print(f"\nFigure saved to {output_file}")
    
    print("\n" + "="*60)
    print("TEST COMPLETED")
    print("="*60)

if __name__ == "__main__":
    test_divergence_display() 
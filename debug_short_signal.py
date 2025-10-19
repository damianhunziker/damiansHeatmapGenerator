#!/usr/bin/env python3
"""
Debug script to investigate the short signal on 9.4.24 12:00
"""

import pandas as pd
import numpy as np
from classes.strategies.live_kama_ssl_strategy import LiveKAMASSLStrategy
from classes.trade_analyzer import TradeAnalyzer
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def debug_short_signal():
    print("\n" + "="*80)
    print("DEBUG: Investigating Short Signal on 9.4.24 12:00")
    print("="*80)
    
    # Load cached data
    print("\n1. Loading cached data...")
    df = pd.read_csv('ohlc_cache/BTCUSDT_4h_ohlc.csv')
    
    # Convert timestamp to datetime index
    df['time_period_start'] = pd.to_datetime(df['time_period_start'])
    df.set_index('time_period_start', inplace=True)
    
    # Use sufficient lead time before the target date
    start_date = '2024-03-01'  # Start 1+ month before target
    end_date = '2024-05-01'    # End after target
    df = df[start_date:end_date]
    
    print(f"Data loaded:")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Target timestamp to investigate
    target_time = pd.Timestamp('2024-04-09 12:00:00')
    print(f"\nTarget time to investigate: {target_time}")
    
    # Check if target time exists in data
    if target_time not in df.index:
        print(f"WARNING: Target time {target_time} not found in data!")
        # Find closest timestamp
        closest_idx = df.index.get_indexer([target_time], method='nearest')[0]
        closest_time = df.index[closest_idx]
        print(f"Closest timestamp: {closest_time}")
        target_time = closest_time
    
    # Initialize strategy with debug mode
    print("\n2. Initializing strategy with debug mode...")
    strategy_params = {
        'debug_mode': True,
        'trade_direction': 'short',  # Focus on short trades
        'use_fusion_for_long': True,  # Should not affect short trades
        'initial_equity': 10000,
        'fee_pct': 0.04,
        'start_date': start_date,
        'end_date': end_date,
        'asset': 'BTCUSDT'
    }
    
    strategy = LiveKAMASSLStrategy(**strategy_params)
    
    print(f"Strategy initialized:")
    print(f"- Debug mode: {strategy.debug_mode}")
    print(f"- Trade direction: {strategy.trade_direction}")
    print(f"- Use fusion for long: {strategy.use_fusion_for_long}")
    
    # Prepare signals with debug output
    print("\n3. Preparing signals and calculating indicators...")
    prepared_data = strategy.prepare_signals(df)
    
    print(f"Prepared data shape: {prepared_data.shape}")
    print(f"Prepared data columns: {prepared_data.columns.tolist()}")
    
    # Find the target index
    target_idx = prepared_data.index.get_loc(target_time)
    print(f"\nTarget index in prepared data: {target_idx}")
    
    # Analyze SSL signals around the target time
    print("\n4. Analyzing SSL signals around target time...")
    
    # Look at a window around the target time
    window_start = max(0, target_idx - 10)
    window_end = min(len(prepared_data), target_idx + 10)
    
    print(f"\nAnalyzing window from index {window_start} to {window_end}")
    print(f"Time range: {prepared_data.index[window_start]} to {prepared_data.index[window_end-1]}")
    
    # Create detailed analysis table
    print("\n" + "="*120)
    print(f"{'Time':<20} | {'SSL_Up':<10} | {'SSL_Down':<10} | {'Close':<10} | {'SSL_Signal':<12} | {'DEMA':<10} | {'Short_Entry':<12}")
    print("="*120)
    
    for i in range(window_start, window_end):
        row = prepared_data.iloc[i]
        time_str = prepared_data.index[i].strftime('%Y-%m-%d %H:%M')
        ssl_up = row['ssl_up']
        ssl_down = row['ssl_down']
        close = row['price_close']
        ssl_signal = row['ssl_signal']
        dema = row['dema']
        short_entry = row['short_entry']
        
        # Highlight the target row
        marker = " >>> TARGET <<<" if i == target_idx else ""
        
        print(f"{time_str:<20} | {ssl_up:<10.2f} | {ssl_down:<10.2f} | {close:<10.2f} | {ssl_signal:<12.0f} | {dema:<10.2f} | {short_entry:<12} {marker}")
    
    print("="*120)
    
    # Detailed analysis of the target candle and previous candle
    print("\n5. Detailed analysis of target candle and previous candle...")
    
    if target_idx > 0:
        prev_candle = prepared_data.iloc[target_idx - 1]
        current_candle = prepared_data.iloc[target_idx]
        
        print(f"\nPREVIOUS CANDLE ({prepared_data.index[target_idx - 1]}):")
        print(f"- SSL Up: {prev_candle['ssl_up']:.2f}")
        print(f"- SSL Down: {prev_candle['ssl_down']:.2f}")
        print(f"- Close: {prev_candle['price_close']:.2f}")
        print(f"- SSL Signal: {prev_candle['ssl_signal']}")
        print(f"- DEMA: {prev_candle['dema']:.2f}")
        
        print(f"\nCURRENT CANDLE ({prepared_data.index[target_idx]}):")
        print(f"- SSL Up: {current_candle['ssl_up']:.2f}")
        print(f"- SSL Down: {current_candle['ssl_down']:.2f}")
        print(f"- Close: {current_candle['price_close']:.2f}")
        print(f"- SSL Signal: {current_candle['ssl_signal']}")
        print(f"- DEMA: {current_candle['dema']:.2f}")
        print(f"- Short Entry: {current_candle['short_entry']}")
        
        # Check the entry conditions manually
        print(f"\n6. Manual check of short entry conditions...")
        
        # SSL crossover condition (corrected version)
        ssl_crossover_down = (
            prev_candle['ssl_signal'] == 1 and  # Was bullish before (above SSL Upper)
            current_candle['ssl_signal'] == -1   # Is bearish now (below SSL Lower)
        )
        
        # Price below DEMA condition
        price_below_dema = current_candle['price_close'] < current_candle['dema']
        
        print(f"SSL Crossover Down (prev=1, curr=-1): {ssl_crossover_down}")
        print(f"  - Previous SSL Signal: {prev_candle['ssl_signal']} (should be 1)")
        print(f"  - Current SSL Signal: {current_candle['ssl_signal']} (should be -1)")
        
        print(f"Price Below DEMA: {price_below_dema}")
        print(f"  - Current Close: {current_candle['price_close']:.2f}")
        print(f"  - Current DEMA: {current_candle['dema']:.2f}")
        print(f"  - Difference: {current_candle['price_close'] - current_candle['dema']:.2f}")
        
        combined_condition = ssl_crossover_down and price_below_dema
        print(f"\nCombined Entry Condition: {combined_condition}")
        print(f"Actual Short Entry in Data: {current_candle['short_entry']}")
        
        if current_candle['short_entry'] and not combined_condition:
            print("\n❌ ERROR: Short entry signal exists but conditions are not met!")
            print("This indicates a bug in the strategy logic.")
        elif not current_candle['short_entry'] and combined_condition:
            print("\n❌ ERROR: Conditions are met but no short entry signal!")
            print("This indicates a bug in the strategy logic.")
        elif current_candle['short_entry'] and combined_condition:
            print("\n✅ CORRECT: Short entry signal matches the conditions.")
        else:
            print("\n✅ CORRECT: No short entry signal and conditions are not met.")
    
    # Check for any short entries in the entire dataset
    print("\n7. Checking all short entries in the dataset...")
    short_entries = prepared_data[prepared_data['short_entry'] == True]
    
    print(f"\nTotal short entries found: {len(short_entries)}")
    
    if len(short_entries) > 0:
        print("\nAll short entry signals:")
        print("="*80)
        for idx, (timestamp, row) in enumerate(short_entries.iterrows()):
            print(f"{idx+1}. {timestamp} - Close: {row['price_close']:.2f}, SSL Signal: {row['ssl_signal']}, DEMA: {row['dema']:.2f}")
        print("="*80)
        
        # Check if our target time is in the list
        if target_time in short_entries.index:
            print(f"\n✅ Target time {target_time} IS in the short entries list.")
        else:
            print(f"\n❌ Target time {target_time} is NOT in the short entries list.")
    
    # Create a visualization
    print("\n8. Creating visualization...")
    
    # Focus on a smaller window for visualization
    viz_start = max(0, target_idx - 50)
    viz_end = min(len(prepared_data), target_idx + 50)
    viz_data = prepared_data.iloc[viz_start:viz_end]
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=['Price with SSL and DEMA', 'SSL Signals', 'Short Entry Signals'],
        row_heights=[0.6, 0.2, 0.2]
    )
    
    # Price chart with SSL and DEMA
    fig.add_trace(go.Candlestick(
        x=viz_data.index,
        open=viz_data['price_open'],
        high=viz_data['price_high'],
        low=viz_data['price_low'],
        close=viz_data['price_close'],
        name='BTCUSDT'
    ), row=1, col=1)
    
    # SSL lines
    fig.add_trace(go.Scatter(
        x=viz_data.index,
        y=viz_data['ssl_up'],
        name='SSL Up',
        line=dict(color='green', width=2)
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=viz_data.index,
        y=viz_data['ssl_down'],
        name='SSL Down',
        line=dict(color='red', width=2)
    ), row=1, col=1)
    
    # DEMA
    fig.add_trace(go.Scatter(
        x=viz_data.index,
        y=viz_data['dema'],
        name='DEMA',
        line=dict(color='blue', width=2)
    ), row=1, col=1)
    
    # SSL Signals
    fig.add_trace(go.Scatter(
        x=viz_data.index,
        y=viz_data['ssl_signal'],
        name='SSL Signal',
        line=dict(color='purple', width=2),
        mode='lines+markers'
    ), row=2, col=1)
    
    # Short entry markers
    short_entry_times = viz_data[viz_data['short_entry']].index
    short_entry_prices = viz_data[viz_data['short_entry']]['price_high']
    
    if len(short_entry_times) > 0:
        fig.add_trace(go.Scatter(
            x=short_entry_times,
            y=short_entry_prices,
            mode='markers',
            marker=dict(symbol='triangle-down', size=15, color='red'),
            name='Short Entry'
        ), row=1, col=1)
        
        # Also add to the signal subplot
        fig.add_trace(go.Scatter(
            x=short_entry_times,
            y=[1] * len(short_entry_times),
            mode='markers',
            marker=dict(symbol='triangle-down', size=15, color='red'),
            name='Short Entry Signal'
        ), row=3, col=1)
    
    # Mark the target time with a shape instead of vline
    fig.add_shape(
        type="line",
        x0=target_time, x1=target_time,
        y0=0, y1=1,
        yref="paper",
        line=dict(color="orange", width=2, dash="dash"),
    )
    
    # Add annotation for target time
    fig.add_annotation(
        x=target_time,
        y=0.95,
        yref="paper",
        text=f"Target: {target_time.strftime('%Y-%m-%d %H:%M')}",
        showarrow=False,
        bgcolor="orange",
        bordercolor="black",
        borderwidth=1
    )
    
    fig.update_layout(
        title=f'Debug Analysis: Short Signal Investigation for {target_time}',
        height=800,
        xaxis_rangeslider_visible=False
    )
    
    # Save the chart
    chart_filename = f'debug_short_signal_{target_time.strftime("%Y%m%d_%H%M")}.html'
    fig.write_html(chart_filename)
    print(f"\nVisualization saved as: {chart_filename}")
    
    print("\n" + "="*80)
    print("DEBUG ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    debug_short_signal() 
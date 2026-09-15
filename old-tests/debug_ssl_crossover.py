#!/usr/bin/env python3

import pandas as pd
import numpy as np
from classes.strategies.live_kama_ssl_strategy import LiveKAMASSLStrategy

def debug_ssl_crossover():
    print("\n" + "="*80)
    print("DEBUGGING SSL SIGNALS AT PROBLEMATIC DATES")
    print("="*80)
    
    # Load data
    df = pd.read_csv('ohlc_cache/BTCUSDT_4h_ohlc.csv')
    df['time_period_start'] = pd.to_datetime(df['time_period_start'])
    df.set_index('time_period_start', inplace=True)
    
    # Initialize strategy with debug mode
    strategy = LiveKAMASSLStrategy(
        trade_direction='short',
        debug_mode=True,
        start_date='2023-10-01',
        end_date='2024-05-01'
    )
    
    # Prepare data
    df_prepared = strategy.prepare_data(df)
    
    # Get signals
    df_signals = strategy.prepare_signals(df)
    
    # Focus on the problematic dates
    target_dates = [
        '2024-04-09 12:00:00',
        '2024-04-13 20:00:00'  # 13.4.24 20:00
    ]
    
    for target_date in target_dates:
        try:
            target_time = pd.to_datetime(target_date)
            
            # Find the index
            if target_time in df_signals.index:
                idx = df_signals.index.get_loc(target_time)
                
                print(f"\n🔍 ANALYZING {target_date}:")
                print(f"Index: {idx}")
                
                # Check surrounding data (5 candles before and after)
                start_idx = max(0, idx - 5)
                end_idx = min(len(df_signals), idx + 6)
                
                window = df_signals.iloc[start_idx:end_idx]
                
                print(f"\nSSL DATA AROUND {target_date}:")
                print("Time                    | SSL_Up      | SSL_Down    | SSL_Signal | Price_Close | DEMA        | Short_Entry")
                print("-" * 110)
                
                for i, (time, row) in enumerate(window.iterrows()):
                    marker = " >>> " if time == target_time else "     "
                    print(f"{marker}{time} | {row['ssl_up']:11.2f} | {row['ssl_down']:11.2f} | {row['ssl_signal']:10.0f} | {row['price_close']:11.2f} | {row['dema']:11.2f} | {row['short_entry']}")
                
                # Check for SSL crossover conditions
                if idx > 0:
                    prev_row = df_signals.iloc[idx-1]
                    curr_row = df_signals.iloc[idx]
                    
                    print(f"\n🔍 SSL CROSSOVER ANALYSIS:")
                    print(f"Previous SSL Signal: {prev_row['ssl_signal']}")
                    print(f"Current SSL Signal: {curr_row['ssl_signal']}")
                    print(f"SSL Crossover Down: {prev_row['ssl_signal'] == 1 and curr_row['ssl_signal'] == -1}")
                    print(f"Price below DEMA: {curr_row['price_close']} < {curr_row['dema']} = {curr_row['price_close'] < curr_row['dema']}")
                    print(f"Short Entry Signal: {curr_row['short_entry']}")
                    
                    # Check SSL channel values
                    print(f"\n🔍 SSL CHANNEL VALUES:")
                    print(f"Previous: SSL_Up={prev_row['ssl_up']:.2f}, SSL_Down={prev_row['ssl_down']:.2f}")
                    print(f"Current:  SSL_Up={curr_row['ssl_up']:.2f}, SSL_Down={curr_row['ssl_down']:.2f}")
                    print(f"Price vs SSL_Up: {curr_row['price_close']} vs {curr_row['ssl_up']} = {'Above' if curr_row['price_close'] > curr_row['ssl_up'] else 'Below'}")
                    print(f"Price vs SSL_Down: {curr_row['price_close']} vs {curr_row['ssl_down']} = {'Above' if curr_row['price_close'] > curr_row['ssl_down'] else 'Below'}")
                    
                    # Check how SSL signal is calculated
                    print(f"\n🔍 SSL SIGNAL CALCULATION:")
                    print(f"Price > SSL_Up: {curr_row['price_close']} > {curr_row['ssl_up']} = {curr_row['price_close'] > curr_row['ssl_up']} -> Signal should be 1")
                    print(f"Price < SSL_Down: {curr_row['price_close']} < {curr_row['ssl_down']} = {curr_row['price_close'] < curr_row['ssl_down']} -> Signal should be -1")
                    print(f"Actual SSL Signal: {curr_row['ssl_signal']}")
                    
            else:
                print(f"\n❌ {target_date} not found in data")
                
        except Exception as e:
            print(f"\n❌ Error analyzing {target_date}: {e}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    debug_ssl_crossover() 
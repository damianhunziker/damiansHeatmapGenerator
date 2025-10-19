#!/usr/bin/env python3
"""
Debug script for Fusion Filter entry issues

This script helps identify why entries might not be happening when the fusion filter
condition changes from active (range) to inactive (non-range).
"""

import pandas as pd
import numpy as np
from classes.strategies.live_kama_ssl_strategy import LiveKAMASSLStrategy
from classes.data_fetcher import OHLCFetcher

def debug_fusion_filter_entries():
    """Debug fusion filter entry behavior"""
    
    # Initialize strategy with debug mode
    strategy = LiveKAMASSLStrategy(
        entry_filter=0.35,
        exit_filter=0.5,
        debug_mode=True,
        trade_direction='long',
        use_fusion_for_long=True,
        atr_length=9,
        hma_mode='VWMA',
        hma_length=50,
        atr_scaling_factor=1.4,
        allow_continuous_entries=True
    )
    
    # Fetch some test data
    fetcher = OHLCFetcher()
    data = fetcher.fetch_data('BTCUSDT', '4h', limit=1000)
    
    if data is None or len(data) < 100:
        print("❌ Could not fetch sufficient data for testing")
        return
    
    print(f"✅ Fetched {len(data)} candles for testing")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    
    # Prepare the data (calculate indicators)
    print("\n🔧 Calculating indicators...")
    df = strategy.prepare_data(data)
    
    # Calculate signals
    print("🔧 Calculating trading signals...")
    df = strategy.prepare_signals(data)
    
    # Calculate fusion filter values
    fusion_ma, fusion_atr, fusion_cond = strategy.fusion_range_filter.calculate(df)
    
    print(f"\n📊 Fusion Filter Analysis:")
    print(f"Total candles: {len(df)}")
    print(f"Range active periods: {fusion_cond.sum()}/{len(fusion_cond)} ({fusion_cond.sum()/len(fusion_cond)*100:.1f}%)")
    print(f"Range inactive periods: {(~fusion_cond).sum()}/{len(fusion_cond)} ({(~fusion_cond).sum()/len(fusion_cond)*100:.1f}%)")
    
    # Find range transitions (when range becomes inactive)
    transitions = []
    for i in range(1, len(fusion_cond)):
        if fusion_cond.iloc[i-1] and not fusion_cond.iloc[i]:  # Range ends
            transitions.append((i, df.index[i], "Range ENDED"))
        elif not fusion_cond.iloc[i-1] and fusion_cond.iloc[i]:  # Range starts
            transitions.append((i, df.index[i], "Range STARTED"))
    
    print(f"\n🔄 Found {len(transitions)} range transitions:")
    for idx, time, transition_type in transitions:
        print(f"   {transition_type} at {time} (index {idx})")
    
    # Analyze potential entry points when range ends
    print(f"\n🎯 Analyzing potential entry points when range ends...")
    
    entry_opportunities = 0
    missed_entries = 0
    actual_entries = 0
    
    for idx, time, transition_type in transitions:
        if transition_type == "Range ENDED":
            # Check the next few candles for potential entries
            for check_idx in range(idx, min(idx + 10, len(df))):
                current_candle = df.iloc[check_idx]
                current_time = df.index[check_idx]
                
                # Check all three conditions
                trend_condition = current_candle['exit_kama'] > current_candle['kama2']
                fusion_condition = not fusion_cond.iloc[check_idx]  # Allow when range inactive
                kama_delta_condition = current_candle['entry_kama_delta'] > current_candle['entry_kama_delta_limit']
                
                all_conditions_met = trend_condition and fusion_condition and kama_delta_condition
                
                print(f"\n   📅 {current_time} (offset +{check_idx-idx} from range end):")
                print(f"      ✅ Trend: {trend_condition} (Exit KAMA: {current_candle['exit_kama']:.6f} > KAMA2: {current_candle['kama2']:.6f})")
                print(f"      {'✅' if fusion_condition else '❌'} Fusion: {fusion_condition} (MA: {fusion_ma.iloc[check_idx]:.6f} {'<=' if fusion_condition else '>'} ATR: {fusion_atr.iloc[check_idx]:.6f})")
                print(f"      {'✅' if kama_delta_condition else '❌'} KAMA: {kama_delta_condition} (Delta: {current_candle['entry_kama_delta']:.6f} > Limit: {current_candle['entry_kama_delta_limit']:.6f})")
                print(f"      🎯 ALL CONDITIONS: {all_conditions_met}")
                
                if all_conditions_met:
                    entry_opportunities += 1
                    print(f"      🚀 ENTRY OPPORTUNITY!")
                    # Check if there's an actual entry signal
                    if current_candle.get('long_entry', False):
                        actual_entries += 1
                        print(f"      ✅ ACTUAL ENTRY FOUND!")
                    else:
                        missed_entries += 1
                        print(f"      ❌ MISSED ENTRY!")
                    break  # Only check first opportunity after range end
                elif kama_delta_condition:  # KAMA is ready but other conditions not met
                    print(f"      ⚠️  KAMA ready but waiting for other conditions")
                    continue
                else:
                    break  # KAMA condition not met, stop checking this transition
    
    print(f"\n📈 ENTRY ANALYSIS SUMMARY:")
    print(f"   Range transitions (range ended): {len([t for t in transitions if 'ENDED' in t[2]])}")
    print(f"   Entry opportunities found: {entry_opportunities}")
    print(f"   Actual entries taken: {actual_entries}")
    print(f"   Missed entries: {missed_entries}")
    print(f"   Success rate: {actual_entries/entry_opportunities*100:.1f}%" if entry_opportunities > 0 else "   No opportunities found")
    
    # Additional analysis: continuous KAMA condition periods
    print(f"\n📊 KAMA DELTA CONDITION ANALYSIS:")
    kama_active_periods = df['entry_kama_delta'] > df['entry_kama_delta_limit']
    print(f"KAMA condition active: {kama_active_periods.sum()}/{len(kama_active_periods)} candles ({kama_active_periods.sum()/len(kama_active_periods)*100:.1f}%)")
    
    # Find KAMA condition starts
    kama_starts = []
    for i in range(1, len(kama_active_periods)):
        if not kama_active_periods.iloc[i-1] and kama_active_periods.iloc[i]:
            kama_starts.append((i, df.index[i]))
    
    print(f"KAMA condition starts: {len(kama_starts)} times")
    if kama_starts:
        print("First few KAMA starts:")
        for i, (idx, time) in enumerate(kama_starts[:5]):
            print(f"   {time} (index {idx})")

if __name__ == "__main__":
    print("🔍 Fusion Filter Debug Script")
    print("=" * 50)
    debug_fusion_filter_entries() 
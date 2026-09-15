#!/usr/bin/env python3
"""
Test script to verify the fusion filter fix works correctly
"""

import pandas as pd
import numpy as np
from classes.strategies.live_kama_ssl_strategy import LiveKAMASSLStrategy
from classes.data_fetcher import OHLCFetcher

def test_fusion_fix():
    """Test the fusion filter fix"""
    
    # Initialize strategy with debug mode
    strategy = LiveKAMASSLStrategy(
        entry_filter=0.35,
        exit_filter=0.5,
        debug_mode=False,  # Keep debug off for cleaner output initially
        trade_direction='long',
        use_fusion_for_long=True,
        atr_length=9,
        hma_mode='VWMA',
        hma_length=50,
        atr_scaling_factor=1.4,
        allow_continuous_entries=True
    )
    
    # Fetch limited data for quick testing
    fetcher = OHLCFetcher()
    data = fetcher.fetch_data('BTCUSDT', '4h', limit=500)
    
    if data is None or len(data) < 100:
        print("❌ Could not fetch sufficient data for testing")
        return
    
    print(f"✅ Fetched {len(data)} candles for testing")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    
    # Calculate signals
    print("\n🔧 Calculating trading signals...")
    df = strategy.prepare_signals(data)
    
    # Count entries
    total_entries = df['long_entry'].sum()
    print(f"\n📊 RESULTS:")
    print(f"Total long entries: {total_entries}")
    
    # Show some example entries
    entry_dates = df[df['long_entry']].index[:5]
    print(f"\nFirst 5 entry dates:")
    for date in entry_dates:
        print(f"   {date}")
    
    # Calculate fusion filter values and check correlation
    fusion_ma, fusion_atr, fusion_cond = strategy.fusion_range_filter.calculate(df)
    
    range_inactive_periods = (~fusion_cond).sum()
    print(f"\nFusion Filter Analysis:")
    print(f"Range inactive periods: {range_inactive_periods}/{len(fusion_cond)} ({range_inactive_periods/len(fusion_cond)*100:.1f}%)")
    print(f"Range active periods: {fusion_cond.sum()}/{len(fusion_cond)} ({fusion_cond.sum()/len(fusion_cond)*100:.1f}%)")
    
    # Check if entries only happen when range is inactive
    entries_during_active_range = 0
    entries_during_inactive_range = 0
    
    for i in range(len(df)):
        if df['long_entry'].iloc[i]:
            if fusion_cond.iloc[i]:
                entries_during_active_range += 1
            else:
                entries_during_inactive_range += 1
    
    print(f"\nEntry Distribution:")
    print(f"Entries during active range (should be 0): {entries_during_active_range}")
    print(f"Entries during inactive range: {entries_during_inactive_range}")
    
    if entries_during_active_range == 0:
        print("✅ GOOD: No entries during active range periods!")
    else:
        print("❌ PROBLEM: Entries happened during active range periods!")
    
    return total_entries > 0

if __name__ == "__main__":
    print("🔧 Testing Fusion Filter Fix")
    print("=" * 50)
    success = test_fusion_fix()
    if success:
        print("\n✅ Test completed successfully!")
    else:
        print("\n❌ Test failed!") 
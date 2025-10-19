#!/usr/bin/env python3

import pandas as pd
import numpy as np
from classes.indicators.divergence import DivergenceDetector

def test_source_parameter():
    """Test if the source parameter actually affects divergence calculation"""
    
    print("="*60)
    print("TESTING DIVERGENCE SOURCE PARAMETER")
    print("="*60)
    
    # Create test data with clear differences between high/low and close
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='4H')
    n = len(dates)
    
    # Create price data where close prices have different patterns than high/low
    base_price = 100
    
    # Create high/low with one pattern
    high_pattern = base_price + 10 * np.sin(np.linspace(0, 4*np.pi, n))
    low_pattern = base_price + 8 * np.sin(np.linspace(0, 4*np.pi, n))
    
    # Create close with a different pattern (phase shifted)
    close_pattern = base_price + 9 * np.sin(np.linspace(0, 4*np.pi, n) + np.pi/2)
    
    # Create test DataFrame
    df = pd.DataFrame({
        'price_open': base_price + np.random.normal(0, 0.1, n),
        'price_high': high_pattern + np.random.normal(0, 0.1, n),
        'price_low': low_pattern + np.random.normal(0, 0.1, n),
        'price_close': close_pattern + np.random.normal(0, 0.1, n),
        'volume_traded': np.random.uniform(1000, 5000, n),
        'momentum': 50 + 20 * np.sin(np.linspace(0, 4*np.pi, n) + np.pi/4),  # Different phase for divergences
        'bb_basis': base_price + np.random.normal(0, 0.5, n)
    }, index=dates)
    
    print(f"Created test data with {len(df)} candles")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    # Test with source='Close'
    print("\n" + "-"*40)
    print("TESTING WITH SOURCE='Close'")
    print("-"*40)
    
    detector_close = DivergenceDetector(
        source='Close',
        debug_mode=True,
        long_pivot_period=5,
        short_pivot_period=5
    )
    
    divergences_close = detector_close.detect_divergences(df, indicator='momentum')
    
    print(f"\nFound {len(divergences_close)} divergences with source='Close'")
    for i, div in enumerate(divergences_close[:3]):  # Show first 3
        print(f"\nDivergence {i+1}:")
        print(f"  Type: {div['type']}")
        print(f"  Price comparison: {div['prev_price']:.2f} -> {div['current_price']:.2f}")
        print(f"  Indicator: {div['prev_ind']:.2f} -> {div['current_ind']:.2f}")
    
    # Test with source='High' (this should be ignored in current implementation)
    print("\n" + "-"*40)
    print("TESTING WITH SOURCE='High'")
    print("-"*40)
    
    detector_high = DivergenceDetector(
        source='High',
        debug_mode=True,
        long_pivot_period=5,
        short_pivot_period=5
    )
    
    divergences_high = detector_high.detect_divergences(df, indicator='momentum')
    
    print(f"\nFound {len(divergences_high)} divergences with source='High'")
    for i, div in enumerate(divergences_high[:3]):  # Show first 3
        print(f"\nDivergence {i+1}:")
        print(f"  Type: {div['type']}")
        print(f"  Price comparison: {div['prev_price']:.2f} -> {div['current_price']:.2f}")
        print(f"  Indicator: {div['prev_ind']:.2f} -> {div['current_ind']:.2f}")
    
    # Compare results
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    print(f"Divergences with source='Close': {len(divergences_close)}")
    print(f"Divergences with source='High': {len(divergences_high)}")
    
    # Check if the price values used are actually different
    if len(divergences_close) > 0 and len(divergences_high) > 0:
        close_prices = [(d['prev_price'], d['current_price']) for d in divergences_close]
        high_prices = [(d['prev_price'], d['current_price']) for d in divergences_high]
        
        print(f"\nFirst divergence price comparison:")
        print(f"  Close source: {close_prices[0]}")
        print(f"  High source: {high_prices[0]}")
        
        if close_prices[0] == high_prices[0]:
            print("  ❌ ISSUE: Same prices used despite different source parameter!")
        else:
            print("  ✅ Different prices used correctly")
    
    # Check what the actual source parameter is being used for
    print("\n" + "-"*40)
    print("ANALYZING SOURCE PARAMETER USAGE")
    print("-"*40)
    
    print(f"detector_close.source = '{detector_close.source}'")
    print(f"detector_high.source = '{detector_high.source}'")
    
    # Look at the actual divergence detection code
    print("\nAnalyzing the detect_divergences method:")
    print("Lines 265-266 in detect_divergences:")
    print("  if (filtered_df['price_high'].iloc[current_idx] < filtered_df['price_high'].iloc[next_idx] and")
    print("      filtered_df[indicator].iloc[current_idx] > filtered_df[indicator].iloc[next_idx]):")
    print("\nLines 295-296 in detect_divergences:")
    print("  if (filtered_df['price_low'].iloc[current_idx] > filtered_df['price_low'].iloc[next_idx] and")
    print("      filtered_df[indicator].iloc[current_idx] < filtered_df[indicator].iloc[next_idx]):")
    
    print("\n🔍 CONCLUSION:")
    print("The detect_divergences method is HARDCODED to use:")
    print("  - 'price_high' for bearish divergences (lines 265, 282-283)")
    print("  - 'price_low' for bullish divergences (lines 295, 308-309)")
    print("The 'source' parameter is IGNORED in the main divergence detection logic!")
    
    print("\n📝 The source parameter is only used in:")
    print("  - pivot() method for finding pivot points")
    print("  - positive_regular_positive_hidden_divergence() method (line 614)")
    print("  - negative_regular_negative_hidden_divergence() method (line 661)")
    print("  - But NOT in the main detect_divergences() method!")

if __name__ == "__main__":
    test_source_parameter() 
import sys
import pandas as pd
from datetime import datetime
from classes.indicators.divergence import DivergenceDetector
from classes.strategies.live_kama_ssl_strategy import LiveKAMASSLStrategy

def test_divergence_detection():
    """Test divergence detection with separate pivot periods for short and long trades"""
    # Initialize strategy with debug mode
    strategy = LiveKAMASSLStrategy(debug_mode=True)
    
    # Create a test DataFrame with sample data
    dates = pd.date_range(start='2024-01-01', end='2024-02-01', freq='4h')
    df = pd.DataFrame({
        'time_period_start': dates,
        'price_open': [100] * len(dates),
        'price_high': [110] * len(dates),
        'price_low': [90] * len(dates),
        'price_close': [105] * len(dates),
        'volume_traded': [1000] * len(dates)  # Add volume for indicator calculation
    })
    df.set_index('time_period_start', inplace=True)
    
    # Calculate indicators
    df = strategy.calculate_indicators(df)
    
    # Initialize divergence detector with debug mode
    div_detector = DivergenceDetector(debug_mode=True)
    div_detector.set_date_range('2024-01-01', '2024-02-01')
    
    print("\n=== Testing Divergence Detection ===")
    print("Testing with different pivot periods:")
    print("- Short trades: pivot_period = 11")
    print("- Long trades: pivot_period = 10")
    
    # Detect divergences with separate pivot periods
    divergences = div_detector.detect_divergences(
        df=df,
        indicator='rsi',
        length=14,
        order=5
    )
    
    print("\nDivergence Results:")
    for div in divergences:
        print(f"Type: {div['type']}")
        print(f"Period: {div['pivot_period']}")
        print(f"Start: {div['start_time']}")
        print(f"End: {div['end_time']}")
        print(f"Price Idx: {div['price_idx']}")
        print(f"Indicator Idx: {div['indicator_idx']}")
        print("---")

if __name__ == "__main__":
    test_divergence_detection() 
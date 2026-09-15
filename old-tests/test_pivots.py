import pandas as pd
import numpy as np
from classes.indicators.divergence import DivergenceDetector

def create_test_data():
    # Create sample data with known pivot points
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='1H')
    n = len(dates)
    
    # Create a sine wave for price movement to ensure we have clear highs and lows
    t = np.linspace(0, 4*np.pi, n)
    base_price = 100 + 10*np.sin(t)
    
    # Add some noise to make it more realistic
    noise = np.random.normal(0, 0.5, n)
    prices = base_price + noise
    
    # Create DataFrame with OHLC data
    df = pd.DataFrame({
        'price_open': prices,
        'price_high': prices + np.random.uniform(0, 0.5, n),
        'price_low': prices - np.random.uniform(0, 0.5, n),
        'price_close': prices + np.random.normal(0, 0.1, n),
        'volume': np.random.uniform(1000, 5000, n),
        'rsi': 50 + 20*np.sin(t + np.pi/4),  # RSI with phase shift
        'momentum': 100 + 30*np.sin(t + np.pi/3),  # Momentum with different phase
        'adp': 200 + 40*np.sin(t + np.pi/6),  # ADP with another phase
        'cci': 0 + 100*np.sin(t + np.pi/5)  # CCI with another phase
    }, index=dates)
    
    return df

def test_pivot_detection():
    print("\nTesting pivot point detection...")
    
    # Create test data
    df = create_test_data()
    print(f"Created test data with shape: {df.shape}")
    
    # Initialize DivergenceDetector with debug mode
    detector = DivergenceDetector(debug_mode=True)
    detector.df = df
    
    # Calculate pivot points
    print("\nCalculating pivot points...")
    detector.calculate_pivot_points()
    
    # Get pivot points
    high_pivots = [(i, p) for i, p in enumerate(detector.highs) if p is not None]
    low_pivots = [(i, p) for i, p in enumerate(detector.lows) if p is not None]
    
    print(f"\nFound {len(high_pivots)} high pivots and {len(low_pivots)} low pivots")
    
    # Analyze pivot points
    print("\nAnalyzing high pivots:")
    for i, pivot in enumerate(high_pivots[:5]):  # Show first 5 high pivots
        idx, data = pivot
        print(f"\nHigh Pivot #{i}:")
        print(f"Index: {idx}")
        print(f"Price: {data[1]:.2f}")
        print(f"RSI: {data[2]:.2f}")
        print(f"Momentum: {data[3]:.2f}")
        print(f"ADP: {data[4]:.2f}")
        
        # Get surrounding prices
        window = 5
        surrounding_prices = df['price_high'].iloc[max(0, idx-window):min(len(df), idx+window+1)]
        print("\nSurrounding prices:")
        print(surrounding_prices)
        
        # Verify if it's a valid pivot
        is_valid = all(data[1] >= p for p in surrounding_prices.iloc[:window]) and \
                  all(data[1] >= p for p in surrounding_prices.iloc[window+1:])
        print(f"Valid pivot: {is_valid}")
    
    print("\nAnalyzing low pivots:")
    for i, pivot in enumerate(low_pivots[:5]):  # Show first 5 low pivots
        idx, data = pivot
        print(f"\nLow Pivot #{i}:")
        print(f"Index: {idx}")
        print(f"Price: {data[1]:.2f}")
        print(f"RSI: {data[2]:.2f}")
        print(f"Momentum: {data[3]:.2f}")
        print(f"ADP: {data[4]:.2f}")
        
        # Get surrounding prices
        window = 5
        surrounding_prices = df['price_low'].iloc[max(0, idx-window):min(len(df), idx+window+1)]
        print("\nSurrounding prices:")
        print(surrounding_prices)
        
        # Verify if it's a valid pivot
        is_valid = all(data[1] <= p for p in surrounding_prices.iloc[:window]) and \
                  all(data[1] <= p for p in surrounding_prices.iloc[window+1:])
        print(f"Valid pivot: {is_valid}")

def test_divergence_detection():
    print("\nTesting divergence detection...")
    
    # Create test data
    df = create_test_data()
    print(f"Created test data with shape: {df.shape}")
    
    # Initialize DivergenceDetector with debug mode
    detector = DivergenceDetector(debug_mode=True)
    
    # Detect divergences
    divergences = detector.detect_divergences(df)
    
    if divergences:
        print(f"\nFound {len(divergences)} divergences:")
        for i, div in enumerate(divergences):
            print(f"\nDivergence #{i+1}:")
            print(f"Type: {div['type']}")
            print(f"Price indices: {div['price_idx']} -> {div['prev_price_idx']}")
            print(f"Prices: {div['current_price']:.2f} -> {div['prev_price']:.2f}")
            print(f"Indicator: {div['current_ind']:.2f} -> {div['prev_ind']:.2f}")
            print(f"Status: {div['status']}")
    else:
        print("\nNo divergences found")

def main():
    test_pivot_detection()
    test_divergence_detection()

if __name__ == "__main__":
    main() 
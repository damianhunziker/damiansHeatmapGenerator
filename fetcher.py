from strategy_utils import print_logo, fetch_data, get_trading_pairs
from classes.data_fetcher import OHLCFetcher
import pandas as pd
from tqdm import tqdm
import os

def fetch_all_data(interval, higher_tf):
    """Fetch data for all trading pairs"""
    pairs = get_trading_pairs()
    print(f"\nFetching data for {len(pairs)} trading pairs")
    
    # Create cache directory if it doesn't exist
    cache_dir = os.path.join('data_cache', interval)
    os.makedirs(cache_dir, exist_ok=True)
    
    # Higher timeframe cache directory
    higher_cache_dir = os.path.join('data_cache', higher_tf)
    os.makedirs(higher_cache_dir, exist_ok=True)
    
    # Fetch data for each pair
    failed_pairs = []
    for pair in pairs:
        try:

            # Delete existing cache files
            cache_dir = "ohlc_cache"
            cache_file = f"{cache_dir}/{pair}_{interval}_ohlc.csv"
            os.remove(cache_file)
            cache_file_htf = f"{cache_dir}/{pair}_{higher_tf}_ohlc.csv"
            os.remove(cache_file_htf)

            # Fetch data for both timeframes with maximum lookback
            timeframe_data = fetch_data(
                pair, 
                interval
            )

            # Fetch data for both timeframes with maximum lookback
            timeframe_data_htf = fetch_data(
                pair, 
                higher_tf
            )
            
        except Exception as e:
            print(f"\nError fetching {pair}: {str(e)}")
            failed_pairs.append(pair)
    
    # Report results
    print("\nData fetching completed!")
    print(f"Successfully fetched: {len(pairs) - len(failed_pairs)} pairs")
    if failed_pairs:
        print(f"Failed to fetch: {len(failed_pairs)} pairs")
        print("Failed pairs:", failed_pairs)

if __name__ == "__main__":
    print_logo()
    print("DATA FETCHER - Heatmap Generator and Strategy Backtester")
    
    # Get timeframe inputs
    interval = input("\nEnter primary timeframe (e.g., 1h): ").lower()
    higher_tf = input("Enter higher timeframe (e.g., 4h): ").lower()
    
    # Fetch data
    fetch_all_data(interval, higher_tf) 
import os
import shutil
from datetime import datetime
import time
import pandas as pd
from heatmap import create_heatmap, get_parameter_ranges
from strategy_utils import get_available_strategies, fetch_data, print_logo, get_valid_date, get_trading_pairs
from fetcher import OHLCFetcher

def get_user_inputs():
    """Get user inputs for automation"""
    print("\nAvailable Strategies:")
    strategies = get_available_strategies()
    for number, (name, _) in strategies.items():
        print(f"{number}: {name}")
    
    while True:
        strategy_number = input("\nEnter strategy number: ")
        if strategy_number in strategies:
            strategy_name, strategy_class = strategies[strategy_number]
            break
        print("Invalid strategy number. Please try again.")
    
    # Get date inputs
    while True:
        start_date = input("\nEnter start date (YYYY-MM-DD): ")
        if get_valid_date(start_date):
            break
        print("Invalid date format. Please use YYYY-MM-DD")
    
    while True:
        end_date = input("Enter end date (YYYY-MM-DD): ")
        if get_valid_date(end_date):
            break
        print("Invalid date format. Please use YYYY-MM-DD")
    
    return strategy_number, strategy_name, strategy_class, start_date, end_date

def run_heatmap_for_pairs():
    # Print logo
    print_logo()
    
    print("AUTOMATOR - Heatmap Generator and Strategy Backtester")
    
    # Get user inputs
    strategy_number, strategy_name, strategy_class, start_date, end_date = get_user_inputs()
    
    # Get pairs from strategy_utils
    pairs = get_trading_pairs()
    
    # Fixed parameters
    interval = "4h"
    higher_tf = "1d"  # Add higher timeframe
    initial_equity = 1000
    fee_pct = 0.04
    
    # Get parameter ranges
    param_ranges = get_parameter_ranges(strategy_class)
    
    # Create timestamp for the run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create main output directory
    output_dir = f"automator_html/run_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a log file
    log_file = os.path.join(output_dir, "automation_log.txt")
    
    with open(log_file, "w") as f:
        f.write(f"Heatmap Automation Run - {timestamp}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Configuration:\n")
        f.write(f"Interval: {interval}\n")
        f.write(f"Higher Timeframe: {higher_tf}\n")
        f.write(f"Initial Equity: {initial_equity}\n")
        f.write(f"Fee Percentage: {fee_pct}\n")
        f.write(f"Date Range: {start_date} to {end_date}\n")
        f.write(f"Strategy: {strategy_name}\n\n")
        f.write("Processing pairs:\n")
    
    # Process each pair
    for pair in pairs:
        print(f"\nProcessing {pair}...")
        
        try:
            # Create pair-specific directory
            pair_dir = os.path.join(output_dir, pair)
            os.makedirs(pair_dir, exist_ok=True)
            
            # Create fresh directory
            os.makedirs("html_cache/pnl_images", exist_ok=True)
            
            # Initialize fetcher and fetch data
            fetcher = OHLCFetcher()
            primary_data = fetcher.fetch_data(pair, interval)
            higher_data = fetcher.fetch_data(pair, higher_tf)
            
            # Create timeframe_data structure
            timeframe_data = {
                'primary': {'data': primary_data, 'interval': interval},
                'higher': {'data': higher_data, 'interval': higher_tf}
            }
            
            # Calculate lookback candles based on date range
            mask = (primary_data.index >= start_date) & (primary_data.index <= end_date)
            filtered_data = primary_data[mask]
            
            start_lookback = len(primary_data[primary_data.index < start_date])
            end_lookback = len(primary_data[primary_data.index <= end_date])
            lookback_candles = end_lookback - start_lookback
            
            # Log the execution start
            with open(log_file, "a") as f:
                f.write(f"\n{pair} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            # Run heatmap creation
            create_heatmap(
                timeframe_data=timeframe_data,
                strategy_class=strategy_class,
                param_ranges=param_ranges,
                initial_equity=initial_equity,
                fee_pct=fee_pct,
                last_n_candles_analyze=lookback_candles,
                last_n_candles_display=end_lookback,
                interval=interval,
                asset=pair,
                strategy_name=strategy_name,
                start_date=start_date,
                end_date=end_date
            )
            
            # Copy the html_cache directory to pair directory
            if os.path.exists("html_cache"):
                # Create the destination directory
                dest_dir = os.path.join(pair_dir, "html")
                # Use copytree with dirs_exist_ok=True to handle existing directories
                shutil.copytree("html_cache", dest_dir, dirs_exist_ok=True)
            
            print(f"Completed {pair}")
            
            # Add a small delay between pairs to prevent rate limiting
            time.sleep(2)
            
        except Exception as e:
            print(f"Error processing {pair}: {str(e)}")
            with open(log_file, "a") as f:
                f.write(f"\nError processing {pair}: {str(e)}\n")
            continue
    
    print(f"\nAutomation complete! Results saved in: {output_dir}")
    print(f"Check {log_file} for detailed execution log")

if __name__ == "__main__":
    run_heatmap_for_pairs() 
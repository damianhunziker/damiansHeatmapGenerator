import os
import importlib
import inspect
from classes.data_fetcher import OHLCFetcher
from classes.base_strategy import BaseStrategy
import numpy as np

def get_available_strategies():
    """Automatically detects all available strategies in the strategies folder"""
    strategies = {}
    strategy_files = [f for f in os.listdir('classes/strategies') if f.endswith('_strategy.py')]
    
    for idx, file in enumerate(strategy_files, 1):
        module_name = file[:-3]  # Remove .py
        module = importlib.import_module(f'classes.strategies.{module_name}')
        
        # Find all classes in the module that inherit from BaseStrategy
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and 
                issubclass(obj, BaseStrategy) and 
                obj != BaseStrategy):
                # Add to selection
                strategies[str(idx)] = (name, obj)
                break
    
    return strategies 

def get_strategy_inputs(strategy_class):
    """Ask user for strategy parameters"""
    params = {}
    print("\nEnter strategy parameters:")
    for param_name, (default_value, description) in strategy_class.get_parameters().items():
        value = input(f"Enter {description} [{default_value}]: ") or default_value
        params[param_name] = type(default_value)(value)  # Convert to correct data type
    return params

def print_logo():
    print("""


Damian's

 ██░ ██ ▓█████ ▄▄▄     ▄▄▄█████▓ ███▄ ▄███▓ ▄▄▄       ██▓███                  
▓██░ ██▒▓█   ▀▒████▄   ▓  ██▒ ▓▒▓██▒▀█▀ ██▒▒████▄    ▓██░  ██▒                
▒██▀▀██░▒███  ▒██  ▀█▄ ▒ ▓██░ ▒░▓██    ▓██░▒██  ▀█▄  ▓██░ ██▓▒                
░▓█ ░██ ▒▓█  ▄░██▄▄▄▄██░ ▓██▓ ░ ▒██    ▒██ ░██▄▄▄▄██ ▒██▄█▓▒ ▒                
░▓█▒░██▓░▒████▒▓█   ▓██▒ ▒██▒ ░ ▒██▒   ░██▒ ▓█   ▓██▒▒██▒ ░  ░                
 ▒ ░░▒░▒░░ ▒░ ░▒▒   ▓▒█░ ▒ ░░   ░ ▒░   ░  ░ ▒▒   ▓▒█░▒▓▒░ ░  ░                
 ▒ ░▒░ ░ ░ ░  ░ ▒   ▒▒ ░   ░    ░  ░      ░  ▒   ▒▒ ░░▒ ░                     
 ░  ░░ ░   ░    ░   ▒    ░      ░      ░     ░   ▒   ░░                       
 ░  ░  ░   ░  ░     ░  ░               ░         ░  ░                         
                                                                              
  ▄████ ▓█████  ███▄    █ ▓█████  ██▀███   ▄▄▄     ▄▄▄█████▓ ▒█████   ██▀███  
 ██▒ ▀█▒▓█   ▀  ██ ▀█   █ ▓█   ▀ ▓██ ▒ ██▒▒████▄   ▓  ██▒ ▓▒▒██▒  ██▒▓██ ▒ ██▒
▒██░▄▄▄░▒███   ▓██  ▀█ ██▒▒███   ▓██ ░▄█ ▒▒██  ▀█▄ ▒ ▓██░ ▒░▒██░  ██▒▓██ ░▄█ ▒
░▓█  ██▓▒▓█  ▄ ▓██▒  ▐▌██▒▒▓█  ▄ ▒██▀▀█▄  ░██▄▄▄▄██░ ▓██▓ ░ ▒██   ██░▒██▀▀█▄  
░▒▓███▀▒░▒████▒▒██░   ▓██░░▒████▒░██▓ ▒██▒ ▓█   ▓██▒ ▒██▒ ░ ░ ████▓▒░░██▓ ▒██▒
 ░▒   ▒ ░░ ▒░ ░░ ▒░   ▒ ▒ ░░ ▒░ ░░ ▒▓ ░▒▓░ ▒▒   ▓▒█░ ▒ ░░   ░ ▒░▒░▒░ ░ ▒▓ ░▒▓░
  ░   ░  ░ ░  ░░ ░░   ░ ▒░ ░ ░  ░  ░▒ ░ ▒░  ▒   ▒▒ ░   ░      ░ ▒ ▒░   ░▒ ░ ▒░
░ ░   ░    ░      ░   ░ ░    ░     ░░   ░   ░   ▒    ░      ░ ░ ░ ▒    ░░   ░ 
      ░    ░  ░         ░    ░  ░   ░           ░  ░            ░ ░     ░     
                                                                              
                                                                                                                                                                  
    """)

def get_user_inputs():
    """Ask user for inputs and return them"""
    # Strategy selection
    strategy_choices = get_available_strategies()
    
    print("\nAvailable strategies:")
    for key, (name, _) in strategy_choices.items():
        print(f"{key}: {name}")
    
    strategy_choice = input("\nSelect strategy [1]: ") or "1"
    strategy_name, strategy_class = strategy_choices[strategy_choice]
    print(f"\nSelected strategy: {strategy_name}")
    
    # Base parameters
    asset = input("Enter trading pair [BTCUSDT]: ") or "BTCUSDT"
    interval = input("Enter time interval [4h]: ") or "4h"
    initial_equity = float(input("Enter initial capital in USD [10000]: ") or "10000")
    last_n_candles = int(input("Enter number of candles to display [16151]: ") or "16151")
    lookback_candles = int(input("Enter number of candles to analyze [16151]: ") or "16151")
    fee_pct = float(input("Enter fee percentage per trade [0.04]: ") or "0.04")
    
    return asset, interval, initial_equity, last_n_candles, lookback_candles, fee_pct, strategy_name, strategy_class

def fetch_data(asset, interval):
    """Load data with OHLCFetcher"""
    fetcher = OHLCFetcher()
    data = fetcher.fetch_data(asset, interval)
    print(f"Loaded {len(data)} datasets.")
    return data

def get_parameter_ranges(strategy_class):
    """Ask user for parameter ranges for the heatmap"""
    param_ranges = {}
    print("\nEnter parameter ranges for the heatmap:")
    
    for param_name, (default_value, description) in strategy_class.get_parameters().items():
        # Print the parameter name
        print(f"Enter ranges for parameter: {param_name}")
        
        # Check if the default value is a float
        if isinstance(default_value, float):
            min_val = float(input(f"Enter minimum value [{default_value / 2}]: ") or default_value / 2)
            max_val = float(input(f"Enter maximum value [{default_value * 2}]: ") or default_value * 2)
            step = float(input(f"Enter step size [0.1]: ") or 0.1)
        else:
            min_val = int(input(f"Enter minimum value [{default_value // 2}]: ") or default_value // 2)
            max_val = int(input(f"Enter maximum value [{default_value * 2}]: ") or default_value * 2)
            step = int(input(f"Enter step size [1]: ") or 1)
        
        param_ranges[param_name] = np.arange(min_val, max_val + step, step)
    
    return param_ranges
import os
import importlib
import inspect
from classes.data_fetcher import OHLCFetcher
from classes.base_strategy import BaseStrategy
import numpy as np
import pandas as pd
import readline  # Enables arrow key navigation in input
from prompt_toolkit import prompt  # Für bessere Input-Handling
from pathlib import Path
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

def get_valid_date(date_string):
    """Validate date string format YYYY-MM-DD"""
    try:
        datetime.strptime(date_string, '%Y-%m-%d')
        return True
    except ValueError:
        return False

def get_available_strategies():
    """Automatically detects all available strategies in the strategies folder"""
    strategies = {}
    strategy_files = [f for f in os.listdir('classes/strategies') if f.endswith('_strategy.py')]
    
    for idx, file in enumerate(strategy_files, 1):
        module_name = file[:-3]  # Remove .py
        module = importlib.import_module(f'classes.strategies.{module_name}')
        
        # Find all classes in the module that inherit from BaseStrategy
        strategy_found = False
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and 
                issubclass(obj, BaseStrategy) and 
                obj != BaseStrategy and
                obj.__module__ == f'classes.strategies.{module_name}'):  # Nur Klassen aus diesem Modul
                strategies[str(idx)] = (name, obj)
                strategy_found = True
                break
                
        if not strategy_found:
            continue
    
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


DAMIAN's

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

def ensure_timestamp_column(df):
    """Ensures the DataFrame has a proper timestamp column"""
    if 'timestamp' not in df.columns:
        if 'date' in df.columns:
            df['timestamp'] = pd.to_datetime(df['date'])
            del df['date']
        elif 'time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['time'])
            del df['time']
        else:
            df['timestamp'] = pd.to_datetime(df.index)
    else:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def get_user_inputs():
    """Gets all necessary user inputs in the specified order"""
    from prompt_toolkit import prompt
    import os
    from pathlib import Path
    
    # Get relative path to cache directory
    cache_dir = "ohlc_cache"
    
    # 1. Strategy Selection
    strategies = get_available_strategies()
    print("\nAvailable Strategies:")
    for key, (name, _) in strategies.items():
        print(f"{key}: {name}")
    
    default_choice = '1'
    choice = input(f"\nSelect strategy number [{default_choice}]: ") or default_choice
    while choice not in strategies:
        print("Invalid selection. Please try again.")
        choice = input(f"Select strategy number [{default_choice}]: ") or default_choice
    
    strategy_name, strategy_class = strategies[choice]
    
    # Get required timeframes from strategy
    required_timeframes = strategy_class.get_required_timeframes()
    timeframe_data = {}
    
    # 2. Asset
    default_asset = "BTCUSDT"
    asset = input(f"\nEnter asset [{default_asset}]: ").upper() or default_asset
    
    # 3. Handle each required timeframe
    for tf_name, default_tf in required_timeframes.items():
        print(f"\nSetting up {tf_name} timeframe:")
        interval = input(f"Enter interval (1m, 5m, 15m, 1h, 4h, 1d) [{default_tf}]: ").lower() or default_tf
        
        # Check cached data
        cache_file = f"{cache_dir}/{asset}_{interval}_ohlc.csv"
        print(f"Looking for cache file: {cache_file}")
        data_exists = os.path.exists(cache_file)
        
        if data_exists:
            cached_data = pd.read_csv(cache_file, index_col='time_period_start', parse_dates=True)
            cached_data = ensure_timestamp_column(cached_data)
            earliest_date = (cached_data['timestamp'].min() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            latest_date = cached_data['timestamp'].max().strftime('%Y-%m-%d')
            
            print(f"\nFound cached data from {earliest_date} to {latest_date}")
            refetch = input("Do you want to refetch data? (y/[N]): ").lower() == 'y'
            
            if refetch:
                print("Removing old cache file...")
                os.remove(cache_file)
                fetch_data(asset, interval)
                print("Loading new data from cache...")
                data = pd.read_csv(cache_file, index_col='time_period_start', parse_dates=True)
                data = ensure_timestamp_column(data)
            else:
                data = cached_data
        else:
            print("\nNo cached data found. Fetching data...")
            fetch_data(asset, interval)
            print("Loading data from cache...")
            data = pd.read_csv(cache_file, index_col='time_period_start', parse_dates=True)
            data = ensure_timestamp_column(data)
        
        timeframe_data[tf_name] = {
            'interval': interval,
            'data': data
        }
    
    # Use primary timeframe for date range selection
    primary_data = timeframe_data['primary']['data']
    
    # 4. Check available date range
    min_date = (primary_data['timestamp'].min() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    max_date = primary_data['timestamp'].max().strftime('%Y-%m-%d')
    
    # 5. Get date range from user
    print(f"\nAvailable date range: {min_date} to {max_date}")
    
    while True:
        try:
            start_date = prompt("Enter start date: ", default=min_date)
            end_date = prompt("Enter end date: ", default=max_date)
            
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            if start_dt < primary_data['timestamp'].min() or end_dt > primary_data['timestamp'].max():
                print(f"Error: Please select dates between {min_date} and {max_date}")
                continue
                
            if start_dt > end_dt:
                print("Error: Start date must be before or equal to end date")
                continue
                
            break
        except ValueError:
            print("Error: Please enter valid dates in YYYY-MM-DD format")
    
    # 6. Get other inputs
    default_equity = "1000"
    initial_equity = float(input(f"\nEnter initial equity [{default_equity}]: ") or default_equity)
    
    default_fee = "0.04"
    fee_pct = float(input(f"Enter fee percentage [{default_fee}]: ") or default_fee)
    
    # 7. Calculate lookback candles for primary timeframe
    mask = (primary_data['timestamp'] >= start_dt) & (primary_data['timestamp'] <= end_dt)
    filtered_data = primary_data[mask].copy()
    
    start_lookback_candles = len(primary_data[primary_data['timestamp'] < start_dt])
    end_lookback_candles = len(primary_data[primary_data['timestamp'] <= end_dt])
    lookback_candles = end_lookback_candles - start_lookback_candles
    
    return {
        'strategy_name': strategy_name,
        'strategy_class': strategy_class,
        'asset': asset,
        'timeframe_data': timeframe_data,
        'initial_equity': initial_equity,
        'fee_pct': fee_pct,
        'start_date': start_date,
        'end_date': end_date,
        'lookback_candles': lookback_candles,
        'start_lookback_candles': start_lookback_candles,
        'end_lookback_candles': end_lookback_candles,
        'interval': interval
    }

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

def create_performance_chart(timestamps, pnl_performance, buy_hold, drawdown, output_type='interactive', params=None, base_dir='pnl_cache'):
    """Creates a performance comparison chart"""
    fig = make_subplots(
        rows=2, cols=1,
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
        subplot_titles=('Performance with Drawdown', 'Performance Only'),
        vertical_spacing=0.2
    )
    
    # Stelle sicher, dass alle Listen die gleiche Länge haben
    if not (len(timestamps) == len(pnl_performance) == len(buy_hold) == len(drawdown)):
        print(f"Warning: Unequal lengths - timestamps: {len(timestamps)}, pnl: {len(pnl_performance)}, buy_hold: {len(buy_hold)}, drawdown: {len(drawdown)}")
        # Nutze die kürzeste Länge
        min_len = min(len(timestamps), len(pnl_performance), len(buy_hold), len(drawdown))
        timestamps = timestamps[:min_len]
        pnl_performance = pnl_performance[:min_len]
        buy_hold = buy_hold[:min_len]
        drawdown = drawdown[:min_len]
    
    # Erstelle separate Linien für positive und negative Bereiche
    x_vals = list(timestamps)
    y_vals = list(pnl_performance)
    
    # Initialisiere Listen für positive und negative Segmente
    pos_x = []
    pos_y = []
    neg_x = []
    neg_y = []
    
    # Fülle die Listen basierend auf den Y-Werten
    for i in range(len(y_vals)):
        if y_vals[i] >= 0:
            pos_x.append(x_vals[i])
            pos_y.append(y_vals[i])
            if neg_x and neg_y:  # Wenn es vorherige negative Werte gab
                neg_x.append(x_vals[i])
                neg_y.append(None)
        else:
            neg_x.append(x_vals[i])
            neg_y.append(y_vals[i])
            if pos_x and pos_y:  # Wenn es vorherige positive Werte gab
                pos_x.append(x_vals[i])
                pos_y.append(None)
    
    # Füge positive Performance hinzu (nur wenn Daten vorhanden) - Subplot 1
    if pos_x and pos_y:
        fig.add_trace(
            go.Scatter(
                x=pos_x,
                y=pos_y,
                name='Strategy Performance (Profit)',
                fill='tozeroy',
                line=dict(color='green', width=2),
                fillcolor='rgba(0,255,0,0.1)'
            ),
            row=1, col=1,
            secondary_y=False
        )
        # Gleiche Trace für Subplot 2
        fig.add_trace(
            go.Scatter(
                x=pos_x,
                y=pos_y,
                name='Strategy Performance (Profit)',
                fill='tozeroy',
                line=dict(color='green', width=2),
                fillcolor='rgba(0,255,0,0.1)',
                showlegend=False  # Keine zusätzliche Legende für den zweiten Plot
            ),
            row=2, col=1
        )
    
    # Füge negative Performance hinzu (nur wenn Daten vorhanden) - Subplot 1
    if neg_x and neg_y:
        fig.add_trace(
            go.Scatter(
                x=neg_x,
                y=neg_y,
                name='Strategy Performance (Loss)',
                fill='tozeroy',
                line=dict(color='red', width=2),
                fillcolor='rgba(255,0,0,0.1)'
            ),
            row=1, col=1,
            secondary_y=False
        )
        # Gleiche Trace für Subplot 2
        fig.add_trace(
            go.Scatter(
                x=neg_x,
                y=neg_y,
                name='Strategy Performance (Loss)',
                fill='tozeroy',
                line=dict(color='red', width=2),
                fillcolor='rgba(255,0,0,0.1)',
                showlegend=False  # Keine zusätzliche Legende für den zweiten Plot
            ),
            row=2, col=1
        )
    
    # Füge Buy & Hold hinzu - nur im ersten Subplot
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=buy_hold,
            name='Buy & Hold',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1,
        secondary_y=False
    )
    
    # Füge Drawdown hinzu - nur im ersten Subplot
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=drawdown,
            name='Drawdown',
            fill='tozeroy',
            fillcolor='rgba(148,0,211,0.15)',
            line=dict(color='rgba(148,0,211,0.5)')
        ),
        row=1, col=1,
        secondary_y=True
    )
    
    # Update Layout
    title = f'Strategy vs Buy & Hold Performance for params: {params}' if params else 'Strategy Performance with Drawdown'
    fig.update_layout(
        title=title,
        height=900,  # Erhöhe die Höhe für zwei Subplots
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.1,  # Position unterhalb des Charts
            xanchor="center",
            x=0.5,   # Zentriert
        ),
        margin=dict(b=100),  # Mehr Platz unten für die Legende
        hovermode='x unified'
    )
    
    # Update y-axes labels
    fig.update_yaxes(title_text="Performance %", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Drawdown %", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Performance %", row=2, col=1)
    
    # Update x-axes labels
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    
    if output_type == 'interactive':
        return fig
    else:
        image_path = os.path.join(base_dir, f'pnl_image_{params}.png')
        fig.write_image(image_path)
        return image_path

def get_trading_pairs():
    """Returns a list of default trading pairs"""
    return [
        "BTCUSDT",
        "ETHUSDT",
        "BNBUSDT",
        "SOLUSDT",
        "DOGEUSDT",
        "AVAXUSDT",
        "LINKUSDT",
        "SHIBUSDT",
        "APEUSDT",
        "TONUSDT"
    ]

def process_chunk_base(data, trade_direction, initial_equity, fee_pct):
    """Base implementation of process_chunk that all strategies can use"""
    trades = []
    in_position = False
    entry_time = None
    entry_price = None
    current_equity = initial_equity
    current_direction = None
    
    for i in range(1, len(data)):
        current_time = data.index[i]
        
        if not in_position:
            if (trade_direction in ['long', 'both'] and data['long_entry'].iloc[i]) or \
               (trade_direction in ['short', 'both'] and data['short_entry'].iloc[i]):
                entry_time = current_time
                entry_price = data['price_close'].iloc[i]
                in_position = True
                current_direction = 'long' if data['long_entry'].iloc[i] else 'short'
                
        elif (current_direction == 'long' and data['long_exit'].iloc[i]) or \
             (current_direction == 'short' and data['short_exit'].iloc[i]):
            exit_price = data['price_close'].iloc[i]
            
            # Calculate position size and contracts
            position_size = current_equity
            contracts = position_size / entry_price
            
            # Calculate profit based on trade direction
            if current_direction == "long":
                gross_profit = contracts * (exit_price - entry_price)
            else:  # short
                gross_profit = contracts * (entry_price - exit_price)
            
            # Calculate fees and net profit
            entry_fee = position_size * fee_pct
            exit_fee = (position_size + gross_profit) * fee_pct
            total_fee = entry_fee + exit_fee
            net_profit = gross_profit - total_fee
            
            # Update equity
            current_equity += net_profit
            
            # Add trade to list
            trade = (entry_time, current_time, entry_price, exit_price, net_profit, gross_profit, total_fee, current_direction)
            trades.append(trade)
            
            in_position = False
            entry_time = None
            entry_price = None
    
    # If the last position is still open
    if in_position:
        trade = (entry_time, data.index[-1], entry_price, data['price_close'].iloc[-1], None, None, None, current_direction)
        trades.append(trade)
    
    return trades
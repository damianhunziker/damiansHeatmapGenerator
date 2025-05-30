import inspect
import importlib
import sys
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import os
from strategy_utils import get_available_strategies
import pandas as pd
import numpy as np
from classes.strategies.live_kama_ssl_strategy import LiveKAMASSLStrategy
import yfinance as yf

def get_class_init_params(module_name: str, class_name: str) -> Dict[str, Any]:
    """Get initialization parameters for a class from a module."""
    try:
        # Handle module paths correctly
        if not module_name.startswith('classes.'):
            module_name = f'classes.{module_name}'
        
        # Import the module
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            print(f"Warning: Could not import module {module_name}: {str(e)}")
            return {}
        
        # Get the class
        try:
            class_obj = getattr(module, class_name)
        except AttributeError:
            print(f"Warning: Class {class_name} not found in module {module_name}")
            return {}
        
        # Get the __init__ signature
        try:
            signature = inspect.signature(class_obj.__init__)
        except ValueError:
            print(f"Warning: Could not get signature for {class_name}.__init__")
            return {}
        
        # Extract parameters and their default values
        params = {}
        for name, param in signature.parameters.items():
            if name != 'self':  # Skip 'self' parameter
                params[name] = {
                    'default': param.default if param.default != inspect.Parameter.empty else None,
                    'annotation': str(param.annotation) if param.annotation != inspect.Parameter.empty else None,
                    'required': param.default == inspect.Parameter.empty
                }
        
        # If this is a strategy class, add its get_parameters() info
        if hasattr(class_obj, 'get_parameters'):
            try:
                strategy_params = class_obj.get_parameters()
                for name, (default, description) in strategy_params.items():
                    if name not in params:
                        params[name] = {
                            'default': default,
                            'description': description,
                            'required': default is None
                        }
            except Exception as e:
                print(f"Warning: Error getting strategy parameters for {class_name}: {str(e)}")
        
        return params
    except Exception as e:
        print(f"Warning: Unexpected error getting parameters for {module_name}.{class_name}: {str(e)}")
        return {}

def get_schema() -> Dict[str, Any]:
    """Get the parameter schema for all executable files."""
    schema = {
        'automator': {
            'description': 'Automates running strategies across multiple pairs',
            'parameters': get_class_init_params('automator', 'Automator')
        },
        'heatmap': {
            'description': 'Generates heatmaps for strategy parameter optimization',
            'parameters': get_class_init_params('heatmap', 'Heatmap')
        },
        'chart_analysis': {
            'description': 'Analyzes and visualizes trading charts',
            'parameters': get_class_init_params('chart_analysis', 'ChartAnalysis')
        },
        'pnl': {
            'description': 'Calculates and visualizes PnL',
            'parameters': get_class_init_params('pnl', 'PnL')
        },
        'fetcher': {
            'description': 'Fetches market data',
            'parameters': get_class_init_params('data_fetcher', 'OHLCFetcher')
        }
    }
    
    # Add common parameters
    common_params = {
        'start_date': {
            'type': 'string',
            'format': 'YYYY-MM-DD',
            'required': True,
            'description': 'Start date for analysis'
        },
        'end_date': {
            'type': 'string',
            'format': 'YYYY-MM-DD',
            'required': True,
            'description': 'End date for analysis'
        },
        'interval': {
            'type': 'string',
            'enum': ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d'],
            'default': '4h',
            'description': 'Trading interval'
        },
        'asset': {
            'type': 'string',
            'description': 'Trading pair (e.g., BTCUSDT)',
            'required': True
        },
        'strategy': {
            'type': 'string',
            'description': 'Strategy name to use',
            'required': False
        }
    }
    
    # Add common parameters to each executable that has parameters
    for exec_name in schema:
        if schema[exec_name]['parameters']:  # Only add if parameters exist
            schema[exec_name]['parameters'].update(common_params)
    
    # Add available strategies and their parameters
    try:
        strategies = get_available_strategies()
        schema['strategies'] = {}
        for number, (name, strategy_class) in strategies.items():
            schema['strategies'][name] = {
                'description': f'Strategy #{number}',
                'parameters': get_class_init_params(strategy_class.__module__, strategy_class.__name__)
            }
    except Exception as e:
        print(f"Warning: Error loading strategies: {str(e)}")
        schema['strategies'] = {}
    
    return schema

def validate_date(date_str: str) -> bool:
    """Validate date string format."""
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return True
    except ValueError:
        return False

def validate_parameters(exec_name: str, params: Dict[str, Any]) -> List[str]:
    """Validate parameters for an executable."""
    errors = []
    
    # Validate required parameters
    if 'start_date' not in params:
        errors.append("Missing required parameter: start_date")
    elif not validate_date(params['start_date']):
        errors.append("Invalid date format for start_date. Use YYYY-MM-DD")
    
    if 'end_date' not in params:
        errors.append("Missing required parameter: end_date")
    elif not validate_date(params['end_date']):
        errors.append("Invalid date format for end_date. Use YYYY-MM-DD")
    
    if 'start_date' in params and 'end_date' in params:
        start = datetime.strptime(params['start_date'], '%Y-%m-%d')
        end = datetime.strptime(params['end_date'], '%Y-%m-%d')
        if start > end:
            errors.append("start_date must be before or equal to end_date")
    
    if 'asset' not in params:
        errors.append("Missing required parameter: asset")
    
    # Validate interval
    valid_intervals = ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d']
    if 'interval' in params and params['interval'] not in valid_intervals:
        errors.append(f"Invalid interval. Must be one of: {', '.join(valid_intervals)}")
    
    # Check strategy-specific parameters if a strategy is specified
    if 'strategy' in params:
        try:
            strategies = get_available_strategies()
            strategy_found = False
            for _, (name, strategy_class) in strategies.items():
                if name == params['strategy']:
                    strategy_found = True
                    # Get strategy parameters
                    strategy_params = strategy_class.get_parameters()
                    for param_name, (default_value, description) in strategy_params.items():
                        if param_name not in params:
                            if default_value is None:
                                errors.append(f"Missing required strategy parameter: {param_name} - {description}")
                        else:
                            try:
                                # Try to convert the value to the correct type
                                params[param_name] = type(default_value)(params[param_name])
                            except (ValueError, TypeError):
                                errors.append(f"Invalid type for parameter {param_name}. Expected {type(default_value).__name__}")
                    break
            if not strategy_found:
                errors.append(f"Unknown strategy: {params['strategy']}")
        except Exception as e:
            errors.append(f"Error validating strategy parameters: {str(e)}")
    
    return errors

def run_executable(exec_name: str, params: Dict[str, Any]) -> None:
    """Run an executable with parameters."""
    # Validate parameters
    errors = validate_parameters(exec_name, params)
    if errors:
        print("Parameter validation errors:")
        for error in errors:
            print(f"- {error}")
        return
    
    # Fetch data if needed
    cache_dir = "ohlc_cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f"{cache_dir}/{params['asset']}_{params['interval']}_ohlc.csv"
    
    if not os.path.exists(cache_file):
        print(f"\nFetching data for {params['asset']} at {params['interval']} interval...")
        fetch_data(params['asset'], params['interval'])
    
    # Load data
    data = pd.read_csv(cache_file, index_col='time_period_start', parse_dates=True)
    
    # Filter data by date range
    data = data[params['start_date']:params['end_date']]
    
    # Create timeframe data structure
    timeframe_data = {
        'primary': {
            'interval': params['interval'],
            'data': data
        }
    }
    
    # Run the appropriate executable
    if exec_name == 'pnl':
        from pnl import create_interactive_chart
        strategies = get_available_strategies()
        strategy_found = False
        for _, (name, strategy_class) in strategies.items():
            if name == params['strategy']:
                strategy_found = True
                # Add default initial_equity and fee_pct if not provided
                if 'initial_equity' not in params:
                    params['initial_equity'] = 10000
                if 'fee_pct' not in params:
                    params['fee_pct'] = 0.04
                create_interactive_chart(
                    timeframe_data=timeframe_data,
                    strategy_class=strategy_class,
                    strategy_params=params,
                    last_n_candles_analyze=None,
                    last_n_candles_display=None
                )
                break
        if not strategy_found:
            print(f"Error: Strategy {params['strategy']} not found")
    
    elif exec_name == 'heatmap':
        from heatmap import create_heatmap
        strategies = get_available_strategies()
        strategy_found = False
        for _, (name, strategy_class) in strategies.items():
            if name == params['strategy']:
                strategy_found = True
                param_ranges = strategy_class.get_parameter_ranges()
                create_heatmap(
                    timeframe_data=timeframe_data,
                    strategy_class=strategy_class,
                    param_ranges=param_ranges,
                    initial_equity=params.get('initial_equity', 10000),
                    fee_pct=params.get('fee_pct', 0.04),
                    last_n_candles_analyze=None,
                    last_n_candles_display=None,
                    interval=params['interval'],
                    asset=params['asset'],
                    strategy_name=params['strategy'],
                    start_date=params['start_date'],
                    end_date=params['end_date']
                )
                break
        if not strategy_found:
            print(f"Error: Strategy {params['strategy']} not found")
    
    elif exec_name == 'chart_analysis':
        from chart_analysis import create_interactive_chart
        strategies = get_available_strategies()
        strategy_found = False
        for _, (name, strategy_class) in strategies.items():
            if name == params['strategy']:
                strategy_found = True
                # Add default initial_equity if not provided
                if 'initial_equity' not in params:
                    params['initial_equity'] = 10000
                create_interactive_chart(
                    timeframe_data=timeframe_data,
                    strategy_class=strategy_class,
                    strategy_params=params,
                    last_n_candles_analyze=None,
                    last_n_candles_display=None
                )
                break
        if not strategy_found:
            print(f"Error: Strategy {params['strategy']} not found")
    
    elif exec_name == 'fetcher':
        print(f"Data fetched and saved to {cache_file}")
    
    elif exec_name == 'automator':
        from automator import run_heatmap_for_pairs
        run_heatmap_for_pairs()
    
    else:
        print(f"Unknown executable: {exec_name}")

def print_schema() -> None:
    """Print the parameter schema in a readable format."""
    schema = get_schema()
    print("\nParameter Schema:")
    print("================")
    
    # Print executables first
    for exec_name, exec_info in {k: v for k, v in schema.items() if k != 'strategies'}.items():
        print(f"\n{exec_name.upper()}")
        print("-" * len(exec_name))
        print(f"Description: {exec_info['description']}")
        print("\nParameters:")
        
        for param_name, param_info in exec_info['parameters'].items():
            required = "Required" if param_info.get('required', False) else "Optional"
            default = f", Default: {param_info['default']}" if param_info.get('default') is not None else ""
            annotation = f", Type: {param_info['annotation']}" if param_info.get('annotation') else ""
            description = f" - {param_info['description']}" if 'description' in param_info else ""
            print(f"- {param_name} ({required}{annotation}{default}){description}")
    
    # Print strategies
    print("\nAVAILABLE STRATEGIES")
    print("===================")
    for strategy_name, strategy_info in schema['strategies'].items():
        print(f"\n{strategy_name}")
        print("-" * len(strategy_name))
        print(f"Description: {strategy_info['description']}")
        print("\nParameters:")
        
        for param_name, param_info in strategy_info['parameters'].items():
            required = "Required" if param_info.get('required', False) else "Optional"
            default = f", Default: {param_info['default']}" if param_info.get('default') is not None else ""
            description = f" - {param_info['description']}" if 'description' in param_info else ""
            print(f"- {param_name} ({required}{default}){description}")

def test_fusion_range_filter():
    print("\n=== Starting Fusion Range Filter Test ===")
    print("="*50)
    
    # Use local cached data instead of downloading
    print("\nLoading cached data...")
    df = pd.read_csv('ohlc_cache/BTCUSDT_4h_ohlc.csv')
    
    # Convert timestamp to datetime index
    df['time_period_start'] = pd.to_datetime(df['time_period_start'])
    df.set_index('time_period_start', inplace=True)
    
    # Filter to a specific date range
    start_date = '2023-01-01'
    end_date = '2024-01-01'
    df = df[start_date:end_date]
    
    print(f"\nData loaded:")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Data shape: {df.shape}")
    
    # Initialize strategy with debug mode and fusion range filter enabled
    strategy = LiveKAMASSLStrategy(
        debug_mode=True,
        use_fusion_for_long=True,
        atr_length=9,
        hma_mode='VWMA',
        hma_length=50,
        atr_scaling_factor=1.4,
        trade_direction='long'  # Test only long trades since fusion filter is for longs
    )
    
    print("\nStrategy initialized with parameters:")
    print("- Debug mode: True")
    print("- Use fusion for long: True")
    print("- ATR length: 9")
    print("- HMA mode: VWMA")
    print("- HMA length: 50")
    print("- ATR scaling factor: 1.4")
    print("- Trade direction: long")
    
    # Prepare data and calculate indicators
    print("\nPreparing data and calculating indicators...")
    prepared_data = strategy.prepare_signals(df)
    
    # Create figure to visualize the signals
    print("\nCreating visualization...")
    fig = strategy.create_figure(prepared_data)
    
    # Analyze fusion range filter conditions
    print("\nAnalyzing Fusion Range Filter conditions:")
    fusion_ma, fusion_atr, fusion_cond = strategy.fusion_range_filter.calculate(prepared_data)
    
    print(f"\nFusion Range Filter Statistics:")
    print(f"MA range: {fusion_ma.min():.2f} to {fusion_ma.max():.2f}")
    print(f"ATR range: {fusion_atr.min():.2f} to {fusion_atr.max():.2f}")
    print(f"True conditions: {fusion_cond.sum()} out of {len(fusion_cond)} ({fusion_cond.mean()*100:.2f}%)")
    
    # Check if we have any long entry signals
    long_entries = prepared_data['long_entry'].sum()
    print(f"\nLong entry signals: {long_entries}")
    
    if long_entries == 0:
        print("\nWARNING: No long entry signals found. This might indicate an issue with the Fusion Range Filter.")
        print("\nAnalyzing potential entry conditions:")
        for i in range(1, len(prepared_data)):
            current_candle = prepared_data.iloc[i]
            
            # Check trend condition
            trend_condition = current_candle['exit_kama'] > current_candle['kama2']
            
            # Check KAMA delta condition
            kama_delta_condition = current_candle['entry_kama_delta'] > current_candle['entry_kama_delta_limit']
            
            # Check BB condition
            bb_condition = current_candle['price_close'] > current_candle['bb_basis']
            
            # Check fusion condition
            fusion_condition = fusion_cond.iloc[i]
            
            # If all conditions except fusion are met
            if trend_condition and kama_delta_condition and bb_condition:
                print(f"\nPotential entry at {prepared_data.index[i]}:")
                print(f"Trend condition (exit_kama > kama2): {trend_condition}")
                print(f"KAMA delta condition: {kama_delta_condition}")
                print(f"BB condition: {bb_condition}")
                print(f"Fusion condition: {fusion_condition}")
                print(f"Exit KAMA: {current_candle['exit_kama']:.2f}")
                print(f"KAMA2: {current_candle['kama2']:.2f}")
                print(f"Entry KAMA Delta: {current_candle['entry_kama_delta']:.2f}")
                print(f"Entry KAMA Delta Limit: {current_candle['entry_kama_delta_limit']:.2f}")
                print(f"Close price: {current_candle['price_close']:.2f}")
                print(f"BB basis: {current_candle['bb_basis']:.2f}")
                print(f"Fusion MA: {fusion_ma.iloc[i]:.2f}")
                print(f"Fusion ATR: {fusion_atr.iloc[i]:.2f}")
    else:
        print("\nAnalyzing long entry signals:")
        entry_indices = prepared_data[prepared_data['long_entry']].index
        for idx in entry_indices:
            i = prepared_data.index.get_loc(idx)
            print(f"\nEntry at {idx}:")
            print(f"Fusion condition: {fusion_cond.iloc[i]}")
            print(f"MA value: {fusion_ma.iloc[i]:.2f}")
            print(f"ATR value: {fusion_atr.iloc[i]:.2f}")
            print(f"Close price: {prepared_data['price_close'].iloc[i]:.2f}")
            print(f"BB basis: {prepared_data['bb_basis'].iloc[i]:.2f}")
            print(f"Exit KAMA: {prepared_data['exit_kama'].iloc[i]:.2f}")
            print(f"KAMA2: {prepared_data['kama2'].iloc[i]:.2f}")
            print(f"Entry KAMA Delta: {prepared_data['entry_kama_delta'].iloc[i]:.2f}")
            print(f"Entry KAMA Delta Limit: {prepared_data['entry_kama_delta_limit'].iloc[i]:.2f}")
    
    print("\nTest completed!")
    print("="*50)

def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) < 2 or sys.argv[1] in ['-h', '--help']:
        print("\nUsage:")
        print("  python test.py <executable> [parameters]")
        print("  python test.py --schema")
        print("\nExecutables:")
        print("  automator, heatmap, chart_analysis, pnl, fetcher")
        print("\nExample:")
        print('  python test.py pnl --start_date="2024-01-01" --end_date="2024-03-01" --asset="BTCUSDT" --strategy="LiveKAMASSLStrategy"')
        return
    
    if sys.argv[1] == '--schema':
        print_schema()
        return
    
    exec_name = sys.argv[1]
    
    # Parse parameters from command line
    params = {}
    for arg in sys.argv[2:]:
        if arg.startswith('--'):
            key_value = arg[2:].split('=', 1)
            if len(key_value) == 2:
                key, value = key_value
                # Handle boolean values
                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
                # Handle numeric values
                elif value.replace('.', '').isdigit():
                    value = float(value) if '.' in value else int(value)
                params[key] = value
    
    run_executable(exec_name, params)

if __name__ == "__main__":
    main()

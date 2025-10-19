import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os

def load_tv_trades():
    """Load TradingView trades from CSV with precise timestamp analysis"""
    df = pd.read_csv('classes/strategies/tv-export-SOLUSDT-4h.csv', delimiter='\t')
    
    trades = []
    trade_numbers = df['Trade #'].unique()
    
    print(f"📊 Loading TradingView trades from {len(trade_numbers)} trade numbers...")
    
    for trade_num in trade_numbers:
        trade_data = df[df['Trade #'] == trade_num]
        entry_row = trade_data[trade_data['Type'].str.contains('Entry', na=False)]
        exit_row = trade_data[trade_data['Type'].str.contains('Exit', na=False)]
        
        if len(entry_row) > 0 and len(exit_row) > 0:
            entry = entry_row.iloc[0]
            exit = exit_row.iloc[0]
            
            # Parse datetime with precise format
            entry_time = pd.to_datetime(entry['Date/Time'])
            exit_time = pd.to_datetime(exit['Date/Time'])
            
            # Determine direction
            signal_lower = entry['Signal'].lower()
            direction = 'LONG' if 'long' in signal_lower else 'SHORT'
            
            trade_data = {
                'trade_num': trade_num,
                'direction': direction,
                'entry_time': entry_time,
                'exit_time': exit_time,
                'entry_price': float(entry['Price USDT']),
                'exit_price': float(exit['Price USDT']),
                'pnl': float(exit['P&L USDT']),
                'source': 'TradingView'
            }
            trades.append(trade_data)
    
    # Sort by entry time
    trades.sort(key=lambda x: x['entry_time'])
    
    print(f"✅ Loaded {len(trades)} TradingView trades")
    print(f"   Date range: {trades[0]['entry_time']} to {trades[-1]['exit_time']}")
    
    return trades

def get_python_trades_actual():
    """
    Load actual Python strategy trades using the same method as chart_analysis.py
    This ensures we're comparing apples to apples with TradingView
    """
    print("\n📊 Loading ACTUAL Python strategy trades (same method as chart_analysis.py)...")
    
    try:
        # Import required modules
        from strategy_utils import fetch_data, get_available_strategies
        from classes.trade_analyzer import TradeAnalyzer
        import pandas as pd
        import os
        
        # Strategy parameters matching the TradingView test
        strategy_params = {
            'entry_filter': 0.7,
            'exit_filter': 1.0,
            'initial_equity': 1000,
            'fee_pct': 0.04,
            'use_divergence_exit_long': True,
            'use_hidden_divergence_exit_long': False,
            'use_divergence_exit_short': True,
            'use_hidden_divergence_exit_short': False,
            'divergence_order': 5,
            'rsi_length': 14,
            'calculate_divergences': False,  # Set to False to match user's command
            'start_date': '2020-09-24',
            'end_date': '2025-05-25',
            'asset': 'SOLUSDT',
            'interval': '4h',
            'use_fusion_for_long': True,
            'atr_length': 9,
            'hma_mode': 'VWMA',
            'hma_length': 50,
            'atr_scaling_factor': 1.4,
            'debug_mode': False,  # Set to False for clean output
            'trade_direction': 'both',
            'show_pivot_points': True,
            'position_sizing_method': 'percentage',
            'position_size_value': 0.25,
            'max_position_size': 10000
        }
        
        print(f"📋 Strategy Parameters:")
        print(f"   Date range: {strategy_params['start_date']} to {strategy_params['end_date']}")
        print(f"   Asset: {strategy_params['asset']}")
        print(f"   Initial equity: ${strategy_params['initial_equity']}")
        print(f"   Trade direction: {strategy_params['trade_direction']}")
        print(f"   Calculate divergences: {strategy_params['calculate_divergences']}")
        
        # Use the EXACT same data loading method as test.py
        print("\n📈 Loading market data using test.py method...")
        
        # Fetch data if needed - same as test.py
        cache_dir = "ohlc_cache"
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = f"{cache_dir}/{strategy_params['asset']}_{strategy_params['interval']}_ohlc.csv"
        
        if not os.path.exists(cache_file):
            print(f"Fetching data for {strategy_params['asset']} at {strategy_params['interval']} interval...")
            fetch_data(strategy_params['asset'], strategy_params['interval'])
        
        # Load data - EXACTLY like test.py
        data = pd.read_csv(cache_file, index_col='time_period_start', parse_dates=True)
        
        # Filter data by date range - EXACTLY like test.py
        data = data[strategy_params['start_date']:strategy_params['end_date']]
        
        # Create timeframe data structure - EXACTLY like test.py
        timeframe_data = {
            'primary': {
                'interval': strategy_params['interval'],
                'data': data
            }
        }
        
        print(f"   Data loaded: {len(data)} candles from {data.index[0]} to {data.index[-1]}")
        
        # Get strategy class - using the same method as chart_analysis.py
        strategies = get_available_strategies()
        strategy_class = None
        for _, (name, strat_class) in strategies.items():
            if name == 'LiveKAMASSLStrategy':
                strategy_class = strat_class
                break
        
        if strategy_class is None:
            raise ValueError("LiveKAMASSLStrategy not found in available strategies")
        
        print(f"✅ Strategy class found: {strategy_class.__name__}")
        
        # Initialize strategy with timeframe data - EXACTLY like chart_analysis.py
        print("\n🔧 Initializing strategy...")
        strategy = strategy_class(**strategy_params)
        strategy.timeframe_data = timeframe_data  # Add timeframe data to strategy
        
        # Create analyzer with strategy and strategy_params - EXACTLY like chart_analysis.py
        analyzer = TradeAnalyzer(strategy, strategy_params)
        
        # Set date range for divergence detector if it exists
        if hasattr(strategy, 'divergence_detector'):
            strategy.divergence_detector.set_date_range(
                strategy_params['start_date'], 
                strategy_params['end_date']
            )
        
        # Analyze data and get trades - EXACTLY like chart_analysis.py
        print("\n⚙️ Analyzing trades...")
        trades, display_data = analyzer.analyze_data(data, None, None)
        
        print(f"📊 Analysis complete!")
        print(f"   Total trades found: {len(trades)}")
        print(f"   Display data shape: {display_data.shape}")
        
        # Convert trades to the format expected by our analysis
        formatted_trades = []
        
        for i, trade in enumerate(trades):
            # Safely unpack trade tuple
            entry_time = trade[0]
            exit_time = trade[1] 
            entry_price = trade[2]
            exit_price = trade[3]
            net_profit = trade[4]
            gross_profit = trade[5]
            fees = trade[6]
            trade_type = trade[7]
            exit_reason = trade[8] if len(trade) > 8 else ''
            
            # Only include completed trades
            if net_profit is not None:
                formatted_trades.append({
                    'trade_id': i + 1,
                    'direction': trade_type,
                    'entry_time': entry_time,
                    'exit_time': exit_time,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'net_profit': net_profit,
                    'gross_profit': gross_profit,
                    'fees': fees,
                    'exit_reason': exit_reason
                })
        
        print(f"\n📈 Formatted trades summary:")
        print(f"   Completed trades: {len(formatted_trades)}")
        
        if len(formatted_trades) > 0:
            long_trades = len([t for t in formatted_trades if t['direction'] == 'LONG'])
            short_trades = len([t for t in formatted_trades if t['direction'] == 'SHORT'])
            print(f"   Long trades: {long_trades}")
            print(f"   Short trades: {short_trades}")
            
            # Show first few trades for verification
            print(f"\n📋 First 3 trades:")
            for i, trade in enumerate(formatted_trades[:3]):
                print(f"   {i+1}. {trade['direction']} {trade['entry_time']} -> {trade['exit_time']} | "
                      f"${trade['entry_price']:.2f} -> ${trade['exit_price']:.2f} | "
                      f"P/L: ${trade['net_profit']:.2f}")
        
        return formatted_trades
        
    except Exception as e:
        print(f"❌ Error loading actual Python trades: {e}")
        import traceback
        traceback.print_exc()
        return []

def generate_representative_python_trades(trade_count):
    """Generate representative Python trades based on actual strategy results"""
    print(f"🔄 Generating {trade_count} representative Python trades...")
    
    trades = []
    
    # Use the actual date range from TradingView data
    start_date = pd.to_datetime('2020-09-24')
    end_date = pd.to_datetime('2025-01-25')
    
    # Generate realistic trade distribution
    np.random.seed(42)  # For reproducible results
    
    for i in range(trade_count):
        # Generate entry time with realistic distribution
        # More trades in volatile periods
        days_range = (end_date - start_date).days
        entry_day = np.random.randint(0, days_range)
        entry_time = start_date + timedelta(days=entry_day)
        
        # Align to 4-hour boundaries
        entry_time = entry_time.replace(
            hour=(entry_time.hour // 4) * 4, 
            minute=0, 
            second=0, 
            microsecond=0
        )
        
        # Random trade duration (realistic for crypto: 1-21 days)
        duration_hours = np.random.choice([4, 8, 12, 24, 48, 72, 96, 168, 336, 504], 
                                        p=[0.1, 0.15, 0.15, 0.2, 0.15, 0.1, 0.07, 0.05, 0.02, 0.01])
        exit_time = entry_time + timedelta(hours=int(duration_hours))
        
        # Random direction with slight bias based on market trends
        direction = np.random.choice(['LONG', 'SHORT'], p=[0.45, 0.55])  # Slightly more shorts
        
        # Generate realistic prices based on SOL historical range
        base_price = 50 + 150 * np.random.random()  # $50-$200 range
        price_change = np.random.uniform(-0.3, 0.4)  # -30% to +40% range
        
        entry_price = base_price
        if direction == 'LONG':
            exit_price = entry_price * (1 + price_change)
            pnl = (exit_price - entry_price) * 1000  # Position size simulation
        else:
            exit_price = entry_price * (1 - price_change) 
            pnl = (entry_price - exit_price) * 1000
        
        trade_data = {
            'trade_num': i + 1,
            'direction': direction,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'source': 'Python'
        }
        trades.append(trade_data)
    
    # Sort by entry time
    trades.sort(key=lambda x: x['entry_time'])
    
    print(f"✅ Generated {len(trades)} representative Python trades")
    print(f"   Date range: {trades[0]['entry_time']} to {trades[-1]['exit_time']}")
    
    return trades

def get_simulated_python_trades():
    """Fallback simulated Python trades based on known strategy performance"""
    print("🔄 Using simulated Python trades based on known performance (203 trades)")
    return generate_representative_python_trades(203)

def analyze_timestamp_sync(tv_trades, python_trades):
    """Analyze timestamp synchronization between strategies"""
    print(f"\n🔍 TIMESTAMP SYNCHRONIZATION ANALYSIS")
    print(f"=" * 50)
    
    # Check for exact matches
    exact_matches = 0
    close_matches = 0  # Within 4 hours (1 candle)
    
    tv_entry_times = [trade['entry_time'] for trade in tv_trades]
    python_entry_times = [trade['entry_time'] for trade in python_trades]
    
    print(f"TradingView trades: {len(tv_trades)}")
    print(f"Python trades: {len(python_trades)}")
    
    # Find time overlaps and differences
    all_times = sorted(set(tv_entry_times + python_entry_times))
    
    print(f"\nTiming Analysis:")
    print(f"Total unique trade times: {len(all_times)}")
    
    # Analyze 4-hour window matches (since this is 4h strategy)
    tolerance = timedelta(hours=4)
    
    for tv_time in tv_entry_times:
        # Find closest Python trade time
        closest_python_time = min(python_entry_times, 
                                key=lambda x: abs((x - tv_time).total_seconds()),
                                default=None)
        
        if closest_python_time:
            time_diff = abs((closest_python_time - tv_time).total_seconds())
            
            if time_diff == 0:
                exact_matches += 1
            elif time_diff <= tolerance.total_seconds():
                close_matches += 1
    
    print(f"Exact timestamp matches: {exact_matches}")
    print(f"Close matches (±4h): {close_matches}")
    print(f"Potential async trades: {len(tv_trades) - exact_matches - close_matches}")
    
    return {
        'exact_matches': exact_matches,
        'close_matches': close_matches,
        'total_tv': len(tv_trades),
        'total_python': len(python_trades),
        'sync_ratio': (exact_matches + close_matches) / max(len(tv_trades), 1)
    }

def find_precise_deviations(tv_trades, python_trades):
    """Find precise timing deviations with 4-hour tolerance"""
    print(f"\n🎯 FINDING PRECISE TRADE DEVIATIONS")
    print(f"=" * 40)
    
    tolerance = timedelta(hours=4)  # 4-hour strategy tolerance
    deviations = []
    sync_periods = []
    
    # Create time grid based on 4-hour intervals
    if tv_trades and python_trades:
        start_time = min(tv_trades[0]['entry_time'], python_trades[0]['entry_time'])
        end_time = max(tv_trades[-1]['exit_time'], python_trades[-1]['exit_time'])
    else:
        return deviations, sync_periods
    
    # Create 4-hour time grid
    current_time = start_time.replace(hour=(start_time.hour // 4) * 4, minute=0, second=0, microsecond=0)
    time_grid = []
    
    while current_time <= end_time:
        time_grid.append(current_time)
        current_time += timedelta(hours=4)
    
    print(f"Created time grid with {len(time_grid)} 4-hour periods")
    
    # Analyze each time period
    for i in range(len(time_grid) - 1):
        period_start = time_grid[i]
        period_end = time_grid[i + 1]
        
        # Count trades in this period for each strategy
        tv_trades_in_period = []
        python_trades_in_period = []
        
        for trade in tv_trades:
            if period_start <= trade['entry_time'] < period_end:
                tv_trades_in_period.append(trade)
        
        for trade in python_trades:
            if period_start <= trade['entry_time'] < period_end:
                python_trades_in_period.append(trade)
        
        tv_count = len(tv_trades_in_period)
        python_count = len(python_trades_in_period)
        
        # Determine period type
        if tv_count == python_count == 0:
            # No trades in either strategy - skip
            continue
        elif tv_count == python_count and tv_count > 0:
            # Synchronized period
            sync_periods.append({
                'start_time': period_start,
                'end_time': period_end,
                'trade_count': tv_count,
                'type': 'synchronized'
            })
        else:
            # Deviation period
            deviation_type = 'missing_python' if tv_count > python_count else 'extra_python'
            deviations.append({
                'start_time': period_start,
                'end_time': period_end,
                'tv_trades': tv_count,
                'python_trades': python_count,
                'deviation_type': deviation_type,
                'magnitude': abs(tv_count - python_count),
                'tv_trade_details': tv_trades_in_period,
                'python_trade_details': python_trades_in_period
            })
    
    print(f"Found {len(deviations)} deviation periods")
    print(f"Found {len(sync_periods)} synchronized periods")
    
    return deviations, sync_periods

def load_real_price_data():
    """Load actual SOLUSDT price data from cache if available"""
    try:
        # Try to load from OHLC cache
        cache_file = 'ohlc_cache/SOLUSDT_4h.csv'
        df = pd.read_csv(cache_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        
        # Rename columns to match expected format
        df = df.rename(columns={
            'open': 'price_open',
            'high': 'price_high', 
            'low': 'price_low',
            'close': 'price_close'
        })
        
        print(f"✅ Loaded real SOLUSDT data: {len(df)} candles")
        print(f"   Date range: {df.index[0]} to {df.index[-1]}")
        
        return df.reset_index().rename(columns={'timestamp': 'time', 'price_close': 'price'})
        
    except FileNotFoundError:
        print("⚠️ OHLC cache not found, using simulated data")
        return load_simulated_price_data()

def load_simulated_price_data():
    """Fallback simulated price data"""
    start_date = pd.to_datetime('2020-09-24')
    end_date = pd.to_datetime('2025-01-25')
    
    times = pd.date_range(start=start_date, end=end_date, freq='4h')
    
    np.random.seed(42)
    base_price = 50
    prices = []
    
    for i in range(len(times)):
        change = np.random.uniform(-0.05, 0.05)
        base_price *= (1 + change)
        base_price = max(0.1, min(300, base_price))
        prices.append(base_price)
    
    return pd.DataFrame({'time': times, 'price': prices})

def calculate_performance_difference(tv_trades, python_trades):
    """Calculate the performance difference between TradingView and Python strategies"""
    print("\n💰 PERFORMANCE COMPARISON ANALYSIS")
    print("=" * 50)
    
    # Calculate TradingView performance
    tv_total_pnl = sum(trade['pnl'] for trade in tv_trades)
    tv_winning_trades = len([trade for trade in tv_trades if trade['pnl'] > 0])
    tv_losing_trades = len([trade for trade in tv_trades if trade['pnl'] < 0])
    tv_win_rate = tv_winning_trades / len(tv_trades) * 100 if tv_trades else 0
    
    # TradingView starts with equivalent equity (we need to estimate this)
    # Based on the trade sizes, it looks like TradingView also started with $1000
    tv_initial_equity = 1000
    tv_final_equity = tv_initial_equity + tv_total_pnl
    tv_return_pct = (tv_total_pnl / tv_initial_equity) * 100
    
    # Calculate Python performance
    python_total_pnl = sum(trade['net_profit'] for trade in python_trades)
    python_winning_trades = len([trade for trade in python_trades if trade['net_profit'] > 0])
    python_losing_trades = len([trade for trade in python_trades if trade['net_profit'] < 0])
    python_win_rate = python_winning_trades / len(python_trades) * 100 if python_trades else 0
    
    python_initial_equity = 1000  # Known from strategy params
    python_final_equity = python_initial_equity + python_total_pnl
    python_return_pct = (python_total_pnl / python_initial_equity) * 100
    
    # Calculate differences
    trade_count_diff = len(python_trades) - len(tv_trades)
    pnl_diff = python_total_pnl - tv_total_pnl
    return_diff = python_return_pct - tv_return_pct
    win_rate_diff = python_win_rate - tv_win_rate
    
    print(f"📈 TRADINGVIEW PERFORMANCE:")
    print(f"   Trades: {len(tv_trades)}")
    print(f"   Total P&L: ${tv_total_pnl:,.2f}")
    print(f"   Return: {tv_return_pct:,.1f}%")
    print(f"   Win Rate: {tv_win_rate:.1f}% ({tv_winning_trades}/{len(tv_trades)})")
    print(f"   Final Equity: ${tv_final_equity:,.2f}")
    
    print(f"\n🐍 PYTHON PERFORMANCE:")
    print(f"   Trades: {len(python_trades)}")
    print(f"   Total P&L: ${python_total_pnl:,.2f}")
    print(f"   Return: {python_return_pct:,.1f}%")
    print(f"   Win Rate: {python_win_rate:.1f}% ({python_winning_trades}/{len(python_trades)})")
    print(f"   Final Equity: ${python_final_equity:,.2f}")
    
    print(f"\n🔄 PERFORMANCE DIFFERENCES:")
    print(f"   Trade Count Difference: {trade_count_diff:+d} trades")
    print(f"   P&L Difference: ${pnl_diff:+,.2f}")
    print(f"   Return Difference: {return_diff:+.1f} percentage points")
    print(f"   Win Rate Difference: {win_rate_diff:+.1f} percentage points")
    
    # Calculate relative performance impact
    if tv_return_pct != 0:
        relative_performance_impact = (return_diff / abs(tv_return_pct)) * 100
        print(f"   Relative Performance Impact: {relative_performance_impact:+.1f}%")
    
    # Show top winning and losing trades comparison
    print(f"\n📊 TRADE ANALYSIS:")
    
    # TradingView best/worst trades
    tv_best_trade = max(tv_trades, key=lambda x: x['pnl'])
    tv_worst_trade = min(tv_trades, key=lambda x: x['pnl'])
    
    print(f"   TradingView Best Trade: ${tv_best_trade['pnl']:,.2f} ({tv_best_trade['direction']})")
    print(f"   TradingView Worst Trade: ${tv_worst_trade['pnl']:,.2f} ({tv_worst_trade['direction']})")
    
    # Python best/worst trades
    python_best_trade = max(python_trades, key=lambda x: x['net_profit'])
    python_worst_trade = min(python_trades, key=lambda x: x['net_profit'])
    
    print(f"   Python Best Trade: ${python_best_trade['net_profit']:,.2f} ({python_best_trade['direction']})")
    print(f"   Python Worst Trade: ${python_worst_trade['net_profit']:,.2f} ({python_worst_trade['direction']})")
    
    return {
        'tv_return_pct': tv_return_pct,
        'python_return_pct': python_return_pct,
        'return_difference': return_diff,
        'pnl_difference': pnl_diff,
        'trade_count_diff': trade_count_diff,
        'win_rate_diff': win_rate_diff,
        'relative_impact': relative_performance_impact if tv_return_pct != 0 else 0
    }

def detailed_trade_comparison(tv_trades, python_trades):
    """Create detailed trade-by-trade comparison statistics"""
    print("\n📊 DETAILED TRADE-BY-TRADE COMPARISON")
    print("=" * 80)
    
    # Create time-based mapping for easier comparison
    tv_by_time = {}
    for trade in tv_trades:
        key = f"{trade['entry_time']}_{trade['direction']}"
        tv_by_time[key] = trade
    
    python_by_time = {}
    for trade in python_trades:
        key = f"{trade['entry_time']}_{trade['direction']}"
        python_by_time[key] = trade
    
    # Find matched trades
    matched_trades = []
    tv_only_trades = []
    python_only_trades = []
    
    # Process TradingView trades
    for key, tv_trade in tv_by_time.items():
        if key in python_by_time:
            python_trade = python_by_time[key]
            matched_trades.append((tv_trade, python_trade))
        else:
            tv_only_trades.append(tv_trade)
    
    # Find Python-only trades
    for key, python_trade in python_by_time.items():
        if key not in tv_by_time:
            python_only_trades.append(python_trade)
    
    print(f"📈 TRADE MATCHING SUMMARY:")
    print(f"   Matched Trades: {len(matched_trades)}")
    print(f"   TradingView Only: {len(tv_only_trades)}")
    print(f"   Python Only: {len(python_only_trades)}")
    print(f"   Total TradingView: {len(tv_trades)}")
    print(f"   Total Python: {len(python_trades)}")
    
    # Analyze matched trades
    if matched_trades:
        print(f"\n🔄 MATCHED TRADES ANALYSIS ({len(matched_trades)} trades):")
        print("=" * 80)
        
        total_tv_pnl = 0
        total_python_pnl = 0
        total_pnl_diff = 0
        entry_price_diffs = []
        exit_price_diffs = []
        duration_diffs = []
        
        print(f"{'#':<3} | {'Type':<5} | {'Entry Time':<16} | {'TV P&L':<12} | {'PY P&L':<12} | {'Diff':<12} | {'TV Entry':<8} | {'PY Entry':<8} | {'TV Exit':<8} | {'PY Exit':<8}")
        print("-" * 120)
        
        for i, (tv_trade, py_trade) in enumerate(matched_trades[:20], 1):  # Show first 20
            tv_pnl = tv_trade['pnl']
            py_pnl = py_trade['net_profit']
            pnl_diff = py_pnl - tv_pnl
            
            total_tv_pnl += tv_pnl
            total_python_pnl += py_pnl
            total_pnl_diff += pnl_diff
            
            # Calculate price differences
            entry_diff = py_trade['entry_price'] - tv_trade['entry_price']
            exit_diff = py_trade['exit_price'] - tv_trade['exit_price']
            entry_price_diffs.append(abs(entry_diff))
            exit_price_diffs.append(abs(exit_diff))
            
            # Calculate duration difference
            tv_duration = (tv_trade['exit_time'] - tv_trade['entry_time']).total_seconds() / 3600
            py_duration = (py_trade['exit_time'] - py_trade['entry_time']).total_seconds() / 3600
            duration_diff = py_duration - tv_duration
            duration_diffs.append(abs(duration_diff))
            
            print(f"{i:<3} | {tv_trade['direction']:<5} | {tv_trade['entry_time'].strftime('%Y-%m-%d %H:%M'):<16} | ${tv_pnl:<11,.0f} | ${py_pnl:<11,.0f} | ${pnl_diff:<11,.0f} | ${tv_trade['entry_price']:<7.2f} | ${py_trade['entry_price']:<7.2f} | ${tv_trade['exit_price']:<7.2f} | ${py_trade['exit_price']:<7.2f}")
        
        if len(matched_trades) > 20:
            print(f"... and {len(matched_trades) - 20} more matched trades")
        
        print("-" * 120)
        print(f"TOTAL| {'MATCHED':<5} | {'SUMMARY':<16} | ${total_tv_pnl:<11,.0f} | ${total_python_pnl:<11,.0f} | ${total_pnl_diff:<11,.0f} |")
        
        # Statistical analysis
        import numpy as np
        pnl_diffs = [py_trade['net_profit'] - tv_trade['pnl'] for tv_trade, py_trade in matched_trades]
        
        print(f"\n📊 MATCHED TRADES STATISTICS:")
        print(f"   Average P&L Difference: ${np.mean(pnl_diffs):,.2f}")
        print(f"   Median P&L Difference: ${np.median(pnl_diffs):,.2f}")
        print(f"   Std Dev P&L Difference: ${np.std(pnl_diffs):,.2f}")
        print(f"   Max P&L Difference: ${np.max(pnl_diffs):,.2f}")
        print(f"   Min P&L Difference: ${np.min(pnl_diffs):,.2f}")
        
        print(f"\n💰 PRICE ACCURACY ANALYSIS:")
        print(f"   Average Entry Price Difference: ${np.mean(entry_price_diffs):,.4f}")
        print(f"   Average Exit Price Difference: ${np.mean(exit_price_diffs):,.4f}")
        print(f"   Average Duration Difference: {np.mean(duration_diffs):.2f} hours")
    
    # Analyze TradingView-only trades
    if tv_only_trades:
        print(f"\n📈 TRADINGVIEW-ONLY TRADES ({len(tv_only_trades)} trades):")
        print("=" * 80)
        
        tv_only_pnl = sum(trade['pnl'] for trade in tv_only_trades)
        print(f"{'#':<3} | {'Type':<5} | {'Entry Time':<16} | {'Exit Time':<16} | {'P&L':<12} | {'Entry $':<8} | {'Exit $':<8}")
        print("-" * 90)
        
        for i, trade in enumerate(tv_only_trades, 1):
            print(f"{i:<3} | {trade['direction']:<5} | {trade['entry_time'].strftime('%Y-%m-%d %H:%M'):<16} | {trade['exit_time'].strftime('%Y-%m-%d %H:%M'):<16} | ${trade['pnl']:<11,.0f} | ${trade['entry_price']:<7.2f} | ${trade['exit_price']:<7.2f}")
        
        print("-" * 90)
        print(f"TOTAL TV-ONLY P&L: ${tv_only_pnl:,.2f}")
        
        # Analyze patterns
        long_only = [t for t in tv_only_trades if t['direction'] == 'LONG']
        short_only = [t for t in tv_only_trades if t['direction'] == 'SHORT']
        winning_only = [t for t in tv_only_trades if t['pnl'] > 0]
        losing_only = [t for t in tv_only_trades if t['pnl'] < 0]
        
        print(f"\n🔍 TV-ONLY TRADE PATTERNS:")
        print(f"   Long Trades: {len(long_only)} (${sum(t['pnl'] for t in long_only):,.2f})")
        print(f"   Short Trades: {len(short_only)} (${sum(t['pnl'] for t in short_only):,.2f})")
        print(f"   Winning Trades: {len(winning_only)} (${sum(t['pnl'] for t in winning_only):,.2f})")
        print(f"   Losing Trades: {len(losing_only)} (${sum(t['pnl'] for t in losing_only):,.2f})")
    
    # Analyze Python-only trades
    if python_only_trades:
        print(f"\n🐍 PYTHON-ONLY TRADES ({len(python_only_trades)} trades):")
        print("=" * 80)
        
        python_only_pnl = sum(trade['net_profit'] for trade in python_only_trades)
        print(f"{'#':<3} | {'Type':<5} | {'Entry Time':<16} | {'Exit Time':<16} | {'P&L':<12} | {'Entry $':<8} | {'Exit $':<8}")
        print("-" * 90)
        
        for i, trade in enumerate(python_only_trades, 1):
            print(f"{i:<3} | {trade['direction']:<5} | {trade['entry_time'].strftime('%Y-%m-%d %H:%M'):<16} | {trade['exit_time'].strftime('%Y-%m-%d %H:%M'):<16} | ${trade['net_profit']:<11,.0f} | ${trade['entry_price']:<7.2f} | ${trade['exit_price']:<7.2f}")
        
        print("-" * 90)
        print(f"TOTAL PYTHON-ONLY P&L: ${python_only_pnl:,.2f}")
        
        # Analyze patterns
        long_only = [t for t in python_only_trades if t['direction'] == 'LONG']
        short_only = [t for t in python_only_trades if t['direction'] == 'SHORT']
        winning_only = [t for t in python_only_trades if t['net_profit'] > 0]
        losing_only = [t for t in python_only_trades if t['net_profit'] < 0]
        
        print(f"\n🔍 PYTHON-ONLY TRADE PATTERNS:")
        print(f"   Long Trades: {len(long_only)} (${sum(t['net_profit'] for t in long_only):,.2f})")
        print(f"   Short Trades: {len(short_only)} (${sum(t['net_profit'] for t in short_only):,.2f})")
        print(f"   Winning Trades: {len(winning_only)} (${sum(t['net_profit'] for t in winning_only):,.2f})")
        print(f"   Losing Trades: {len(losing_only)} (${sum(t['net_profit'] for t in losing_only):,.2f})")
    
    # Summary impact analysis
    print(f"\n💥 IMPACT ANALYSIS:")
    print("=" * 50)
    
    missed_tv_pnl = sum(trade['pnl'] for trade in tv_only_trades) if tv_only_trades else 0
    extra_python_pnl = sum(trade['net_profit'] for trade in python_only_trades) if python_only_trades else 0
    matched_pnl_diff = sum(py_trade['net_profit'] - tv_trade['pnl'] for tv_trade, py_trade in matched_trades) if matched_trades else 0
    
    print(f"   Missed TradingView P&L: ${missed_tv_pnl:,.2f}")
    print(f"   Extra Python P&L: ${extra_python_pnl:,.2f}")
    print(f"   Matched Trades P&L Diff: ${matched_pnl_diff:,.2f}")
    print(f"   Total Difference: ${extra_python_pnl - missed_tv_pnl + matched_pnl_diff:,.2f}")
    
    # Performance attribution
    print(f"\n🎯 PERFORMANCE ATTRIBUTION:")
    print(f"   Missing TradingView trades cost: ${-missed_tv_pnl:,.2f}")
    print(f"   Extra Python trades gained: ${extra_python_pnl:,.2f}")
    print(f"   Execution differences on matched trades: ${matched_pnl_diff:,.2f}")
    
    return {
        'matched_count': len(matched_trades),
        'tv_only_count': len(tv_only_trades),
        'python_only_count': len(python_only_trades),
        'missed_tv_pnl': missed_tv_pnl,
        'extra_python_pnl': extra_python_pnl,
        'matched_pnl_diff': matched_pnl_diff,
        'total_impact': extra_python_pnl - missed_tv_pnl + matched_pnl_diff
    }

def generate_deviation_percentage_report(tv_trades, python_trades):
    """Generate a comprehensive report of all trade deviations with percentage differences"""
    print("\n📊 COMPREHENSIVE DEVIATION REPORT WITH PERCENTAGES")
    print("=" * 100)
    
    # Create time-based mapping for easier comparison
    tv_by_time = {}
    for trade in tv_trades:
        key = f"{trade['entry_time']}_{trade['direction']}"
        tv_by_time[key] = trade
    
    python_by_time = {}
    for trade in python_trades:
        key = f"{trade['entry_time']}_{trade['direction']}"
        python_by_time[key] = trade
    
    all_deviations = []
    
    # 1. Process matched trades (execution differences)
    for key, tv_trade in tv_by_time.items():
        if key in python_by_time:
            python_trade = python_by_time[key]
            tv_pnl = tv_trade['pnl']
            py_pnl = python_trade['net_profit']
            pnl_diff = py_pnl - tv_pnl
            
            # Calculate percentage difference
            if tv_pnl != 0:
                pct_diff = (pnl_diff / abs(tv_pnl)) * 100
            else:
                pct_diff = 0 if pnl_diff == 0 else (float('inf') if pnl_diff > 0 else float('-inf'))
            
            all_deviations.append({
                'type': 'EXECUTION_DIFF',
                'entry_time': tv_trade['entry_time'],
                'exit_time': tv_trade['exit_time'],
                'direction': tv_trade['direction'],
                'tv_pnl': tv_pnl,
                'py_pnl': py_pnl,
                'pnl_diff': pnl_diff,
                'pct_diff': pct_diff,
                'tv_entry': tv_trade['entry_price'],
                'py_entry': python_trade['entry_price'],
                'tv_exit': tv_trade['exit_price'],
                'py_exit': python_trade['exit_price'],
                'description': f"Execution difference: TV ${tv_pnl:,.0f} vs PY ${py_pnl:,.0f}"
            })
    
    # 2. Process TradingView-only trades (missed opportunities)
    for key, tv_trade in tv_by_time.items():
        if key not in python_by_time:
            tv_pnl = tv_trade['pnl']
            # For missed trades, Python gets 0, so the percentage loss is 100% of the TradingView gain
            pct_diff = -100.0 if tv_pnl > 0 else 100.0  # If TV made profit, Python lost 100% of that opportunity
            
            all_deviations.append({
                'type': 'MISSED_TV_TRADE',
                'entry_time': tv_trade['entry_time'],
                'exit_time': tv_trade['exit_time'],
                'direction': tv_trade['direction'],
                'tv_pnl': tv_pnl,
                'py_pnl': 0,
                'pnl_diff': -tv_pnl,  # Python loses the entire TradingView profit
                'pct_diff': pct_diff,
                'tv_entry': tv_trade['entry_price'],
                'py_entry': None,
                'tv_exit': tv_trade['exit_price'],
                'py_exit': None,
                'description': f"Missed TV trade: Lost ${tv_pnl:,.0f} opportunity"
            })
    
    # 3. Process Python-only trades (extra trades)
    for key, python_trade in python_by_time.items():
        if key not in tv_by_time:
            py_pnl = python_trade['net_profit']
            # For extra trades, this is pure gain/loss for Python vs TradingView's 0
            pct_diff = float('inf') if py_pnl > 0 else float('-inf')  # Infinite percentage since TV had 0
            
            all_deviations.append({
                'type': 'EXTRA_PY_TRADE',
                'entry_time': python_trade['entry_time'],
                'exit_time': python_trade['exit_time'],
                'direction': python_trade['direction'],
                'tv_pnl': 0,
                'py_pnl': py_pnl,
                'pnl_diff': py_pnl,
                'pct_diff': pct_diff,
                'tv_entry': None,
                'py_entry': python_trade['entry_price'],
                'tv_exit': None,
                'py_exit': python_trade['exit_price'],
                'description': f"Extra PY trade: {'Gained' if py_pnl > 0 else 'Lost'} ${py_pnl:,.0f}"
            })
    
    # Sort by absolute percentage difference (finite values first, then by absolute difference)
    def sort_key(item):
        pct = item['pct_diff']
        if pct == float('inf') or pct == float('-inf'):
            return (1, abs(item['pnl_diff']))  # Sort infinite values by absolute dollar difference
        else:
            return (0, abs(pct))  # Sort finite values by absolute percentage
    
    all_deviations.sort(key=sort_key, reverse=True)
    
    # Print detailed report
    print(f"\n📋 ALL TRADE DEVIATIONS SORTED BY IMPACT ({len(all_deviations)} total)")
    print("=" * 100)
    print(f"{'#':<3} | {'Type':<15} | {'Dir':<5} | {'Entry Time':<16} | {'TV P&L':<12} | {'PY P&L':<12} | {'Diff':<12} | {'%Diff':<10} | {'Description':<30}")
    print("-" * 130)
    
    for i, deviation in enumerate(all_deviations[:50], 1):  # Show top 50
        pct_str = f"{deviation['pct_diff']:+.1f}%" if abs(deviation['pct_diff']) != float('inf') else "±∞%"
        print(f"{i:<3} | {deviation['type']:<15} | {deviation['direction']:<5} | {deviation['entry_time'].strftime('%Y-%m-%d %H:%M'):<16} | ${deviation['tv_pnl']:<11,.0f} | ${deviation['py_pnl']:<11,.0f} | ${deviation['pnl_diff']:<11,.0f} | {pct_str:<10} | {deviation['description'][:30]:<30}")
    
    if len(all_deviations) > 50:
        print(f"... and {len(all_deviations) - 50} more deviations")
    
    # Summary statistics
    execution_diffs = [d for d in all_deviations if d['type'] == 'EXECUTION_DIFF']
    missed_trades = [d for d in all_deviations if d['type'] == 'MISSED_TV_TRADE']
    extra_trades = [d for d in all_deviations if d['type'] == 'EXTRA_PY_TRADE']
    
    print("\n📊 DEVIATION SUMMARY BY TYPE:")
    print("=" * 50)
    print(f"Execution Differences: {len(execution_diffs)} trades")
    print(f"   Total Impact: ${sum(d['pnl_diff'] for d in execution_diffs):,.2f}")
    print(f"   Average Impact: ${sum(d['pnl_diff'] for d in execution_diffs)/len(execution_diffs):,.2f}" if execution_diffs else "   Average Impact: $0.00")
    
    print(f"\nMissed TradingView Trades: {len(missed_trades)} trades")
    print(f"   Total Impact: ${sum(d['pnl_diff'] for d in missed_trades):,.2f}")
    print(f"   Average Impact: ${sum(d['pnl_diff'] for d in missed_trades)/len(missed_trades):,.2f}" if missed_trades else "   Average Impact: $0.00")
    
    print(f"\nExtra Python Trades: {len(extra_trades)} trades")
    print(f"   Total Impact: ${sum(d['pnl_diff'] for d in extra_trades):,.2f}")
    print(f"   Average Impact: ${sum(d['pnl_diff'] for d in extra_trades)/len(extra_trades):,.2f}" if extra_trades else "   Average Impact: $0.00")
    
    # Top 10 biggest negative impacts
    negative_impacts = [d for d in all_deviations if d['pnl_diff'] < 0]
    negative_impacts.sort(key=lambda x: x['pnl_diff'])
    
    print(f"\n🔴 TOP 10 BIGGEST NEGATIVE IMPACTS:")
    print("=" * 80)
    print(f"{'#':<3} | {'Type':<15} | {'Date':<12} | {'Impact':<15} | {'%Diff':<10} | {'Description':<25}")
    print("-" * 80)
    
    for i, impact in enumerate(negative_impacts[:10], 1):
        pct_str = f"{impact['pct_diff']:+.1f}%" if abs(impact['pct_diff']) != float('inf') else "±∞%"
        print(f"{i:<3} | {impact['type']:<15} | {impact['entry_time'].strftime('%Y-%m-%d'):<12} | ${impact['pnl_diff']:<14,.0f} | {pct_str:<10} | {impact['description'][:25]:<25}")
    
    # Top 10 biggest positive impacts
    positive_impacts = [d for d in all_deviations if d['pnl_diff'] > 0]
    positive_impacts.sort(key=lambda x: x['pnl_diff'], reverse=True)
    
    print(f"\n🟢 TOP 10 BIGGEST POSITIVE IMPACTS:")
    print("=" * 80)
    print(f"{'#':<3} | {'Type':<15} | {'Date':<12} | {'Impact':<15} | {'%Diff':<10} | {'Description':<25}")
    print("-" * 80)
    
    for i, impact in enumerate(positive_impacts[:10], 1):
        pct_str = f"{impact['pct_diff']:+.1f}%" if abs(impact['pct_diff']) != float('inf') else "+∞%"
        print(f"{i:<3} | {impact['type']:<15} | {impact['entry_time'].strftime('%Y-%m-%d'):<12} | ${impact['pnl_diff']:<14,.0f} | {pct_str:<10} | {impact['description'][:25]:<25}")
    
    # Percentage distribution analysis
    finite_execution_diffs = [d for d in execution_diffs if abs(d['pct_diff']) != float('inf')]
    if finite_execution_diffs:
        import numpy as np
        pct_diffs = [d['pct_diff'] for d in finite_execution_diffs]
        
        print(f"\n📈 EXECUTION DIFFERENCE PERCENTAGE STATISTICS:")
        print("=" * 50)
        print(f"   Mean Percentage Difference: {np.mean(pct_diffs):+.2f}%")
        print(f"   Median Percentage Difference: {np.median(pct_diffs):+.2f}%")
        print(f"   Std Dev Percentage Difference: {np.std(pct_diffs):.2f}%")
        print(f"   Max Percentage Difference: {np.max(pct_diffs):+.2f}%")
        print(f"   Min Percentage Difference: {np.min(pct_diffs):+.2f}%")
        
        # Distribution buckets
        buckets = {
            'Large Negative (< -50%)': len([p for p in pct_diffs if p < -50]),
            'Moderate Negative (-50% to -10%)': len([p for p in pct_diffs if -50 <= p < -10]),
            'Small Negative (-10% to 0%)': len([p for p in pct_diffs if -10 <= p < 0]),
            'Small Positive (0% to 10%)': len([p for p in pct_diffs if 0 <= p < 10]),
            'Moderate Positive (10% to 50%)': len([p for p in pct_diffs if 10 <= p < 50]),
            'Large Positive (> 50%)': len([p for p in pct_diffs if p >= 50])
        }
        
        print(f"\n📊 EXECUTION DIFFERENCE DISTRIBUTION:")
        for bucket, count in buckets.items():
            percentage = (count / len(pct_diffs)) * 100 if pct_diffs else 0
            print(f"   {bucket}: {count} trades ({percentage:.1f}%)")
    
    return all_deviations

def main():
    """Main function to run the precise timing analysis"""
    print("=" * 80)
    print("🎯 PRECISE TIMESTAMP ANALYSIS - TradingView vs Python Strategy")
    print("=" * 80)
    print("🔧 Using ACTUAL chart_analysis.py method for Python trades")
    print("=" * 80)
    
    # Load TradingView trades
    tv_trades = load_tv_trades()
    if not tv_trades:
        print("❌ Failed to load TradingView trades")
        return
    
    # Load ACTUAL Python trades using the same method as chart_analysis.py
    python_trades = get_python_trades_actual()
    if not python_trades:
        print("❌ Failed to load Python trades")
        return
    
    # Analyze synchronization
    sync_analysis = analyze_timestamp_sync(tv_trades, python_trades)
    
    # Find precise deviations
    deviations, sync_periods = find_precise_deviations(tv_trades, python_trades)
    
    # Create the chart with the loaded data
    print("\n📊 Creating precise timing deviation chart with actual data...")
    create_precise_deviation_chart_with_data(tv_trades, python_trades, sync_analysis, deviations, sync_periods)
    
    # Calculate performance difference
    performance_diff = calculate_performance_difference(tv_trades, python_trades)
    
    # Detailed trade comparison
    detailed_comparison = detailed_trade_comparison(tv_trades, python_trades)
    
    # Generate deviation percentage report
    deviation_report = generate_deviation_percentage_report(tv_trades, python_trades)
    
    print(f"\n🎯 FINAL ANALYSIS COMPLETE!")
    print(f"📈 Chart saved as: precise_timing_analysis.html")
    print(f"🔍 Sync Ratio: {sync_analysis['sync_ratio']:.1%}")
    print(f"📊 Deviation Periods: {len(deviations)}")
    print(f"🎯 Synchronized Periods: {len(sync_periods)}")
    print(f"💰 Performance Differences: {performance_diff}")
    print(f"🔄 Detailed Trade Comparison: {detailed_comparison}")
    print(f"🎯 Deviation Percentage Report: {deviation_report}")

def create_precise_deviation_chart_with_data(tv_trades, python_trades, sync_analysis, deviations, sync_periods):
    """Create a precise timing deviation chart with pre-loaded data"""
    print("\n📊 Creating precise timing deviation chart with provided data...")
    
    # Load price data
    price_data = load_real_price_data()
    
    print(f"\n📊 CHART DATA SUMMARY:")
    print(f"TradingView trades: {len(tv_trades)}")
    print(f"Python trades: {len(python_trades)}")
    print(f"Deviation periods: {len(deviations)}")
    print(f"Synchronized periods: {len(sync_periods)}")
    print(f"Sync ratio: {sync_analysis['sync_ratio']:.1%}")
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.6, 0.25, 0.15],
        subplot_titles=[
            'SOLUSDT Price with Precise Trade Timing Analysis',
            'Trade Synchronization Comparison', 
            'Deviation Details'
        ]
    )
    
    # Add price chart
    fig.add_trace(
        go.Scatter(
            x=price_data['time'],
            y=price_data['price'],
            mode='lines',
            name='💰 SOLUSDT Price',
            line=dict(color='blue', width=1),
            opacity=0.7
        ),
        row=1, col=1
    )
    
    # Get chart bounds
    chart_min = price_data['price'].min() * 0.95
    chart_max = price_data['price'].max() * 1.05
    
    # Add synchronized periods (gray background)
    print(f"\n🎨 Adding {len(sync_periods)} synchronized backgrounds...")
    for sync_period in sync_periods:
        fig.add_shape(
            type="rect",
            x0=sync_period['start_time'],
            x1=sync_period['end_time'], 
            y0=chart_min,
            y1=chart_max,
            fillcolor="rgba(128, 128, 128, 0.2)",  # Light gray
            line=dict(width=0),
            layer="below",
            row=1, col=1
        )
    
    # Add deviation periods (colored backgrounds)
    print(f"🎨 Adding {len(deviations)} deviation backgrounds...")
    for deviation in deviations:
        if deviation['deviation_type'] == 'missing_python':
            fill_color = "rgba(255, 0, 0, 0.3)"  # Red
        else:
            fill_color = "rgba(0, 255, 0, 0.3)"  # Green
        
        fig.add_shape(
            type="rect",
            x0=deviation['start_time'],
            x1=deviation['end_time'],
            y0=chart_min,
            y1=chart_max,
            fillcolor=fill_color,
            line=dict(width=0),
            layer="below",
            row=1, col=1
        )
        
        # Add annotation for significant deviations
        if deviation['magnitude'] > 1:
            mid_time = deviation['start_time'] + (deviation['end_time'] - deviation['start_time']) / 2
            mid_price = (chart_min + chart_max) / 2
            
            fig.add_annotation(
                x=mid_time,
                y=mid_price,
                text=f"Δ{deviation['magnitude']}",
                showarrow=False,
                font=dict(color="white", size=10),
                bgcolor="rgba(0,0,0,0.5)",
                row=1, col=1
            )
    
    # Add trade markers
    tv_entry_times = [trade['entry_time'] for trade in tv_trades]
    tv_entry_prices = []
    
    for trade in tv_trades:
        # Find corresponding price
        closest_idx = price_data['time'].sub(trade['entry_time']).abs().idxmin()
        tv_entry_prices.append(price_data.loc[closest_idx, 'price'])
    
    fig.add_trace(
        go.Scatter(
            x=tv_entry_times,
            y=tv_entry_prices,
            mode='markers',
            name=f'📈 TradingView ({len(tv_trades)})',
            marker=dict(symbol='circle', size=6, color='red'),
        ),
        row=1, col=1
    )
    
    python_entry_times = [trade['entry_time'] for trade in python_trades]
    python_entry_prices = []
    
    for trade in python_trades:
        closest_idx = price_data['time'].sub(trade['entry_time']).abs().idxmin()
        python_entry_prices.append(price_data.loc[closest_idx, 'price'])
    
    fig.add_trace(
        go.Scatter(
            x=python_entry_times,
            y=python_entry_prices,
            mode='markers',
            name=f'🐍 Python ({len(python_trades)})',
            marker=dict(symbol='circle', size=6, color='blue'),
        ),
        row=1, col=1
    )
    
    # Add legend indicators
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None],
            mode='markers',
            name='⚪ Synchronized Periods',
            marker=dict(color='gray', size=10, symbol='square'),
            showlegend=True
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None],
            mode='markers', 
            name='🔴 Missing Python Trades',
            marker=dict(color='red', size=10, symbol='square'),
            showlegend=True
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None],
            mode='markers',
            name='🟢 Extra Python Trades', 
            marker=dict(color='green', size=10, symbol='square'),
            showlegend=True
        ),
        row=1, col=1
    )
    
    # Update layout
    fig.update_layout(
        title={
            'text': f'Precise Trade Timing Analysis - Sync Ratio: {sync_analysis["sync_ratio"]:.1%}<br><sub>Gray: Synchronized | Red: Missing Python | Green: Extra Python</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        height=1000,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left", 
            x=1.02,
            font=dict(size=12),
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1
        ),
        margin=dict(l=50, r=150, t=100, b=50)
    )
    
    # Update axis labels
    fig.update_yaxes(title_text="SOLUSDT Price", row=1, col=1)
    
    # Save chart
    output_file = "precise_timing_analysis.html"
    fig.write_html(output_file)
    print(f"\n✅ Chart saved as: {output_file}")
    
    return fig

if __name__ == "__main__":
    main() 
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def load_tv_trades():
    """Load TradingView trades with detailed parsing"""
    df = pd.read_csv('classes/strategies/tv-export-SOLUSDT-4h.csv', delimiter='\t')
    
    trades = []
    trade_numbers = df['Trade #'].unique()
    
    for trade_num in trade_numbers:
        trade_data = df[df['Trade #'] == trade_num]
        entry_row = trade_data[trade_data['Type'].str.contains('Entry', na=False)]
        exit_row = trade_data[trade_data['Type'].str.contains('Exit', na=False)]
        
        if len(entry_row) > 0 and len(exit_row) > 0:
            entry = entry_row.iloc[0]
            exit = exit_row.iloc[0]
            
            # Parse datetime
            entry_time = pd.to_datetime(entry['Date/Time'])
            exit_time = pd.to_datetime(exit['Date/Time'])
            
            trades.append({
                'trade_num': trade_num,
                'direction': 'LONG' if 'long' in entry['Signal'].lower() else 'SHORT',
                'entry_time': entry_time,
                'exit_time': exit_time,
                'entry_price': entry['Price USDT'],
                'exit_price': exit['Price USDT'],
                'pnl_usdt': exit['P&L USDT'],
                'quantity': entry['Quantity'],
                'signal': entry['Signal'],
                'duration_hours': (exit_time - entry_time).total_seconds() / 3600
            })
    
    return pd.DataFrame(trades)

def get_python_trades_sync():
    """Get Python trades from synchronised run (2020-09-24 start)"""
    # Based on the synchronised test output
    python_trades = [
        {'trade_num': 1, 'direction': 'LONG', 'entry_time': '2020-09-26 20:00', 'exit_time': '2020-09-30 12:00', 'entry_price': 3.14, 'exit_price': 2.92, 'pnl': -70.84},
        {'trade_num': 2, 'direction': 'SHORT', 'entry_time': '2020-11-04 20:00', 'exit_time': '2020-11-05 00:00', 'entry_price': 1.30, 'exit_price': 1.39, 'pnl': -66.41},
        {'trade_num': 3, 'direction': 'LONG', 'entry_time': '2020-11-06 20:00', 'exit_time': '2020-11-12 00:00', 'entry_price': 1.69, 'exit_price': 2.03, 'pnl': 174.46},
        {'trade_num': 4, 'direction': 'SHORT', 'entry_time': '2020-11-12 00:00', 'exit_time': '2020-11-16 12:00', 'entry_price': 2.03, 'exit_price': 2.19, 'pnl': -79.62},
        {'trade_num': 5, 'direction': 'LONG', 'entry_time': '2020-11-16 12:00', 'exit_time': '2020-11-19 12:00', 'entry_price': 2.19, 'exit_price': 2.06, 'pnl': -54.76},
        {'trade_num': 6, 'direction': 'LONG', 'entry_time': '2020-11-22 00:00', 'exit_time': '2020-11-26 12:00', 'entry_price': 2.45, 'exit_price': 1.83, 'pnl': -229.51},
        {'trade_num': 7, 'direction': 'LONG', 'entry_time': '2020-12-02 20:00', 'exit_time': '2020-12-05 00:00', 'entry_price': 2.12, 'exit_price': 1.94, 'pnl': -55.31},
        {'trade_num': 8, 'direction': 'LONG', 'entry_time': '2020-12-16 00:00', 'exit_time': '2020-12-20 04:00', 'entry_price': 1.75, 'exit_price': 1.67, 'pnl': -27.48},
        {'trade_num': 9, 'direction': 'SHORT', 'entry_time': '2020-12-21 00:00', 'exit_time': '2020-12-27 20:00', 'entry_price': 1.56, 'exit_price': 1.40, 'pnl': 60.66},
        {'trade_num': 10, 'direction': 'SHORT', 'entry_time': '2020-12-28 04:00', 'exit_time': '2020-12-28 16:00', 'entry_price': 1.30, 'exit_price': 1.38, 'pnl': -40.71},
        {'trade_num': 11, 'direction': 'LONG', 'entry_time': '2020-12-30 00:00', 'exit_time': '2021-01-15 20:00', 'entry_price': 1.65, 'exit_price': 3.10, 'pnl': 534.96},
        {'trade_num': 12, 'direction': 'LONG', 'entry_time': '2021-01-17 12:00', 'exit_time': '2021-01-21 16:00', 'entry_price': 3.39, 'exit_price': 3.48, 'pnl': 30.00},
        {'trade_num': 13, 'direction': 'LONG', 'entry_time': '2021-01-26 00:00', 'exit_time': '2021-01-29 12:00', 'entry_price': 3.89, 'exit_price': 3.67, 'pnl': -69.33},
        {'trade_num': 14, 'direction': 'LONG', 'entry_time': '2021-01-30 20:00', 'exit_time': '2021-02-15 08:00', 'entry_price': 4.04, 'exit_price': 8.11, 'pnl': 1112.88},
        {'trade_num': 15, 'direction': 'LONG', 'entry_time': '2021-02-15 12:00', 'exit_time': '2021-02-17 08:00', 'entry_price': 8.29, 'exit_price': 7.71, 'pnl': -156.09},
    ]
    
    # Convert to DataFrame with proper datetime
    df = pd.DataFrame(python_trades)
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df['exit_time'] = pd.to_datetime(df['exit_time'])
    df['duration_hours'] = (df['exit_time'] - df['entry_time']).dt.total_seconds() / 3600
    
    return df

def compare_timing_differences():
    """Compare timing differences between TradingView and Python"""
    print("🕒 DETAILED TIMING DIFFERENCES ANALYSIS")
    print("=" * 80)
    
    tv_trades = load_tv_trades()
    py_trades = get_python_trades_sync()
    
    # Filter to same time period for comparison
    start_date = '2020-09-24'
    end_date = '2021-03-01'  # First few months for detailed analysis
    
    tv_filtered = tv_trades[
        (tv_trades['entry_time'] >= start_date) & 
        (tv_trades['entry_time'] <= end_date)
    ].copy().reset_index(drop=True)
    
    py_filtered = py_trades[
        (py_trades['entry_time'] >= start_date) & 
        (py_trades['entry_time'] <= end_date)
    ].copy().reset_index(drop=True)
    
    print(f"📊 COMPARISON PERIOD: {start_date} to {end_date}")
    print(f"TradingView trades: {len(tv_filtered)}")
    print(f"Python trades: {len(py_filtered)}")
    print(f"Trade count difference: {len(tv_filtered) - len(py_filtered)}")
    
    return tv_filtered, py_filtered

def analyze_entry_timing_differences(tv_trades, py_trades):
    """Analyze entry timing differences in detail"""
    print(f"\n🎯 ENTRY TIMING DIFFERENCES ANALYSIS")
    print("=" * 80)
    
    # Find timing differences for similar trades
    timing_diffs = []
    
    print(f"{'Trade':<5} {'TV Entry':<16} {'PY Entry':<16} {'Diff (h)':<8} {'TV Dir':<5} {'PY Dir':<5} {'Match':<5}")
    print("-" * 75)
    
    # Compare first 15 trades for detailed analysis
    max_compare = min(len(tv_trades), len(py_trades), 15)
    
    for i in range(max_compare):
        tv_trade = tv_trades.iloc[i]
        
        # Find closest Python trade by time
        closest_py_idx = None
        min_time_diff = float('inf')
        
        for j in range(len(py_trades)):
            py_trade = py_trades.iloc[j]
            time_diff = abs((tv_trade['entry_time'] - py_trade['entry_time']).total_seconds() / 3600)
            
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                closest_py_idx = j
        
        if closest_py_idx is not None:
            py_trade = py_trades.iloc[closest_py_idx]
            
            # Calculate timing difference
            time_diff_hours = (py_trade['entry_time'] - tv_trade['entry_time']).total_seconds() / 3600
            
            # Check if directions match
            direction_match = tv_trade['direction'] == py_trade['direction']
            
            timing_diffs.append({
                'trade_num': i + 1,
                'tv_entry': tv_trade['entry_time'],
                'py_entry': py_trade['entry_time'],
                'time_diff_hours': time_diff_hours,
                'tv_direction': tv_trade['direction'],
                'py_direction': py_trade['direction'],
                'direction_match': direction_match,
                'tv_price': tv_trade['entry_price'],
                'py_price': py_trade['entry_price']
            })
            
            print(f"{i+1:<5} {tv_trade['entry_time'].strftime('%Y-%m-%d %H:%M'):<16} "
                  f"{py_trade['entry_time'].strftime('%Y-%m-%d %H:%M'):<16} "
                  f"{time_diff_hours:+7.1f}h {tv_trade['direction']:<5} {py_trade['direction']:<5} "
                  f"{'✅' if direction_match else '❌':<5}")
    
    return timing_diffs

def analyze_timing_statistics(timing_diffs):
    """Analyze timing difference statistics"""
    print(f"\n📈 TIMING STATISTICS")
    print("=" * 50)
    
    df = pd.DataFrame(timing_diffs)
    
    if len(df) == 0:
        print("No timing differences to analyze")
        return
    
    # Overall statistics
    print(f"📊 OVERALL TIMING STATISTICS:")
    print(f"Total compared trades: {len(df)}")
    print(f"Direction matches: {df['direction_match'].sum()}/{len(df)} ({df['direction_match'].mean()*100:.1f}%)")
    print(f"Average timing difference: {df['time_diff_hours'].mean():+.1f} hours")
    print(f"Median timing difference: {df['time_diff_hours'].median():+.1f} hours")
    print(f"Std dev timing difference: {df['time_diff_hours'].std():.1f} hours")
    print(f"Min timing difference: {df['time_diff_hours'].min():+.1f} hours")
    print(f"Max timing difference: {df['time_diff_hours'].max():+.1f} hours")
    
    # Timing difference distribution
    print(f"\n📊 TIMING DIFFERENCE DISTRIBUTION:")
    bins = [-float('inf'), -24, -8, -4, -1, 1, 4, 8, 24, float('inf')]
    labels = ['<-24h', '-24h to -8h', '-8h to -4h', '-4h to -1h', '-1h to +1h', '+1h to +4h', '+4h to +8h', '+8h to +24h', '>+24h']
    
    df['time_diff_category'] = pd.cut(df['time_diff_hours'], bins=bins, labels=labels)
    time_dist = df['time_diff_category'].value_counts().sort_index()
    
    for category, count in time_dist.items():
        percentage = count / len(df) * 100
        print(f"{category:<12}: {count:2d} trades ({percentage:4.1f}%)")
    
    # Direction match analysis
    print(f"\n📊 DIRECTION MATCH ANALYSIS:")
    match_by_timing = df.groupby(df['time_diff_hours'].abs() <= 4)['direction_match'].agg(['count', 'sum', 'mean'])
    
    for timing_close, stats in match_by_timing.iterrows():
        timing_desc = "Within ±4h" if timing_close else "Beyond ±4h"
        print(f"{timing_desc:<12}: {stats['sum']}/{stats['count']} matches ({stats['mean']*100:.1f}%)")

def identify_timing_patterns(tv_trades, py_trades, timing_diffs):
    """Identify patterns in timing differences"""
    print(f"\n🔍 TIMING PATTERN ANALYSIS")
    print("=" * 50)
    
    df = pd.DataFrame(timing_diffs)
    
    if len(df) == 0:
        print("No data for pattern analysis")
        return
    
    # Pattern 1: Early vs Late entries
    early_entries = df[df['time_diff_hours'] < -4]  # Python earlier than TV
    late_entries = df[df['time_diff_hours'] > 4]    # Python later than TV
    close_entries = df[abs(df['time_diff_hours']) <= 4]  # Within 4 hours
    
    print(f"🎯 ENTRY TIMING PATTERNS:")
    print(f"Python earlier (>4h): {len(early_entries)} trades ({len(early_entries)/len(df)*100:.1f}%)")
    print(f"Close timing (±4h): {len(close_entries)} trades ({len(close_entries)/len(df)*100:.1f}%)")
    print(f"Python later (>4h): {len(late_entries)} trades ({len(late_entries)/len(df)*100:.1f}%)")
    
    # Pattern 2: Direction-specific timing
    print(f"\n🎯 DIRECTION-SPECIFIC TIMING:")
    for direction in ['LONG', 'SHORT']:
        dir_data = df[df['tv_direction'] == direction]
        if len(dir_data) > 0:
            avg_diff = dir_data['time_diff_hours'].mean()
            print(f"{direction} trades: {len(dir_data)} trades, avg timing diff: {avg_diff:+.1f}h")
    
    # Pattern 3: Consecutive differences
    consecutive_early = 0
    consecutive_late = 0
    current_streak = 0
    current_type = None
    
    for diff in df['time_diff_hours']:
        if diff < -4:  # Early
            if current_type == 'early':
                current_streak += 1
            else:
                consecutive_early = max(consecutive_early, current_streak)
                current_streak = 1
                current_type = 'early'
        elif diff > 4:  # Late
            if current_type == 'late':
                current_streak += 1
            else:
                consecutive_late = max(consecutive_late, current_streak)
                current_streak = 1
                current_type = 'late'
        else:  # Close
            consecutive_early = max(consecutive_early, current_streak)
            consecutive_late = max(consecutive_late, current_streak)
            current_streak = 0
            current_type = None
    
    print(f"\n🎯 CONSECUTIVE PATTERNS:")
    print(f"Max consecutive early entries: {consecutive_early}")
    print(f"Max consecutive late entries: {consecutive_late}")

def analyze_missing_trades(tv_trades, py_trades):
    """Analyze trades that appear in one system but not the other"""
    print(f"\n❌ MISSING TRADES ANALYSIS")
    print("=" * 50)
    
    # Find TV trades that don't have Python equivalents
    tv_times = set(tv_trades['entry_time'].dt.floor('4H'))  # Round to 4h intervals
    py_times = set(py_trades['entry_time'].dt.floor('4H'))
    
    missing_in_python = tv_times - py_times
    missing_in_tv = py_times - tv_times
    
    print(f"📊 MISSING TRADE SUMMARY:")
    print(f"TradingView trades missing in Python: {len(missing_in_python)}")
    print(f"Python trades missing in TradingView: {len(missing_in_tv)}")
    
    if missing_in_python:
        print(f"\n📋 TV TRADES MISSING IN PYTHON:")
        for time in sorted(missing_in_python)[:10]:  # Show first 10
            tv_trade = tv_trades[tv_trades['entry_time'].dt.floor('4H') == time].iloc[0]
            print(f"  {time.strftime('%Y-%m-%d %H:%M')} - {tv_trade['direction']} @ ${tv_trade['entry_price']:.2f}")
    
    if missing_in_tv:
        print(f"\n📋 PYTHON TRADES MISSING IN TV:")
        for time in sorted(missing_in_tv)[:10]:  # Show first 10
            py_trade = py_trades[py_trades['entry_time'].dt.floor('4H') == time].iloc[0]
            print(f"  {time.strftime('%Y-%m-%d %H:%M')} - {py_trade['direction']} @ ${py_trade['entry_price']:.2f}")

def main():
    """Run complete timing analysis"""
    # Load and compare data
    tv_trades, py_trades = compare_timing_differences()
    
    # Detailed timing analysis
    timing_diffs = analyze_entry_timing_differences(tv_trades, py_trades)
    
    # Statistical analysis
    analyze_timing_statistics(timing_diffs)
    
    # Pattern identification
    identify_timing_patterns(tv_trades, py_trades, timing_diffs)
    
    # Missing trades analysis
    analyze_missing_trades(tv_trades, py_trades)
    
    print(f"\n🎯 SUMMARY:")
    print(f"The main timing differences appear to be:")
    print(f"1. Python strategy triggers signals at slightly different times")
    print(f"2. Some trades are missing in one system vs the other") 
    print(f"3. Direction accuracy is generally good when timing is close")
    print(f"4. Systematic timing shifts suggest indicator calculation differences")

if __name__ == "__main__":
    main() 
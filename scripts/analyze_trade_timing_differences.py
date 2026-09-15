import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def load_tv_trades():
    """Load and analyze TradingView trades with proper timing"""
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
                'signal': entry['Signal']
            })
    
    return pd.DataFrame(trades)

def parse_python_trades_from_output():
    """Parse Python trades from the test output"""
    # Based on the test output, extract key trades for comparison
    python_trades = [
        {'trade_num': 1, 'direction': 'LONG', 'entry_time': '2020-08-13 12:00', 'exit_time': '2020-08-16 08:00', 'entry_price': 4.08, 'exit_price': 3.20, 'pnl': -217.21},
        {'trade_num': 2, 'direction': 'LONG', 'entry_time': '2020-08-16 12:00', 'exit_time': '2020-08-17 20:00', 'entry_price': 3.44, 'exit_price': 3.13, 'pnl': -71.98},
        {'trade_num': 3, 'direction': 'LONG', 'entry_time': '2020-08-18 20:00', 'exit_time': '2020-08-19 16:00', 'entry_price': 3.36, 'exit_price': 3.01, 'pnl': -74.98},
        {'trade_num': 4, 'direction': 'LONG', 'entry_time': '2020-08-20 20:00', 'exit_time': '2020-08-21 12:00', 'entry_price': 3.33, 'exit_price': 3.07, 'pnl': -48.98},
        {'trade_num': 5, 'direction': 'LONG', 'entry_time': '2020-08-24 04:00', 'exit_time': '2020-09-03 08:00', 'entry_price': 3.25, 'exit_price': 3.90, 'pnl': 116.46},
        {'trade_num': 6, 'direction': 'SHORT', 'entry_time': '2020-09-16 08:00', 'exit_time': '2020-09-18 16:00', 'entry_price': 2.69, 'exit_price': 3.06, 'pnl': -97.95},
        # Last few trades for comparison
        {'trade_num': 201, 'direction': 'LONG', 'entry_time': '2025-01-02 16:00', 'exit_time': '2025-01-07 20:00', 'entry_price': 208.51, 'exit_price': 207.76, 'pnl': -1443.35},
        {'trade_num': 202, 'direction': 'SHORT', 'entry_time': '2025-01-07 20:00', 'exit_time': '2025-01-15 08:00', 'entry_price': 207.76, 'exit_price': 189.73, 'pnl': 28098.60},
        {'trade_num': 203, 'direction': 'LONG', 'entry_time': '2025-01-16 00:00', 'exit_time': '2025-01-25 12:00', 'entry_price': 201.90, 'exit_price': 247.28, 'pnl': 79480.88},
    ]
    
    # Convert to DataFrame with proper datetime
    df = pd.DataFrame(python_trades)
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df['exit_time'] = pd.to_datetime(df['exit_time'])
    
    return df

def compare_trade_timing():
    """Compare trade timing between TradingView and Python"""
    print("🕒 TRADE TIMING ANALYSIS")
    print("=" * 70)
    
    tv_trades = load_tv_trades()
    python_trades = parse_python_trades_from_output()
    
    print(f"📊 BASIC COMPARISON:")
    print(f"TradingView trades: {len(tv_trades)}")
    print(f"Python trades: {len(python_trades)} (shown sample)")
    print(f"TradingView date range: {tv_trades['entry_time'].min()} to {tv_trades['exit_time'].max()}")
    print(f"Python date range: {python_trades['entry_time'].min()} to {python_trades['exit_time'].max()}")
    
    # Check if there are overlapping time periods
    tv_start = tv_trades['entry_time'].min()
    tv_end = tv_trades['exit_time'].max()
    py_start = python_trades['entry_time'].min()
    py_end = python_trades['exit_time'].max()
    
    print(f"\n📅 TIME PERIOD OVERLAP ANALYSIS:")
    print(f"TradingView period: {tv_start.strftime('%Y-%m-%d')} to {tv_end.strftime('%Y-%m-%d')}")
    print(f"Python period: {py_start.strftime('%Y-%m-%d')} to {py_end.strftime('%Y-%m-%d')}")
    
    # Check for overlapping periods
    overlap_start = max(tv_start, py_start)
    overlap_end = min(tv_end, py_end)
    
    if overlap_start <= overlap_end:
        print(f"✅ OVERLAP PERIOD: {overlap_start.strftime('%Y-%m-%d')} to {overlap_end.strftime('%Y-%m-%d')}")
        
        # Filter trades in overlap period
        tv_overlap = tv_trades[
            (tv_trades['entry_time'] >= overlap_start) & 
            (tv_trades['entry_time'] <= overlap_end)
        ].copy()
        
        py_overlap = python_trades[
            (python_trades['entry_time'] >= overlap_start) & 
            (python_trades['entry_time'] <= overlap_end)
        ].copy()
        
        print(f"TradingView trades in overlap: {len(tv_overlap)}")
        print(f"Python trades in overlap: {len(py_overlap)}")
        
        return tv_overlap, py_overlap
    else:
        print(f"❌ NO OVERLAP - strategies tested on different periods!")
        return tv_trades, python_trades

def analyze_first_trades():
    """Analyze the first few trades from both systems"""
    print(f"\n🔍 FIRST TRADES COMPARISON")
    print("=" * 70)
    
    tv_trades = load_tv_trades()
    
    print(f"📈 TRADINGVIEW FIRST 10 TRADES:")
    print(f"{'#':<3} {'Dir':<5} {'Entry Time':<16} {'Exit Time':<16} {'Entry $':<8} {'Exit $':<8} {'P&L $':<10}")
    print("-" * 80)
    
    for i, trade in tv_trades.head(10).iterrows():
        print(f"{trade['trade_num']:<3} {trade['direction']:<5} "
              f"{trade['entry_time'].strftime('%Y-%m-%d %H:%M'):<16} "
              f"{trade['exit_time'].strftime('%Y-%m-%d %H:%M'):<16} "
              f"{trade['entry_price']:<8.2f} {trade['exit_price']:<8.2f} {trade['pnl_usdt']:<10.2f}")
    
    # Compare with Python trades
    python_trades = parse_python_trades_from_output()
    
    print(f"\n🐍 PYTHON FIRST 10 TRADES:")
    print(f"{'#':<3} {'Dir':<5} {'Entry Time':<16} {'Exit Time':<16} {'Entry $':<8} {'Exit $':<8} {'P&L $':<10}")
    print("-" * 80)
    
    for i, trade in python_trades.head(10).iterrows():
        print(f"{trade['trade_num']:<3} {trade['direction']:<5} "
              f"{trade['entry_time'].strftime('%Y-%m-%d %H:%M'):<16} "
              f"{trade['exit_time'].strftime('%Y-%m-%d %H:%M'):<16} "
              f"{trade['entry_price']:<8.2f} {trade['exit_price']:<8.2f} {trade['pnl']:<10.2f}")

def identify_key_differences():
    """Identify the key differences causing performance discrepancy"""
    print(f"\n🔍 KEY DIFFERENCES ANALYSIS")
    print("=" * 70)
    
    tv_trades = load_tv_trades()
    
    print(f"🎯 POTENTIAL CAUSES OF PERFORMANCE DIFFERENCES:")
    print(f"")
    print(f"1. DIFFERENT TIME PERIODS:")
    print(f"   - TradingView: {tv_trades['entry_time'].min().strftime('%Y-%m-%d')} to {tv_trades['exit_time'].max().strftime('%Y-%m-%d')}")
    print(f"   - Python: 2020-08-13 to 2025-01-25")
    print(f"   - Impact: Different market conditions, volatility")
    print(f"")
    print(f"2. TRADE EXECUTION TIMING:")
    print(f"   - TradingView may use different candle timing")
    print(f"   - Signal confirmation differences")
    print(f"   - Order execution within the 4h timeframe")
    print(f"")
    print(f"3. INDICATOR CALCULATION DIFFERENCES:")
    print(f"   - KAMA calculation precision")
    print(f"   - SSL Channel calculation")
    print(f"   - DEMA calculation methodology")
    print(f"")
    print(f"4. SIGNAL LOGIC DIFFERENCES:")
    print(f"   - Entry condition combinations")
    print(f"   - Exit trigger timing")
    print(f"   - Fusion Range Filter implementation")
    print(f"")
    print(f"5. DATA DIFFERENCES:")
    print(f"   - TradingView uses different OHLC data source")
    print(f"   - Timestamp alignment issues")
    print(f"   - Volume data differences")

def recommend_debugging_steps():
    """Recommend specific debugging steps"""
    print(f"\n🛠️ RECOMMENDED DEBUGGING STEPS")
    print("=" * 70)
    
    print(f"1. SYNCHRONIZE TIME PERIODS:")
    print(f"   - Test Python strategy on exact same date range as TradingView")
    print(f"   - Use same start/end dates")
    print(f"")
    print(f"2. COMPARE INDIVIDUAL INDICATORS:")
    print(f"   - Export KAMA values from TradingView")
    print(f"   - Compare with Python KAMA calculations")
    print(f"   - Check SSL Channel values")
    print(f"")
    print(f"3. TRADE-BY-TRADE COMPARISON:")
    print(f"   - Focus on first 10-20 trades")
    print(f"   - Compare entry/exit signals exactly")
    print(f"   - Check for off-by-one errors in timing")
    print(f"")
    print(f"4. DATA VALIDATION:")
    print(f"   - Compare OHLC data between sources")
    print(f"   - Check for gaps or missing candles")
    print(f"   - Verify timestamp alignment")
    print(f"")
    print(f"5. IMPLEMENT STEP-BY-STEP DEBUG:")
    print(f"   - Add detailed logging to Python strategy")
    print(f"   - Log every indicator value and signal")
    print(f"   - Compare with TradingView step by step")

if __name__ == "__main__":
    # Run all analyses
    compare_trade_timing()
    analyze_first_trades()
    identify_key_differences()
    recommend_debugging_steps()
    
    print(f"\n🎯 CONCLUSION:")
    print(f"The performance difference is likely due to:")
    print(f"1. Different time periods being tested")
    print(f"2. Subtle differences in indicator calculations")
    print(f"3. Trade execution timing differences")
    print(f"4. Data source variations")
    print(f"Need detailed trade-by-trade comparison to identify root cause.") 
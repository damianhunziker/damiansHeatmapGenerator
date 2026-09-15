import pandas as pd
import numpy as np
from datetime import datetime
import re

def load_tv_trades():
    """Load TradingView trades from test.csv"""
    df = pd.read_csv('test.csv', delimiter='\t')
    
    # Group by trade number to get complete trades
    trades = []
    trade_numbers = df['Trade #'].unique()
    
    for trade_num in trade_numbers:
        trade_data = df[df['Trade #'] == trade_num]
        
        # Find entry and exit rows
        entry_row = trade_data[trade_data['Type'].str.contains('Entry', na=False)]
        exit_row = trade_data[trade_data['Type'].str.contains('Exit', na=False)]
        
        if len(entry_row) > 0 and len(exit_row) > 0:
            entry = entry_row.iloc[0]
            exit = exit_row.iloc[0]
            
            # Determine trade direction
            if 'long' in entry['Signal'].lower():
                trade_type = 'LONG'
            elif 'short' in entry['Signal'].lower():
                trade_type = 'SHORT'
            else:
                trade_type = 'UNKNOWN'
            
            trades.append({
                'trade_num': trade_num,
                'type': trade_type,
                'entry_time': entry['Date/Time'],
                'exit_time': exit['Date/Time'],
                'entry_price': entry['Price USDT'],
                'exit_price': exit['Price USDT'],
                'net_pl': exit['P&L USDT'],
                'pct': float(exit['P&L %'].replace('%', '')),
                'exit_reason': exit['Signal'],
                'cumulative_pl': exit['Cumulative P&L USDT']
            })
    
    return trades

def create_python_trades():
    """Manually create Python trades from the output data"""
    python_trades = [
        {'trade_num': 1, 'type': 'LONG', 'entry_time': '2018-04-21 04:00', 'exit_time': '2018-05-07 20:00', 'entry_price': 0.30, 'exit_price': 0.33, 'net_pl': 799.44, 'pct': 8.08, 'exit_reason': 'KAMA exit signal'},
        {'trade_num': 2, 'type': 'SHORT', 'entry_time': '2018-05-23 12:00', 'exit_time': '2018-05-23 16:00', 'entry_price': 0.21, 'exit_price': 0.21, 'net_pl': 13.03, 'pct': 0.20, 'exit_reason': 'SSL crossover up exit'},
        {'trade_num': 3, 'type': 'SHORT', 'entry_time': '2018-06-01 20:00', 'exit_time': '2018-06-06 16:00', 'entry_price': 0.22, 'exit_price': 0.21, 'net_pl': 275.36, 'pct': 2.63, 'exit_reason': 'SSL crossover up exit'},
        {'trade_num': 4, 'type': 'SHORT', 'entry_time': '2018-06-17 08:00', 'exit_time': '2018-06-24 12:00', 'entry_price': 0.16, 'exit_price': 0.12, 'net_pl': 2738.45, 'pct': 24.79, 'exit_reason': 'SSL crossover up exit'},
        {'trade_num': 5, 'type': 'LONG', 'entry_time': '2018-07-01 20:00', 'exit_time': '2018-07-02 08:00', 'entry_price': 0.14, 'exit_price': 0.14, 'net_pl': 332.80, 'pct': 2.49, 'exit_reason': 'Opposite Entry'},
        {'trade_num': 6, 'type': 'SHORT', 'entry_time': '2018-07-02 08:00', 'exit_time': '2018-07-12 16:00', 'entry_price': 0.14, 'exit_price': 0.12, 'net_pl': 1814.54, 'pct': 12.90, 'exit_reason': 'SSL crossover up exit'},
        {'trade_num': 7, 'type': 'LONG', 'entry_time': '2018-07-18 04:00', 'exit_time': '2018-07-30 20:00', 'entry_price': 0.17, 'exit_price': 0.16, 'net_pl': -1512.97, 'pct': -9.40, 'exit_reason': 'KAMA exit signal'},
        {'trade_num': 8, 'type': 'SHORT', 'entry_time': '2018-08-14 08:00', 'exit_time': '2018-08-14 20:00', 'entry_price': 0.09, 'exit_price': 0.09, 'net_pl': 461.90, 'pct': 3.28, 'exit_reason': 'SSL crossover up exit'},
        {'trade_num': 9, 'type': 'SHORT', 'entry_time': '2018-08-19 16:00', 'exit_time': '2018-09-07 12:00', 'entry_price': 0.10, 'exit_price': 0.09, 'net_pl': 2344.77, 'pct': 15.80, 'exit_reason': 'SSL crossover up exit'},
        {'trade_num': 10, 'type': 'SHORT', 'entry_time': '2018-09-21 12:00', 'exit_time': '2018-09-22 00:00', 'entry_price': 0.08, 'exit_price': 0.09, 'net_pl': -484.98, 'pct': -2.73, 'exit_reason': 'DEMA crossover exit (with KAMA condition)'},
        # Add first 10 trades for comparison
    ]
    return python_trades

def compare_trades():
    """Compare the trades and find differences"""
    print("=== LOADING DATA ===")
    
    # Load TradingView trades
    tv_trades = load_tv_trades()
    
    # Use simplified Python trades for comparison
    python_trades = create_python_trades()
    
    print(f"TradingView Trades: {len(tv_trades)}")
    print(f"Python Trades (sample): {len(python_trades)}")
    print()
    
    # Summary statistics
    tv_total_pl = sum(t['net_pl'] for t in tv_trades)
    python_total_pl = sum(t['net_pl'] for t in python_trades)
    
    print("=== OVERALL COMPARISON ===")
    print(f"TradingView Total P/L: ${tv_total_pl:,.2f}")
    print(f"Python Sample P/L: ${python_total_pl:,.2f}")
    print()
    
    # Trade type breakdown for TV
    tv_long = [t for t in tv_trades if t['type'] == 'LONG']
    tv_short = [t for t in tv_trades if t['type'] == 'SHORT']
    
    print("=== TRADINGVIEW BREAKDOWN ===")
    print(f"Total Trades: {len(tv_trades)}")
    print(f"Long Trades: {len(tv_long)} ({len(tv_long)/len(tv_trades)*100:.1f}%)")
    print(f"Short Trades: {len(tv_short)} ({len(tv_short)/len(tv_trades)*100:.1f}%)")
    
    tv_long_pl = sum(t['net_pl'] for t in tv_long)
    tv_short_pl = sum(t['net_pl'] for t in tv_short)
    
    print(f"Long P/L: ${tv_long_pl:,.2f}")
    print(f"Short P/L: ${tv_short_pl:,.2f}")
    print()
    
    # Detailed first trades comparison
    print("=== DETAILED TRADE COMPARISON ===")
    print()
    
    max_compare = min(len(python_trades), len(tv_trades))
    
    for i in range(max_compare):
        py_trade = python_trades[i]
        tv_trade = tv_trades[i]
        
        print(f"Trade #{i+1}:")
        print(f"  Python:     {py_trade['type']:<5} | Entry: {py_trade['entry_time']} | Exit: {py_trade['exit_time']}")
        print(f"  TradingView: {tv_trade['type']:<5} | Entry: {tv_trade['entry_time']} | Exit: {tv_trade['exit_time']}")
        print(f"  Entry Price - Python: ${py_trade['entry_price']:.4f}, TV: ${tv_trade['entry_price']:.4f}")
        print(f"  Exit Price  - Python: ${py_trade['exit_price']:.4f}, TV: ${tv_trade['exit_price']:.4f}")
        print(f"  P&L         - Python: ${py_trade['net_pl']:,.2f}, TV: ${tv_trade['net_pl']:,.2f}")
        
        # Check for differences
        type_match = py_trade['type'] == tv_trade['type']
        time_match = py_trade['entry_time'] == tv_trade['entry_time']
        price_diff = abs(py_trade['entry_price'] - tv_trade['entry_price'])
        pl_diff = abs(py_trade['net_pl'] - tv_trade['net_pl'])
        
        print(f"  Differences:")
        if not type_match:
            print(f"    ⚠️  TYPE MISMATCH: Python {py_trade['type']} vs TV {tv_trade['type']}")
        if not time_match:
            print(f"    ⚠️  TIME MISMATCH: Python {py_trade['entry_time']} vs TV {tv_trade['entry_time']}")
        if price_diff > 0.01:
            print(f"    ⚠️  PRICE DIFFERENCE: ${price_diff:.4f}")
        if pl_diff > 50:
            print(f"    ⚠️  P&L DIFFERENCE: ${pl_diff:,.2f}")
        
        if type_match and time_match and price_diff <= 0.01 and pl_diff <= 50:
            print(f"    ✅ TRADE MATCHES WELL")
        
        print()
    
    # Show some key TradingView trades for reference
    print("=== TRADINGVIEW TRADE SAMPLE ===")
    for i in range(min(5, len(tv_trades))):
        trade = tv_trades[i]
        print(f"TV Trade #{i+1}: {trade['type']} | {trade['entry_time']} to {trade['exit_time']} | P&L: ${trade['net_pl']:,.2f}")
    
    print(f"\nFull TradingView performance: ${tv_total_pl:,.2f}")
    print(f"Python strategy generated: 155 trades with ${59036.48:,.2f} net P/L")
    print(f"Major difference: ${abs(59036.48 - tv_total_pl):,.2f}")

if __name__ == "__main__":
    compare_trades() 
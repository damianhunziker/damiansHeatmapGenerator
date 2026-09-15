import pandas as pd
import numpy as np
from datetime import datetime
import re

def parse_trade_log_from_output(output_text):
    """Parse Python trades from the terminal output"""
    trades = []
    
    # Find the trade list section
    trade_section_start = output_text.find("Trade List:")
    if trade_section_start == -1:
        return trades
    
    lines = output_text[trade_section_start:].split('\n')
    
    # Find trade data lines (they start with a number followed by |)
    for line in lines:
        if re.match(r'^\s*\d+\s*\|', line):
            parts = [part.strip() for part in line.split('|')]
            if len(parts) >= 10:
                try:
                    trade_num = int(parts[0].strip())
                    trade_type = parts[1].strip()
                    entry_time = parts[2].strip()
                    exit_time = parts[3].strip()
                    entry_price = float(parts[4].strip().replace('$', '').replace(',', ''))
                    exit_price = float(parts[5].strip().replace('$', '').replace(',', ''))
                    gross_pl = float(parts[6].strip().replace('$', '').replace(',', ''))
                    fees = float(parts[7].strip().replace('$', '').replace(',', ''))
                    net_pl = float(parts[8].strip().replace('$', '').replace(',', ''))
                    pct = parts[9].strip().replace('%', '')
                    exit_reason = parts[10].strip() if len(parts) > 10 else ''
                    
                    trades.append({
                        'trade_num': trade_num,
                        'type': trade_type,
                        'entry_time': entry_time,
                        'exit_time': exit_time,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'gross_pl': gross_pl,
                        'fees': fees,
                        'net_pl': net_pl,
                        'pct': pct,
                        'exit_reason': exit_reason
                    })
                except (ValueError, IndexError) as e:
                    print(f"Error parsing line: {line}")
                    print(f"Error: {e}")
                    continue
    
    return trades

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
                'pct': exit['P&L %'].replace('%', ''),
                'exit_reason': exit['Signal'],
                'cumulative_pl': exit['Cumulative P&L USDT']
            })
    
    return trades

def compare_trades(python_trades, tv_trades):
    """Compare Python trades with TradingView trades"""
    print("=== TRADE COMPARISON ===")
    print(f"Python Trades: {len(python_trades)}")
    print(f"TradingView Trades: {len(tv_trades)}")
    print()
    
    # Summary statistics
    python_total_pl = sum(t['net_pl'] for t in python_trades)
    tv_total_pl = sum(t['net_pl'] for t in tv_trades)
    
    print(f"Python Total P/L: ${python_total_pl:,.2f}")
    print(f"TradingView Total P/L: ${tv_total_pl:,.2f}")
    print(f"Difference: ${python_total_pl - tv_total_pl:,.2f}")
    print()
    
    # Trade type breakdown
    python_long = [t for t in python_trades if t['type'] == 'LONG']
    python_short = [t for t in python_trades if t['type'] == 'SHORT']
    tv_long = [t for t in tv_trades if t['type'] == 'LONG']
    tv_short = [t for t in tv_trades if t['type'] == 'SHORT']
    
    print("Trade Type Breakdown:")
    print(f"Python - Long: {len(python_long)}, Short: {len(python_short)}")
    print(f"TradingView - Long: {len(tv_long)}, Short: {len(tv_short)}")
    print()
    
    # Compare first 20 trades in detail
    print("=== DETAILED TRADE COMPARISON (First 20 Trades) ===")
    print()
    
    max_compare = min(20, len(python_trades), len(tv_trades))
    
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
        pl_diff = abs(py_trade['net_pl'] - tv_trade['net_pl'])
        
        if not type_match:
            print(f"  ⚠️  TYPE MISMATCH!")
        if pl_diff > 100:  # Significant P&L difference
            print(f"  ⚠️  LARGE P&L DIFFERENCE: ${pl_diff:,.2f}")
        
        print()
    
    # Find missing or extra trades
    print("=== TRADE TIMING ANALYSIS ===")
    
    # Convert times to datetime for comparison
    py_times = []
    tv_times = []
    
    for trade in python_trades:
        try:
            entry_dt = datetime.strptime(trade['entry_time'], '%Y-%m-%d %H:%M')
            py_times.append(entry_dt)
        except:
            pass
    
    for trade in tv_trades:
        try:
            entry_dt = datetime.strptime(trade['entry_time'], '%Y-%m-%d %H:%M')
            tv_times.append(entry_dt)
        except:
            pass
    
    # Find unique times
    py_unique = set(py_times) - set(tv_times)
    tv_unique = set(tv_times) - set(py_times)
    
    if py_unique:
        print(f"Python-only entry times ({len(py_unique)}):")
        for dt in sorted(py_unique)[:10]:  # Show first 10
            print(f"  {dt}")
        print()
    
    if tv_unique:
        print(f"TradingView-only entry times ({len(tv_unique)}):")
        for dt in sorted(tv_unique)[:10]:  # Show first 10
            print(f"  {dt}")
        print()

def main():
    # Read the terminal output from the previous command
    # Since we can't access the previous output directly, we'll run the test again
    # and capture the output
    
    import subprocess
    
    print("Running Python strategy test...")
    result = subprocess.run([
        'python3', 'test.py', 'pnl',
        '--start_date=2018-03-01',
        '--end_date=2025-04-01',
        '--asset=ADAUSDT',
        '--strategy=LiveKAMASSLStrategy',
        '--interval=4h',
        '--debug_mode=False',  # Disable debug for cleaner output
        '--trade_direction=both',
        '--calculate_divergences=False',
        '--use_divergence_exit_long=True',
        '--use_divergence_exit_short=True'
    ], capture_output=True, text=True, cwd='.')
    
    print("Parsing Python trades...")
    python_trades = parse_trade_log_from_output(result.stdout)
    
    print("Loading TradingView trades...")
    tv_trades = load_tv_trades()
    
    print("Comparing trades...")
    compare_trades(python_trades, tv_trades)

if __name__ == "__main__":
    main() 
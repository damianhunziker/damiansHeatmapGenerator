import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re

def load_tv_trades():
    """Load TradingView trades from CSV"""
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
            
            # Determine trade direction
            signal_lower = entry['Signal'].lower()
            if 'long' in signal_lower:
                direction = 'LONG'
            elif 'short' in signal_lower:
                direction = 'SHORT'
            else:
                direction = 'UNKNOWN'
            
            # Parse exit signal
            exit_signal = exit['Signal']
            
            trade = {
                'trade_num': trade_num,
                'direction': direction,
                'entry_time': pd.to_datetime(entry['Date/Time']),
                'entry_price': entry['Price USDT'],
                'exit_time': pd.to_datetime(exit['Date/Time']),
                'exit_price': exit['Price USDT'],
                'pnl_usdt': exit['P&L USDT'],
                'pnl_pct': exit['P&L %'].replace('%', '').replace(',', '') if isinstance(exit['P&L %'], str) else exit['P&L %'],
                'exit_signal': exit_signal,
                'quantity': entry['Quantity']
            }
            trades.append(trade)
    
    return pd.DataFrame(trades)

def parse_python_trades():
    """Parse Python strategy trades from the terminal output"""
    print("📊 ANALYZING PYTHON STRATEGY OUTPUT")
    print("=" * 50)
    
    # CORRECTED Python results from user's actual test:
    python_stats = {
        'total_trades': 203,
        'net_profit': 4335046.18,
        'performance_pct': 43350.46,
        'long_trades': 125,
        'short_trades': 78,
        'win_rate': 41.38,
        'profit_factor': 2.04,
        'max_drawdown': 20056.09,
        'max_drawdown_pct': 50.82,
        'average_trade': 21354.91,
        'average_trade_pct': 213.55,
        'winning_trades': 84,
        'average_win': 101029.14,
        'average_loss': 34885.72,
        'start_date': '2018-03-01',
        'end_date': '2025-04-01',
        'asset': 'SOLUSDT',
        'interval': '4h'
    }
    
    return python_stats

def analyze_trade_differences():
    """Analyze the differences between TradingView and Python strategies"""
    
    print("\n" + "="*80)
    print("🔍 SOLUSDT 4H STRATEGY COMPARISON ANALYSIS")
    print("="*80)
    
    # Load TradingView data
    tv_trades = load_tv_trades()
    python_stats = parse_python_trades()
    
    print(f"\n📈 TRADINGVIEW STRATEGY RESULTS:")
    print(f"   Total Trades: {len(tv_trades)}")
    print(f"   Date Range: {tv_trades['entry_time'].min()} to {tv_trades['exit_time'].max()}")
    print(f"   Long Trades: {len(tv_trades[tv_trades['direction'] == 'LONG'])}")
    print(f"   Short Trades: {len(tv_trades[tv_trades['direction'] == 'SHORT'])}")
    
    # Calculate TradingView performance
    total_pnl = tv_trades['pnl_usdt'].sum()
    initial_balance = 1000  # Assuming starting balance
    final_balance = initial_balance + total_pnl
    tv_performance = ((final_balance - initial_balance) / initial_balance) * 100
    
    print(f"   Total P&L: ${total_pnl:,.2f}")
    print(f"   Performance: {tv_performance:.1f}%")
    print(f"   Final Balance: ${final_balance:,.2f}")
    
    print(f"\n🐍 PYTHON STRATEGY RESULTS:")
    print(f"   Total Trades: {python_stats['total_trades']}")
    print(f"   Date Range: {python_stats['start_date']} to {python_stats['end_date']}")
    print(f"   Net Profit: ${python_stats['net_profit']:,.2f}")
    print(f"   Performance: {python_stats['performance_pct']:.1f}%")
    print(f"   Long Trades: {python_stats['long_trades']}")
    print(f"   Short Trades: {python_stats['short_trades']}")
    print(f"   Win Rate: {python_stats['win_rate']:.2f}%")
    print(f"   Profit Factor: {python_stats['profit_factor']:.2f}")
    print(f"   Max Drawdown: ${python_stats['max_drawdown']:,.2f} ({python_stats['max_drawdown_pct']:.2f}%)")
    print(f"   Average Trade: ${python_stats['average_trade']:,.2f}")
    print(f"   Average Win: ${python_stats['average_win']:,.2f}")
    print(f"   Average Loss: ${python_stats['average_loss']:,.2f}")
    
    print(f"\n⚡ KEY DIFFERENCES:")
    trade_diff = len(tv_trades) - python_stats['total_trades']
    profit_diff = python_stats['net_profit'] - total_pnl
    perf_diff = python_stats['performance_pct'] - tv_performance
    
    print(f"   Trade Count Difference: {trade_diff:+d} trades")
    print(f"   Profit Difference: ${profit_diff:+,.2f}")
    print(f"   Performance Difference: {perf_diff:+.1f}%")
    
    if trade_diff > 0:
        print(f"   ➤ TradingView generated {abs(trade_diff)} MORE trades")
    elif trade_diff < 0:
        print(f"   ➤ Python generated {abs(trade_diff)} MORE trades")
    else:
        print(f"   ➤ Same number of trades")
    
    if profit_diff > 0:
        print(f"   ➤ Python generated ${abs(profit_diff):,.2f} MORE profit")
    elif profit_diff < 0:
        print(f"   ➤ TradingView generated ${abs(profit_diff):,.2f} MORE profit")
    
    if perf_diff > 0:
        print(f"   ➤ Python performed {abs(perf_diff):.1f}% BETTER")
    elif perf_diff < 0:
        print(f"   ➤ TradingView performed {abs(perf_diff):.1f}% BETTER")
    
    print(f"\n📊 TRADINGVIEW TRADE BREAKDOWN BY YEAR:")
    tv_trades['year'] = tv_trades['entry_time'].dt.year
    yearly_trades = tv_trades.groupby('year').agg({
        'trade_num': 'count',
        'pnl_usdt': 'sum',
        'direction': lambda x: f"L:{len(x[x=='LONG'])}/S:{len(x[x=='SHORT'])}"
    }).round(2)
    yearly_trades.columns = ['Trades', 'P&L_USD', 'Long/Short']
    print(yearly_trades)
    
    print(f"\n🔍 TRADINGVIEW EXIT SIGNAL ANALYSIS:")
    exit_signals = tv_trades['exit_signal'].value_counts()
    print("Exit Signal Distribution:")
    for signal, count in exit_signals.items():
        percentage = (count / len(tv_trades)) * 100
        print(f"   {signal}: {count} trades ({percentage:.1f}%)")
    
    print(f"\n⏰ TRADINGVIEW TRADE DURATION ANALYSIS:")
    tv_trades['duration_hours'] = (tv_trades['exit_time'] - tv_trades['entry_time']).dt.total_seconds() / 3600
    print(f"   Average Trade Duration: {tv_trades['duration_hours'].mean():.1f} hours")
    print(f"   Median Trade Duration: {tv_trades['duration_hours'].median():.1f} hours")
    print(f"   Min Duration: {tv_trades['duration_hours'].min():.1f} hours")
    print(f"   Max Duration: {tv_trades['duration_hours'].max():.1f} hours")
    
    print(f"\n💰 TRADINGVIEW PROFITABILITY ANALYSIS:")
    profitable_trades = len(tv_trades[tv_trades['pnl_usdt'] > 0])
    losing_trades = len(tv_trades[tv_trades['pnl_usdt'] < 0])
    win_rate = (profitable_trades / len(tv_trades)) * 100
    
    print(f"   Profitable Trades: {profitable_trades}")
    print(f"   Losing Trades: {losing_trades}")
    print(f"   Win Rate: {win_rate:.1f}%")
    print(f"   Average Win: ${tv_trades[tv_trades['pnl_usdt'] > 0]['pnl_usdt'].mean():.2f}")
    print(f"   Average Loss: ${tv_trades[tv_trades['pnl_usdt'] < 0]['pnl_usdt'].mean():.2f}")
    
    print(f"\n📈 LIKELY REASONS FOR DIFFERENCES:")
    print("1. 🕐 TIMING DIFFERENCES:")
    print("   - TradingView uses real-time bar closes")
    print("   - Python processes historical data sequentially")
    print("   - Different handling of 4h candle boundaries")
    
    print("\n2. 📊 INDICATOR CALCULATION DIFFERENCES:")
    print("   - KAMA calculation variations")
    print("   - SSL Channel implementation differences")
    print("   - DEMA calculation precision")
    print("   - Bollinger Bands calculation")
    
    print("\n3. 🎯 SIGNAL GENERATION DIFFERENCES:")
    print("   - Entry condition logic variations")
    print("   - Exit condition timing")
    print("   - Divergence detection algorithms")
    print("   - Real-time vs historical signal processing")
    
    print("\n4. 📝 TRADE EXECUTION DIFFERENCES:")
    print("   - Position sizing methods")
    print("   - Fee calculation approaches")
    print("   - Entry/exit price determination")
    print("   - Slippage handling")
    
    print(f"\n🔧 SPECIFIC TECHNICAL ISSUES TO INVESTIGATE:")
    print("1. SSL Channel crossover detection accuracy")
    print("2. DEMA calculation (2*EMA - EMA(EMA) vs standard)")
    print("3. KAMA delta threshold calculations")
    print("4. Bollinger Band basis crossover timing")
    print("5. Divergence confirmation logic")
    print("6. Real-time vs historical data processing")
    
    print(f"\n✅ RECOMMENDATIONS:")
    print("1. Implement exact TradingView indicator formulas")
    print("2. Add debug logging for signal generation")
    print("3. Compare indicator values at specific timestamps")
    print("4. Verify candle data alignment between sources")
    print("5. Test with smaller date ranges for precise comparison")
    print("6. Add trade-by-trade comparison functionality")
    
    return tv_trades, python_stats

if __name__ == "__main__":
    tv_trades, python_stats = analyze_trade_differences() 
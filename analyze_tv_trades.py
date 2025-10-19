import pandas as pd
import numpy as np
from datetime import datetime

# Read TradingView data
df = pd.read_csv('tv-export-adausdt-4h.txt', delimiter='\t')

# Filter to get only entry trades (since each trade has entry and exit rows)
entries = df[df['Type'].str.contains('Entry', na=False)]
exits = df[df['Type'].str.contains('Exit', na=False)]

# Combine entry and exit data
trades = []
for i in range(len(entries)):
    entry = entries.iloc[i]
    exit_row = exits.iloc[i]
    
    trade = {
        'trade_num': entry['Trade #'],
        'direction': 'LONG' if 'long' in entry['Signal'] else 'SHORT',
        'entry_time': entry['Date/Time'],
        'entry_price': entry['Price USDT'],
        'exit_time': exit_row['Date/Time'],
        'exit_price': exit_row['Price USDT'],
        'pnl_usdt': exit_row['P&L USDT'],
        'pnl_pct': exit_row['P&L %'].replace('%', ''),
        'exit_signal': exit_row['Signal'],
        'cumulative_pnl': exit_row['Cumulative P&L USDT']
    }
    trades.append(trade)

trades_df = pd.DataFrame(trades)

# Convert dates and numeric values
trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
trades_df['pnl_pct'] = trades_df['pnl_pct'].astype(float)
trades_df['pnl_usdt'] = trades_df['pnl_usdt'].astype(float)

print('=== TRADINGVIEW TRADE ANALYSIS ===')
print()
print(f'Gesamtzeitraum: {trades_df["entry_time"].min()} bis {trades_df["exit_time"].max()}')
print(f'Gesamtanzahl Trades: {len(trades_df)}')
print()

# Direction breakdown
long_trades = trades_df[trades_df['direction'] == 'LONG']
short_trades = trades_df[trades_df['direction'] == 'SHORT']

print(f'Long Trades: {len(long_trades)} ({len(long_trades)/len(trades_df)*100:.1f}%)')
print(f'Short Trades: {len(short_trades)} ({len(short_trades)/len(trades_df)*100:.1f}%)')
print()

# Performance by direction
print('=== PERFORMANCE BY DIRECTION ===')
print('Long Trades:')
print(f'  Winning trades: {len(long_trades[long_trades["pnl_usdt"] > 0])}/{len(long_trades)} ({len(long_trades[long_trades["pnl_usdt"] > 0])/len(long_trades)*100:.1f}%)')
print(f'  Total PnL: {long_trades["pnl_usdt"].sum():.2f} USDT')
print(f'  Average PnL per trade: {long_trades["pnl_usdt"].mean():.2f} USDT')
print(f'  Best trade: {long_trades["pnl_usdt"].max():.2f} USDT')
print(f'  Worst trade: {long_trades["pnl_usdt"].min():.2f} USDT')
print()

print('Short Trades:')
print(f'  Winning trades: {len(short_trades[short_trades["pnl_usdt"] > 0])}/{len(short_trades)} ({len(short_trades[short_trades["pnl_usdt"] > 0])/len(short_trades)*100:.1f}%)')
print(f'  Total PnL: {short_trades["pnl_usdt"].sum():.2f} USDT')
print(f'  Average PnL per trade: {short_trades["pnl_usdt"].mean():.2f} USDT')
print(f'  Best trade: {short_trades["pnl_usdt"].max():.2f} USDT')
print(f'  Worst trade: {short_trades["pnl_usdt"].min():.2f} USDT')
print()

# Exit signals analysis
print('=== EXIT SIGNALS ANALYSIS ===')
exit_signals = trades_df['exit_signal'].value_counts()
for signal, count in exit_signals.items():
    pct = count / len(trades_df) * 100
    signal_trades = trades_df[trades_df['exit_signal'] == signal]
    avg_pnl = signal_trades['pnl_usdt'].mean()
    print(f'{signal}: {count} trades ({pct:.1f}%) - Avg PnL: {avg_pnl:.2f} USDT')
print()

# Time period breakdown
print('=== YEARLY BREAKDOWN ===')
trades_df['year'] = trades_df['entry_time'].dt.year
yearly_stats = trades_df.groupby('year').agg({
    'trade_num': 'count',
    'pnl_usdt': ['sum', 'mean'],
    'direction': lambda x: (x == 'LONG').sum()
}).round(2)

yearly_stats.columns = ['Total_Trades', 'Total_PnL', 'Avg_PnL', 'Long_Trades']
yearly_stats['Short_Trades'] = yearly_stats['Total_Trades'] - yearly_stats['Long_Trades']
yearly_stats['Long_Pct'] = (yearly_stats['Long_Trades'] / yearly_stats['Total_Trades'] * 100).round(1)

print(yearly_stats)
print()

# Final cumulative result
final_pnl = trades_df['cumulative_pnl'].iloc[-1]
print(f'=== FINAL RESULT ===')
print(f'Final Cumulative P&L: {final_pnl:.2f} USDT')
print(f'Total Return: {(final_pnl / 10000) * 100:.1f}% (assuming 10k initial)')
print(f'Win Rate: {len(trades_df[trades_df["pnl_usdt"] > 0]) / len(trades_df) * 100:.1f}%')

# First few trades for detailed comparison
print()
print('=== FIRST 20 TRADES FOR COMPARISON ===')
first_20 = trades_df.head(20)[['trade_num', 'direction', 'entry_time', 'entry_price', 'exit_time', 'exit_price', 'pnl_usdt', 'exit_signal']]
for _, trade in first_20.iterrows():
    print(f'Trade {int(trade["trade_num"])}: {trade["direction"]} {trade["entry_time"]} -> {trade["exit_time"]} | {trade["entry_price"]:.4f} -> {trade["exit_price"]:.4f} | PnL: {trade["pnl_usdt"]:.2f} | Exit: {trade["exit_signal"]}')

# Compare 2018 data specifically
print()
print('=== 2018 COMPARISON PERIOD ===')
trades_2018 = trades_df[trades_df['year'] == 2018]
print(f'2018 Trades: {len(trades_2018)}')
print(f'2018 Long: {len(trades_2018[trades_2018["direction"] == "LONG"])}')
print(f'2018 Short: {len(trades_2018[trades_2018["direction"] == "SHORT"])}')
print(f'2018 Total PnL: {trades_2018["pnl_usdt"].sum():.2f} USDT')
print(f'2018 Win Rate: {len(trades_2018[trades_2018["pnl_usdt"] > 0]) / len(trades_2018) * 100:.1f}%') 
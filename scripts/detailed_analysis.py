import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def load_tv_trades():
    """Load and analyze TradingView trades"""
    df = pd.read_csv('test.csv', delimiter='\t')
    
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
                trade_type = 'LONG'
            elif 'short' in signal_lower:
                trade_type = 'SHORT'
            else:
                trade_type = 'UNKNOWN'
            
            # Parse datetime
            try:
                entry_time = datetime.strptime(entry['Date/Time'], '%Y-%m-%d %H:%M')
                exit_time = datetime.strptime(exit['Date/Time'], '%Y-%m-%d %H:%M')
            except:
                try:
                    entry_time = datetime.strptime(entry['Date/Time'], '%Y-%m-%d %H:%M:%S')
                    exit_time = datetime.strptime(exit['Date/Time'], '%Y-%m-%d %H:%M:%S')
                except:
                    continue
            
            trades.append({
                'trade_num': trade_num,
                'type': trade_type,
                'entry_time': entry_time,
                'exit_time': exit_time,
                'entry_price': float(entry['Price USDT']),
                'exit_price': float(exit['Price USDT']),
                'net_pl': float(exit['P&L USDT']),
                'pct': float(exit['P&L %'].replace('%', '')),
                'entry_signal': entry['Signal'],
                'exit_signal': exit['Signal'],
                'duration_hours': (exit_time - entry_time).total_seconds() / 3600
            })
    
    return trades

def analyze_performance_differences():
    """Analyze the main performance differences"""
    
    print("=== UMFASSENDE TRADE-ANALYSE ===")
    print()
    
    # Load TradingView data
    tv_trades = load_tv_trades()
    
    print(f"📊 TRADINGVIEW STATISTIKEN:")
    print(f"   Gesamte Trades: {len(tv_trades)}")
    print(f"   Zeitraum: {tv_trades[0]['entry_time'].strftime('%Y-%m-%d')} bis {tv_trades[-1]['exit_time'].strftime('%Y-%m-%d')}")
    print(f"   Gesamtgewinn: ${sum(t['net_pl'] for t in tv_trades):,.2f}")
    print()
    
    # Trade type breakdown
    tv_long = [t for t in tv_trades if t['type'] == 'LONG']
    tv_short = [t for t in tv_trades if t['type'] == 'SHORT']
    
    tv_long_pl = sum(t['net_pl'] for t in tv_long)
    tv_short_pl = sum(t['net_pl'] for t in tv_short)
    
    print(f"📈 TRADE-VERTEILUNG (TradingView):")
    print(f"   Long Trades: {len(tv_long)} ({len(tv_long)/len(tv_trades)*100:.1f}%)")
    print(f"   Short Trades: {len(tv_short)} ({len(tv_short)/len(tv_trades)*100:.1f}%)")
    print(f"   Long P/L: ${tv_long_pl:,.2f}")
    print(f"   Short P/L: ${tv_short_pl:,.2f}")
    print()
    
    # Python results (from the output)
    python_stats = {
        'total_trades': 155,
        'net_pl': 59036.48,
        'long_trades': 72,
        'short_trades': 83,
        'long_pl': 52988.81,
        'short_pl': 6047.67
    }
    
    print(f"🐍 PYTHON STATISTIKEN:")
    print(f"   Gesamte Trades: {python_stats['total_trades']}")
    print(f"   Gesamtgewinn: ${python_stats['net_pl']:,.2f}")
    print(f"   Long Trades: {python_stats['long_trades']} ({python_stats['long_trades']/python_stats['total_trades']*100:.1f}%)")
    print(f"   Short Trades: {python_stats['short_trades']} ({python_stats['short_trades']/python_stats['total_trades']*100:.1f}%)")
    print(f"   Long P/L: ${python_stats['long_pl']:,.2f}")
    print(f"   Short P/L: ${python_stats['short_pl']:,.2f}")
    print()
    
    # Key differences analysis
    print(f"⚠️  HAUPTUNTERSCHIEDE:")
    print(f"   Trade-Anzahl: TV {len(tv_trades)} vs Python {python_stats['total_trades']} (Differenz: {len(tv_trades) - python_stats['total_trades']})")
    print(f"   Gesamtgewinn: TV ${sum(t['net_pl'] for t in tv_trades):,.2f} vs Python ${python_stats['net_pl']:,.2f}")
    print(f"   Gewinn-Differenz: ${sum(t['net_pl'] for t in tv_trades) - python_stats['net_pl']:,.2f}")
    print()
    
    # Analyze trade timing patterns
    print(f"📅 ZEITANALYSE (TradingView):")
    
    # Monthly breakdown
    monthly_trades = {}
    for trade in tv_trades:
        month_key = trade['entry_time'].strftime('%Y-%m')
        if month_key not in monthly_trades:
            monthly_trades[month_key] = {'count': 0, 'pl': 0}
        monthly_trades[month_key]['count'] += 1
        monthly_trades[month_key]['pl'] += trade['net_pl']
    
    print(f"   Trades pro Monat (Top 10 profitabelste):")
    sorted_months = sorted(monthly_trades.items(), key=lambda x: x[1]['pl'], reverse=True)
    for month, data in sorted_months[:10]:
        print(f"     {month}: {data['count']} Trades, ${data['pl']:,.2f}")
    print()
    
    # Trade duration analysis
    durations = [t['duration_hours'] for t in tv_trades]
    avg_duration = np.mean(durations)
    median_duration = np.median(durations)
    
    print(f"⏱️  TRADE-DAUER (TradingView):")
    print(f"   Durchschnittlich: {avg_duration:.1f} Stunden ({avg_duration/24:.1f} Tage)")
    print(f"   Median: {median_duration:.1f} Stunden ({median_duration/24:.1f} Tage)")
    print(f"   Kürzester Trade: {min(durations):.1f} Stunden")
    print(f"   Längster Trade: {max(durations):.1f} Stunden ({max(durations)/24:.1f} Tage)")
    print()
    
    # Win rate analysis
    winning_trades = [t for t in tv_trades if t['net_pl'] > 0]
    losing_trades = [t for t in tv_trades if t['net_pl'] < 0]
    
    print(f"🎯 ERFOLGSQUOTE (TradingView):")
    print(f"   Gewinnende Trades: {len(winning_trades)} ({len(winning_trades)/len(tv_trades)*100:.1f}%)")
    print(f"   Verlierende Trades: {len(losing_trades)} ({len(losing_trades)/len(tv_trades)*100:.1f}%)")
    print(f"   Durchschnittlicher Gewinn: ${np.mean([t['net_pl'] for t in winning_trades]):,.2f}")
    print(f"   Durchschnittlicher Verlust: ${np.mean([t['net_pl'] for t in losing_trades]):,.2f}")
    print()
    
    # Analyze signal types
    print(f"📡 SIGNAL-ANALYSE (TradingView):")
    entry_signals = {}
    exit_signals = {}
    
    for trade in tv_trades:
        entry_sig = trade['entry_signal']
        exit_sig = trade['exit_signal']
        
        if entry_sig not in entry_signals:
            entry_signals[entry_sig] = 0
        if exit_sig not in exit_signals:
            exit_signals[exit_sig] = 0
            
        entry_signals[entry_sig] += 1
        exit_signals[exit_sig] += 1
    
    print(f"   Entry Signals:")
    for signal, count in sorted(entry_signals.items(), key=lambda x: x[1], reverse=True):
        print(f"     {signal}: {count} mal ({count/len(tv_trades)*100:.1f}%)")
    
    print(f"   Exit Signals:")
    for signal, count in sorted(exit_signals.items(), key=lambda x: x[1], reverse=True):
        print(f"     {signal}: {count} mal ({count/len(tv_trades)*100:.1f}%)")
    print()
    
    # Key findings summary
    print(f"🔍 WICHTIGSTE ERKENNTNISSE:")
    print(f"   1. TradingView generiert {len(tv_trades) - python_stats['total_trades']} mehr Trades")
    print(f"   2. TradingView hat einen ${sum(t['net_pl'] for t in tv_trades) - python_stats['net_pl']:,.2f} höheren Gesamtgewinn")
    print(f"   3. TradingView: {len(tv_long)/len(tv_trades)*100:.1f}% Long vs Python: {python_stats['long_trades']/python_stats['total_trades']*100:.1f}% Long")
    print(f"   4. TradingView Long Performance: ${tv_long_pl:,.2f} vs Python: ${python_stats['long_pl']:,.2f}")
    print(f"   5. TradingView Short Performance: ${tv_short_pl:,.2f} vs Python: ${python_stats['short_pl']:,.2f}")
    
    # Calculate performance metrics
    tv_total_pl = sum(t['net_pl'] for t in tv_trades)
    tv_win_rate = len(winning_trades) / len(tv_trades) * 100
    
    print(f"\n📈 PERFORMANCE-VERGLEICH:")
    print(f"   TradingView: {tv_win_rate:.1f}% Erfolgsquote, ${tv_total_pl:,.2f} Gewinn")
    print(f"   Python: ~47.1% Erfolgsquote (73/155), ${python_stats['net_pl']:,.2f} Gewinn")
    print(f"   Performance-Differenz: {tv_win_rate - 47.1:.1f} Prozentpunkte besser bei TradingView")

if __name__ == "__main__":
    analyze_performance_differences() 
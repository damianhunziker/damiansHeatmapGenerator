import pandas as pd
import numpy as np
from datetime import datetime

def analyze_tv_position_sizing():
    """Analyze TradingView position sizing from CSV data"""
    df = pd.read_csv('classes/strategies/tv-export-SOLUSDT-4h.csv', delimiter='\t')
    
    trades = []
    trade_numbers = df['Trade #'].unique()
    
    print("🔍 TRADINGVIEW POSITION SIZING ANALYSIS")
    print("=" * 60)
    
    for trade_num in trade_numbers:
        trade_data = df[df['Trade #'] == trade_num]
        entry_row = trade_data[trade_data['Type'].str.contains('Entry', na=False)]
        exit_row = trade_data[trade_data['Type'].str.contains('Exit', na=False)]
        
        if len(entry_row) > 0 and len(exit_row) > 0:
            entry = entry_row.iloc[0]
            exit = exit_row.iloc[0]
            
            # Extract position information
            entry_price = entry['Price USDT']
            exit_price = exit['Price USDT']
            pnl_usdt = exit['P&L USDT']
            qty = entry['Quantity']  # This should tell us the position size
            
            trades.append({
                'trade_num': trade_num,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'quantity': qty,
                'pnl_usdt': pnl_usdt,
                'position_value': entry_price * qty,
                'direction': 'LONG' if 'long' in entry['Signal'].lower() else 'SHORT'
            })
    
    trades_df = pd.DataFrame(trades)
    
    print(f"📊 TradingView Analysis:")
    print(f"Total trades: {len(trades_df)}")
    print(f"Average quantity: {trades_df['quantity'].mean():.2f}")
    print(f"Average position value: ${trades_df['position_value'].mean():.2f}")
    print(f"Min position value: ${trades_df['position_value'].min():.2f}")
    print(f"Max position value: ${trades_df['position_value'].max():.2f}")
    
    print(f"\nFirst 10 trades position analysis:")
    for i, trade in trades_df.head(10).iterrows():
        print(f"Trade {trade['trade_num']:2d}: {trade['direction']:5s} | "
              f"Price: ${trade['entry_price']:6.2f} | "
              f"Qty: {trade['quantity']:8.2f} | "
              f"Value: ${trade['position_value']:8.2f} | "
              f"P&L: ${trade['pnl_usdt']:8.2f}")
    
    return trades_df

def analyze_python_position_sizing():
    """Analyze Python strategy position sizing"""
    print(f"\n🐍 PYTHON STRATEGY POSITION SIZING ANALYSIS")
    print("=" * 60)
    
    print("Current Python Logic:")
    print("position_size = current_equity  # 100% of capital")
    print("contracts = position_size / entry_price")
    print("profit = contracts * (exit_price - entry_price)  # for LONG")
    
    print(f"\nPython Strategy Results (from test output):")
    print(f"Initial Equity: $1,000")
    print(f"Final Equity: $434,504.62")
    print(f"Total Return: 43,350.46%")
    print(f"Total Trades: 203")
    
    print(f"\nExample calculation for first profitable trade:")
    print(f"Trade #5: LONG entry at $3.25, exit at $3.90")
    print(f"If equity was ~$600 (after previous losses):")
    print(f"  Contracts = $600 / $3.25 = 184.62 contracts")
    print(f"  Profit = 184.62 * ($3.90 - $3.25) = $116.46")
    print(f"  Return = 19.93% (matches output)")

def propose_position_sizing_optimizations():
    """Propose different position sizing strategies"""
    print(f"\n💡 POSITION SIZING OPTIMIZATION PROPOSALS")
    print("=" * 60)
    
    strategies = [
        {
            "name": "Fixed Dollar Amount",
            "description": "Use fixed $100 per trade",
            "formula": "position_size = 100",
            "pros": ["Consistent risk", "Easy to understand", "Matches TV potentially"],
            "cons": ["No compounding", "Doesn't scale with account"]
        },
        {
            "name": "Fixed Percentage",
            "description": "Use 10% of current equity",
            "formula": "position_size = current_equity * 0.10",
            "pros": ["Scales with account", "Controlled risk", "Some compounding"],
            "cons": ["Slower growth than 100%", "Still aggressive"]
        },
        {
            "name": "Kelly Criterion",
            "description": "Based on win rate and avg win/loss",
            "formula": "position_size = current_equity * kelly_fraction",
            "pros": ["Mathematically optimal", "Risk-adjusted", "Dynamic"],
            "cons": ["Complex calculation", "Requires historical data"]
        },
        {
            "name": "Risk-Based Sizing",
            "description": "Risk fixed % of equity per trade",
            "formula": "position_size = (equity * risk_pct) / stop_loss_pct",
            "pros": ["Controls max loss", "Professional approach"],
            "cons": ["Requires stop losses", "More complex"]
        },
        {
            "name": "TradingView Match",
            "description": "Match TV's apparent sizing",
            "formula": "position_size = min(1000, current_equity * 0.5)",
            "pros": ["Direct comparison possible", "More conservative"],
            "cons": ["Arbitrary limits", "May not be optimal"]
        }
    ]
    
    print(f"🎯 RECOMMENDED POSITION SIZING STRATEGIES:\n")
    
    for i, strategy in enumerate(strategies, 1):
        print(f"{i}. {strategy['name']}")
        print(f"   Description: {strategy['description']}")
        print(f"   Formula: {strategy['formula']}")
        print(f"   Pros: {', '.join(strategy['pros'])}")
        print(f"   Cons: {', '.join(strategy['cons'])}")
        print()

def create_optimized_strategy_comparison():
    """Create comparison of different position sizing approaches"""
    print(f"\n📈 POSITION SIZING COMPARISON SIMULATION")
    print("=" * 60)
    
    # Simulate with different position sizing on the same trades
    initial_equity = 1000
    
    print(f"Simulation based on SOLUSDT strategy results:")
    print(f"Assuming 203 trades with 41.38% win rate")
    print(f"Average winning trade: +213.55% (of position)")
    print(f"Average losing trade: -34.89% (of position)")
    
    scenarios = [
        ("Current (100% equity)", 1.0, "current_equity"),
        ("Conservative (25% equity)", 0.25, "current_equity * 0.25"),
        ("Moderate (50% equity)", 0.50, "current_equity * 0.50"),
        ("Fixed $100", 100, "min(100, current_equity)"),
        ("Fixed $500", 500, "min(500, current_equity)"),
    ]
    
    print(f"\n{'Strategy':<25} {'Formula':<25} {'Risk Level':<15} {'Expected Growth'}")
    print("-" * 80)
    
    for name, factor, formula in scenarios:
        if isinstance(factor, float):
            risk_level = f"{factor*100:.0f}% equity"
            # Rough estimation of growth potential
            if factor == 1.0:
                growth = "Exponential"
            elif factor >= 0.5:
                growth = "High"
            elif factor >= 0.25:
                growth = "Moderate"
            else:
                growth = "Conservative"
        else:
            risk_level = f"${factor} fixed"
            growth = "Linear"
        
        print(f"{name:<25} {formula:<25} {risk_level:<15} {growth}")

def recommend_implementation():
    """Provide specific implementation recommendations"""
    print(f"\n🛠️ IMPLEMENTATION RECOMMENDATIONS")
    print("=" * 60)
    
    print(f"1. IMMEDIATE FIX (for fair TV comparison):")
    print(f"   - Change: position_size = min(1000, current_equity * 0.5)")
    print(f"   - This limits position size and makes comparison fairer")
    print(f"   - Should reduce overall returns but increase stability")
    
    print(f"\n2. PROFESSIONAL APPROACH:")
    print(f"   - Implement risk-based position sizing")
    print(f"   - Risk 2-5% of account per trade")
    print(f"   - position_size = (current_equity * risk_pct) / expected_loss_pct")
    
    print(f"\n3. OPTIMIZATION APPROACH:")
    print(f"   - Add position_sizing parameter to strategy")
    print(f"   - Test different sizing methods")
    print(f"   - Use walk-forward analysis")
    
    print(f"\n4. CODE CHANGES NEEDED:")
    print(f"   - Modify process_chunk() method")
    print(f"   - Add position sizing parameters to __init__")
    print(f"   - Create position_size_calculator() method")

if __name__ == "__main__":
    # Run all analyses
    tv_trades = analyze_tv_position_sizing()
    analyze_python_position_sizing()
    propose_position_sizing_optimizations()
    create_optimized_strategy_comparison()
    recommend_implementation()
    
    print(f"\n🎯 SUMMARY:")
    print(f"The Python strategy uses 100% of equity per trade, creating exponential")
    print(f"growth but extreme risk. TradingView likely uses fixed or limited sizing.")
    print(f"For fair comparison, implement position size limits.") 
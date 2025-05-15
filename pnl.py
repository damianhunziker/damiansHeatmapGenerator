import plotly.graph_objects as go
from plotly.subplots import make_subplots
from strategy_utils import get_user_inputs, fetch_data, get_strategy_inputs, print_logo, create_performance_chart
from classes.data_fetcher import OHLCFetcher
import numpy as np
from classes.trade_analyzer import TradeAnalyzer
import webbrowser
import os
import pandas as pd

def create_interactive_chart(timeframe_data, strategy_class, strategy_params, last_n_candles_analyze=None, last_n_candles_display=None):
    """Creates an interactive Plotly chart with price movement and PnL curve"""
    # Get date range from strategy parameters
    start_date = strategy_params.get('start_date')
    end_date = strategy_params.get('end_date')
    print(f"\nPNL create_interactive_chart - Using date range: {start_date} to {end_date}")
    
    # Initialize strategy with timeframe data and parameters
    strategy = strategy_class(**strategy_params)
    strategy.timeframe_data = timeframe_data
    
    # Ensure the strategy's divergence detector has the date range
    if hasattr(strategy, 'divergence_detector'):
        strategy.divergence_detector.set_date_range(start_date, end_date)
        print(f"Set date range in divergence detector: {start_date} to {end_date}")
    
    # Create analyzer with strategy and strategy_params
    analyzer = TradeAnalyzer(strategy, strategy_params)
    
    # Get primary timeframe data
    data = timeframe_data['primary']['data']
    
    # Filter data by date range before analysis
    if start_date and end_date:
        data = data[start_date:end_date].copy()
        print(f"Filtered data to {len(data)} candles")
    
    # Analyze data and get trades
    trades, display_data = analyzer.analyze_data(data, last_n_candles_analyze, last_n_candles_display)
    
    # Berechne tägliche Rendite und Buy & Hold Performance
    display_data['Daily Return'] = display_data['price_close'].pct_change()
    display_data['Buy & Hold'] = (1 + display_data['Daily Return']).cumprod()
    
    # Berechne Strategie Performance
    equity_curve = [strategy_params['initial_equity']]
    equity_curve_timestamps = [display_data.index[0]]
    for trade in trades:
        # Handle both old and new trade tuple formats
        net_profit = trade[4] if trade[4] is not None else 0  # Index 4 is net_profit in both formats
        equity_curve.append(equity_curve[-1] + net_profit)
        equity_curve_timestamps.append(trade[1])  # Index 1 is exit_time in both formats
    
    # Konvertiere Equity Curve zu Prozent-Performance
    initial_equity = equity_curve[0]
    equity_performance = [(eq / initial_equity) for eq in equity_curve]
    
    # Buy & Hold resampled auf die Trade-Zeitpunkte
    buy_hold = display_data['Buy & Hold'].reindex(equity_curve_timestamps, method='ffill')
    
    # Berechne Drawdown
    peak = np.maximum.accumulate(equity_performance)
    drawdown = (np.array(equity_performance) - peak) / peak * 100
    
    # Generiere Chart
    fig = create_performance_chart(equity_curve_timestamps, equity_performance, buy_hold, drawdown, 'interactive')
    
    # Berechne und zeige die Kennzahlen
    metrics = analyzer.calculate_metrics(equity_curve, trades)
    
    # Erstelle HTML-Tabelle mit Statistiken
    stats_table = f"""
    <div style="display: flex; justify-content: center;">
        <table style="border-collapse: collapse; border: 1px solid black; font-family: Arial;">
            <tr>
                <th style="padding: 5px;">Metric</th>
                <th style="padding: 5px;">Value</th>
            </tr>
            <tr>
                <td style="padding: 5px;">Net Profit</td>
                <td style="padding: 5px;">${metrics['total_net_profit']:.2f} ({metrics['profit_pct']:.2f}%)</td>
            </tr>
            <tr>
                <td style="padding: 5px;">Number of Closed Trades</td>
                <td style="padding: 5px;">{len([t for t in trades if t[4] is not None])}</td>
            </tr>
            <tr>
                <td style="padding: 5px;">Win Rate</td>
                <td style="padding: 5px;">{metrics['win_rate']:.2f}%</td>
            </tr>
            <tr>
                <td style="padding: 5px;">Profit Factor</td>
                <td style="padding: 5px;">{metrics['profit_factor']:.2f}</td>
            </tr>
            <tr>
                <td style="padding: 5px;">Max Drawdown</td>
                <td style="padding: 5px;">${metrics['max_drawdown_usd']:.2f} ({metrics['max_drawdown_pct']:.2f}%)</td>
            </tr>
            <tr>
                <td style="padding: 5px;">Average Trade</td>
                <td style="padding: 5px;">${metrics['avg_trade_profit']:.2f} ({metrics['avg_trade_profit_pct']:.2f}%)</td>
            </tr>
            <tr>
                <td style="padding: 5px;">Average Trade Duration</td>
                <td style="padding: 5px;">{metrics['avg_trade_duration']:.2f} seconds</td>
            </tr>
        </table>
    </div>
    """
    
    # Add trade list table with exit reasons if available
    trade_rows = []
    for i, trade in enumerate(trades, 1):
        if len(trade) >= 9:  # New format with exit_reason
            entry_time, exit_time, entry_price, exit_price, net_profit, gross_profit, fees, trade_type, exit_reason = trade
        else:  # Old format without exit_reason
            entry_time, exit_time, entry_price, exit_price, net_profit, gross_profit, fees, trade_type = trade
            exit_reason = "N/A"
            
        if net_profit is not None:
            profit_pct = (net_profit / equity_curve[i-1]) * 100
            trade_rows.append(f"""
                <tr>
                    <td style="padding: 5px;">{i}</td>
                    <td style="padding: 5px;">{trade_type}</td>
                    <td style="padding: 5px;">{entry_time.strftime('%Y-%m-%d %H:%M')}</td>
                    <td style="padding: 5px;">{exit_time.strftime('%Y-%m-%d %H:%M')}</td>
                    <td style="padding: 5px;">${entry_price:.2f}</td>
                    <td style="padding: 5px;">${exit_price:.2f}</td>
                    <td style="padding: 5px;">${gross_profit:.2f}</td>
                    <td style="padding: 5px;">${fees:.2f}</td>
                    <td style="padding: 5px;">${net_profit:.2f}</td>
                    <td style="padding: 5px;">{profit_pct:.2f}%</td>
                    <td style="padding: 5px;">{exit_reason}</td>
                </tr>
            """)
    
    trade_table = f"""
    <div style="margin-top: 20px;">
        <table style="border-collapse: collapse; border: 1px solid black; font-family: Arial; margin: 0 auto;">
            <tr>
                <th style="padding: 5px;">#</th>
                <th style="padding: 5px;">Type</th>
                <th style="padding: 5px;">Entry Time</th>
                <th style="padding: 5px;">Exit Time</th>
                <th style="padding: 5px;">Entry</th>
                <th style="padding: 5px;">Exit</th>
                <th style="padding: 5px;">Gross P/L</th>
                <th style="padding: 5px;">Fees</th>
                <th style="padding: 5px;">Net P/L</th>
                <th style="padding: 5px;">%</th>
                <th style="padding: 5px;">Exit Reason</th>
            </tr>
            {"".join(trade_rows)}
        </table>
    </div>
    """
    
    # Speichere und zeige das Ergebnis
    if not os.path.exists("html_cache"):
        os.makedirs("html_cache")
    
    fig.write_html("html_cache/chart.html", full_html=False, include_plotlyjs='cdn')
    
    with open("html_cache/chart.html", "r") as f:
        chart_html = f.read()
    
    html = f"""
    <html>
        <head>
            <title>Strategy Results</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                {stats_table}
                {chart_html}
                {trade_table}
            </div>
        </body>
    </html>
    """
    
    with open("html_cache/results.html", "w") as f:
        f.write(html)
    
    webbrowser.open("file://" + os.path.realpath("html_cache/results.html"))

if __name__ == "__main__":
    print_logo()
   
    print("PNL - Heatmap Generator und Strategie Backtester")
    
    # Get user inputs
    user_inputs = get_user_inputs()
    print(f"\nPNL - User Inputs - start_date: {user_inputs.get('start_date')}, end_date: {user_inputs.get('end_date')}")
    
    # Get strategy-specific parameters
    strategy_params = get_strategy_inputs(user_inputs['strategy_class'])
    strategy_params.update({
        'initial_equity': user_inputs['initial_equity'],
        'fee_pct': user_inputs['fee_pct'],
        'start_date': user_inputs['start_date'],  # Add date range to strategy params
        'end_date': user_inputs['end_date']       # Add date range to strategy params
    })
    print(f"PNL - Strategy Params - start_date: {strategy_params.get('start_date')}, end_date: {strategy_params.get('end_date')}")

    # Create interactive chart
    create_interactive_chart(
        user_inputs['timeframe_data'],  # Pass entire timeframe_data dictionary
        user_inputs['strategy_class'], 
        strategy_params, 
        user_inputs['lookback_candles'], 
        user_inputs['lookback_candles']
    )
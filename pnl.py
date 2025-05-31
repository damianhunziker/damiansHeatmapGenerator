import plotly.graph_objects as go
from plotly.subplots import make_subplots
from strategy_utils import get_user_inputs, fetch_data, get_strategy_inputs, print_logo, create_performance_chart
from classes.data_fetcher import OHLCFetcher
import numpy as np
from classes.trade_analyzer import TradeAnalyzer
import webbrowser
import os
import pandas as pd

def analyze_direction(timeframe_data, strategy_class, strategy_params, direction):
    """Analyze strategy for a specific direction (both, long, short)"""
    # Create strategy params for this direction
    direction_params = strategy_params.copy()
    direction_params['trade_direction'] = direction
    
    # Initialize strategy
    strategy = strategy_class(**direction_params)
    strategy.timeframe_data = timeframe_data
    
    # Ensure the strategy's divergence detector has the date range
    if hasattr(strategy, 'divergence_detector'):
        strategy.divergence_detector.set_date_range(
            strategy_params.get('start_date'), 
            strategy_params.get('end_date')
        )
    
    # Create analyzer
    analyzer = TradeAnalyzer(strategy, direction_params)
    
    # Get and filter data
    data = timeframe_data['primary']['data']
    start_date = strategy_params.get('start_date')
    end_date = strategy_params.get('end_date')
    
    if start_date and end_date:
        data = data[start_date:end_date].copy()
    
    # Analyze data and get trades
    trades, display_data = analyzer.analyze_data(data)
    
    # Calculate equity curve with proper timestamps
    initial_equity = strategy_params['initial_equity']
    equity_curve = [initial_equity]
    equity_curve_timestamps = [display_data.index[0]]
    
    # Add equity points for each completed trade
    for trade in trades:
        if trade[4] is not None:  # Only completed trades
            net_profit = trade[4]
            equity_curve.append(equity_curve[-1] + net_profit)
            equity_curve_timestamps.append(trade[1])  # Exit time
    
    # Calculate performance metrics for the original performance chart
    pnl_performance = [(eq / initial_equity - 1) * 100 for eq in equity_curve]
    
    # Calculate buy & hold performance
    if len(equity_curve_timestamps) > 0:
        # Get price data for the same timestamps
        price_data = []
        for timestamp in equity_curve_timestamps:
            # Find closest timestamp in data
            closest_idx = data.index.get_indexer([timestamp], method='nearest')[0]
            price_data.append(data['price_close'].iloc[closest_idx])
        
        # Calculate buy & hold performance
        initial_price = price_data[0]
        buy_hold = [(price / initial_price - 1) * 100 for price in price_data]
    else:
        buy_hold = []
    
    # Calculate drawdown
    peak = np.maximum.accumulate([x/100 + 1 for x in pnl_performance])
    drawdown = (np.array([x/100 + 1 for x in pnl_performance]) - peak) / peak * 100
    
    # Calculate metrics
    metrics = analyzer.calculate_metrics(equity_curve, trades)
    
    return {
        'trades': trades,
        'metrics': metrics,
        'equity_curve_timestamps': equity_curve_timestamps,
        'pnl_performance': pnl_performance,
        'buy_hold': buy_hold,
        'drawdown': drawdown,
        'direction': direction
    }

def calculate_realistic_direction_metrics(both_results, target_direction):
    """
    Berechnet realistische Metriken für Long/Short Only basierend auf der realen Both-Equity-Kurve
    """
    if not both_results or not both_results['trades']:
        return None
    
    # Filter trades für die gewünschte Richtung aus der Both-Analyse
    direction_trades = []
    for trade in both_results['trades']:
        if len(trade) >= 8:  # Vollständige Trade-Information
            trade_type = trade[7]  # LONG oder SHORT
            if (target_direction == 'long' and trade_type == 'LONG') or \
               (target_direction == 'short' and trade_type == 'SHORT'):
                direction_trades.append(trade)
    
    if not direction_trades:
        return None
    
    # Berechne realistische Equity-Kurve basierend auf der Both-Analyse
    # Starte mit der initialen Equity
    initial_equity = 10000  # Standard Initial Equity
    current_equity = initial_equity
    direction_equity_curve = [initial_equity]
    direction_timestamps = [both_results['equity_curve_timestamps'][0]]
    
    # Gehe durch alle Both-Trades und berücksichtige nur die der gewünschten Richtung
    for trade in both_results['trades']:
        if trade[4] is not None:  # Nur abgeschlossene Trades
            trade_type = trade[7]
            # Wenn dieser Trade zur gewünschten Richtung gehört, füge ihn zur Equity hinzu
            if (target_direction == 'long' and trade_type == 'LONG') or \
               (target_direction == 'short' and trade_type == 'SHORT'):
                net_profit = trade[4]
                current_equity += net_profit
                direction_equity_curve.append(current_equity)
                direction_timestamps.append(trade[1])  # Exit time
    
    # Berechne Performance-Metriken für diese realistische Equity-Kurve
    pnl_performance = [(eq / initial_equity - 1) * 100 for eq in direction_equity_curve]
    
    # Berechne Buy & Hold Performance basierend auf den Both-Results Timestamps
    # Verwende die gleichen Zeitpunkte wie in both_results, aber nur für direction trades
    buy_hold = []
    if len(direction_timestamps) > 0 and len(both_results['buy_hold']) > 0:
        # Erstelle buy_hold basierend auf den direction timestamps
        # Finde die entsprechenden Indizes in both_results für unsere timestamps
        both_timestamps = both_results['equity_curve_timestamps']
        both_buy_hold = both_results['buy_hold']
        
        for direction_timestamp in direction_timestamps:
            # Finde den nächsten entsprechenden Zeitstempel in both_results
            closest_idx = None
            min_diff = None
            for i, both_timestamp in enumerate(both_timestamps):
                if i < len(both_buy_hold):
                    diff = abs((direction_timestamp - both_timestamp).total_seconds())
                    if closest_idx is None or diff < min_diff:
                        closest_idx = i
                        min_diff = diff
            
            if closest_idx is not None and closest_idx < len(both_buy_hold):
                buy_hold.append(both_buy_hold[closest_idx])
            elif len(buy_hold) > 0:
                # Verwende den letzten verfügbaren Wert
                buy_hold.append(buy_hold[-1])
            else:
                # Fallback: Starte bei 0
                buy_hold.append(0.0)
    
    # Fallback: Wenn buy_hold immer noch leer ist, erstelle eine einfache Nulllinie
    if not buy_hold:
        buy_hold = [0.0] * len(direction_timestamps)
    
    # Stelle sicher, dass buy_hold und andere Listen die gleiche Länge haben
    while len(buy_hold) < len(direction_timestamps):
        buy_hold.append(buy_hold[-1] if buy_hold else 0.0)
    while len(buy_hold) > len(direction_timestamps):
        buy_hold.pop()
    
    # Berechne Drawdown basierend auf der realistischen Equity-Kurve
    peak = np.maximum.accumulate([x/100 + 1 for x in pnl_performance])
    drawdown = (np.array([x/100 + 1 for x in pnl_performance]) - peak) / peak * 100
    
    # Berechne Metriken mit einer Mock-Analyzer-Instanz
    class MockAnalyzer:
        def calculate_metrics(self, equity_curve, trades):
            """Calculate various financial metrics from an equity curve and trades."""
            # Initialize variables
            max_equity = equity_curve[0]
            max_drawdown = 0
            max_drawdown_usd = 0
            total_fees = 0
            total_gross_profit = 0
            total_net_profit = 0
            total_trade_duration = 0
            winning_trades = 0
            total_profit = 0
            total_loss = 0

            # Calculate drawdown using the provided formula
            for equity in equity_curve:
                if equity > max_equity:
                    max_equity = equity
                current_drawdown = (max_equity - equity) / max_equity
                current_drawdown_usd = max_equity - equity
                if current_drawdown > max_drawdown:
                    max_drawdown = current_drawdown
                    max_drawdown_usd = current_drawdown_usd
            max_drawdown_pct = max_drawdown * 100

            # Calculate returns
            if len(equity_curve) > 1:
                returns = np.diff(equity_curve) / equity_curve[:-1]
                if len(returns) > 1:
                    returns_mean = np.mean(returns)
                    returns_std = np.std(returns, ddof=1) if len(returns) > 1 else 0
                    sharpe_ratio = returns_mean / returns_std if returns_std != 0 else 0
                    downside_returns = returns[returns < 0]
                    downside_std = np.std(downside_returns, ddof=1) if len(downside_returns) > 1 else 0
                    sortino_ratio = returns_mean / downside_std if downside_std != 0 else 0
                    volatility = returns_std
                else:
                    sharpe_ratio = 0
                    sortino_ratio = 0
                    volatility = 0
            else:
                sharpe_ratio = 0
                sortino_ratio = 0
                volatility = 0

            # Calculate trade-related metrics
            for trade in trades:
                if len(trade) >= 9:
                    entry_time, exit_time, entry_price, exit_price, net_profit, gross_profit, fees, trade_type, exit_reason = trade
                else:
                    entry_time, exit_time, entry_price, exit_price, net_profit, gross_profit, fees, trade_type = trade[:8]
                    
                if net_profit is not None:  # Only consider completed trades
                    total_fees += fees
                    total_gross_profit += gross_profit
                    total_net_profit += net_profit
                    total_trade_duration += (exit_time - entry_time).total_seconds()
                    if net_profit > 0:
                        winning_trades += 1
                        total_profit += net_profit
                    else:
                        total_loss += abs(net_profit)

            num_trades = sum(1 for trade in trades if len(trade) > 4 and trade[4] is not None)  # Count only completed trades
            avg_trade_profit = total_net_profit / num_trades if num_trades > 0 else 0
            avg_trade_profit_pct = (avg_trade_profit / equity_curve[0]) * 100 if equity_curve[0] != 0 else 0
            avg_trade_duration = total_trade_duration / num_trades if num_trades > 0 else 0
            profit_pct = (total_net_profit / equity_curve[0]) * 100 if equity_curve[0] != 0 else 0
            win_rate = (winning_trades / num_trades) * 100 if num_trades > 0 else 0
            profit_factor = total_profit / total_loss if total_loss != 0 else float('inf')

            return {
                'max_drawdown': max_drawdown,
                'max_drawdown_usd': max_drawdown_usd,
                'max_drawdown_pct': max_drawdown_pct,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'volatility': volatility,
                'total_fees': total_fees,
                'total_gross_profit': total_gross_profit,
                'total_net_profit': total_net_profit,
                'avg_trade_profit': avg_trade_profit,
                'avg_trade_profit_pct': avg_trade_profit_pct,
                'avg_trade_duration': avg_trade_duration,
                'profit_pct': profit_pct,
                'win_rate': win_rate,
                'profit_factor': profit_factor
            }
    
    mock_analyzer = MockAnalyzer()
    metrics = mock_analyzer.calculate_metrics(direction_equity_curve, direction_trades)
    
    return {
        'trades': direction_trades,
        'metrics': metrics,
        'equity_curve_timestamps': direction_timestamps,
        'pnl_performance': pnl_performance,
        'buy_hold': buy_hold,
        'drawdown': drawdown,
        'direction': target_direction
    }

def create_metrics_table(results_both, results_long, results_short):
    """Create side-by-side metrics table for available directions"""
    
    def format_metric(value, is_percentage=False, is_currency=False):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "N/A"
        if is_currency:
            return f"${value:.2f}"
        elif is_percentage:
            return f"{value:.2f}%"
        else:
            return f"{value:.2f}"
    
    # Get metrics for available directions
    metrics_both = results_both['metrics'] if results_both else None
    metrics_long = results_long['metrics'] if results_long else None
    metrics_short = results_short['metrics'] if results_short else None
    
    # Determine which columns to show
    show_both = results_both is not None
    show_long = results_long is not None
    show_short = results_short is not None
    
    # Create header row
    header_cols = []
    if show_both:
        header_cols.append('<th style="padding: 10px; border: 1px solid black; text-align: center;">Both</th>')
    if show_long:
        header_cols.append('<th style="padding: 10px; border: 1px solid black; text-align: center;">Long Only</th>')
    if show_short:
        header_cols.append('<th style="padding: 10px; border: 1px solid black; text-align: center;">Short Only</th>')
    
    header_row = f"""
                    <th style="padding: 10px; border: 1px solid black; text-align: left;">Metric</th>
                    {"".join(header_cols)}
    """
    
    # Helper function to create data cells for a row
    def create_data_cells(metric_name, both_value, long_value, short_value, is_percentage=False, is_currency=False, is_count=False):
        cells = []
        if show_both:
            if is_count:
                cells.append(f'<td style="padding: 8px; border: 1px solid black; text-align: center;">{both_value}</td>')
            else:
                cells.append(f'<td style="padding: 8px; border: 1px solid black; text-align: center;">{format_metric(both_value, is_percentage, is_currency)}</td>')
        if show_long:
            if is_count:
                cells.append(f'<td style="padding: 8px; border: 1px solid black; text-align: center;">{long_value}</td>')
            else:
                cells.append(f'<td style="padding: 8px; border: 1px solid black; text-align: center;">{format_metric(long_value, is_percentage, is_currency)}</td>')
        if show_short:
            if is_count:
                cells.append(f'<td style="padding: 8px; border: 1px solid black; text-align: center;">{short_value}</td>')
            else:
                cells.append(f'<td style="padding: 8px; border: 1px solid black; text-align: center;">{format_metric(short_value, is_percentage, is_currency)}</td>')
        return "".join(cells)
    
    # Create table rows
    rows = []
    
    # Net Profit row
    both_net_profit = f"{format_metric(metrics_both['total_net_profit'], is_currency=True)} ({format_metric(metrics_both['profit_pct'], is_percentage=True)})" if metrics_both else "N/A"
    long_net_profit = f"{format_metric(metrics_long['total_net_profit'], is_currency=True)} ({format_metric(metrics_long['profit_pct'], is_percentage=True)})" if metrics_long else "N/A"
    short_net_profit = f"{format_metric(metrics_short['total_net_profit'], is_currency=True)} ({format_metric(metrics_short['profit_pct'], is_percentage=True)})" if metrics_short else "N/A"
    
    rows.append(f"""
        <tr>
            <td style="padding: 8px; border: 1px solid black; font-weight: bold;">Net Profit</td>
            {create_data_cells("Net Profit", both_net_profit, long_net_profit, short_net_profit, is_count=True)}
        </tr>
    """)
    
    # Total Trades row
    both_trades = len([t for t in results_both['trades'] if t[4] is not None]) if results_both else 0
    long_trades = len([t for t in results_long['trades'] if t[4] is not None]) if results_long else 0
    short_trades = len([t for t in results_short['trades'] if t[4] is not None]) if results_short else 0
    
    rows.append(f"""
        <tr style="background-color: #f9f9f9;">
            <td style="padding: 8px; border: 1px solid black; font-weight: bold;">Total Trades</td>
            {create_data_cells("Total Trades", both_trades, long_trades, short_trades, is_count=True)}
        </tr>
    """)
    
    # Win Rate row
    rows.append(f"""
        <tr>
            <td style="padding: 8px; border: 1px solid black; font-weight: bold;">Win Rate</td>
            {create_data_cells("Win Rate", metrics_both['win_rate'] if metrics_both else None, metrics_long['win_rate'] if metrics_long else None, metrics_short['win_rate'] if metrics_short else None, is_percentage=True)}
        </tr>
    """)
    
    # Profit Factor row
    rows.append(f"""
        <tr style="background-color: #f9f9f9;">
            <td style="padding: 8px; border: 1px solid black; font-weight: bold;">Profit Factor</td>
            {create_data_cells("Profit Factor", metrics_both['profit_factor'] if metrics_both else None, metrics_long['profit_factor'] if metrics_long else None, metrics_short['profit_factor'] if metrics_short else None)}
        </tr>
    """)
    
    # Max Drawdown row
    both_drawdown = f"{format_metric(metrics_both['max_drawdown_usd'], is_currency=True)} ({format_metric(metrics_both['max_drawdown_pct'], is_percentage=True)})" if metrics_both else "N/A"
    long_drawdown = f"{format_metric(metrics_long['max_drawdown_usd'], is_currency=True)} ({format_metric(metrics_long['max_drawdown_pct'], is_percentage=True)})" if metrics_long else "N/A"
    short_drawdown = f"{format_metric(metrics_short['max_drawdown_usd'], is_currency=True)} ({format_metric(metrics_short['max_drawdown_pct'], is_percentage=True)})" if metrics_short else "N/A"
    
    rows.append(f"""
        <tr>
            <td style="padding: 8px; border: 1px solid black; font-weight: bold;">Max Drawdown</td>
            {create_data_cells("Max Drawdown", both_drawdown, long_drawdown, short_drawdown, is_count=True)}
        </tr>
    """)
    
    # Average Trade row
    both_avg_trade = f"{format_metric(metrics_both['avg_trade_profit'], is_currency=True)} ({format_metric(metrics_both['avg_trade_profit_pct'], is_percentage=True)})" if metrics_both else "N/A"
    long_avg_trade = f"{format_metric(metrics_long['avg_trade_profit'], is_currency=True)} ({format_metric(metrics_long['avg_trade_profit_pct'], is_percentage=True)})" if metrics_long else "N/A"
    short_avg_trade = f"{format_metric(metrics_short['avg_trade_profit'], is_currency=True)} ({format_metric(metrics_short['avg_trade_profit_pct'], is_percentage=True)})" if metrics_short else "N/A"
    
    rows.append(f"""
        <tr style="background-color: #f9f9f9;">
            <td style="padding: 8px; border: 1px solid black; font-weight: bold;">Average Trade</td>
            {create_data_cells("Average Trade", both_avg_trade, long_avg_trade, short_avg_trade, is_count=True)}
        </tr>
    """)
    
    # Winning Trades row
    both_winning = f"{len([t for t in results_both['trades'] if t[4] is not None and t[4] > 0])} of {len([t for t in results_both['trades'] if t[4] is not None])}" if results_both else "N/A"
    long_winning = f"{len([t for t in results_long['trades'] if t[4] is not None and t[4] > 0])} of {len([t for t in results_long['trades'] if t[4] is not None])}" if results_long else "N/A"
    short_winning = f"{len([t for t in results_short['trades'] if t[4] is not None and t[4] > 0])} of {len([t for t in results_short['trades'] if t[4] is not None])}" if results_short else "N/A"
    
    rows.append(f"""
        <tr>
            <td style="padding: 8px; border: 1px solid black; font-weight: bold;">Winning Trades</td>
            {create_data_cells("Winning Trades", both_winning, long_winning, short_winning, is_count=True)}
        </tr>
    """)
    
    # Average Win row
    both_avg_win = sum([t[4] for t in results_both['trades'] if t[4] is not None and t[4] > 0]) / len([t for t in results_both['trades'] if t[4] is not None and t[4] > 0]) if results_both and len([t for t in results_both['trades'] if t[4] is not None and t[4] > 0]) > 0 else 0
    long_avg_win = sum([t[4] for t in results_long['trades'] if t[4] is not None and t[4] > 0]) / len([t for t in results_long['trades'] if t[4] is not None and t[4] > 0]) if results_long and len([t for t in results_long['trades'] if t[4] is not None and t[4] > 0]) > 0 else 0
    short_avg_win = sum([t[4] for t in results_short['trades'] if t[4] is not None and t[4] > 0]) / len([t for t in results_short['trades'] if t[4] is not None and t[4] > 0]) if results_short and len([t for t in results_short['trades'] if t[4] is not None and t[4] > 0]) > 0 else 0
    
    rows.append(f"""
        <tr style="background-color: #f9f9f9;">
            <td style="padding: 8px; border: 1px solid black; font-weight: bold;">Average Win</td>
            {create_data_cells("Average Win", both_avg_win, long_avg_win, short_avg_win, is_currency=True)}
        </tr>
    """)
    
    # Average Loss row
    both_avg_loss = sum([abs(t[4]) for t in results_both['trades'] if t[4] is not None and t[4] < 0]) / len([t for t in results_both['trades'] if t[4] is not None and t[4] < 0]) if results_both and len([t for t in results_both['trades'] if t[4] is not None and t[4] < 0]) > 0 else 0
    long_avg_loss = sum([abs(t[4]) for t in results_long['trades'] if t[4] is not None and t[4] < 0]) / len([t for t in results_long['trades'] if t[4] is not None and t[4] < 0]) if results_long and len([t for t in results_long['trades'] if t[4] is not None and t[4] < 0]) > 0 else 0
    short_avg_loss = sum([abs(t[4]) for t in results_short['trades'] if t[4] is not None and t[4] < 0]) / len([t for t in results_short['trades'] if t[4] is not None and t[4] < 0]) if results_short and len([t for t in results_short['trades'] if t[4] is not None and t[4] < 0]) > 0 else 0
    
    rows.append(f"""
        <tr>
            <td style="padding: 8px; border: 1px solid black; font-weight: bold;">Average Loss</td>
            {create_data_cells("Average Loss", both_avg_loss, long_avg_loss, short_avg_loss, is_currency=True)}
        </tr>
    """)
    
    table_html = f"""
    <div style="margin: 20px 0;">
        <h2 style="text-align: center;">Strategy Performance Comparison</h2>
        <table style="border-collapse: collapse; border: 1px solid black; font-family: Arial; margin: 0 auto; width: 100%;">
            <thead>
                <tr style="background-color: #f0f0f0;">
                    {header_row}
                </tr>
            </thead>
            <tbody>
                {"".join(rows)}
            </tbody>
        </table>
    </div>
    """
    
    return table_html

def create_trade_list_table(trades, direction="Both"):
    """Create trade list table for specified direction"""
    trade_rows = []
    
    for i, trade in enumerate(trades, 1):
        if len(trade) >= 9:  # New format with exit_reason
            entry_time, exit_time, entry_price, exit_price, net_profit, gross_profit, fees, trade_type, exit_reason = trade
        else:  # Old format without exit_reason
            entry_time, exit_time, entry_price, exit_price, net_profit, gross_profit, fees, trade_type = trade
            exit_reason = "N/A"
            
        if net_profit is not None:
            # Korrekte Prozent-Berechnung: Preisveränderung des Assets
            if trade_type == "LONG":
                profit_pct = ((exit_price - entry_price) / entry_price) * 100
            else:  # SHORT
                profit_pct = ((entry_price - exit_price) / entry_price) * 100
            
            row_style = "background-color: #e8f5e8;" if net_profit > 0 else "background-color: #ffe8e8;" if net_profit < 0 else ""
            
            trade_rows.append(f"""
                <tr style="{row_style}">
                    <td style="padding: 5px; border: 1px solid #ddd; text-align: center;">{i}</td>
                    <td style="padding: 5px; border: 1px solid #ddd; text-align: center;">{trade_type}</td>
                    <td style="padding: 5px; border: 1px solid #ddd; text-align: center;">{entry_time.strftime('%Y-%m-%d %H:%M')}</td>
                    <td style="padding: 5px; border: 1px solid #ddd; text-align: center;">{exit_time.strftime('%Y-%m-%d %H:%M')}</td>
                    <td style="padding: 5px; border: 1px solid #ddd; text-align: center;">${entry_price:.2f}</td>
                    <td style="padding: 5px; border: 1px solid #ddd; text-align: center;">${exit_price:.2f}</td>
                    <td style="padding: 5px; border: 1px solid #ddd; text-align: center;">${gross_profit:.2f}</td>
                    <td style="padding: 5px; border: 1px solid #ddd; text-align: center;">${fees:.2f}</td>
                    <td style="padding: 5px; border: 1px solid #ddd; text-align: center;">${net_profit:.2f}</td>
                    <td style="padding: 5px; border: 1px solid #ddd; text-align: center;">{profit_pct:.2f}%</td>
                    <td style="padding: 5px; border: 1px solid #ddd; text-align: center;">{exit_reason}</td>
                </tr>
            """)
    
    trade_table = f"""
    <div style="margin-top: 30px;">
        <h2 style="text-align: center;">Trade List ({direction} Direction{'s' if direction == 'Both' else ''})</h2>
        <table style="border-collapse: collapse; border: 1px solid black; font-family: Arial; margin: 0 auto; width: 100%;">
            <thead>
                <tr style="background-color: #f0f0f0;">
                    <th style="padding: 8px; border: 1px solid black;">#</th>
                    <th style="padding: 8px; border: 1px solid black;">Type</th>
                    <th style="padding: 8px; border: 1px solid black;">Entry Time</th>
                    <th style="padding: 8px; border: 1px solid black;">Exit Time</th>
                    <th style="padding: 8px; border: 1px solid black;">Entry</th>
                    <th style="padding: 8px; border: 1px solid black;">Exit</th>
                    <th style="padding: 8px; border: 1px solid black;">Gross P/L</th>
                    <th style="padding: 8px; border: 1px solid black;">Fees</th>
                    <th style="padding: 8px; border: 1px solid black;">Net P/L</th>
                    <th style="padding: 8px; border: 1px solid black;">%</th>
                    <th style="padding: 8px; border: 1px solid black;">Exit Reason</th>
                </tr>
            </thead>
            <tbody>
                {"".join(trade_rows)}
            </tbody>
        </table>
    </div>
    """
    
    return trade_table

def create_interactive_chart(timeframe_data, strategy_class, strategy_params, last_n_candles_analyze=None, last_n_candles_display=None):
    """Creates an interactive analysis based on selected trade direction"""
    
    # Get the selected trade direction from strategy parameters
    selected_direction = strategy_params.get('trade_direction', 'both')
    
    print(f"\n=== PnL Analysis: Analyzing '{selected_direction}' direction ===")
    
    # Initialize results variables
    results_both = None
    results_long = None
    results_short = None
    
    # Analyze based on selected direction
    if selected_direction == 'both':
        print("Analyzing 'both' direction...")
        results_both = analyze_direction(timeframe_data, strategy_class, strategy_params, 'both')
        
        print("Calculating realistic 'long' metrics from 'both' analysis...")
        results_long = calculate_realistic_direction_metrics(results_both, 'long')
        
        print("Calculating realistic 'short' metrics from 'both' analysis...")
        results_short = calculate_realistic_direction_metrics(results_both, 'short')
        
    elif selected_direction == 'long':
        print("Analyzing 'long' direction only...")
        results_long = analyze_direction(timeframe_data, strategy_class, strategy_params, 'long')
        
    elif selected_direction == 'short':
        print("Analyzing 'short' direction only...")
        results_short = analyze_direction(timeframe_data, strategy_class, strategy_params, 'short')
    
    # Create performance charts based on analyzed directions
    fig_both = None
    fig_long = None
    fig_short = None
    
    # Get asset name for chart titles
    asset_name = strategy_params.get('asset', 'Asset')
    
    if results_both is not None:
        fig_both = create_performance_chart(
            timestamps=results_both['equity_curve_timestamps'], 
            pnl_performance=results_both['pnl_performance'],
            buy_hold=results_both['buy_hold'],
            drawdown=results_both['drawdown'],
            output_type='interactive',
            params='Both Strategies'
        )
    
    if results_long is not None:
        fig_long = create_performance_chart(
            timestamps=results_long['equity_curve_timestamps'], 
            pnl_performance=results_long['pnl_performance'],
            buy_hold=results_long['buy_hold'],
            drawdown=results_long['drawdown'],
            output_type='interactive',
            params='Long Strategy'
        )
    
    if results_short is not None:
        fig_short = create_performance_chart(
            timestamps=results_short['equity_curve_timestamps'], 
            pnl_performance=results_short['pnl_performance'],
            buy_hold=results_short['buy_hold'],
            drawdown=results_short['drawdown'],
            output_type='interactive',
            params='Short Strategy'
        )
    
    # Create metrics comparison table
    metrics_table = create_metrics_table(results_both, results_long, results_short)
    
    # Create trade list table for the primary direction
    trade_table = ""
    if selected_direction == 'both' and results_both:
        trade_table = create_trade_list_table(results_both['trades'], "Both")
    elif selected_direction == 'long' and results_long:
        trade_table = create_trade_list_table(results_long['trades'], "Long")
    elif selected_direction == 'short' and results_short:
        trade_table = create_trade_list_table(results_short['trades'], "Short")
    
    # Save charts as HTML
    if not os.path.exists("html_cache"):
        os.makedirs("html_cache")
    
    chart_both_html = ""
    chart_long_html = ""
    chart_short_html = ""
    
    if fig_both is not None:
        fig_both.write_html("html_cache/chart_both.html", full_html=False, include_plotlyjs='cdn')
        with open("html_cache/chart_both.html", "r") as f:
            chart_both_html = f.read()
    
    if fig_long is not None:
        fig_long.write_html("html_cache/chart_long.html", full_html=False, include_plotlyjs='cdn')
        with open("html_cache/chart_long.html", "r") as f:
            chart_long_html = f.read()
    
    if fig_short is not None:
        fig_short.write_html("html_cache/chart_short.html", full_html=False, include_plotlyjs='cdn')
        with open("html_cache/chart_short.html", "r") as f:
            chart_short_html = f.read()
    
    # Create final HTML with side-by-side layout
    html = f"""
    <html>
        <head>
            <title>Strategy Performance Analysis</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1400px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }}
                .charts-container {{
                    display: grid;
                    grid-template-columns: {f"1fr" if sum([fig_both is not None, fig_long is not None, fig_short is not None]) == 1 else f"1fr 1fr" if sum([fig_both is not None, fig_long is not None, fig_short is not None]) == 2 else "1fr 1fr 1fr"};
                    gap: 20px;
                    margin: 20px 0;
                }}
                .chart-box {{
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 10px;
                    background-color: #fafafa;
                    height: 600px;
                    overflow: auto;
                }}
                h1 {{
                    text-align: center;
                    color: #333;
                    margin-bottom: 30px;
                }}
                .summary {{
                    background-color: #e8f4fd;
                    padding: 15px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                    border-left: 4px solid #2196F3;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Strategy Performance Analysis</h1>
                
                <div class="summary">
                    <h3>Analysis Summary</h3>
                    <p><strong>Asset:</strong> {asset_name}</p>
                    <p><strong>Strategy:</strong> {strategy_class.__name__}</p>
                    <p><strong>Period:</strong> {strategy_params.get('start_date')} to {strategy_params.get('end_date')}</p>
                    <p><strong>Initial Equity:</strong> ${strategy_params.get('initial_equity', 10000):,.2f}</p>
                    <p><strong>Fee:</strong> {strategy_params.get('fee_pct', 0.04):.2f}%</p>
                </div>
                
                {metrics_table}
                
                <div class="charts-container">
                    {f'<div class="chart-box">{chart_both_html}</div>' if fig_both is not None else ''}
                    {f'<div class="chart-box">{chart_long_html}</div>' if fig_long is not None else ''}
                    {f'<div class="chart-box">{chart_short_html}</div>' if fig_short is not None else ''}
                </div>
                
                {trade_table}
            </div>
        </body>
    </html>
    """
    
    # Save and open final HTML
    with open("html_cache/results.html", "w") as f:
        f.write(html)
    
    print(f"\n=== Analysis Complete ===")
    if results_both:
        print(f"Both Direction: {len([t for t in results_both['trades'] if t[4] is not None])} trades, Net P/L: ${results_both['metrics']['total_net_profit']:.2f}")
    if results_long:
        print(f"Long Only: {len([t for t in results_long['trades'] if t[4] is not None])} trades, Net P/L: ${results_long['metrics']['total_net_profit']:.2f}")
    if results_short:
        print(f"Short Only: {len([t for t in results_short['trades'] if t[4] is not None])} trades, Net P/L: ${results_short['metrics']['total_net_profit']:.2f}")
    
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
        'start_date': user_inputs['start_date'],
        'end_date': user_inputs['end_date'],
        'asset': user_inputs['asset']
    })
    print(f"PNL - Strategy Params - start_date: {strategy_params.get('start_date')}, end_date: {strategy_params.get('end_date')}")

    # Create interactive chart
    create_interactive_chart(
        user_inputs['timeframe_data'],
        user_inputs['strategy_class'], 
        strategy_params, 
        user_inputs['lookback_candles'], 
        user_inputs['lookback_candles']
    )
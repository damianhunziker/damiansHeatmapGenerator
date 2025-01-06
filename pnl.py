import plotly.graph_objects as go
from plotly.subplots import make_subplots
from strategy_utils import get_user_inputs, fetch_data, get_strategy_inputs, print_logo
from classes.data_fetcher import OHLCFetcher
import numpy as np
from classes.trade_analyzer import TradeAnalyzer
import webbrowser
import os

def create_interactive_chart(data, strategy_class, strategy_params, last_n_candles_analyze=None, last_n_candles_display=None, lookback_candles=20):
    """Creates an interactive Plotly chart with price movement and PnL curve"""
    # Initialize Trade Analyzer
    analyzer = TradeAnalyzer(strategy_class, strategy_params)
    trades, display_data = analyzer.analyze_data(data, last_n_candles_analyze, last_n_candles_display)
    
    print("\nCreating interactive chart...")
    
    # Erstelle Plotly Figure
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Berechne Equity-Kurve
    equity_curve = [strategy_params['initial_equity']]
    equity_curve_timestamps = [display_data.index[0]]  # Startzeitpunkt hinzufügen
    for trade in trades:
        equity_curve.append(equity_curve[-1] + trade[4])  # trade[4] ist der Nettogewinn
        equity_curve_timestamps.append(trade[1])  # Endzeitpunkt des Trades hinzufügen
    
    # Debugging: Check equity_curve content
    if len(equity_curve) < 2:
        print("Error: Equity curve has less than 2 data points.")
        return
    
    # Finde den Zeitraum, in dem Trades stattgefunden haben
    start_time = equity_curve_timestamps[0]
    end_time = equity_curve_timestamps[-1]
    
    # Filtere den Preisverlauf auf den Zeitraum der Trades
    price_data = display_data.loc[start_time:end_time]
    
    # Füge Preisverlauf hinzu
    fig.add_trace(go.Scatter(x=price_data.index, y=price_data['price_close'], name='Schlusskurs'), secondary_y=False)
    
    # Füge PnL-Kurve hinzu
    fig.add_trace(go.Scatter(x=equity_curve_timestamps, y=equity_curve, name='PnL'), secondary_y=True)
    
    # Aktualisiere Layout
    fig.update_layout(
        title='Preisverlauf und PnL',
        xaxis_title='Zeit',
        yaxis_title='Preis',
        yaxis2_title='PnL',
        legend=dict(x=0, y=1, orientation='h'),
        hovermode='x unified'
    )
    
    # Berechne und zeige die Kennzahlen
    metrics = analyzer.calculate_metrics(equity_curve, trades)
    print(f"Max Drawdown: {metrics['max_drawdown_pct']}%")
    print(f"Max Drawdown (USD): {metrics['max_drawdown_usd']}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']}")
    print(f"Sortino Ratio: {metrics['sortino_ratio']}")
    print(f"Volatility: {metrics['volatility']}")
    print(f"Total Fees: {metrics['total_fees']}")
    print(f"Total Gross Profit: {metrics['total_gross_profit']}")
    print(f"Total Net Profit: {metrics['total_net_profit']}")
    print(f"Average Trade Profit: {metrics['avg_trade_profit']}")
    print(f"Average Trade Profit (%): {metrics['avg_trade_profit_pct']}%")
    print(f"Average Trade Duration: {metrics['avg_trade_duration']} seconds")
    print(f"Profit (%): {metrics['profit_pct']}%")
    
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
                <td style="padding: 5px;">{len(trades)}</td>
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
    
    # Erstelle den Ordner "html_cache", falls er nicht existiert
    if not os.path.exists("html_cache"):
        os.makedirs("html_cache")
    
    # Speichere das Diagramm als HTML-Datei im Ordner "html_cache"
    fig.write_html("html_cache/chart.html", full_html=False, include_plotlyjs='cdn')
    
    # Lese den Inhalt der chart.html-Datei
    with open("html_cache/chart.html", "r") as f:
        chart_html = f.read()
    
    # Erstelle eine vollständige HTML-Datei mit Statistiktabelle und Diagramm
    html = f"""
    <html>
        <head>
            <title>Strategie-Ergebnisse</title>
        </head>
        <body>
            {stats_table}
            {chart_html}
        </body>
    </html>
    """
    
    # Speichere die vollständige HTML-Datei im Ordner "html_cache"
    with open("html_cache/results.html", "w") as f:
        f.write(html)
    
    # Öffne die HTML-Datei im Standardbrowser
    webbrowser.open("file://" + os.path.realpath("html_cache/results.html"))

if __name__ == "__main__":
    print_logo()
   
    print("PNL - Heatmap Generator und Strategie Backtester")
    
    # Benutzer-Eingaben abrufen
    asset, interval, initial_equity, last_n_candles, lookback_candles, fee_pct, strategy_name, strategy_class = get_user_inputs()
    
    # Strategie-spezifische Parameter abrufen
    strategy_params = get_strategy_inputs(strategy_class)
    strategy_params.update({
        'initial_equity': initial_equity,
        'fee_pct': fee_pct
    })
    
    # Lade Daten
    data = fetch_data(asset, interval)

    # Erstelle interaktives Diagramm
    create_interactive_chart(data, strategy_class, strategy_params, last_n_candles, lookback_candles)
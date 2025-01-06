import plotly.graph_objects as go
from plotly.subplots import make_subplots
from classes.data_fetcher import OHLCFetcher
from classes.trade_analyzer import TradeAnalyzer
from strategy_utils import get_user_inputs, fetch_data, get_strategy_inputs, print_logo
import pandas as pd

def create_interactive_chart(data, strategy_class, strategy_params, last_n_candles_analyze=None, last_n_candles_display=None, lookback_candles=20):
    """Erstellt ein interaktives Chart mit Plotly"""
    # Initialisiere Trade Analyzer
    analyzer = TradeAnalyzer(strategy_class, strategy_params)
    trades, display_data = analyzer.analyze_data(data, last_n_candles_analyze, last_n_candles_display)
    
    print("\nErstelle Chart...")
    # Erstelle Figure mit Subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.03, 
                       row_heights=[0.7, 0.3])
    
    # Candlestick Chart
    fig.add_trace(go.Candlestick(
        x=display_data.index,
        open=display_data['price_open'],
        high=display_data['price_high'],
        low=display_data['price_low'],
        close=display_data['price_close'],
        name='OHLC'
    ), row=1, col=1)
    
    # Füge Trade Entry/Exit Markierungen hinzu
    entry_times = [trade[0] for trade in trades]
    entry_prices = [trade[2] for trade in trades]
    exit_times = [trade[1] for trade in trades]
    exit_prices = [trade[3] for trade in trades]
    trade_types = [trade[7] for trade in trades]
    
    # Calculate the width of one candle
    if data.index.freq is not None:
        candle_width = pd.tseries.frequencies.to_offset(data.index.freq)
    else:
        # Manually calculate the candle width if frequency is not set
        candle_width = data.index[1] - data.index[0]

    for i in range(len(trades)):
        if trade_types[i] == "LONG":
            fig.add_trace(go.Scatter(
                x=[entry_times[i]+ (0.5 * candle_width)],
                y=[entry_prices[i]],
                mode='markers',
                marker=dict(
                    symbol='triangle-right',
                    size=12,
                    color='blue',
                    line=dict(width=1)
                ),
                showlegend=False
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=[exit_times[i] + (2 * candle_width)],
                y=[exit_prices[i]],
                mode='markers',
                marker=dict(
                    symbol='triangle-left',
                    size=12,
                    color='green',
                    line=dict(width=1)
                ),
                showlegend=False
            ), row=1, col=1)
        
        elif trade_types[i] == "SHORT":
            fig.add_trace(go.Scatter(
                x=[entry_times[i] + (0.5 * candle_width)],
                y=[entry_prices[i]],
                mode='markers',
                marker=dict(
                    symbol='triangle-right',
                    size=12,
                    color='red',
                    line=dict(width=1)
                ),
                showlegend=False
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=[exit_times[i] + (2 * candle_width)],
                y=[exit_prices[i]],
                mode='markers',
                marker=dict(
                    symbol='triangle-left',
                    size=12,
                    color='green',
                    line=dict(width=1)
                ),
                showlegend=False
            ), row=1, col=1)
    
    # Füge strategie-spezifische Traces hinzu
    strategy = strategy_class(**strategy_params)
    strategy.add_strategy_traces(fig, display_data)
    
    # Indikatoren Chart
    strategy.add_indicator_traces(fig, display_data, row=2, col=1)
    
    # Layout anpassen
    fig.update_layout(
        title='Price Chart with Signals and Indicators',
        yaxis_title='Price',
        yaxis2_title='Indicators',
        xaxis_rangeslider_visible=False
    )
    
    # Y-Achsen-Format anpassen
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Indicators", row=2, col=1)
    
    # Chart anzeigen
    fig.show()
    
    return fig

if __name__ == "__main__":
    print_logo()
   
    print("CHART ANALYSE - Heatmap Generator and Strategy Backtester")
    
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
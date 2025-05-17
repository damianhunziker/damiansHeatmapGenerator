from tqdm import tqdm
import sys
import os
import pandas as pd
from strategy_utils import print_logo, get_user_inputs, get_strategy_inputs, fetch_data, get_available_strategies
from classes.trade_analyzer import TradeAnalyzer
from plotly.subplots import make_subplots
import plotly.graph_objects as go

print("Initializing Chart Analysis...")

def create_interactive_chart(timeframe_data, strategy_class, strategy_params, last_n_candles_analyze, last_n_candles_display):
    """Creates an interactive Plotly chart with price movement and PnL curve"""
    print("\n=== Debug: Strategy Initialization ===")
    # Initialize strategy with timeframe data
    strategy = strategy_class(**strategy_params)
    strategy.timeframe_data = timeframe_data  # Add timeframe data to strategy
    print(f"Strategy class: {strategy.__class__.__name__}")
    print(f"Strategy params: {strategy_params}")
    
    # Create analyzer with strategy and strategy_params
    analyzer = TradeAnalyzer(strategy, strategy_params)
    
    # Get primary timeframe data
    data = timeframe_data['primary']['data']
    print(f"\nData shape: {data.shape}")
    print(f"Data columns: {data.columns.tolist()}")
    
    # Get date range from strategy parameters
    start_date = strategy_params.get('start_date')
    end_date = strategy_params.get('end_date')
    print(f"Date range: {start_date} to {end_date}")
    
    # If the strategy uses DivergenceDetector, set the date range
    if hasattr(strategy, 'divergence_detector'):
        strategy.divergence_detector.set_date_range(start_date, end_date)
    
    print("\n=== Debug: Data Analysis ===")
    # Analyze data and get trades
    trades, display_data = analyzer.analyze_data(data, last_n_candles_analyze, last_n_candles_display)
    print(f"Display data shape: {display_data.shape}")
    print(f"Display data columns: {display_data.columns.tolist()}")
    
    # Check KAMA values
    if 'entry_kama' in display_data.columns:
        print("\n=== Debug: KAMA Values ===")
        print(f"entry_kama first 5 values: {display_data['entry_kama'].head()}")
        print(f"entry_kama last 5 values: {display_data['entry_kama'].tail()}")
        print(f"entry_kama NaN count: {display_data['entry_kama'].isna().sum()}")
    else:
        print("\nWARNING: entry_kama not found in display_data columns!")
    
    if 'kama2' in display_data.columns:
        print(f"\nkama2 first 5 values: {display_data['kama2'].head()}")
        print(f"kama2 last 5 values: {display_data['kama2'].tail()}")
        print(f"kama2 NaN count: {display_data['kama2'].isna().sum()}")
    else:
        print("\nWARNING: kama2 not found in display_data columns!")
    
    print(f"\nNumber of trades found: {len(trades)}")
    
    # Calculate equity curve for metrics
    equity_curve = [strategy_params['initial_equity']]
    for trade in trades:
        if trade[4] is not None:  # if net_profit is not None
            equity_curve.append(equity_curve[-1] + trade[4])
    
    # Calculate and print metrics
    metrics = analyzer.calculate_metrics(equity_curve, trades)
    
    print("\n=== Performance Metrics ===")
    print(f"Total Net Profit: ${metrics['total_net_profit']:.2f} ({metrics['profit_pct']:.2f}%)")
    print(f"Total Gross Profit: ${metrics['total_gross_profit']:.2f}")
    print(f"Total Fees: ${metrics['total_fees']:.2f}")
    print(f"Win Rate: {metrics['win_rate']:.2f}%")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"Average Trade Profit: ${metrics['avg_trade_profit']:.2f} ({metrics['avg_trade_profit_pct']:.2f}%)")
    print(f"Maximum Drawdown: ${metrics['max_drawdown_usd']:.2f} ({metrics['max_drawdown_pct']:.2f}%)")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
    print(f"Volatility: {metrics['volatility']:.2f}")
    
    print("\n=== Trade List ===")
    analyzer._print_trade_statistics(trades)
    
    print("\n=== Debug: Chart Creation ===")
    # Count available indicators for subplot layout
    available_indicators = [ind for ind in strategy.indicators if ind in display_data.columns]
    num_indicators = len(available_indicators)
    print(f"Available indicators: {available_indicators}")
    print(f"Number of indicators: {num_indicators}")
    
    # Calculate total number of rows needed (price chart + 3 KAMA delta charts + one row per indicator)
    total_rows = 1 + 3 + num_indicators  # Changed from 2 to 3 KAMA delta charts
    print(f"Total subplot rows: {total_rows}")
    
    # Calculate row heights
    row_heights = [0.4]  # Price chart
    kama_delta_height = 0.3 / 3  # Divide 30% among 3 KAMA delta charts
    row_heights.extend([kama_delta_height] * 3)  # Three KAMA delta charts
    if num_indicators > 0:
        indicator_height = 0.3 / num_indicators
        row_heights.extend([indicator_height] * num_indicators)
    print(f"Row heights: {row_heights}")
    
    print("\nCreating subplots...")
    # Create initial subplot layout
    fig = make_subplots(
        rows=total_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights
    )
    
    print("\nAdding candlestick chart...")
    # Candlestick Chart
    fig.add_trace(go.Candlestick(
        x=display_data.index,
        open=display_data['price_open'],
        high=display_data['price_high'],
        low=display_data['price_low'],
        close=display_data['price_close'],
        name='OHLC'
    ), row=1, col=1)
    
    print("\nAdding trade markers...")
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
    
    print("\nAdding strategy traces...")
    # Füge strategie-spezifische Traces hinzu
    strategy.add_strategy_traces(fig, display_data)
    
    print("\nAdding indicator traces...")
    # Indikatoren Chart (starting from row 2)
    strategy.add_indicator_traces(fig, display_data, row=2, col=1)
    
    print("\nUpdating layout...")
    # Layout anpassen
    fig.update_layout(
        title='Price Chart with Signals and Indicators',
        yaxis_title='Price',
        height=300 * total_rows,
        xaxis_rangeslider_visible=False
    )
    
    # Y-Achsen-Format anpassen
    fig.update_yaxes(title_text="Price", row=1, col=1)
    
    print("\nShowing chart...")
    # Chart anzeigen
    fig.show()
    
    return fig

def create_chart(timeframe_data, params):
    """Creates a chart using the provided timeframe data and parameters."""
    # Get strategy class
    strategies = get_available_strategies()
    strategy_found = False
    
    for _, (name, strategy_class) in strategies.items():
        if name == params['strategy']:
            strategy_found = True
            # Create strategy parameters
            strategy_params = {
                'initial_equity': params.get('initial_equity', 10000),
                'fee_pct': params.get('fee_pct', 0.04),
                'start_date': params['start_date'],
                'end_date': params['end_date']
            }
            
            # Add any strategy-specific parameters
            if hasattr(strategy_class, 'get_parameters'):
                for param_name, (default_value, _) in strategy_class.get_parameters().items():
                    strategy_params[param_name] = params.get(param_name, default_value)
            
            # Create chart
            create_interactive_chart(
                timeframe_data=timeframe_data,
                strategy_class=strategy_class,
                strategy_params=strategy_params,
                last_n_candles_analyze=None,
                last_n_candles_display=None
            )
            break
    
    if not strategy_found:
        print(f"Error: Strategy {params['strategy']} not found")

if __name__ == "__main__":
    print_logo()
    print("CHART ANALYSE - Heatmap Generator and Strategy Backtester")
    
    # Get user inputs
    user_inputs = get_user_inputs()
    
    # Get strategy-specific parameters
    strategy_params = get_strategy_inputs(user_inputs['strategy_class'])
    strategy_params.update({
        'initial_equity': user_inputs['initial_equity'],
        'fee_pct': user_inputs['fee_pct'],
        'start_date': user_inputs['start_date'],
        'end_date': user_inputs['end_date']
    })

    # Create interactive chart
    create_interactive_chart(
        user_inputs['timeframe_data'],
        user_inputs['strategy_class'], 
        strategy_params, 
        user_inputs['lookback_candles'], 
        user_inputs['lookback_candles']
    ) 
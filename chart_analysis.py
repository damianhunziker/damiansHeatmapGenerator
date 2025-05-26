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
    
    # Print trade statistics
    print("\n=== Trade Statistics ===")
    print(f"Total number of trades: {len(trades)}")
    
    if len(trades) > 0:
        # Calculate basic metrics
        initial_equity = strategy_params.get('initial_equity', 10000)
        current_equity = initial_equity
        winning_trades = 0
        total_profit = 0
        total_fees = 0
        
        print("\nTrade List:")
        print("=" * 140)
        print(f"{'#':3} | {'Type':<6} | {'Entry Time':<19} | {'Exit Time':<19} | {'Entry':<10} | {'Exit':<10} | "
              f"{'Gross P/L':<10} | {'Fees':<8} | {'Net P/L':<10} | {'%':<8} | {'Exit Reason':<25}")
        print("=" * 140)
        
        for i, trade in enumerate(trades, 1):
            # Safely unpack trade tuple, handling potential extra values
            entry_time = trade[0]
            exit_time = trade[1]
            entry_price = trade[2]
            exit_price = trade[3]
            net_profit = trade[4]
            gross_profit = trade[5]
            fees = trade[6]
            trade_type = trade[7]
            exit_reason = trade[8] if len(trade) > 8 else ''  # Get exit reason from tuple
            
            if net_profit is not None:
                profit_pct = (net_profit / current_equity) * 100
                current_equity += net_profit
                total_profit += net_profit
                total_fees += fees
                if net_profit > 0:
                    winning_trades += 1
                
                print(f"{i:3d} | {trade_type:<6} | {entry_time.strftime('%Y-%m-%d %H:%M'):<19} | "
                      f"{exit_time.strftime('%Y-%m-%d %H:%M'):<19} | ${entry_price:<10.2f} | "
                      f"${exit_price:<10.2f} | ${gross_profit:<10.2f} | ${fees:<8.2f} | "
                      f"${net_profit:<10.2f} | {profit_pct:8.2f}% | {exit_reason:<25}")
        
        print("=" * 140)
        
        # Calculate and print summary statistics
        completed_trades = sum(1 for t in trades if t[4] is not None)
        win_rate = (winning_trades / completed_trades * 100) if completed_trades > 0 else 0
        total_return = ((current_equity - initial_equity) / initial_equity * 100)
        
        print("\nSummary Statistics:")
        print(f"Initial Equity: ${initial_equity:,.2f}")
        print(f"Final Equity: ${current_equity:,.2f}")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Total Profit: ${total_profit:,.2f}")
        print(f"Total Fees: ${total_fees:,.2f}")
        print(f"Number of Trades: {completed_trades}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Average Profit per Trade: ${(total_profit/completed_trades):,.2f}" if completed_trades > 0 else "No completed trades")
    else:
        print("\nNo trades found in the analyzed period.")
    
    # --- Divergence subplot logic ---
    # Collect all divergence indicators from both regular and short divergence results
    divergence_indicators = set()
    if hasattr(strategy, 'divergence_results'):
        divergence_indicators.update(strategy.divergence_results.keys())
    if hasattr(strategy, 'short_divergence_results'):
        divergence_indicators.update(strategy.short_divergence_results.keys())
    
    # Remove SSL and DEMA from divergence indicators
    divergence_indicators = [ind for ind in divergence_indicators if ind not in ['ssl', 'dema']]
    
    # Special handling for ADP - add it twice if it exists
    if 'adp' in divergence_indicators:
        divergence_indicators.remove('adp')
        divergence_indicators = ['adp_long', 'adp_short'] + list(divergence_indicators)
    else:
        divergence_indicators = list(divergence_indicators)
    
    print(f"Divergence indicators: {divergence_indicators}")
    
    # Calculate total number of rows needed (price chart + 2 KAMA delta charts + one row per divergence indicator)
    total_rows = 1 + 2 + len(divergence_indicators)
    print(f"Total subplot rows: {total_rows}")
    
    # Calculate row heights
    row_heights = [0.4]  # Price chart
    kama_delta_height = 0.3 / 2  # Divide 30% among 2 KAMA delta charts
    row_heights.extend([kama_delta_height] * 2)  # Two KAMA delta charts
    if len(divergence_indicators) > 0:
        indicator_height = 0.3 / len(divergence_indicators)
        row_heights.extend([indicator_height] * len(divergence_indicators))
    print(f"Row heights: {row_heights}")
    
    print("\nCreating subplots...")
    # Create initial subplot layout
    fig = make_subplots(
        rows=total_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        subplot_titles=(['Price'] + ['Entry KAMA Delta', 'Exit KAMA Delta'] + 
                       [f"{ind.split('_')[0]} (Long)" if ind.endswith('_long') else 
                        f"{ind.split('_')[0]} (Short)" if ind.endswith('_short') else
                        f"{ind} (Long)" if ind in strategy.indicators else f"{ind} (Short)" 
                        for ind in divergence_indicators])
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
    
    # Add SSL and DEMA to main price chart
    if 'ssl_up' in display_data.columns:
        fig.add_trace(go.Scatter(
            x=display_data.index,
            y=display_data['ssl_up'],
            name='SSL Up',
            line=dict(color='green', width=1)
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=display_data.index,
            y=display_data['ssl_down'],
            name='SSL Down',
            line=dict(color='red', width=1)
        ), row=1, col=1)
    
    if 'dema' in display_data.columns:
        fig.add_trace(go.Scatter(
            x=display_data.index,
            y=display_data['dema'],
            name='DEMA',
            line=dict(color='purple', width=1)
        ), row=1, col=1)
    
    print("\nAdding trade markers...")
    # Add Trade Entry/Exit markers
    entry_times = [trade[0] for trade in trades]
    entry_prices = [trade[2] for trade in trades]
    exit_times = [trade[1] for trade in trades]
    exit_prices = [trade[3] for trade in trades]
    trade_types = [trade[7] for trade in trades]
    exit_reasons = [trade[8] if len(trade) > 8 else '' for trade in trades]  # Get exit reasons
    
    # Calculate the width of one candle
    if data.index.freq is not None:
        candle_width = pd.tseries.frequencies.to_offset(data.index.freq)
    else:
        # Manually calculate the candle width if frequency is not set
        candle_width = data.index[1] - data.index[0]

    # Convert timestamps to pandas Timestamps if they aren't already
    entry_times = [pd.Timestamp(t) if not isinstance(t, pd.Timestamp) else t for t in entry_times]
    exit_times = [pd.Timestamp(t) if not isinstance(t, pd.Timestamp) else t for t in exit_times]
    
    for i in range(len(trades)):
        if trade_types[i] == "LONG":
            # Entry marker
            fig.add_trace(go.Scatter(
                x=[entry_times[i] + candle_width/2],
                y=[entry_prices[i]],
                mode='markers',
                marker=dict(
                    symbol='triangle-right',
                    size=12,
                    color='blue',
                    line=dict(width=1)
                ),
                name='Long Entry',
                text=[f'Long Entry<br>Price: ${entry_prices[i]:.2f}'],
                hoverinfo='text',
                showlegend=False
            ), row=1, col=1)
            
            # Exit marker with tooltip
            fig.add_trace(go.Scatter(
                x=[exit_times[i] + candle_width],
                y=[exit_prices[i]],
                mode='markers',
                marker=dict(
                    symbol='triangle-left',
                    size=12,
                    color='green',
                    line=dict(width=1)
                ),
                name='Long Exit',
                text=[f'Long Exit<br>Price: ${exit_prices[i]:.2f}<br>Reason: {exit_reasons[i]}'],
                hoverinfo='text',
                showlegend=False
            ), row=1, col=1)
        
        elif trade_types[i] == "SHORT":
            # Entry marker
            fig.add_trace(go.Scatter(
                x=[entry_times[i] + candle_width/2],
                y=[entry_prices[i]],
                mode='markers',
                marker=dict(
                    symbol='triangle-right',
                    size=12,
                    color='red',
                    line=dict(width=1)
                ),
                name='Short Entry',
                text=[f'Short Entry<br>Price: ${entry_prices[i]:.2f}'],
                hoverinfo='text',
                showlegend=False
            ), row=1, col=1)
            
            # Exit marker with tooltip
            fig.add_trace(go.Scatter(
                x=[exit_times[i] + candle_width],
                y=[exit_prices[i]],
                mode='markers',
                marker=dict(
                    symbol='triangle-left',
                    size=12,
                    color='green',
                    line=dict(width=1)
                ),
                name='Short Exit',
                text=[f'Short Exit<br>Price: ${exit_prices[i]:.2f}<br>Reason: {exit_reasons[i]}'],
                hoverinfo='text',
                showlegend=False
            ), row=1, col=1)
    
    print("\nAdding strategy traces...")
    # Add strategy-specific traces
    strategy.add_strategy_traces(fig, display_data)
    
    print("\nAdding divergence lines to price chart...")
    # Add divergence lines to price chart
    for indicator in divergence_indicators:
        # Handle special case for ADP
        if indicator == 'adp_long':
            base_indicator = 'adp'
            is_long = True
        elif indicator == 'adp_short':
            base_indicator = 'adp'
            is_long = False
        else:
            base_indicator = indicator
            is_long = indicator in strategy.indicators
        
        # Add long profile divergences to price chart - only BEARISH divergences
        if is_long and hasattr(strategy, 'divergence_results') and base_indicator in strategy.divergence_results:
            for div in strategy.divergence_results[base_indicator]:
                if div['type'] == 'bearish':  # Long profile shows bearish divergences for exit signals
                    prev_time = strategy.divergence_detector.df.index[div['prev_price_idx']]
                    current_time = strategy.divergence_detector.df.index[div['price_idx']]
                    if prev_time not in display_data.index or current_time not in display_data.index:
                        continue
                    prev_price = display_data['price_high'].loc[prev_time]  # Use high for bearish
                    curr_price = display_data['price_high'].loc[current_time]
                    line_color = 'darkred' if div['status'] == 'confirmed' else 'pink'
                    
                    # Add line with hover text
                    fig.add_trace(go.Scatter(
                        x=[prev_time, current_time],
                        y=[prev_price, curr_price],
                        mode='lines',
                        line=dict(color=line_color, width=2, dash='dash'),
                        name=f"Bearish Divergence ({base_indicator})",
                        text=[f"Long profile {div['type']} divergence<br>Indicator: {base_indicator}<br>Status: {div['status']}"],
                        hoverinfo='text',
                        showlegend=False
                    ), row=1, col=1)
        
        # Add short profile divergences to price chart - only BULLISH divergences
        if not is_long and hasattr(strategy, 'short_divergence_results') and base_indicator in strategy.short_divergence_results:
            for div in strategy.short_divergence_results[base_indicator]:
                if div['type'] == 'bullish':  # Short profile shows bullish divergences for exit signals
                    prev_time = strategy.short_divergence_detector.df.index[div['prev_price_idx']]
                    current_time = strategy.short_divergence_detector.df.index[div['price_idx']]
                    if prev_time not in display_data.index or current_time not in display_data.index:
                        continue
                    prev_price = display_data['price_low'].loc[prev_time]  # Use low for bullish
                    curr_price = display_data['price_low'].loc[current_time]
                    line_color = 'darkgreen' if div['status'] == 'confirmed' else 'lightgreen'
                    
                    # Add line with hover text
                    fig.add_trace(go.Scatter(
                        x=[prev_time, current_time],
                        y=[prev_price, curr_price],
                        mode='lines',
                        line=dict(color=line_color, width=2, dash='dash'),
                        name=f"Bullish Divergence ({base_indicator})",
                        text=[f"Short profile {div['type']} divergence<br>Indicator: {base_indicator}<br>Status: {div['status']}"],
                        hoverinfo='text',
                        showlegend=False
                    ), row=1, col=1)
    
    print("\nAdding KAMA delta traces...")
    # Add KAMA delta traces (rows 2 and 3)
    # Entry KAMA Delta
    fig.add_trace(go.Scatter(
        x=display_data.index,
        y=display_data['entry_kama_delta'],
        name='Entry KAMA Delta',
        line=dict(color='blue', width=1)
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=display_data.index,
        y=display_data['entry_kama_delta_limit'],
        name='Entry KAMA Limit',
        line=dict(color='red', width=1, dash='dash')
    ), row=2, col=1)
    
    # Exit KAMA Delta
    fig.add_trace(go.Scatter(
        x=display_data.index,
        y=display_data['exit_kama_delta'],
        name='Exit KAMA Delta',
        line=dict(color='blue', width=1)
    ), row=3, col=1)
    
    fig.add_trace(go.Scatter(
        x=display_data.index,
        y=display_data['exit_kama_delta_limit'],
        name='Exit KAMA Limit',
        line=dict(color='red', width=1, dash='dash')
    ), row=3, col=1)

    print("\nAdding divergence indicator traces...")
    # --- Add divergence indicator subplots ---
    for idx, indicator in enumerate(divergence_indicators):
        subplot_row = 4 + idx  # Start after KAMA delta plots (1 price + 2 KAMA deltas + idx)
        
        # Handle special case for ADP
        if indicator == 'adp_long':
            base_indicator = 'adp'
            is_long = True
        elif indicator == 'adp_short':
            base_indicator = 'adp'
            is_long = False
        else:
            base_indicator = indicator
            is_long = indicator in strategy.indicators
        
        # Plot the indicator values
        if base_indicator in display_data.columns:
            # Add indicator line with appropriate label
            fig.add_trace(go.Scatter(
                x=display_data.index,
                y=display_data[base_indicator],
                name=f"{base_indicator} ({'Long' if is_long else 'Short'})",
                line=dict(width=1)
            ), row=subplot_row, col=1)
            
            # Add divergence lines based on whether this is long or short
            if is_long and hasattr(strategy, 'divergence_results') and base_indicator in strategy.divergence_results:
                # Long profile divergences - show only BEARISH divergences (for exit signals)
                for div in strategy.divergence_results[base_indicator]:
                    if div['type'] == 'bearish':  # Only show bearish for long profile
                        prev_time = strategy.divergence_detector.df.index[div['prev_price_idx']]
                        current_time = strategy.divergence_detector.df.index[div['price_idx']]
                        if prev_time not in display_data.index or current_time not in display_data.index:
                            continue
                        prev_val = display_data[base_indicator].loc[prev_time]
                        curr_val = display_data[base_indicator].loc[current_time]
                        line_color = 'darkred' if div['status'] == 'confirmed' else 'pink'
                        fig.add_shape(
                            type='line',
                            x0=prev_time,
                            y0=prev_val,
                            x1=current_time,
                            y1=curr_val,
                            line=dict(color=line_color, width=2, dash='dash'),
                            row=subplot_row, col=1
                        )
            
            if not is_long and hasattr(strategy, 'short_divergence_results') and base_indicator in strategy.short_divergence_results:
                # Short profile divergences - show only BULLISH divergences (for exit signals)
                for div in strategy.short_divergence_results[base_indicator]:
                    if div['type'] == 'bullish':  # Only show bullish for short profile
                        prev_time = strategy.short_divergence_detector.df.index[div['prev_price_idx']]
                        current_time = strategy.short_divergence_detector.df.index[div['price_idx']]
                        if prev_time not in display_data.index or current_time not in display_data.index:
                            continue
                        prev_val = display_data[base_indicator].loc[prev_time]
                        curr_val = display_data[base_indicator].loc[current_time]
                        line_color = 'darkgreen' if div['status'] == 'confirmed' else 'lightgreen'
                        fig.add_shape(
                            type='line',
                            x0=prev_time,
                            y0=prev_val,
                            x1=current_time,
                            y1=curr_val,
                            line=dict(color=line_color, width=2, dash='dash'),
                            row=subplot_row, col=1
                        )
    
    print("\nUpdating layout...")
    # Update layout
    fig.update_layout(
        title='Price Chart with Signals and Indicators',
        yaxis_title='Price',
        height=300 * total_rows,
        xaxis_rangeslider_visible=False
    )
    
    # Update y-axis titles
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Entry KAMA Delta", row=2, col=1)
    fig.update_yaxes(title_text="Exit KAMA Delta", row=3, col=1)
    
    # Update y-axis titles for divergence indicators
    for idx, indicator in enumerate(divergence_indicators):
        subplot_row = 4 + idx
        # Get base indicator name without the _long/_short suffix
        base_indicator = indicator.split('_')[0] if '_' in indicator else indicator
        indicator_type = 'Long' if indicator.endswith('_long') or indicator in strategy.indicators else 'Short'
        fig.update_yaxes(title_text=f"{base_indicator} ({indicator_type})", row=subplot_row, col=1)
    
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
                'end_date': params['end_date'],
                'trade_direction': params.get('trade_direction', 'short')  # Add trade_direction parameter
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
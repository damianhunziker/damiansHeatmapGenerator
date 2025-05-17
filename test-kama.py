import pandas as pd
import numpy as np
import json
from classes.strategies.kama_strategy import KAMAStrategy
from classes.data_fetcher import OHLCFetcher
from datetime import datetime, timezone, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import argparse

def pine_change(series, length=1):
    """Implement Pine Script's ta.change() function."""
    result = series - series.shift(length)
    result.iloc[:length] = 0
    return result

def pine_sum(series, length):
    """Implement Pine Script's math.sum() function."""
    series = series.fillna(0)
    return series.rolling(window=length, min_periods=1).sum()

def load_tv_data(json_str=None):
    """Load TradingView data from file or JSON string."""
    try:
        if json_str:
            data = json.loads(json_str)
        else:
            with open('TVBtc4hkama16-4-24', 'r') as file:
                data = json.load(file)
        
        # Extract timestamps and values
        timestamps = [datetime.fromtimestamp(ts/1000, tz=timezone.utc) for ts, _ in data['data']]
        values = [value for _, value in data['data']]
        
        # Create DataFrame
        df = pd.DataFrame({
            'price_close': values,
            'kama_tv': values  # Rename to kama_tv for clarity
        }, index=timestamps)
        
        # Shift the TradingView KAMA data two candles to the right to align with our calculation
        df['kama_tv'] = df['kama_tv'].shift(2)
        
        print(f"\nLoaded {len(values)} data points from TradingView")
        print("Date range:", timestamps[0], "to", timestamps[-1])
        
        return df
    except Exception as e:
        print(f"Error loading JSON data: {e}")
        return None

def pine_stdev(series, length):
    """Implement Pine Script's standard deviation calculation."""
    # Pine's stdev uses population standard deviation (ddof=0)
    # and handles NaN values differently
    return series.rolling(window=length, min_periods=1).std(ddof=0)

def pine_abs_sum(series, length):
    """Implement Pine Script's math.sum(math.abs()) combination."""
    return series.abs().rolling(window=length, min_periods=1).sum()

def calculate_kama(data, length=16, fast_length=4, slow_length=24, src='price_close'):
    """Calculate KAMA with proper warmup period."""
    df = data.copy()
    
    # Convert to float if needed
    if df[src].dtype == 'object':
        df[src] = df[src].astype(float)
    
    # Calculate momentum (absolute price change over length period)
    price_change = df[src] - df[src].shift(length)
    mom = price_change.abs()
    mom = mom.fillna(0)  # Fill NaN values with 0
    
    # Calculate volatility
    volatility = df[src].diff().abs().rolling(window=length, min_periods=1).sum()
    volatility = volatility.fillna(0)  # Fill NaN values with 0
    
    # Calculate efficiency ratio (ER)
    er = pd.Series(0.0, index=df.index)
    mask = volatility > 0
    er[mask] = (mom[mask] / volatility[mask]).clip(0, 1)
    
    # Calculate smoothing constant (SC)
    fast_alpha = 2 / (fast_length + 1)
    slow_alpha = 2 / (slow_length + 1)
    sc = np.power(er * (fast_alpha - slow_alpha) + slow_alpha, 2)
    
    # Calculate KAMA
    kama = pd.Series(index=df.index, dtype=float)
    kama.iloc[0] = df[src].iloc[0]
    
    for i in range(1, len(df)):
        kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (df[src].iloc[i] - kama.iloc[i-1])
    
    return kama

def plot_kama_comparison(comparison_df):
    """Create an interactive plot comparing price and KAMA values."""
    fig = make_subplots(rows=2, cols=1, 
                       shared_xaxes=True,
                       vertical_spacing=0.03,
                       row_heights=[0.7, 0.3])

    # Add price and KAMA lines to top subplot
    fig.add_trace(
        go.Scatter(x=comparison_df.index, y=comparison_df['price'],
                  name='Price', line=dict(color='gray', width=1)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=comparison_df.index, y=comparison_df['tv_kama'],
                  name='TV KAMA', line=dict(color='blue', width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=comparison_df.index, y=comparison_df['our_kama'],
                  name='Our KAMA', line=dict(color='red', width=2, dash='dot')),
        row=1, col=1
    )

    # Add difference plot to bottom subplot
    fig.add_trace(
        go.Scatter(x=comparison_df.index, y=comparison_df['diff_pct'],
                  name='Difference %', line=dict(color='orange', width=1),
                  fill='tozeroy'),
        row=2, col=1
    )

    # Update layout
    date_range = f"{comparison_df.index.min().strftime('%Y-%m-%d %H:%M')} to {comparison_df.index.max().strftime('%Y-%m-%d %H:%M')}"
    fig.update_layout(
        title=f'KAMA Comparison with TradingView<br><sub>{date_range}</sub>',
        xaxis_title='Date',
        yaxis_title='Price',
        yaxis2_title='Difference %',
        height=800,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    # Update y-axes labels
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Difference %", row=2, col=1)

    # Show the plot
    fig.show()

def plot_comparison(comparison_df):
    """Create a plot comparing TradingView KAMA with our KAMA implementation."""
    # Calculate differences
    comparison_df['diff'] = abs(comparison_df['kama_tv'] - comparison_df['our_kama'])
    comparison_df['diff_pct'] = (comparison_df['diff'] / comparison_df['kama_tv']) * 100

    # Create figure with secondary y-axis
    fig = make_subplots(rows=2, cols=1,
                       shared_xaxes=True,
                       vertical_spacing=0.03,
                       row_heights=[0.7, 0.3])

    # Add price and KAMA lines to top subplot
    fig.add_trace(
        go.Scatter(x=comparison_df.index, y=comparison_df['price'],
                  name='ETH Price', line=dict(color='gray', width=1)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=comparison_df.index, y=comparison_df['kama_tv'],
                  name='TV KAMA', line=dict(color='blue', width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=comparison_df.index, y=comparison_df['our_kama'],
                  name='Our KAMA', line=dict(color='red', width=2, dash='dot')),
        row=1, col=1
    )

    # Add difference plot to bottom subplot
    fig.add_trace(
        go.Scatter(x=comparison_df.index, y=comparison_df['diff_pct'],
                  name='Difference %', line=dict(color='orange', width=1),
                  fill='tozeroy'),
        row=2, col=1
    )

    # Update layout
    date_range = f"{comparison_df.index.min().strftime('%Y-%m-%d %H:%M')} to {comparison_df.index.max().strftime('%Y-%m-%d %H:%M')}"
    fig.update_layout(
        title=f'KAMA Comparison: TradingView vs Our Implementation<br><sub>{date_range}</sub>',
        xaxis_title='Date',
        yaxis_title='Price (USDT)',
        yaxis2_title='Difference %',
        height=800,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    # Update y-axes labels
    fig.update_yaxes(title_text="Price (USDT)", row=1, col=1)
    fig.update_yaxes(title_text="Difference %", row=2, col=1)

    # Show the plot
    fig.show()

def test_kama_calculation():
    print("\nStarting KAMA calculation test...")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test KAMA calculation')
    parser.add_argument('--plot', action='store_true', help='Show interactive plot')
    parser.add_argument('--json-data', type=str, help='JSON string containing TradingView data')
    args = parser.parse_args()
    
    # Load TradingView data
    tv_data = load_tv_data(args.json_data if args.json_data else None)
    if tv_data is None:
        print("Failed to load TradingView data.")
        return
    
    # Get historical data for warmup
    print("\nFetching historical Bitcoin data for warmup...")
    fetcher = OHLCFetcher()
    
    # Calculate start date (1000 4h candles before TV data start)
    tv_start = tv_data.index[0]
    warmup_start = tv_start - timedelta(hours=4 * 1000)
    
    # Fetch historical data
    historical_data = fetcher.fetch_data(
        asset="ETHUSDT",
        interval="4h",
        start_date=warmup_start.strftime('%Y-%m-%d'),
        limit=2000  # Ensure we get enough data
    )
    
    if historical_data is None:
        print("Failed to fetch historical data.")
        return
    
    print(f"Fetched {len(historical_data)} historical data points")
    print("Historical date range:", historical_data.index[0], "to", historical_data.index[-1])
    
    # Convert historical data index to UTC
    historical_data.index = historical_data.index.tz_localize('UTC')
    
    # Calculate KAMA on full historical data
    print("\nCalculating KAMA with warmup period...")
    historical_kama = calculate_kama(
        data=historical_data,
        length=16,
        fast_length=4,
        slow_length=24,
        src='price_close'
    )
    
    # Debug output
    print("\nSample of historical KAMA values:")
    print(historical_kama[-10:])
    
    print("\nTV Data timestamps:")
    print(tv_data.index)
    print("\nHistorical Data timestamps (sample):")
    print(historical_data.index[-10:])
    
    # Create comparison DataFrame by resampling historical data to match TV data
    comparison = pd.DataFrame(index=tv_data.index)
    comparison['price'] = historical_data['price_close']
    comparison['kama_tv'] = tv_data['kama_tv']
    comparison['our_kama'] = historical_kama
    
    # Debug output
    print("\nComparison Data:")
    print(comparison)
    
    # Print non-NaN values
    print("\nNon-NaN values in comparison:")
    print(comparison.dropna())
    
    # Calculate differences for non-NaN values
    valid_mask = ~comparison['our_kama'].isna()
    if valid_mask.any():
        valid_comparison = comparison[valid_mask]
        valid_comparison['diff'] = abs(valid_comparison['kama_tv'] - valid_comparison['our_kama'])
        valid_comparison['diff_pct'] = (valid_comparison['diff'] / valid_comparison['kama_tv']) * 100
        
        print("\nDifference statistics (non-NaN values):")
        print(f"Mean absolute difference: {valid_comparison['diff'].mean():.8f}")
        print(f"Max absolute difference: {valid_comparison['diff'].max():.8f}")
        print(f"Mean percentage difference: {valid_comparison['diff_pct'].mean():.8f}%")
        print(f"Max percentage difference: {valid_comparison['diff_pct'].max():.8f}%")
    else:
        print("\nNo valid comparison data available.")
    
    # Show plot if requested
    if args.plot:
        plot_comparison(comparison)

if __name__ == "__main__":
    test_kama_calculation() 
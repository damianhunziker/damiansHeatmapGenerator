import pandas as pd
import numpy as np
import json
from classes.strategies.kama_strategy import KAMAStrategy
from classes.data_fetcher import OHLCFetcher
from datetime import datetime, timezone, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import argparse

def load_tv_data():
    # Load the TradingView price and KAMA values
    base_date = datetime(2024, 3, 1, 0, 0)
    
    # Create synthetic price data that would result in these KAMA values
    prices = [
        2364.62, 2365.73, 2367.29, 2374.15, 2383.51,
        2388.33, 2395.95, 2400.74, 2403.77, 2406.93,
        2409.42, 2411.22, 2412.52, 2415.00, 2417.12,
        2419.54, 2421.95, 2424.46, 2426.43, 2430.57  # Updated last price
    ]
    
    # TradingView KAMA values from JSON data
    kama_values = [
        2364.623946279, 2365.7333258806, 2367.2905937362, 2374.1455089988, 2383.5051055709,
        2388.3326911944, 2395.9485977621, 2400.7372826081, 2403.7735643326, 2406.9305672517,
        2409.4181765235, 2411.2204829458, 2412.5209602587, 2415.0002976797, 2417.1226604396,
        2419.5440862925, 2421.9516743376, 2424.462468047, 2426.4278618928, 2430.5728103391
    ]
    
    # Create warmup data - 1000 candles before the actual data
    # Using a simple random walk for the warmup period
    warmup_prices = []
    start_price = prices[0]  # Start from the first actual price
    current_price = start_price
    
    # Generate more realistic warmup data
    np.random.seed(42)  # For reproducibility
    volatility = np.std([p2 - p1 for p1, p2 in zip(prices[:-1], prices[1:])])
    
    for _ in range(1000):
        # Random walk with volatility matching the actual data
        step = np.random.normal(0, volatility)
        current_price += step
        warmup_prices.append(current_price)
    
    # Reverse the warmup prices so they lead up to the actual data
    warmup_prices.reverse()
    
    # Ensure smooth transition to actual data
    # Adjust last few warmup prices to smoothly approach the first actual price
    transition_length = 10
    for i in range(transition_length):
        weight = i / transition_length
        warmup_prices[-i-1] = warmup_prices[-i-1] * (1 - weight) + start_price * weight
    
    # Combine warmup and actual prices
    all_prices = warmup_prices + prices
    
    # Create timestamps for all data
    all_dates = [(base_date - timedelta(hours=4*(1000-i))) for i in range(1000)] + \
                [base_date + timedelta(hours=4*i) for i in range(len(prices))]
    
    # Create DataFrame with all data
    df = pd.DataFrame({
        'price_close': all_prices,
        'kama': [np.nan] * 1000 + kama_values  # NaN for warmup period
    }, index=all_dates)
    
    return df

def load_tv_data_from_json(json_str):
    """Load TradingView data from a JSON string."""
    try:
        # Clean up the JSON string - it might contain multiple concatenated JSON objects
        # Find all JSON objects in the string
        json_objects = []
        depth = 0
        start = 0
        for i, char in enumerate(json_str):
            if char == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    json_objects.append(json_str[start:i+1])
        
        # Parse each JSON object and combine data
        all_data = []
        for json_obj in json_objects:
            try:
                data = json.loads(json_obj)
                all_data.extend(data['data'])
            except:
                continue
        
        # Remove duplicates while preserving order
        seen = set()
        unique_data = []
        for item in all_data:
            item_tuple = tuple(item)  # Convert to tuple for hashability
            if item_tuple not in seen:
                seen.add(item_tuple)
                unique_data.append(item)
        
        # Sort by timestamp
        unique_data.sort(key=lambda x: x[0])
        
        # Extract timestamps and KAMA values
        timestamps = [entry[0] for entry in unique_data]
        kama_values = [entry[1] for entry in unique_data]
        
        # Convert timestamps to datetime
        dates = [datetime.fromtimestamp(ts/1000, tz=timezone.utc) for ts in timestamps]
        
        # Create synthetic prices close to KAMA values
        # This is just an approximation since we don't have actual prices
        prices = [kama + np.random.normal(0, 0.1) for kama in kama_values]
        
        # Create DataFrame
        df = pd.DataFrame({
            'price_close': prices,
            'kama': kama_values
        }, index=dates)
        
        print(f"\nLoaded {len(df)} unique data points from TradingView")
        print("Date range:", df.index.min(), "to", df.index.max())
        
        return df
    except Exception as e:
        print(f"Error loading JSON data: {e}")
        return None

def debug_kama_calculation(data, length, fast_length, slow_length, src='price_close'):
    """Debug version of KAMA calculation that prints intermediate values"""
    df = data.copy()
    
    # Convert Decimal to float if needed
    if df[src].dtype == 'object':  # Decimal values are stored as object dtype
        df[src] = df[src].astype(float)
    
    # Calculate period-by-period changes
    changes = df[src].diff()
    abs_changes = abs(changes)
    
    # Calculate momentum (absolute price change over length period)
    # Pine: mom = math.abs(ta.change(src, length))
    mom = abs(df[src] - df[src].shift(length))
    
    # Calculate volatility as sum of absolute changes
    # Pine: volatility = math.sum(math.abs(ta.change(src)), length)
    volatility = pd.Series(np.nan, index=df.index)
    
    # Calculate rolling sum for volatility, matching Pine Script's behavior
    for i in range(length, len(df)):
        # Sum absolute changes for the current window
        window_changes = abs_changes.iloc[i-length+1:i+1]
        volatility.iloc[i] = window_changes.sum()
    
    # Calculate efficiency ratio with proper zero handling
    # Pine: er = volatility != 0 ? mom / volatility : 0
    er = pd.Series(0.0, index=df.index)
    mask = (volatility != 0) & (~pd.isna(volatility)) & (~pd.isna(mom))
    er[mask] = mom[mask] / volatility[mask]
    
    # Calculate smoothing constant
    # Pine: sc = math.pow(er * (fastAlpha - slowAlpha) + slowAlpha, 2)
    fast_alpha = 2 / (fast_length + 1)
    slow_alpha = 2 / (slow_length + 1)
    sc = np.power(er * (fast_alpha - slow_alpha) + slow_alpha, 2)
    
    # Initialize KAMA series
    kama = pd.Series(index=df.index, dtype=float)
    
    # Special handling for initialization period
    for i in range(len(df)):
        if i == 0:
            # Initialize first value with price
            kama.iloc[i] = df[src].iloc[i]
        elif i < length:
            # During initialization period, use a weighted approach
            weight = i / length
            alpha = weight * fast_alpha + (1 - weight) * slow_alpha
            sc_init = alpha * alpha
            kama.iloc[i] = sc_init * df[src].iloc[i] + (1 - sc_init) * kama.iloc[i-1]
        else:
            # Normal KAMA calculation after initialization
            if pd.isna(sc.iloc[i]):
                kama.iloc[i] = kama.iloc[i-1]
            else:
                kama.iloc[i] = sc.iloc[i] * df[src].iloc[i] + (1 - sc.iloc[i]) * kama.iloc[i-1]
    
    # Print diagnostic information
    print("\nInitialization period details:")
    for i in range(min(5, len(df))):
        print(f"\nBar {i}:")
        print(f"Price: {df[src].iloc[i]:.6f}")
        if i == 0:
            print("Initial value")
        else:
            weight = min(i / length, 1.0)
            alpha = weight * fast_alpha + (1 - weight) * slow_alpha
            print(f"Weight: {weight:.6f}")
            print(f"Alpha: {alpha:.6f}")
            print(f"SC: {alpha * alpha:.6f}")
        print(f"KAMA: {kama.iloc[i]:.6f}")
    
    print("\nFinal period details:")
    for i in range(max(0, len(df)-5), len(df)):
        print(f"\nBar {i}:")
        print(f"Price: {df[src].iloc[i]:.6f}")
        print(f"ER: {er.iloc[i]:.6f}")
        print(f"SC: {sc.iloc[i]:.6f}")
        print(f"KAMA: {kama.iloc[i]:.6f}")
    
    return kama

def plot_kama_comparison(comparison_df, warmup_df):
    """Create an interactive plot comparing price and KAMA values."""
    # Only plot the non-warmup period
    plot_df = comparison_df.copy()
    
    fig = make_subplots(rows=2, cols=1, 
                       shared_xaxes=True,
                       vertical_spacing=0.03,
                       row_heights=[0.7, 0.3])

    # Add price and KAMA lines to top subplot
    fig.add_trace(
        go.Scatter(x=plot_df.index, y=plot_df['price'],
                  name='Price', line=dict(color='gray', width=1)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=plot_df.index, y=plot_df['tv_kama'],
                  name='TV KAMA', line=dict(color='blue', width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=plot_df.index, y=plot_df['our_kama'],
                  name='Our KAMA', line=dict(color='red', width=2, dash='dot')),
        row=1, col=1
    )

    # Add difference plot to bottom subplot
    fig.add_trace(
        go.Scatter(x=plot_df.index, y=plot_df['diff_pct'],
                  name='Difference %', line=dict(color='orange', width=1),
                  fill='tozeroy'),
        row=2, col=1
    )

    # Update layout
    date_range = f"{plot_df.index.min().strftime('%Y-%m-%d %H:%M')} to {plot_df.index.max().strftime('%Y-%m-%d %H:%M')}"
    fig.update_layout(
        title=f'KAMA Comparison with TradingView (with 1000-candle warmup)<br><sub>{date_range}</sub>',
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

def test_kama_calculation():
    print("\nStarting KAMA calculation test...")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test KAMA calculation')
    parser.add_argument('--plot', action='store_true', help='Show interactive plot')
    parser.add_argument('--json-data', type=str, help='JSON string containing TradingView data')
    args = parser.parse_args()
    
    # Load TradingView data
    if args.json_data:
        tv_data_original = load_tv_data_from_json(args.json_data)
        if tv_data_original is None:
            print("Failed to load JSON data. Using default data.")
            full_data = load_tv_data()
        else:
            # Create warmup data
            warmup_length = 1000
            first_price = tv_data_original['price_close'].iloc[0]
            
            # Generate warmup prices
            np.random.seed(42)
            volatility = tv_data_original['price_close'].diff().std()
            
            warmup_prices = []
            current_price = first_price
            for _ in range(warmup_length):
                step = np.random.normal(0, volatility)
                current_price += step
                warmup_prices.append(current_price)
            
            warmup_prices.reverse()
            
            # Create warmup DataFrame
            warmup_dates = [(tv_data_original.index[0] - timedelta(hours=4*(warmup_length-i))) 
                          for i in range(warmup_length)]
            warmup_df = pd.DataFrame({
                'price_close': warmup_prices,
                'kama': [np.nan] * warmup_length
            }, index=warmup_dates)
            
            # Combine warmup and actual data
            full_data = pd.concat([warmup_df, tv_data_original])
    else:
        full_data = load_tv_data()
    
    # Split data into warmup and actual periods
    warmup_data = full_data.iloc[:1000]
    tv_data = full_data.iloc[1000:]
    
    print("\nData Summary:")
    print("Warmup period length:", len(warmup_data))
    print("Actual data length:", len(tv_data))
    print("Total data points:", len(full_data))
    print("\nTime range:")
    print("Start:", tv_data.index.min())
    print("End:", tv_data.index.max())
    
    # Calculate KAMA with debug output using all data
    our_kama = debug_kama_calculation(
        data=full_data,
        length=16,
        fast_length=4,
        slow_length=24
    )
    
    # Create comparison DataFrame (only for the actual period)
    comparison = pd.DataFrame({
        'tv_kama': tv_data['kama'],
        'our_kama': our_kama[1000:],  # Skip warmup period
        'price': tv_data['price_close']
    }).dropna()
    
    print("\nComparison of KAMA values (excluding warmup period):")
    print(comparison)
    
    # Calculate differences
    comparison['diff'] = abs(comparison['tv_kama'] - comparison['our_kama'])
    comparison['diff_pct'] = (comparison['diff'] / comparison['tv_kama']) * 100
    
    print("\nDifference statistics:")
    print(f"Mean absolute difference: {comparison['diff'].mean():.8f}")
    print(f"Max absolute difference: {comparison['diff'].max():.8f}")
    print(f"Mean percentage difference: {comparison['diff_pct'].mean():.8f}%")
    print(f"Max percentage difference: {comparison['diff_pct'].max():.8f}%")
    
    # Show plot if requested
    if args.plot:
        plot_kama_comparison(comparison, warmup_data)

if __name__ == "__main__":
    test_kama_calculation() 
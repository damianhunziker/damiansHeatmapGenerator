import altair as alt
import pandas as pd
import numpy as np
from core.strategy_utils import get_user_inputs, fetch_data, get_parameter_ranges, print_logo, create_performance_chart, instantiate_strategy
from core.html_viewer import publish_file, print_viewer_info
from classes.trade_analyzer import TradeAnalyzer
import itertools
from tqdm import tqdm
import multiprocessing
import os
import altair_saver
import matplotlib.pyplot as plt

def analyze_strategy(params, timeframe_data, strategy_class, initial_equity, fee_pct, last_n_candles_analyze, last_n_candles_display, param_ranges, start_date=None, end_date=None):
    """Analyze strategy with given parameters"""
    # Initialize strategy with parameters
    strategy_params = dict(zip(param_ranges.keys(), params))
    strategy_params.update({
        'initial_equity': initial_equity,
        'fee_pct': fee_pct
    })
    
    # Initialize strategy with timeframe data
    strategy = instantiate_strategy(strategy_class, strategy_params)
    strategy.timeframe_data = timeframe_data
    
    # Create analyzer with strategy
    analyzer = TradeAnalyzer(strategy, strategy_params)
    
    # Get primary timeframe data
    data = timeframe_data['primary']['data']
    
    # Analyze data and get trades
    trades, _ = analyzer.analyze_data(data, last_n_candles_analyze, last_n_candles_display)

    # Calculate metrics using the TradeAnalyzer
    equity_curve = [initial_equity]
    trade_timestamps = []
    
    for trade in trades:
        equity_curve.append(equity_curve[-1] + trade[4])
        trade_timestamps.append(trade[1])

    metrics = analyzer.calculate_metrics(equity_curve, trades)
    
    # Replace zero drawdowns with max drawdown
    max_dd = metrics['max_drawdown_pct']
    drawdown_pct = max_dd if max_dd > 0 else 0

    return {
        'profit_pct': metrics['profit_pct'],
        'total_net_profit': metrics['total_net_profit'],
        'num_trades': len(trades),
        'win_rate': metrics['win_rate'],
        'profit_factor': metrics['profit_factor'],
        'max_drawdown': metrics['max_drawdown'],
        'drawdown_pct': drawdown_pct,
        'avg_trade_profit': metrics['avg_trade_profit'],
        'avg_trade_profit_pct': metrics['avg_trade_profit_pct'],
        'avg_trade_duration': metrics['avg_trade_duration'],
        'sharpe_ratio': metrics['sharpe_ratio'],
        'sortino_ratio': metrics['sortino_ratio'],
        'volatility': metrics['volatility'],
        'equity_curve': equity_curve,
        'trade_timestamps': trade_timestamps,
    }

def analyze_strategy_wrapper(args):
    return analyze_strategy(*args)

def _pool_map(func, args_list, workers, desc):
    """Run ``func`` over ``args_list``, serially when ``workers == 1``."""
    if workers == 1:
        return [func(a) for a in tqdm(args_list, desc=desc)]
    with multiprocessing.Pool(workers) as pool:
        return list(tqdm(pool.imap(func, args_list), total=len(args_list), desc=desc))

def create_tooltip_fields(param_ranges):
    """Helper function to create common tooltip fields."""
    return [
        alt.Tooltip('profit:Q', title='**Profit %**', format='.2f'),
        alt.Tooltip('sharpe_ratio:Q', title='**Sharpe Ratio**', format='.2f'),
        alt.Tooltip('drawdown_pct:Q', title='**Max Drawdown %**', format='.2f'),
        alt.Tooltip('x:O', title=list(param_ranges.keys())[0]),
        alt.Tooltip('y:O', title=list(param_ranges.keys())[1]),
        alt.Tooltip('net_profit:Q', title='Net Profit $', format=',.2f'),
        alt.Tooltip('num_trades:Q', title='Number of Trades'),
        alt.Tooltip('win_rate:Q', title='Win Rate', format='.2%'),
        alt.Tooltip('profit_factor:Q', title='Profit Factor', format='.2f'),
        alt.Tooltip('drawdown:Q', title='Max Drawdown $', format=',.2f'),
        alt.Tooltip('avg_trade_profit:Q', title='Avg Trade Profit $', format=',.2f'),
        alt.Tooltip('avg_trade_profit_pct:Q', title='Avg Trade Profit %', format='.2f'),
        alt.Tooltip('avg_trade_duration:Q', title='Avg Trade Duration'),
        alt.Tooltip('volatility:Q', title='Volatility', format='.2f'),
        alt.Tooltip('sortino_ratio:Q', title='Sortino Ratio'),
        alt.Tooltip('pnl_image:N', title='PnL Image')
    ]

def generate_pnl_image(args):
    """Helper function to generate performance comparison chart"""
    idx, params, equity_curve, full_price_data, start_idx, trade_timestamps, data = args
    
    # Check if there are any trades
    if not trade_timestamps:
        # Return a default image path with updated directory structure
        image_path = os.path.join('html_cache/pnl_images', f'no_trades_{idx}.png')
        plt.figure()
        plt.text(0.5, 0.5, 'No trades in selected period', 
                horizontalalignment='center', verticalalignment='center')
        plt.savefig(image_path)
        plt.close()
        return idx, image_path
    
    # Konvertiere die Zeitstempel in DataFrame-Indizes
    df = pd.DataFrame({'price': full_price_data})
    df.index = data.index  # Verwende den gleichen Index wie das originale Dataframe
    
    # Hole die Positionen der Zeitstempel im Index
    trade_indices = [df.index.get_loc(ts) for ts in trade_timestamps]

    
    # Berechne prozentuale Änderungen für PnL
    initial_equity = equity_curve[0]
    pnl_performance = [(eq/initial_equity - 1) * 100 for eq in equity_curve]  # In Prozent
    
    # Hole die entsprechenden Preise zu den Trade-Indizes
    prices_at_trades = [full_price_data[idx] for idx in trade_indices]
    
    # Berechne Buy & Hold Performance basierend auf den Trade-Zeitpunkten
    initial_price = prices_at_trades[0]
    buy_hold = [(price/initial_price - 1) * 100 for price in prices_at_trades]  # In Prozent
    
    # Stelle sicher, dass alle Listen die gleiche Länge haben
    min_len = min(len(trade_timestamps), len(pnl_performance))
    trade_timestamps = trade_timestamps[:min_len]
    pnl_performance = pnl_performance[:min_len]
    buy_hold = buy_hold[:min_len]
    
    # Berechne Drawdown (bleibt in Prozent)
    peak = np.maximum.accumulate([x/100 + 1 for x in pnl_performance])
    drawdown = (np.array([x/100 + 1 for x in pnl_performance]) - peak) / peak * 100

    
    # Generiere Chart als Bild mit den originalen Zeitstempeln
    image_path = create_performance_chart(
        timestamps=trade_timestamps,
        pnl_performance=pnl_performance,
        buy_hold=buy_hold,
        drawdown=drawdown,
        output_type='image',
        params=idx,
        base_dir='html_cache/pnl_images'
    )
    
    return idx, image_path

def _build_structured_results(df, param_ranges):
    """Build the JSON-friendly grid, best-cells and robustness payload."""
    param_keys = list(param_ranges.keys())
    grid = df.to_dict(orient='records')

    def _top(metric, n=10, ascending=False):
        ranked = df.sort_values(metric, ascending=ascending).head(n)
        return [
            {'x': row['x'], 'y': row['y'], metric: row[metric], 'num_trades': row['num_trades']}
            for _, row in ranked.iterrows()
        ]

    best = {
        'by_profit': _top('profit'),
        'by_net_profit': _top('net_profit'),
        'by_sharpe': _top('sharpe_ratio'),
        'by_drawdown': _top('drawdown_pct', ascending=True),
    }

    robustness = None
    try:
        pivot = df.pivot(index='y', columns='x', values='profit')
        xs = list(pivot.columns)
        ys = list(pivot.index)
        robustness = []
        for iy, y in enumerate(ys):
            for ix, x in enumerate(xs):
                values = []
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        jy, jx = iy + dy, ix + dx
                        if 0 <= jy < len(ys) and 0 <= jx < len(xs):
                            val = pivot.iloc[jy, jx]
                            if pd.notna(val):
                                values.append(float(val))
                robustness.append({
                    'x': x,
                    'y': y,
                    'profit': float(pivot.iloc[iy, ix]) if pd.notna(pivot.iloc[iy, ix]) else None,
                    'neighbor_mean': float(np.mean(values)) if values else None,
                    'neighbor_min': float(np.min(values)) if values else None,
                    'neighbor_count': len(values),
                })
    except Exception:
        robustness = None

    return {
        'grid': grid,
        'best': best,
        'robustness': robustness,
        'param_ranges': {k: np.asarray(v).tolist() for k, v in param_ranges.items()},
        'x_param': param_keys[0] if param_keys else None,
        'y_param': param_keys[1] if len(param_keys) > 1 else None,
    }


def create_heatmap(timeframe_data, strategy_class, param_ranges, initial_equity, fee_pct, last_n_candles_analyze, last_n_candles_display, interval, asset, strategy_name, start_date=None, end_date=None, workers=None, no_images=False):
    """Creates a heatmap of strategy results for different parameter combinations.

    Returns a dict with the parameter grid (one entry per combination), the
    best cells and a neighbourhood-robustness summary, plus the artifact path.
    """
    print("\nCreating Heatmap...")
    
    # Ensure directories exist with correct structure
    os.makedirs("html_cache/pnl_images", exist_ok=True)
    
    # Print start and end dates
    print(f"\nAnalysis Period:")
    print(f"Start Date: {start_date}")
    print(f"End Date: {end_date}")
    
    # Get primary timeframe data
    data = timeframe_data['primary']['data']
    
    # Generate all possible parameter combinations
    param_combinations = list(itertools.product(*param_ranges.values()))
    
    print(f"\nThere are {len(param_combinations)} possible parameter combinations.")

    args_list = [(params, timeframe_data, strategy_class, initial_equity, fee_pct, 
                 last_n_candles_analyze, last_n_candles_display, param_ranges, 
                 start_date, end_date) for params in param_combinations]
    results = _pool_map(analyze_strategy_wrapper, args_list, workers, "Analyzing strategies")
    
    # Ensure the pnl_cache directory exists
    os.makedirs('pnl_cache', exist_ok=True)

    # Prepare arguments for parallel image generation
    image_args = [
        (idx, params, result['equity_curve'], data['price_close'].values, len(data) - len(result['equity_curve']), result['trade_timestamps'], data)
        for idx, (params, result) in enumerate(zip(param_combinations, results))
    ]
    
    # Generate images in parallel (skipped in API mode via no_images)
    if no_images:
        image_paths = {}
    else:
        print("\nGenerating PnL images...")
        image_results = _pool_map(generate_pnl_image, image_args, workers, "Creating PnL images")
        # Update image paths to use relative path from html directory
        image_paths = {idx: f"pnl_images/{os.path.basename(path)}" for idx, path in image_results}
    
    # Create HTML image elements
    image_elements = ""
    for idx, image_path in image_paths.items():  # Changed from .values() to .items()
        image_elements += f'<img id="pnl-img-{idx}" src="{image_path}" alt="PnL Image" style="display:none;" width="70%">\n'

    # Create a DataFrame to store profits and parameters
    results_data = []
    for idx, (params, result) in enumerate(zip(param_combinations, results)):
        results_data.append({
            'x': params[0],
            'y': params[1],
            'profit': result['profit_pct'],
            'net_profit': result['total_net_profit'],
            'num_trades': result['num_trades'],
            'win_rate': result['win_rate'],
            'profit_factor': result['profit_factor'],
            'drawdown': result['max_drawdown'],
            'drawdown_pct': result['drawdown_pct'],
            'avg_trade_profit': result['avg_trade_profit'],
            'avg_trade_profit_pct': result['avg_trade_profit_pct'],
            'avg_trade_duration': result['avg_trade_duration'],
            'sharpe_ratio': result['sharpe_ratio'],
            'sortino_ratio': result['sortino_ratio'],
            'volatility': result['volatility'],
            'pnl_image': f'pnl-img-{idx}' if idx in image_paths else ''  # Use the current index
        })

    df = pd.DataFrame(results_data)

    # Structured results for the machine-readable API (see core/api.py)
    structured = _build_structured_results(df, param_ranges)

    # Determine color scale domains (using all combinations)
    profit_domain = [df['profit'].min(), df['profit'].max()]
    sharpe_domain = [df['sharpe_ratio'].min(), df['sharpe_ratio'].max()]
    drawdown_domain = [df['drawdown_pct'].min(), df['drawdown_pct'].max()]

    # Create heatmap with Altair for Profit
    profit_heatmap = alt.Chart(df).mark_rect().encode(
        x=alt.X('x:O', title=list(param_ranges.keys())[0], axis=alt.Axis(format=".3f")),
        y=alt.Y('y:O', title=list(param_ranges.keys())[1], axis=alt.Axis(format=".3f")),
        color=alt.Color('profit:Q', 
                       scale=alt.Scale(domain=profit_domain, scheme='viridis'), 
                       title='Profit %'),
        tooltip=create_tooltip_fields(param_ranges)
    ).properties(
        title=f'Strategy Results Heatmap: {strategy_name} - Profit',
        width=300,
        height=400
    )

    # Create heatmap with Altair for Sharpe Ratio
    sharpe_heatmap = alt.Chart(df).mark_rect().encode(
        x=alt.X('x:O', title=list(param_ranges.keys())[0], axis=alt.Axis(format=".3f")),
        y=alt.Y('y:O', title=list(param_ranges.keys())[1], axis=alt.Axis(format=".3f")),
        color=alt.Color('sharpe_ratio:Q', 
                       scale=alt.Scale(domain=sharpe_domain, scheme='plasma'), 
                       title='Sharpe Ratio'),
        tooltip=create_tooltip_fields(param_ranges)
    ).properties(
        title=f'Strategy Results Heatmap: {strategy_name} - Sharpe Ratio',
        width=300,
        height=400
    )

    # Create heatmap with Altair for Max Drawdown
    drawdown_heatmap = alt.Chart(df).mark_rect().encode(
        x=alt.X('x:O', title=list(param_ranges.keys())[0], axis=alt.Axis(format=".3f")),
        y=alt.Y('y:O', title=list(param_ranges.keys())[1], axis=alt.Axis(format=".3f")),
        color=alt.Color('drawdown_pct:Q', 
                       scale=alt.Scale(
                           domain=drawdown_domain,
                           scheme='inferno',
                           reverse=True
                       ),
                       title='Max Drawdown %'),
        tooltip=create_tooltip_fields(param_ranges)
    ).properties(
        title=f'Strategy Results Heatmap: {strategy_name} - Max Drawdown',
        width=300,
        height=400
    )

    # Combine the three heatmaps side by side
    combined_heatmap = alt.hconcat(profit_heatmap, sharpe_heatmap, drawdown_heatmap).resolve_scale(
        color='independent'
    )

    # Create the html directory if it doesn't exist
    os.makedirs('html_cache', exist_ok=True)

    # Construct the file name using the strategy name and min/max values of the first two parameter ranges
    param_keys = list(param_ranges.keys())
    param1_name, param2_name = param_keys[0], param_keys[1]
    param1_min, param1_max = min(param_ranges[param1_name]), max(param_ranges[param1_name])
    param2_min, param2_max = min(param_ranges[param2_name]), max(param_ranges[param2_name])
    file_name = f"{strategy_name}_{asset}_{interval}_{param1_name}_{param1_min}-{param1_max}_{param2_name}_{param2_min}-{param2_max}_candles_{last_n_candles_display}.html"
    file_path = os.path.join('html_cache', file_name)

    # Save the combined heatmap as an HTML file with the correct format
    try:
        # First attempt: Try saving with vega format
        altair_saver.save(combined_heatmap, file_path, format="vega")
    except Exception as e:
        print(f"Warning: Could not save with vega format ({str(e)})")
        try:
            # Second attempt: Try disabling vegafusion temporarily
            alt.data_transformers.disable_max_rows()
            altair_saver.save(combined_heatmap, file_path)
            alt.data_transformers.enable('vegafusion')
        except Exception as e:
            print(f"Error saving chart: {str(e)}")
            structured['artifact'] = None
            return structured

    # Modify the HTML to include custom JavaScript and image elements
    with open(file_path, 'r') as file:
        html_content = file.read()

    # Add custom JavaScript for hover functionality
    custom_js = f"""
    .then(function(result) {{
        const view = result.view;
        view.addEventListener('mouseover', function(event, item) {{
            if (item && item.datum && item.datum.pnl_image) {{
                const imgId = item.datum.pnl_image;
                const img = document.getElementById(imgId);
                img.style.display = 'block';
            }}
        }});
        view.addEventListener('mouseout', function(event, item) {{
            if (item && item.datum && item.datum.pnl_image) {{
                const imgId = item.datum.pnl_image;
                const img = document.getElementById(imgId);
                img.style.display = 'none';
            }}
        }});
    }})"""

    # Create an instance of the strategy class to access its attributes
    strategy_instance = strategy_class()

    # Define a list of base strategy parameters to exclude
    base_strategy_params = set(['initial_equity', 'fee_pct'])

    # Extract fixed parameters from the strategy instance, excluding those in param_ranges and base strategy parameters
    fixed_parameters = {attr: getattr(strategy_instance, attr) for attr in dir(strategy_instance) 
                        if not callable(getattr(strategy_instance, attr)) 
                        and not attr.startswith("__") 
                        and attr not in param_ranges
                        and attr not in base_strategy_params}

    # Add a box with parameter range, user inputs, and investigation period
    parameter_info_html = f"""
    <div id="parameter-info" style="margin-top: 20px; padding: 10px; border: 1px solid #ccc; background-color: #f9f9f9;">
        <h3>Parameter Information</h3>
        <p><strong>Strategy Name:</strong> {strategy_name}</p>
        <p><strong>Asset:</strong> {asset}</p>
        <p><strong>Initial Equity:</strong> {initial_equity}</p>
        <p><strong>Fee Percentage:</strong> {fee_pct}%</p>
        <p><strong>Interval:</strong> {interval}</p>
        <p><strong>Investigation Period:</strong> {start_date} to {end_date}</p>
        <h4>Parameter Ranges:</h4>
        <ul>
            {''.join([f'<li><strong>{key}:</strong> {value}</li>' for key, value in param_ranges.items()])}
        </ul>
        <h4>Fixed Parameters:</h4>
        <ul>
            {''.join([f'<li><strong>{key}:</strong> {value}</li>' for key, value in fixed_parameters.items()])}
        </ul>
    </div>
    """

    custom_html = f"""
    <style>
        .vega-visualization canvas {{
            width: 100%;
        }}
        #heatmap-container {{
            display: flex;
            width: 100%;
        }}
        #heatmap {{
            width: 50%;
        }}
        #pnl-image-container {{
            width: 30%;
            position: absolute;
            top: 0;
            right: 0;
        }}
        #pnl-image-container img {{
            position: absolute;
            top: 0;
            right: 0;
            width: 100%;
        }}
    </style>
    <div id="heatmap-container">
        <div id="heatmap"></div>
        <div id="pnl-image-container">
            {image_elements}
        </div>
    </div>
    {parameter_info_html}
    """

    # Insert the custom JavaScript and image elements before the closing </body> tag
    html_content = html_content.replace(', spec, embedOpt)', ', spec, embedOpt)' + custom_js)
    html_content = html_content.replace('</body>', custom_html + '</body>')

    # Write the modified HTML back to the file
    with open(file_path, 'w') as file:
        file.write(html_content)

    print(f"\nHeatmap saved as: {file_path}")
    publish_file(file_path, label=f"Heatmap - {strategy_name} {asset} {interval}")

    structured['artifact'] = file_path
    return structured

if __name__ == "__main__":
    print_logo()
    print_viewer_info()
    
    print("HEATMAP - Heatmap Generator and Strategy Backtester")
    
    # Get user inputs
    user_inputs = get_user_inputs()
    
    # Get parameter ranges for the heatmap
    param_ranges = get_parameter_ranges(user_inputs['strategy_class'])
    
    # Create heatmap
    create_heatmap(
        user_inputs['timeframe_data'],
        user_inputs['strategy_class'],
        param_ranges,
        user_inputs['initial_equity'],
        user_inputs['fee_pct'],
        user_inputs['lookback_candles'],
        user_inputs['end_lookback_candles'],
        user_inputs['interval'],
        user_inputs['asset'],
        user_inputs['strategy_name'],
        user_inputs['start_date'],
        user_inputs['end_date']
    )
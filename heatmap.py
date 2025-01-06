import altair as alt
import pandas as pd
import numpy as np
from strategy_utils import get_user_inputs, fetch_data, get_parameter_ranges, print_logo
from classes.trade_analyzer import TradeAnalyzer
import itertools
from tqdm import tqdm
import multiprocessing
import os
import webbrowser
import altair_saver
import matplotlib.pyplot as plt

def analyze_strategy(params, data, strategy_class, initial_equity, fee_pct, last_n_candles_analyze, last_n_candles_display, param_ranges):
    # Use last_n_candles_analyze to calculate indicators
    data_for_indicators = data.iloc[-(last_n_candles_analyze + last_n_candles_display):]
    
    strategy_params = dict(zip(param_ranges.keys(), params))
    strategy_params.update({
        'initial_equity': initial_equity,
        'fee_pct': fee_pct
    })

    analyzer = TradeAnalyzer(strategy_class, strategy_params)
    
    # Use last_n_candles_display to generate entry and exit signals
    trades, _ = analyzer.analyze_data(data_for_indicators, last_n_candles_display, silent=True)

    # Calculate metrics using the TradeAnalyzer
    equity_curve = [initial_equity]
    for trade in trades:
        equity_curve.append(equity_curve[-1] + trade[4])

    metrics = analyzer.calculate_metrics(equity_curve, trades)
    
    # Replace zero drawdowns with max drawdown
    max_dd = metrics['max_drawdown_pct']
    drawdown_pct = max_dd if max_dd > 0 else 0

    return (
        metrics['profit_pct'], 
        metrics['total_net_profit'], 
        len(trades), 
        metrics['win_rate'], 
        metrics['profit_factor'], 
        metrics['max_drawdown'], 
        drawdown_pct,  # Use the processed drawdown value
        metrics['avg_trade_profit'], 
        metrics['avg_trade_profit_pct'], 
        metrics['avg_trade_duration'], 
        metrics['sharpe_ratio'],
        metrics['sortino_ratio'],
        metrics['volatility'],
        equity_curve
    )

def analyze_strategy_wrapper(args):
    return analyze_strategy(*args)


def create_tooltip_fields(param_ranges):
    """Helper function to create common tooltip fields."""
    return [
        alt.Tooltip('x:O', title=list(param_ranges.keys())[0]),
        alt.Tooltip('y:O', title=list(param_ranges.keys())[1]),
        alt.Tooltip('net_profit:Q', title='Net Profit'),
        alt.Tooltip('num_trades:Q', title='Number of Trades'),
        alt.Tooltip('win_rate:Q', title='Win Rate'),
        alt.Tooltip('profit_factor:Q', title='Profit Factor'),
        alt.Tooltip('drawdown:Q', title='Max Drawdown'),
        alt.Tooltip('drawdown_pct:Q', title='**Max Drawdown %**'),
        alt.Tooltip('avg_trade_profit:Q', title='Avg Trade Profit'),
        alt.Tooltip('avg_trade_profit_pct:Q', title='Avg Trade Profit %'),
        alt.Tooltip('avg_trade_duration:Q', title='Avg Trade Duration'),
        alt.Tooltip('sharpe_ratio:Q', title='Sharpe Ratio'),
        alt.Tooltip('sortino_ratio:Q', title='Sortino Ratio'),
        alt.Tooltip('volatility:Q', title='Volatility'),
        alt.Tooltip('pnl_image:N', title='PnL Image')
    ]

def generate_pnl_image(args):
    """Helper function to generate PnL images in parallel"""
    idx, params, equity_curve = args
    plt.figure()
    plt.plot(equity_curve)
    plt.title(f'PnL for params: {params}')
    plt.xlabel('Trade Number')
    plt.ylabel('Equity')
    image_path = f'pnl_cache/pnl_image_{idx}.png'
    plt.savefig(image_path)
    plt.close()
    return idx, image_path

def open_file(file_path):
    """Open a file with the default application on any operating system"""
    import platform
    import subprocess
    
    try:
        system = platform.system().lower()
        
        if system == 'darwin':  # macOS
            subprocess.run(['open', file_path], check=True)
        elif system == 'windows':
            os.startfile(file_path)  # Windows-specific function
        elif system == 'linux':
            subprocess.run(['xdg-open', file_path], check=True)
        else:
            print(f"Unsupported operating system: {system}")
            print(f"Please open the file manually: {file_path}")
    except Exception as e:
        print(f"Could not open file automatically: {e}")
        print(f"Please open the file manually: {file_path}")

def create_heatmap(data, strategy_class, param_ranges, initial_equity, fee_pct, last_n_candles_analyze, last_n_candles_display, interval, asset, strategy_name):
    """Creates a heatmap of strategy results for different parameter combinations"""
    print("\nCreating Heatmap...")
    
    # Generate all possible parameter combinations
    param_combinations = list(itertools.product(*param_ranges.values()))
    
    print(f"There are {len(param_combinations)} possible parameter combinations.")

    with multiprocessing.Pool() as pool:
        args_list = [(params, data, strategy_class, initial_equity, fee_pct, last_n_candles_analyze, last_n_candles_display, param_ranges) for params in param_combinations]
        results = list(tqdm(pool.imap(analyze_strategy_wrapper, args_list), total=len(param_combinations), desc="Analyzing strategies"))
    
    # Ensure the pnl_cache directory exists
    os.makedirs('pnl_cache', exist_ok=True)

    # Calculate the investigation period using the display period
    start_date = data.index[-last_n_candles_display]
    end_date = data.index[-1]

    # Prepare arguments for parallel image generation
    image_args = [(idx, params, result[-1]) for idx, (params, result) in enumerate(zip(param_combinations, results))]
    
    # Generate images in parallel
    print("\nGenerating PnL images...")
    with multiprocessing.Pool() as pool:
        image_results = list(tqdm(pool.imap(generate_pnl_image, image_args), 
                                total=len(image_args), 
                                desc="Creating PnL images"))
    
    # Create HTML image elements
    image_elements = ""
    for idx, image_path in image_results:
        image_elements += f'<img id="pnl-img-{idx}" src="../{image_path}" alt="PnL Image" style="display:none;" width="70%">\n'

    # Create a DataFrame to store profits and parameters
    results_data = []
    for idx, ((params, result), (_, image_path)) in enumerate(zip(zip(param_combinations, results), image_results)):
        results_data.append({
            'x': params[0],
            'y': params[1],
            'profit': result[0],
            'net_profit': result[1],
            'num_trades': result[2],
            'win_rate': result[3],
            'profit_factor': result[4],
            'drawdown': result[5],
            'drawdown_pct': result[6],
            'avg_trade_profit': result[7],
            'avg_trade_profit_pct': result[8],
            'avg_trade_duration': result[9],
            'sharpe_ratio': result[10],
            'sortino_ratio': result[11],
            'volatility': result[12],
            'pnl_image': f'pnl-img-{idx}'  # Use the current index
        })

    df = pd.DataFrame(results_data)

    # Handle invalid parameter combinations (where slow_length <= length)
    mask_invalid = df['x'] >= df['y']  # Create mask for invalid combinations
    
    # Set invalid combinations to None
    df.loc[mask_invalid, 'drawdown_pct'] = None
    df.loc[mask_invalid, 'profit'] = None
    df.loc[mask_invalid, 'sharpe_ratio'] = None
    
    # Determine color scale domains (using only valid combinations)
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
        tooltip=[
            alt.Tooltip('profit:Q', title='**Profit %**'),
            *create_tooltip_fields(param_ranges)
        ]
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
        tooltip=[
            alt.Tooltip('sharpe_ratio:Q', title='**Sharpe Ratio**'),
            *create_tooltip_fields(param_ranges)
        ]
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
                           reverse=True  # Keeps yellow for best (lowest) drawdown
                       ),
                       title='Max Drawdown %'),
        tooltip=[
            alt.Tooltip('drawdown_pct:Q', title='**Max Drawdown %**'),
            *create_tooltip_fields(param_ranges)
        ]
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
    os.makedirs('html', exist_ok=True)

    # Construct the file name using the strategy name and min/max values of the first two parameter ranges
    param_keys = list(param_ranges.keys())
    param1_name, param2_name = param_keys[0], param_keys[1]
    param1_min, param1_max = min(param_ranges[param1_name]), max(param_ranges[param1_name])
    param2_min, param2_max = min(param_ranges[param2_name]), max(param_ranges[param2_name])
    file_name = f"{strategy_name}_{asset}_{interval}_{param1_name}_{param1_min}-{param1_max}_{param2_name}_{param2_min}-{param2_max}_candles_{last_n_candles_display}.html"
    file_path = os.path.join('html', file_name)

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
            return

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
    open_file(file_path)  # Use the new open_file function instead of os.system

if __name__ == "__main__":
    print_logo()
    
    print("HEATMAP - Heatmap Generator and Strategy Backtester")
    
    # Get user inputs
    asset, interval, initial_equity, last_n_candles_display, lookback_candles, fee_pct, strategy_name, strategy_class = get_user_inputs()
    
    # Get parameter ranges for the heatmap
    param_ranges = get_parameter_ranges(strategy_class)
    
    # Load data
    data = fetch_data(asset, interval)

    print("lookback_candles", lookback_candles)

    # Create heatmap
    create_heatmap(data, strategy_class, param_ranges, initial_equity, fee_pct, lookback_candles, last_n_candles_display, interval, asset, strategy_name)
from tqdm import tqdm
import sys
import time
import inspect
import os
import pandas as pd

print("Initializing Chart Analysis...")
print_logo = None  # Placeholder für späteres Laden
get_user_inputs = None  # Placeholder für späteres Laden
get_strategy_inputs = None  # Placeholder für späteres Laden
TradeAnalyzer = None  # Placeholder für TradeAnalyzer
make_subplots = None  # Placeholder für make_subplots
go = None  # Placeholder für plotly.graph_objects

def show_loaded_modules():
    """Zeigt bereits geladene System-Module"""
    system_modules = [
        name for name, module in sys.modules.items() 
        if module and 
        hasattr(module, '__file__') and 
        module.__file__ and 
        'site-packages' not in module.__file__ and
        'lib/python' in module.__file__
    ]
    
    print("\nLoaded System Modules:")
    for module in tqdm(system_modules, desc="System Modules"):
        tqdm.write(f"  - {module}")  # Verwende tqdm.write statt print
        time.sleep(0.05)  # Kleine Verzögerung für bessere Lesbarkeit

def get_imports_from_file(filename):
    with open(filename, 'r') as file:
        content = file.read()
        
    # Finde alle import Statements
    imports = []
    lines = content.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('from') or line.startswith('import'):
            # Überspringe die tqdm, sys, time imports und leere Zeilen
            if not any(x in line for x in ['tqdm', 'sys', 'time', 'inspect', 'os', 'plotly']) and '=' not in line and 'append' not in line:
                # Entferne Kommentare
                line = line.split('#')[0].strip()
                if line:  # Nur wenn die Zeile nicht leer ist
                    imports.append(line)
    
    # Konvertiere import statements in Module-Alias-Paare
    modules = []
    for imp in imports:
        try:
            if imp.startswith('from'):
                # from module import name
                parts = imp.split('import')
                module = parts[0].replace('from', '').strip()
                names = [n.strip() for n in parts[1].split(',')]
                for name in names:
                    if 'as' in name:
                        # Handle "import x as y"
                        name = name.split('as')[1].strip()
                    if name and not name.startswith('=') and not 'append' in name:  # Prüfe auf gültige Namen
                        modules.append((module, name))
            else:
                # import module
                module = imp.replace('import', '').strip()
                if 'as' in module:
                    parts = module.split('as')
                    if len(parts) == 2 and all(parts) and not 'append' in module:  # Prüfe auf gültige Module/Alias
                        modules.append((parts[0].strip(), parts[1].strip()))
                else:
                    if module and not 'append' in module:  # Prüfe auf gültiges Modul
                        modules.append((module, module))
        except Exception as e:
            print(f"Skipping invalid import statement: {imp}")
            continue

    return modules

def initialize_modules():
    global print_logo, get_user_inputs, get_strategy_inputs, TradeAnalyzer, make_subplots, go
    
    # Zeige zuerst die System-Module
    show_loaded_modules()
    
    # Hole den Pfad der aktuellen Datei
    current_file = os.path.abspath(__file__)

    # Extrahiere die Module aus der Datei
    modules = get_imports_from_file(current_file)

    print("\nLoading modules:")
    # Lade Module mit detailliertem Fortschrittsbalken
    with tqdm(total=len(modules), desc="Progress", unit="module") as pbar:
        # Lade zuerst die wichtigsten Module
        from strategy_utils import print_logo, get_user_inputs, get_strategy_inputs
        from classes.trade_analyzer import TradeAnalyzer
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        
        for module, alias in modules:
            try:
                pbar.set_description(f"Loading {module}")
                if isinstance(alias, list):
                    # Für mehrere Imports aus einem Modul
                    imports = ', '.join(alias)
                    exec(f"from {module} import {imports}")
                else:
                    # Für einzelne Imports
                    exec(f"from {module} import {alias}")
                time.sleep(0.2)  # Längere Verzögerung für bessere Sichtbarkeit
                pbar.update(1)
            except ImportError as e:
                print(f"\nError loading {module}: {e}")
                sys.exit(1)

    print("\nInitialization complete!\n")

def create_interactive_chart(timeframe_data, strategy_class, strategy_params, last_n_candles_analyze, last_n_candles_display):
    """Creates an interactive Plotly chart with price movement and PnL curve"""
    # Initialize strategy with timeframe data
    strategy = strategy_class(**strategy_params)
    strategy.timeframe_data = timeframe_data  # Add timeframe data to strategy
    
    # Create analyzer with strategy and strategy_params
    analyzer = TradeAnalyzer(strategy, strategy_params)
    
    # Get primary timeframe data
    data = timeframe_data['primary']['data']
    
    # Get date range from strategy parameters
    start_date = strategy_params.get('start_date')
    end_date = strategy_params.get('end_date')
    
    # If the strategy uses DivergenceDetector, set the date range
    if hasattr(strategy, 'divergence_detector'):
        strategy.divergence_detector.set_date_range(start_date, end_date)
    
    # Analyze data and get trades
    trades, display_data = analyzer.analyze_data(data, last_n_candles_analyze, last_n_candles_display)
    
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
    
    print("\nErstelle Chart...")
    
    # Count available indicators for subplot layout
    available_indicators = [ind for ind in strategy.indicators if ind in display_data.columns]
    num_indicators = len(available_indicators)
    
    # Calculate total number of rows needed (price chart + 3 KAMA delta charts + one row per indicator)
    total_rows = 1 + 3 + num_indicators  # Changed from 2 to 3 KAMA delta charts
    
    # Calculate row heights (40% for price chart, 30% for KAMA deltas, remaining 30% divided among indicators)
    row_heights = [0.4]  # Price chart
    kama_delta_height = 0.3 / 3  # Divide 30% among 3 KAMA delta charts
    row_heights.extend([kama_delta_height] * 3)  # Three KAMA delta charts
    if num_indicators > 0:
        indicator_height = 0.3 / num_indicators
        row_heights.extend([indicator_height] * num_indicators)
    
    # Create initial subplot layout
    fig = make_subplots(
        rows=total_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights
    )
    
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
    strategy.add_strategy_traces(fig, display_data)
    
    # Indikatoren Chart (starting from row 2)
    strategy.add_indicator_traces(fig, display_data, row=2, col=1)
    
    # Layout anpassen
    fig.update_layout(
        title='Price Chart with Signals and Indicators',
        yaxis_title='Price',
        height=300 * total_rows,  # Increase base height per row for even better visibility
        xaxis_rangeslider_visible=False
    )
    
    # Y-Achsen-Format anpassen
    fig.update_yaxes(title_text="Price", row=1, col=1)
    
    # Chart anzeigen
    fig.show()
    
    return fig

if __name__ == "__main__":
    initialize_modules()
    print_logo()
   
    print("CHART ANALYSE - Heatmap Generator and Strategy Backtester")
    
    # Get user inputs
    user_inputs = get_user_inputs()
    print(f"\nChart Analysis - User Inputs - start_date: {user_inputs.get('start_date')}, end_date: {user_inputs.get('end_date')}")
    
    # Strategie-spezifische Parameter abrufen
    strategy_params = get_strategy_inputs(user_inputs['strategy_class'])
    strategy_params.update({
        'initial_equity': user_inputs['initial_equity'],
        'fee_pct': user_inputs['fee_pct'],
        'start_date': user_inputs['start_date'],
        'end_date': user_inputs['end_date']
    })
    print(f"Chart Analysis - Strategy Params - start_date: {strategy_params.get('start_date')}, end_date: {strategy_params.get('end_date')}")

    # Erstelle interaktives Diagramm
    create_interactive_chart(
        user_inputs['timeframe_data'],
        user_inputs['strategy_class'], 
        strategy_params, 
        user_inputs['lookback_candles'], 
        user_inputs['lookback_candles']
    ) 
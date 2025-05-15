# Damian's Heatmap Generator

![Example Heatmap](example-heatmap.png)

Welcome to Damian's Heatmap Generator, a sophisticated tool designed to analyze trading strategies and visualize their performance through interactive heatmaps.

## Project Background

As a PHP open-source developer venturing into Python, I created this project in collaboration with AI tools (GPT and Claude) and the Cursor IDE. The goal was to build a framework that could convert trading strategies from platforms like TradingView, MetaTrader, or ProRealTime into Python. The framework allows for debugging with `pnl.py` and `chart_analysis.py` to ensure identical signal generation and performance before creating parameter-based heatmaps.

## Overview

Damian's Heatmap Generator allows users to:
- Convert trading strategies from various platforms to Python using AI
- Verify strategy performance through profit/loss analysis
- Compare trade signals with original strategies
- Generate performance heatmaps by varying two parameters while keeping others fixed

## Strategy Conversion Process

1. **AI-Assisted Conversion**
   - Use AI to convert strategies to Python using `macd_strategy.py` as a template
   - Place new strategies in `classes/strategies` directory
   - Support for TradingView, MetaTrader, ProRealTime, and other platforms

2. **Debugging Process**
   - Use `pnl.py` for profit/loss verification
   - Use `chart_analysis.py` for signal comparison
   - Iterate until results match original strategy
   - Prefer simplified strategies for faster heatmap generation

3. **Parameter Configuration**
   When modifying dynamic parameters or long/short behavior, adjust these methods in the strategy class:
   - `__init__`
   - `parameter_ranges`
   - `get_parameters`

## Features

### Analysis Tools
- **PnL Analysis**: Comprehensive profit and loss calculations via `pnl.py`
- **Visual Verification**: Plotly charts showing indicators and entry/exit signals overlaid on price data
- **Strategy Comparison**: One-to-one trade comparison with original strategy

### Visualization
- **Multi-Metric Heatmaps**: 2D heatmaps for Profit, Sharpe Ratio, and Drawdown
- **Interactive Elements**: PnL calculations displayed on heatmap hover
- **Browser Integration**: Results viewed through Plotly and Altair visualizations

### Technical Features
- **Data Management**: OHLC data fetching and caching via Coincopy and Binance APIs
- **Performance**: Multi-core parallelization for heatmap generation
- **Strategy Support**: Comprehensive long and short strategy capabilities

## Getting Started

### Prerequisites

- Python 3.7 or higher
- Required Python packages:
```bash
pandas
numpy
matplotlib
altair
altair_saver
tqdm
requests
plotly
webbrowser
platform
subprocess
decimal
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/damianhunziker/damiansHeatmapGenerator
cd heatmap_generator
```

2. Install the required packages using `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Usage

1. **Run the Heatmap Generator**:
```bash
python heatmap.py
```
   - The heatmap and analysis results will be saved as an HTML file and automatically opened in your default web browser.

2. **Run Profit and Loss Analysis**:
```bash
python pnl.py
```
   - This script will calculate and display the profit and loss of your strategy.

3. **Run Chart Analysis**:
```bash
python chart_analysis.py
```
   - This script will generate visualizations to compare entry and exit signals with the original strategy.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your improvements.

## Acknowledgments

This project was developed with extensive use of AI, which provided valuable insights and optimizations throughout the development process. Special thanks to GPT, Claude and Cursor IDE, that made this project possible.

## License

This project is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License - see the [LICENSE](LICENSE) file for details.

# Test Script for Trading Strategy Tools

This script provides a unified interface to run various trading strategy tools with proper parameter validation and schema information.

## Available Tools

1. **automator.py** - Automates running strategies across multiple pairs
2. **heatmap.py** - Generates heatmaps for strategy parameter optimization
3. **chart_analysis.py** - Analyzes and visualizes trading charts
4. **pnl.py** - Calculates and visualizes PnL
5. **fetcher.py** - Fetches market data

## Usage

### View Parameter Schema

To see the required parameters and their formats for all tools:

```bash
python test.py --schema
```

### Run a Tool

General format:
```bash
python test.py <tool_name> [parameters]
```

Example:
```bash
python test.py pnl --start_date="2024-01-01" --end_date="2024-03-01" --asset="BTCUSDT" --interval="4h"
```

### Common Parameters

All tools accept these common parameters:

- `start_date` (Required) - Analysis start date in YYYY-MM-DD format
- `end_date` (Required) - Analysis end date in YYYY-MM-DD format
- `interval` (Optional) - Trading interval (default: 4h)
  - Valid values: 1m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d
- `asset` (Required) - Trading pair (e.g., BTCUSDT)

### Tool-Specific Parameters

Each tool may have additional required or optional parameters. Use the `--schema` option to see the complete parameter list for each tool.

## Error Handling

The script will:
1. Validate all required parameters are provided
2. Check date formats are correct
3. Verify interval values are valid
4. Display helpful error messages if validation fails

## Examples

1. Run heatmap analysis:
```bash
python test.py heatmap --start_date="2024-01-01" --end_date="2024-03-01" --asset="BTCUSDT" --interval="4h"
```

2. Run PnL analysis:
```bash
python test.py pnl --start_date="2024-01-01" --end_date="2024-03-01" --asset="ETHUSDT" --interval="1h"
```

3. Run chart analysis:
```bash
python test.py chart_analysis --start_date="2024-01-01" --end_date="2024-03-01" --asset="BTCUSDT"
```

4. Fetch market data:
```bash
python test.py fetcher --start_date="2024-01-01" --end_date="2024-03-01" --asset="BTCUSDT" --interval="1d"
```

5. Run automated analysis:
```bash
python test.py automator --start_date="2024-01-01" --end_date="2024-03-01" --asset="BTCUSDT" --interval="4h"
```
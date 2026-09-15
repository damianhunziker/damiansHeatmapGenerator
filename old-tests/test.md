# Test.py Documentation

## Overview

test.py is a command-line interface tool that provides a unified way to execute various trading strategy analysis tools. It automatically discovers parameters from executable modules and provides a schema-based approach for parameter validation and execution.

## Key Features

- Automatic parameter discovery from executable modules
- Schema-based parameter validation
- Support for multiple executables (automator, heatmap, chart_analysis, pnl, fetcher)
- Flexible parameter handling with defaults
- Command-line interface with help system

## Available Executables

### 1. Automator

Automates running strategies across multiple pairs.

### 2. Heatmap

Generates heatmaps for strategy parameter optimization.

### 3. Chart Analysis

Analyzes and visualizes trading charts.

### 4. PnL (Profit and Loss)

Calculates and visualizes PnL metrics.

### 5. Fetcher

Fetches market data from exchanges.

## Common Parameters

All executables share these common parameters:

- `start_date`: Start date for analysis (format: YYYY-MM-DD)
- `end_date`: End date for analysis (format: YYYY-MM-DD)
- `interval`: Trading interval (options: 1m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d)
- `asset`: Trading pair (e.g., BTCUSDT)
- `strategy`: Strategy name to use (optional)

## Usage

### Basic Command Structure

```bash
python test.py <executable> [parameters]
```

### View Schema

To view the complete parameter schema:
```bash
python test.py --schema
```

### Examples

1. Run PnL Analysis:

```bash
python test.py pnl --start_date="2024-01-01" --end_date="2024-03-01" --asset="BTCUSDT" --strategy="LiveKAMASSLStrategy"
```

2. Generate Heatmap:

```bash
python test.py heatmap --start_date="2024-01-01" --end_date="2024-03-01" --asset="BTCUSDT" --strategy="KAMAStrategy"
```

3. Fetch Data:

```bash
python test.py fetcher --asset="BTCUSDT" --interval="4h"
```

## Key Functions

### get_schema()

Discovers and returns the parameter schema for all executables. The schema includes:

- Parameter names
- Default values
- Required/optional status
- Parameter descriptions
- Valid parameter types

### validate_parameters(exec_name, params)

Validates parameters against the schema:

- Checks required parameters
- Validates date formats
- Validates date ranges
- Checks asset parameter
- Validates intervals
- Validates strategy-specific parameters

### run_executable(exec_name, params)

Executes the specified tool with validated parameters:

1. Validates parameters
2. Fetches/loads data if needed
3. Filters data by date range
4. Creates timeframe data structure
5. Executes the appropriate tool

### get_class_init_params(module_name, class_name)

Extracts initialization parameters from class definitions:

- Gets __init__ signature
- Extracts parameter defaults
- Includes strategy-specific parameters
- Handles required vs optional parameters

## Error Handling

The system provides clear error messages for:

- Missing required parameters
- Invalid date formats
- Invalid date ranges
- Unknown strategies
- Invalid parameter types
- Data fetching errors

## Data Caching

- Data is cached in the `ohlc_cache` directory
- Cache files follow the format: `{asset}_{interval}_ohlc.csv`
- Cached data is reused when available

## Strategy Parameters

Strategy-specific parameters are automatically discovered and included in the schema. Each strategy can define:

- Required parameters
- Default values
- Parameter descriptions
- Parameter ranges for optimization

## Adding New Executables

To add a new executable:

1. Create a new module with a main class
2. Define the class's __init__ parameters
3. Optionally implement get_parameters() for strategy-specific parameters
4. Add the executable to the schema in test.py

## Best Practices

1. Always use the --schema flag to check available parameters
2. Provide required parameters explicitly
3. Use date ranges appropriate for the analysis
4. Check cached data for the requested timeframe
5. Monitor system resources for large data operations
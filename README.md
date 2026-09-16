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

## Viewing generated reports

Every tool now writes its HTML output to disk and prints a clickable link instead of relying on `fig.show()` / a local browser (which does not exist inside the container):

```
========================================================================
  Chart Analysis
  http://localhost:8900/html_cache/chart_analysis_LiveKAMASSLStrategy_BTCUSDT_4h_20240101_120000.html
========================================================================
```

The link is served by a small static viewer. On the host the file is opened in the default browser automatically; inside Docker the printed URL is the way to reach it. An index page links to all generated reports:

- Reports index: `http://localhost:8900/html_cache/viewer.html`
- Browse all files: `http://localhost:8900/`

Output locations:

| Tool | Output |
|------|--------|
| `chart_analysis.py` | `html_cache/chart_analysis_<Strategy>_<Asset>_<Interval>_<timestamp>.html` |
| `pnl.py` | `html_cache/results.html` (plus `chart_both/long/short.html`) |
| `heatmap.py` | `html_cache/<Strategy>_<Asset>_<Interval>_..._candles_<n>.html` |
| `automator.py` | `automator_html/run_<timestamp>/<PAIR>/html/...` |

The viewer port is configurable via `HEATMAP_VIEWER_PORT` (default `8900`) and the public host via `HEATMAP_VIEWER_HOST` (default `localhost`).

## Docker / Dev Container

The project ships with a `Dockerfile`, `docker-compose.yml` and a `.devcontainer/` setup. Two services are defined:

| Service | Purpose |
|---------|---------|
| `app` | The Python toolchain (kept alive with `sleep infinity`); run the scripts here |
| `viewer` | Persistent static server for the generated HTML reports, bound to `127.0.0.1:8900` |

Start both:

```bash
docker compose up -d
```

Run a tool inside the app container and open the printed link on the host:

```bash
docker compose exec app python test.py chart_analysis \
    --start_date="2024-01-01" --end_date="2024-03-01" \
    --asset="BTCUSDT" --strategy="LiveKAMASSLStrategy" --interval="4h"
```

The viewer only binds to `127.0.0.1` on the host; it is never exposed to the network.

## Chart Analysis (`chart_analysis.py`)

`chart_analysis.py` is the visual verification tool of the framework. It replays a strategy over historical OHLC data, simulates the resulting trades, and renders everything as a single interactive Plotly chart so you can visually compare the strategy's entry/exit logic against the price action, its indicators, divergences and pivots. It is the main tool for the "Debugging Process" described above (signal comparison) before running a heatmap.

### What it does

1. Loads OHLC data for the chosen asset/interval from `ohlc_cache/` and fetches it via `OHLCFetcher` if it is missing.
2. Discovers all strategies in `classes/strategies/*_strategy.py` and instantiates the selected one.
3. Runs the strategy (`prepare_signals`) to compute indicators and entry/exit signals.
4. Simulates trades with `TradeAnalyzer` (`classes/trade_analyzer.py`), including fees and opposite-direction exits, and records an exit reason per trade.
5. Prints a full trade list and summary statistics to the console.
6. Builds the interactive chart, opens it in the default browser (`fig.show()`), and returns the Plotly `Figure` object.

### Chart components (subplot layout)

The chart is a vertically stacked Plotly figure with a shared x-axis. Row order:

| Row | Content |
|-----|---------|
| 1 | **Price** — candlesticks plus overlays and trade/divergence/pivot markers |
| 2 | **Entry KAMA Delta** — delta line + limit (dashed) |
| 3 | **Exit KAMA Delta** — delta line + limit (dashed) |
| 4 | **Fusion Range Filter** — only when `use_fusion_for_long` is enabled |
| 5…n | one row per **divergence indicator** (long and short profile) |

Row 1 overlays and markers:

- SSL Up (green) / SSL Down (red) and DEMA (purple) when available.
- Bollinger Bands (basis/upper/lower) and, for long trades, Entry KAMA / KAMA2 / Exit KAMA.
- Trade markers: long entry (blue triangle right), long exit (green triangle left), short entry (red triangle right), short exit (green triangle left); hover shows price and exit reason.
- Divergence lines: bearish divergences for the long profile (dark red = confirmed, pink = unconfirmed) and bullish divergences for the short profile (dark green = confirmed, light green = unconfirmed).
- Pivot points: long-profile pivot highs (dark red triangle down) and short-profile pivot lows (dark green triangle up).
- Fusion Range Filter background: gray rectangles mark candles where the filter condition (MA > ATR) is active.

The Fusion Range Filter row (row 4) plots the Fusion MA line (blue), the scaled ATR limit (red dashed) and a gray fill for active periods. Each divergence row plots the indicator values and draws its divergence segments as line shapes.

Controls: two buttons (`Hide Pivot Points` / `Show Pivot Points`) toggle the pivot markers via Plotly `restyle`. The figure height is computed from the number of subplots (price chart 800 px, each indicator 400 px, capped at 3000 px).

### Options

`chart_analysis.py` can be run in two ways.

#### 1. Interactive

```bash
python chart_analysis.py
```

Prompts (in order):

| Prompt | Default | Notes |
|--------|---------|-------|
| Select strategy number | 1 | from the auto-discovered list |
| Enter asset | `BTCUSDT` | e.g. `SOLUSDT` |
| Enter interval per timeframe | strategy default (e.g. `4h`) | `1m, 5m, 15m, 1h, 4h, 1d` |
| Refetch data? | `N` | only asked if a cache file exists |
| Enter start date | earliest cache date | validated against the cache range |
| Enter end date | latest cache date | must be ≥ start date |
| Enter initial equity | `1000` | |
| Enter fee percentage | `0.04` | in percent |
| Strategy parameters | from `get_parameters()` | e.g. entry/exit filter |

#### 2. Non-interactive via `test.py`

```bash
python test.py chart_analysis --start_date="2024-01-01" --end_date="2024-03-01" \
    --asset="BTCUSDT" --strategy="LiveKAMASSLStrategy" --interval="4h"
```

Common parameters:

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `--start_date` | yes | – | `YYYY-MM-DD` |
| `--end_date` | yes | – | `YYYY-MM-DD` |
| `--asset` | yes | – | e.g. `BTCUSDT` |
| `--strategy` | yes | – | strategy class name, e.g. `LiveKAMASSLStrategy` |
| `--interval` | no | `4h` | one of `1m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d` |
| `--initial_equity` | no | `10000` | starting equity |
| `--fee_pct` | no | `0.04` | fee per side in percent |

Strategy-specific parameters are read from `get_parameters()` and defaulted automatically; any of them can be overridden on the command line:

```bash
python test.py chart_analysis --start_date="2024-01-01" --end_date="2024-03-01" \
    --asset="BTCUSDT" --strategy="LiveKAMASSLStrategy" \
    --entry_filter=0.7 --exit_filter=1.0 --use_fusion_for_long=true \
    --atr_length=9 --hma_mode=VWMA --hma_length=50 --atr_scaling_factor=1.4
```

For `LiveKAMASSLStrategy`, `get_parameters()` exposes: `entry_filter`, `exit_filter`, `calculate_divergences`, `use_fusion_for_long`, `atr_length`, `hma_mode`, `hma_length`, `atr_scaling_factor`. The constructor additionally understands `trade_direction` (`long`/`short`/`both`), `debug_mode`, `indicators`, `short_indicators`, `show_pivot_points`, `position_sizing_method`, `position_size_value`, `max_position_size`, `allow_continuous_entries` and the divergence exit flags (`use_divergence_exit_long`, `use_hidden_divergence_exit_long`, `use_divergence_exit_short`, `use_hidden_divergence_exit_short`).

> Note: `python test.py --schema` currently does not list `chart_analysis` parameters because `chart_analysis.py` does not define a `ChartAnalysis` class. Use the tables above as the reference.

### What is printed / output

Console output includes:

- Strategy/initialization and data diagnostics (shape, columns, date range).
- A **trade list** table: `#`, `Type` (LONG/SHORT), `Entry Time`, `Exit Time`, `Entry`, `Exit`, `Gross P/L`, `Fees`, `Net P/L`, `%`, `Exit Reason`.
- **Summary statistics**: initial equity, final equity, total return, total profit, total fees, number of trades, win rate, average profit per trade.
- The divergence indicator list, pivot counts and Fusion Range Filter statistics.

Exit reasons can be e.g. `KAMA exit signal`, `SSL exit signal`, `Realtime BB Cross Exit`, `Bearish divergence exit`, `Bullish divergence exit`, `SSL crossover up exit`, `DEMA crossover exit`, or `Opposite Entry` (the position was closed because the opposite direction signalled).

Generated artifacts:

- `ohlc_cache/<ASSET>_<INTERVAL>_ohlc.csv` — the data cache (created/updated by the fetcher).
- `html_cache/chart_analysis_<Strategy>_<Asset>_<Interval>_<timestamp>.html` — the interactive chart, also linked in the viewer index.

### Components

| Component | Role |
|-----------|------|
| `chart_analysis.py` | `create_interactive_chart()` builds and shows the figure; `create_chart()` helper; interactive `__main__` |
| `core/strategy_utils.py` | `get_user_inputs()`, `get_strategy_inputs()`, `get_available_strategies()`, `fetch_data()` |
| `classes/trade_analyzer.py` | `TradeAnalyzer.analyze_data()` / `_generate_trades()` — signal replay, trade simulation, fees, exit reasons |
| `classes/base_strategy.py` | `BaseStrategy` contract (`get_parameters`, `prepare_signals`, …) |
| `classes/strategies/*_strategy.py` | concrete strategy; provides `prepare_signals()`, `add_strategy_traces()`, `get_parameters()`, `get_required_timeframes()` |
| `classes/indicators/*` | divergence detection, Fusion Range Filter, volume zones |
| `classes/data_fetcher.py` | `OHLCFetcher` for exchange data and caching |

### Example

```bash
# Interactive
python chart_analysis.py

# Non-interactive
python test.py chart_analysis --start_date="2024-01-01" --end_date="2024-03-01" \
    --asset="BTCUSDT" --strategy="LiveKAMASSLStrategy" --interval="4h"
```

## Machine-readable API (`--api`)

Every tool can be run as a JSON API by adding `--api` to any `test.py` call.
With `--api` the process writes **exactly one JSON object to stdout** (all
normal console output is redirected to stderr), so it can be consumed by
scripts, agents and MCP servers.

```bash
python test.py pnl --asset=BTCUSDT --strategy=LiveKAMASSLStrategy \
    --start_date=2024-01-01 --end_date=2024-03-01 --api
```

The response is always wrapped in the same envelope:

```json
{
  "ok": true,
  "tool": "pnl",
  "params": { "...": "..." },
  "data": { "...": "..." },
  "artifacts": [{"path": "/app/html_cache/results.html",
                 "url": "http://localhost:8900/html_cache/results.html"}],
  "warnings": [],
  "error": null,
  "duration_ms": 3719
}
```

Schema (tools, strategy parameters, parameter ranges, intervals):

```bash
python test.py --schema --api
```

### Tools and their `data` payload

| Tool | `data` contents |
|------|-----------------|
| `pnl` | `directions.{both,long,short}` with `metrics`, `trades`, `equity_curve`, `equity_curve_timestamps`, `pnl_performance`, `buy_hold`, `drawdown` |
| `chart_analysis` | `summary` (equity, return, win rate, …), `trades`, `divergence_indicators` |
| `heatmap` | `grid` (one entry per parameter combination), `best` (top-N by profit/net_profit/sharpe/drawdown), `robustness` (3×3 neighbourhood mean/min), `plateaus` (algorithmic region detection, see below), `sidecar` (JSON path with grid + plateaus), `param_ranges`, `x_param`, `y_param` |
| `plateaus` | algorithmic plateau/region detection over a grid (`grid`, `grid_path` sidecar or a fresh heatmap): ranked `regions`, `best_region`, `representative`, `largest_rectangle` |
| `validate_plateau` | walk-forward validation of a parameter region: per-window representative + `summary.consistency` |
| `plot_plateau` | score matrix with region outlines rendered to a PNG |
| `automator` | `output_dir`, `runs[]` (per pair: `ok`, `error`, `best`, `grid`, `artifact`) |
| `fetcher` | `asset`, `interval`, `rows`, `start`, `end`, `cache_path` |
| `series` | per-candle indicator/signal series (see below) |

Heatmap parameter grids are passed as JSON. A grid of `0.7..0.9` (step `0.1`)
× `1.0..1.1` (step `0.1`):

```bash
python test.py heatmap --asset=BTCUSDT --strategy=LiveKAMASSLStrategy \
    --start_date=2024-01-01 --end_date=2024-03-01 --workers=1 \
    '--param_ranges={"entry_filter":{"min":0.7,"max":0.9,"step":0.1},"exit_filter":{"min":1.0,"max":1.1,"step":0.1}}' \
    --api
```

Per-cell PnL PNGs are always generated now (they drive the HTML hover overlay).
`--workers N` controls the multiprocessing pool for local runs; when a run is
offloaded to the home server the server sets the parallelism itself (its own CPU
count) and any client value is ignored, so cores are never left idle.
`--max_combos` (default `400`) guards against accidental huge grids.

### Choosing heatmap parameters (free selection)

The swept parameters are **passed at call time** via `--param_ranges` / the
`param_ranges` argument — they are **not hardcoded**. Any keyword argument the
strategy constructor accepts can be swept; the strategy's
`get_parameter_ranges()` is only the default when `param_ranges` is omitted.

Each value may be a list of explicit values, or a spec:

| Form | Example |
|------|---------|
| list of values (numeric, categorical or boolean) | `{"hma_mode": ["VWMA","HMA"], "use_fusion_for_long": [true,false]}` |
| `{min, max, step}` | `{"entry_filter": {"min":0.5,"max":1.5,"step":0.25}}` |
| `{min, max, count}` | `{"entry_filter": {"min":0.5,"max":1.5,"count":5}}` |

Constraints to keep in mind:

- **Exactly 2 parameters.** The heatmap is 2D: the first two keys become the x/y
  axes (`x_param`, `y_param`). One parameter raises an error; three or more are
  computed but only the first two are visualised, so the `robustness` pivot
  becomes ambiguous. Pass exactly two.
- **Unknown names are silently dropped.** The strategy is instantiated with only
  the kwargs it accepts, so a typo makes that axis constant (all cells
  identical) instead of raising. Sanity-check the grid: if `x`/`y` (or the
  metric column) has a single distinct value, the parameter was not applied.
- **Categorical / boolean** values work as lists; the axis number formatting is
  cosmetic only.
- **The human CLI (`test.py heatmap` without `--api`) ignores `--param_ranges`**
  and always uses `strategy_class.get_parameter_ranges()`. Free selection is
  available via `--api` and the MCP tool.

The result contains `grid` (all combinations with metrics), `best` (top-N by
profit / net_profit / sharpe / drawdown), `robustness` (3×3 neighbourhood
`neighbor_mean` / `neighbor_min`) and `plateaus` (algorithmic region detection).

### Plateau detection

`plateaus` is computed automatically from the underlying data matrix — never by
reading pixels of the rendered heatmap.  The pipeline is: composite score
(robust z-scores of sharpe / profit_factor / drawdown / trades, with a hard
`min_trades` filter) → smoothing → local stability → threshold → morphology →
connected regions → region scoring → representative cell.  Each region carries
its parameter span, score statistics, `cv`, `boundary_penalty` and a
`representative` cell (default: the `medoid`, i.e. the robust centre).  Tune it
per call with `--plateau_config` (or the `plateau_config` argument):

```bash
python test.py heatmap --asset=BTCUSDT --strategy=LiveKAMASSLStrategy \
    --start_date=2024-01-01 --end_date=2024-03-01 --workers=1 \
    '--param_ranges={"entry_filter":{"min":0.7,"max":0.9,"step":0.1},"exit_filter":{"min":1.0,"max":1.1,"step":0.1}}' \
    '--plateau_config={"min_trades":30,"threshold_k":0.5,"representative":"medoid"}' \
    --api
```

Every heatmap also writes a JSON **sidecar** (`data.sidecar`) with the grid and
the plateau analysis, so a later call can re-rank without re-running the
backtest.  Dedicated tools:

```bash
# re-rank an existing grid with different weights (no backtest)
python test.py plateaus --grid_path=html_cache/BTCUSDT_....json \
    '--plateau_config={"weights":{"sharpe_ratio":1.0,"drawdown_pct":-0.8}}' --api

# validate the region across walk-forward windows
python test.py validate_plateau --asset=BTCUSDT --strategy=LiveKAMASSLStrategy \
    --start_date=2024-01-01 --end_date=2024-06-01 --windows=3 \
    '--region={"entry_filter":{"min":0.7,"max":0.9},"exit_filter":{"min":1.0,"max":1.1}}' --api

# presentation overlay
python test.py plot_plateau --grid_path=html_cache/BTCUSDT_....json
```

Prefer **plateaus** over spikes: a stable region has a small `cv` and a
`representative` far from the search-space edge.  Confirm it on a different date
range (`validate_plateau`) and across pairs with `automator`.  Full algorithm and
config reference: [`docs/plateau-detection.md`](docs/plateau-detection.md).

### Derived parameters (exact KAMA scaling)

KAMA has **three** bar-based parameters (the Efficiency-Ratio `length` and the
EMA bounds `fast` / `slow`). Scaling KAMA cleanly means scaling **all three
together** so it behaves like the same indicator on `k`-times longer bars —
changing only one value does *not* scale it, it just distorts the behaviour.
Because the EMA alpha is `2/(period+1)`, the exact rule is:

```
length' = k * length
period' = k * (period + 1) - 1        # exact, for fast / slow
```

The naive approximation `period' ≈ k * period` is only correct for large
periods (base 4, k=6 → 24 vs exact **29**).

Instead of sweeping the three lengths by hand (which needs 3 axes, and half the
grid violates `fast < slow`), sweep **one** `kama_scale` axis and *derive* the
lengths from it with `derived_params`:

```bash
python test.py heatmap --asset=BTCUSDT --strategy=LiveKAMASSLStrategy \
    --start_date=2024-01-01 --end_date=2024-03-01 --workers=1 \
    '--param_ranges={"kama_scale":[1,2,3,4,6,8],"entry_filter":{"min":0.5,"max":1.5,"step":0.25}}' \
    '--derived_params={
        "entry_kama_length":{"scale":"kama_scale","base":16,"kind":"window"},
        "entry_kama_fast":{"scale":"kama_scale","base":4,"kind":"alpha"},
        "entry_kama_slow":{"scale":"kama_scale","base":24,"kind":"alpha"},
        "kama2_length":{"scale":"kama_scale","base":15,"kind":"window"},
        "kama2_fast":{"scale":"kama_scale","base":3,"kind":"alpha"},
        "kama2_slow":{"scale":"kama_scale","base":22,"kind":"alpha"},
        "exit_kama_length":{"scale":"kama_scale","base":14,"kind":"window"},
        "exit_kama_fast":{"scale":"kama_scale","base":2,"kind":"alpha"},
        "exit_kama_slow":{"scale":"kama_scale","base":20,"kind":"alpha"}}' \
    --api
```

Each entry: `scale` is the swept axis name, `base` the unscaled value, `kind`
is `window` (plain rolling window, e.g. the ER length) or `alpha` (EMA bound,
scaled exactly). The derived values are visible in `data.derived_params`.

Notes:

- **Warmup:** KAMA needs ~`6 × longest period` bars to converge. If the
  available history before `start_date` is too short, the envelope returns a
  `warnings` entry (e.g. `KAMA warmup may be insufficient ...`) — a warning, not
  an error, so you can still inspect the result.
- Prefer **log-spaced** scale values (`1,2,3,4,6,8,12,...`); scaling is
  multiplicative, so linear steps sample the space unevenly.
- `derived_params` is accepted by `run_heatmap` / `run_automator` (MCP) and the
  `heatmap` / `automator` `--api` tools.

### Generic parameter resolution (resolvers)

`derived_params` is the KAMA special case of a general mechanism: any **virtual**
parameter can be resolved into any number of **concrete** strategy parameters
via declarative, whitelisted op-trees (no code execution). Configure it in
`configs/param_resolvers.json` and/or inline via `resolvers` /
`resolver_config`, then sweep the virtual axis directly:

```bash
python test.py heatmap --asset=BTCUSDT --strategy=LiveKAMASSLStrategy \
    --start_date=2024-01-01 --end_date=2024-06-01 --workers=1 \
    '--param_ranges={"kama_normalized_length":[1,2,3,4,6,8],"entry_filter":{"min":0.5,"max":1.5,"step":0.25}}' \
    --api
```

`kama_normalized_length` is resolved (default config, **all_exact**) into
`entry_kama_length/fast/slow`, `kama2_*`, `exit_kama_*` (LiveKAMASSLStrategy) or
`slow_kama_length/fast/slow` (DMXStrategy). At the API/MCP boundary the grid is
normalised: a swept resolver **`id`** is converted to its canonical virtual
**`input`** (in `param_ranges` and `derived_params`), and the mapping is
returned in `data.param_aliases`. So sweeping `slow_kama_normalized_length`
(DMX resolver id) is converted to `kama_normalized_length` automatically. A
virtual parameter that no resolver resolves for the running strategy raises a
clear error instead of silently producing identical cells. Inline resolvers
override by `id`; `--resolver_config=/path.json` loads a different file. The
applied resolvers are returned in `data.resolvers`. Extend via hooks,
middlewares and custom ops (`core.params.hook` / `middleware` / `register_op`).

Virtual parameters accept the same sweep specs as any parameter — a list or
`{"min","max","step"}` / `{"min","max","count"}` / `{"values":[...]}`:

```bash
# explicit min/max/step on the virtual axis
'--param_ranges={"kama_normalized_length":{"min":1,"max":8,"step":1},"entry_filter":{"min":0.5,"max":1.5,"step":0.25}}'

# or use the range declared in the resolver config
'--param_ranges={"kama_normalized_length":"config","entry_filter":{"min":0.5,"max":1.5,"step":0.25}}'

# or auto-inject configured virtual ranges (only for resolvers matching the strategy)
--param_ranges='{"entry_filter":{"min":0.5,"max":1.5,"step":0.25}}' --use_config_ranges=true
```

Declare defaults in the config (`virtual_params`, or a `range` on a resolver):
```json
"virtual_params": {
  "kama_normalized_length": {"range": {"min": 1, "max": 8, "step": 1}}
}
```

Over MCP the same works via `run_heatmap` / `run_automator`
(`resolvers`, `resolver_config`, `use_config_ranges`); `get_schema` returns a
`resolvers` section listing the default virtual parameters and their ranges:

```python
run_heatmap(asset="BTCUSDT", strategy="LiveKAMASSLStrategy",
            start_date="2024-01-01", end_date="2024-06-01",
            param_ranges={"kama_normalized_length": {"min": 1, "max": 8, "step": 1},
                          "entry_filter": {"min": 0.5, "max": 1.5, "step": 0.25}},
            workers=1)
```

See [`docs/parameter-resolution-suite.md`](docs/parameter-resolution-suite.md)
for the full design (op catalogue, hooks/middlewares, data flow).

### Runtime debugging with `series`

`series` returns per-candle indicator and signal columns so a model can answer
"why did this trade / no trade happen?". Small results are returned inline;
large results are written to a Parquet file (falls back to CSV if `pyarrow` is
missing) and only the path is returned — query it with DuckDB.

```bash
python test.py series --asset=BTCUSDT --strategy=LiveKAMASSLStrategy \
    --start_date=2024-01-01 --end_date=2024-02-01 \
    --columns price_close,long_entry,short_entry,long_exit,short_exit,exit_reason \
    --tail 500 --api
```

Options: `--columns a,b,c`, `--tail N`, `--format json|parquet`,
`--inline_max_cells N` (default `5000`).

## MCP server

`mcp_server/server.py` exposes the API over the **Model Context Protocol**
(stdio transport). Each tool call runs `python test.py <tool> ... --api` as a
subprocess and returns the JSON envelope. This keeps the MCP stdout channel
clean (the tools print heavily and use multiprocessing). The server opens **no
network ports** — it only speaks JSON-RPC on stdin/stdout.

Tools exposed:

| Tool | Purpose |
|------|---------|
| `get_schema` | strategies, parameters, parameter ranges, intervals |
| `fetch_data` | ensure/refresh the OHLC cache for an asset/interval |
| `run_pnl` | PnL metrics + trade list (both/long/short) |
| `run_chart_analysis` | interactive chart + summary + trades |
| `run_heatmap` | parameter sweep: `grid`, `best`, `robustness`, `plateaus` |
| `analyze_plateaus` | algorithmic plateau/region detection on a grid (no re-backtest) |
| `validate_plateau` | walk-forward validation of a parameter region |
| `plot_plateau` | score matrix + region outlines rendered to a PNG |
| `run_automator` | heatmap across multiple pairs |
| `get_series` | per-candle indicator/signal series (debugging) |
| `run_tool` | generic escape hatch for any tool |
| `list_artifacts` / `read_artifact` | list/read generated reports |

### 1. Prerequisites

- The Docker container is running:

  ```bash
  docker compose up -d
  ```

- The `mcp` package is installed **inside the container** (it is listed in
  `requirements.txt`). If your image predates it:

  ```bash
  # quick (running container only, lost on recreate)
  docker compose exec app pip install mcp

  # permanent (rebuild the image)
  docker compose build app && docker compose up -d
  ```

Because the strategy dependencies (TA-Lib, ccxt, ib_async) and the data caches
live in the container, the MCP server must run **inside the container** — the
client launches it via `docker exec -i`.

### 2. Verify the server starts

```bash
# starts the stdio server and waits for JSON-RPC on stdin (Ctrl-C to stop)
docker exec -i damians-heatmap-dev python mcp_server/server.py

# or just list the registered tools
docker exec -i damians-heatmap-dev python -c \
  "import mcp_server.server as s; print(sorted(t.name for t in s.mcp._tool_manager.list_tools()))"
```

### 3. Register with your MCP client

The only difference between clients is the config file and the key
(`mcp` vs `mcpServers`); the launch command is always
`docker exec -i damians-heatmap-dev python mcp_server/server.py`.

**opencode** — `~/.config/opencode/opencode.json`:

```json
{
  "mcp": {
    "damians-heatmap": {
      "type": "local",
      "command": ["docker", "exec", "-i", "damians-heatmap-dev", "python", "mcp_server/server.py"],
      "enabled": true,
      "timeout": 600000,
      "autoApprove": ["get_schema", "fetch_data", "run_pnl", "run_chart_analysis", "get_series", "list_artifacts", "read_artifact", "analyze_plateaus", "plot_plateau"]
    }
  }
}
```

**Claude Desktop** — `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "damians-heatmap": {
      "command": "docker",
      "args": ["exec", "-i", "damians-heatmap-dev", "python", "mcp_server/server.py"]
    }
  }
}
```

**Cursor / other stdio clients** — command `docker`, args
`exec -i damians-heatmap-dev python mcp_server/server.py`.

After editing the config, **restart the client** so it picks up the new server.

### 4. Use it

Once connected, the agent can call the tools directly, e.g.:

```text
run_pnl(asset="BTCUSDT", strategy="LiveKAMASSLStrategy",
        start_date="2024-01-01", end_date="2024-03-01", direction="both")

run_heatmap(asset="BTCUSDT", strategy="LiveKAMASSLStrategy",
            start_date="2024-01-01", end_date="2024-03-01",
            param_ranges={"entry_filter": {"min": 0.5, "max": 1.5, "step": 0.25},
                          "exit_filter": {"min": 0.75, "max": 1.75, "step": 0.25}},
            workers=1)

get_series(asset="BTCUSDT", strategy="LiveKAMASSLStrategy",
           start_date="2024-01-01", end_date="2024-02-01",
           columns=["price_close", "long_entry", "long_exit", "exit_reason"], tail=500)
```

The strategy-debugging and heatmap-optimization skills (`strategy-debugging`,
`heatmap-optimization`) document the full workflows.

### 5. Configuration & environment

| Variable | Default | Purpose |
|----------|---------|---------|
| `HEATMAP_PROJECT_ROOT` | parent of `mcp_server/` | project directory the tools run in |
| `HEATMAP_PYTHON` | current interpreter | Python used to spawn `test.py` |
| `HEATMAP_MCP_TIMEOUT` | `1500` | subprocess timeout in seconds (heavy heatmaps need more) |

Set them in the client config (e.g. opencode `"environment": {...}`) if needed.

### 6. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `No module named 'mcp'` | install it in the container (see step 1) |
| `no output from test.py` | run the CLI directly (`python test.py <tool> ... --api`) and read `stderr` |
| Client shows no tools | restart the client after editing the config |
| Timeout on `run_heatmap` | lower the grid / `max_combos` or raise the timeout |
| Viewer URLs unreachable | the viewer binds `127.0.0.1:8900`; open the URL on the host |


## Tests

The test suite uses `pytest` (dev dependency, see `requirements-dev.txt`).

```bash
# inside the container (has all strategy dependencies)
docker compose exec app python -m pytest                 # all tests
docker compose exec app python -m pytest -m "not slow"   # skip the heatmap sweep
docker compose exec app python -m pytest -m slow         # only heavy tests
```

Coverage:

| Area | Tests |
|------|-------|
| JSON serialization | `tests/test_serialize.py` (numpy/pandas/timestamps, inf/NaN, trades) |
| CLI argument parsing | `tests/test_cli_args.py` (`--k=v`, `--flag`, JSON, list keys) |
| API helpers | `tests/test_api_unit.py` (ranges, strategy params, lookback, envelope, schema) |
| Heatmap results | `tests/test_heatmap_structured.py` (grid/best/robustness) |
| `--api` end-to-end | `tests/test_api_integration.py` (pnl, chart_analysis, series, fetcher, schema) |
| Heatmap sweep | `tests/test_api_slow.py` (marked `slow`) |
| MCP server | `tests/test_mcp_server.py` (tool registration; skipped without `mcp`) |

Tests are marked `integration` (use cached OHLC data) and `slow` (parameter
sweeps). Data-dependent tests are skipped automatically when the cache is
missing.

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
5. **core/fetcher.py** - Fetches market data

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
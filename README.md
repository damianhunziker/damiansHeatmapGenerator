# Heatmap Generator

Welcome to the Heatmap Generator, a sophisticated tool designed to analyze trading strategies and visualize their performance through interactive heatmaps. This project leverages advanced AI techniques to streamline development and enhance functionality.

## Overview

The Heatmap Generator allows users to:

- Analyze trading strategies using historical data.
- Visualize strategy performance with interactive heatmaps.
- Customize parameters to optimize trading strategies.

## Features

- **AI-Enhanced Development**: This project was developed with significant input from AI, ensuring efficient code and innovative solutions.
- **Interactive Heatmaps**: Visualize strategy performance across different parameter combinations.
- **Comprehensive Analysis**: Generate detailed reports on strategy performance, including profit, drawdown, and trade statistics.

## Getting Started

### Prerequisites

- Python 3.7 or higher
- Required Python packages: `pandas`, `numpy`, `matplotlib`, `altair`, `tqdm`, `requests`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/heatmap_generator.git
   cd heatmap_generator
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. **Prepare Your Data**: Ensure your data is formatted correctly and available for analysis.

2. **Configure Parameters**: Adjust the parameters in `heatmap.py` to suit your analysis needs, including `initial_equity`, `fee_pct`, `last_n_candles_analyze`, and `last_n_candles_display`.

3. **Run the Heatmap Generator**:
   ```bash
   python heatmap.py
   ```

4. **View Results**: The heatmap and analysis results will be saved as an HTML file and automatically opened in your default web browser.

### Example

To analyze a strategy with specific parameters, modify the `create_heatmap` function in `heatmap.py`:

python
create_heatmap(data, KAMAStrategy, param_ranges, initial_equity=10000, fee_pct=0.1, last_n_candles_analyze=12000, last_n_candles_display=9000, interval='1h', asset='BTC/USD', strategy_name='KAMA Strategy')

# heatmap_generator

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your improvements.

## Acknowledgments

This project was developed with extensive use of AI, which provided valuable insights and optimizations throughout the development process. Special thanks to the AI tools and platforms that made this project possible.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
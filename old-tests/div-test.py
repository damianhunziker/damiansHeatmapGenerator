import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classes.indicators.divergence import DivergenceDetector
from strategy_utils import fetch_data
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta

# Fetch data
df = fetch_data("ETHUSDT", "4h")

# Only show last 2 years
two_years_ago = datetime.now() - timedelta(days=730)
start_date = two_years_ago.strftime('%Y-%m-%d')
end_date = datetime.now().strftime('%Y-%m-%d')

# Initialize DivergenceDetector
detector = DivergenceDetector(lookback=20, source='Close', dontconfirm=False)
detector.df = df

# Set date range
detector.set_date_range(start_date, end_date)

# Calculate indicators
df['momentum'] = df['price_close'] - df['price_close'].shift(14)
df['momentum'] = df['momentum'].cumsum()

# Calculate Accumulation/Distribution
df['ad_raw'] = ((2 * df['price_close'] - df['price_low'] - df['price_high']) / 
                (df['price_high'] - df['price_low'])) * df['volume_traded']
df['ad'] = df['ad_raw'].cumsum()

# Detect divergences
divergences = detector.detect_divergences(df, indicator='momentum', length=14, order=5)

# Create figure with subplots
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                   vertical_spacing=0.03, 
                   row_heights=[0.5, 0.25, 0.25])

# Filter data for plotting
df_plot = df[start_date:end_date]

# Candlestick Chart
fig.add_trace(go.Candlestick(
    x=df_plot.index,
    open=df_plot['price_open'],
    high=df_plot['price_high'],
    low=df_plot['price_low'],
    close=df_plot['price_close'],
    name='OHLC'
), row=1, col=1)

# Momentum and AD
fig.add_trace(go.Scatter(
    x=df_plot.index,
    y=df_plot['momentum'],
    name='Momentum'
), row=2, col=1)

fig.add_trace(go.Scatter(
    x=df_plot.index,
    y=df_plot['ad'],
    name='AD'
), row=3, col=1)

# Add divergence lines
for div in divergences:
        if div['type'] == 'bearish':
            color = 'red'
        price_y0 = df_plot.iloc[div['prev_price_idx']]['price_high']
        price_y1 = df_plot.iloc[div['price_idx']]['price_high']
        else:
            color = 'green'
        price_y0 = df_plot.iloc[div['prev_price_idx']]['price_low']
        price_y1 = df_plot.iloc[div['price_idx']]['price_low']
        
    # Price lines
        fig.add_shape(
            type='line',
        x0=df_plot.index[div['prev_price_idx']],
            y0=price_y0,
        x1=df_plot.index[div['price_idx']],
            y1=price_y1,
            line=dict(color=color, width=3),
            row=1, col=1
        )
        
    # Indicator lines
            fig.add_shape(
                type='line',
        x0=df_plot.index[div['prev_price_idx']],
        y0=df_plot['momentum'].iloc[div['prev_price_idx']],
        x1=df_plot.index[div['price_idx']],
        y1=df_plot['momentum'].iloc[div['price_idx']],
                line=dict(color=color, width=2),
                row=2, col=1
            )
        
# Update layout
fig.update_layout(
    title='ETH/USDT 4H with Divergences',
    yaxis_title='Price',
    yaxis2_title='Momentum',
    yaxis3_title='AD',
    template='plotly_dark',
    xaxis_rangeslider_visible=False
)

# Show chart
fig.show()

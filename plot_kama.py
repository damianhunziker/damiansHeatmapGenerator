import json
from datetime import datetime
import plotly.graph_objects as go

# Read the JSON data
with open('TVBtc4hkama16-4-24', 'r') as file:
    data = json.load(file)

# Extract timestamps and values
timestamps = [datetime.fromtimestamp(ts/1000) for ts, _ in data['data']]
values = [value for _, value in data['data']]

# Create the plot
fig = go.Figure()

# Add the line plot
fig.add_trace(go.Scatter(
    x=timestamps,
    y=values,
    mode='lines',
    name='KAMA',
    line=dict(color='blue', width=2)
))

# Update layout
fig.update_layout(
    title='KAMA Values Over Time',
    xaxis_title='Time',
    yaxis_title='Value',
    template='plotly_white',
    hovermode='x unified',
    showlegend=True
)

# Show the plot
fig.show() 
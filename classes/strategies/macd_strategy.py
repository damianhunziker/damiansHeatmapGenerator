import numpy as np
import pandas as pd
import plotly.graph_objects as go
from ..base_strategy import BaseStrategy

class SMACrossoverStrategy(BaseStrategy):
    def __init__(self, fast_length=20, slow_length=50, initial_equity=10000, fee_pct=0.04, trade_direction="both"):
        super().__init__(initial_equity=initial_equity, fee_pct=fee_pct)
        self.fast_length = fast_length
        self.slow_length = slow_length
        self.trade_direction = "long"  # "long", "short", or "both"
    
    @staticmethod
    def get_parameter_ranges():
        return {
            'fast_length': np.arange(5, 50, 1),
            'slow_length': np.arange(20, 200, 1),
        }
    
    @staticmethod
    def get_parameters():
        return {
            'fast_length': (20, "Fast SMA Length"),
            'slow_length': (50, "Slow SMA Length"),
        }
    
    def calculate_sma(self, data, length):
        """Calculate Simple Moving Average"""
        return data['price_close'].rolling(window=length).mean()
    
    def prepare_signals(self, data):
        df = data.copy()
        
        # Calculate SMAs
        df['fast_sma'] = self.calculate_sma(df, self.fast_length)
        df['slow_sma'] = self.calculate_sma(df, self.slow_length)
        
        # Calculate crossover signals
        df['fast_above_slow'] = df['fast_sma'] > df['slow_sma']
        
        # Initialize all signals as False
        df['long_entry'] = False
        df['long_exit'] = False
        df['short_entry'] = False
        df['short_exit'] = False
        
        # Generate crossover signals
        for i in range(1, len(df)):
            # Crossover (crossing up)
            if df['fast_above_slow'].iloc[i] and not df['fast_above_slow'].iloc[i-1]:
                df.iloc[i, df.columns.get_loc('long_entry')] = True
                df.iloc[i, df.columns.get_loc('short_exit')] = True
                
            # Crossunder (crossing down)
            if not df['fast_above_slow'].iloc[i] and df['fast_above_slow'].iloc[i-1]:
                df.iloc[i, df.columns.get_loc('long_exit')] = True
                df.iloc[i, df.columns.get_loc('short_entry')] = True
        
        # Filter signals based on trade direction
        if self.trade_direction == "long":
            df['short_entry'] = False
            df['short_exit'] = False
        elif self.trade_direction == "short":
            df['long_entry'] = False
            df['long_exit'] = False
        
        # Color coding for visualization
        df['trend_color'] = 'gray'
        long_position = False
        short_position = False
        trend_colors = []
        
        for i in range(len(df)):
            if self.trade_direction in ["long", "both"]:
                if not long_position and df['long_entry'].iloc[i]:
                    long_position = True
                elif long_position and df['long_exit'].iloc[i]:
                    long_position = False
                    
            if self.trade_direction in ["short", "both"]:
                if not short_position and df['short_entry'].iloc[i]:
                    short_position = True
                elif short_position and df['short_exit'].iloc[i]:
                    short_position = False
            
            if long_position:
                color = 'green'
            elif short_position:
                color = 'red'
            else:
                color = 'gray'
            trend_colors.append(color)
        
        df['trend_color'] = trend_colors
        return df
    
    def process_chunk(self, data):
        """Process data chunk and return trades"""
        trades = []
        long_position = False
        short_position = False
        entry_time = None
        entry_price = None
        current_equity = self.initial_equity
        
        for i in range(1, len(data)):
            current_time = data.index[i]
            
            # Check for entries
            if not long_position and not short_position:
                if self.trade_direction in ["long", "both"] and data['long_entry'].iloc[i]:
                    long_position = True
                    entry_time = current_time
                    entry_price = data['price_close'].iloc[i]
                elif self.trade_direction in ["short", "both"] and data['short_entry'].iloc[i]:
                    short_position = True
                    entry_time = current_time
                    entry_price = data['price_close'].iloc[i]
            
            # Check for exits
            elif long_position and data['long_exit'].iloc[i]:
                exit_price = data['price_close'].iloc[i]
                trades.append(self.create_trade(entry_time, current_time, entry_price, exit_price, current_equity, "long"))
                current_equity = trades[-1][4] + current_equity if trades[-1][4] is not None else current_equity
                long_position = False
                entry_time = None
                entry_price = None
                
            elif short_position and data['short_exit'].iloc[i]:
                exit_price = data['price_close'].iloc[i]
                trades.append(self.create_trade(entry_time, current_time, entry_price, exit_price, current_equity, "short"))
                current_equity = trades[-1][4] + current_equity if trades[-1][4] is not None else current_equity
                short_position = False
                entry_time = None
                entry_price = None
        
        # Handle open positions at the end
        if long_position or short_position:
            trade_type = "long" if long_position else "short"
            trades.append((entry_time, data.index[-1], entry_price, data['price_close'].iloc[-1], None, None, None, trade_type))
        
        return trades
    
    def create_trade(self, entry_time, exit_time, entry_price, exit_price, current_equity, trade_type):
        """Helper method to create a trade with calculated profits"""
        position_size = current_equity
        contracts = position_size / entry_price
        
        if trade_type == "long":
            gross_profit = contracts * (exit_price - entry_price)
        else:  # short
            gross_profit = contracts * (entry_price - exit_price)
        
        entry_fee = position_size * self.fee_pct
        exit_fee = (position_size + gross_profit) * self.fee_pct
        total_fee = entry_fee + exit_fee
        net_profit = gross_profit - total_fee # gross_profit - total_fee
        
        return (entry_time, exit_time, entry_price, exit_price, net_profit, gross_profit, total_fee, trade_type)
    
    def add_indicator_traces(self, fig, display_data, row, col):
        # Add Fast SMA
        fig.add_trace(go.Scatter(
            x=display_data.index,
            y=display_data['fast_sma'],
            name=f'Fast SMA ({self.fast_length})',
            line=dict(color='blue')
        ), row=row, col=col)
        
        # Add Slow SMA
        fig.add_trace(go.Scatter(
            x=display_data.index,
            y=display_data['slow_sma'],
            name=f'Slow SMA ({self.slow_length})',
            line=dict(color='red')
        ), row=row, col=col)
        
        return fig
    
    def add_strategy_traces(self, fig, display_data):
        """Add strategy-specific traces to the chart"""
        # Add entry points
        
        # Add indicator traces
        self.add_indicator_traces(fig, display_data, row=2, col=1)
        
        return fig
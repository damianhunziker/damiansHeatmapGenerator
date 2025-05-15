from tqdm import tqdm
import pandas as pd
import numpy as np

class TradeAnalyzer:
    def __init__(self, strategy, strategy_params):
        """Initialize the analyzer with a strategy instance"""
        self.strategy = strategy  # Use the passed strategy instance directly
        self.strategy_params = strategy_params
        self.initial_equity = strategy_params.get('initial_equity', 10000)
        self.fee_pct = strategy_params.get('fee_pct', 0.04)
        self.price_multiplier = strategy_params.get('price_multiplier', 1.0)
        print(f"TradeAnalyzer - Init - start_date: {strategy_params.get('start_date')}, end_date: {strategy_params.get('end_date')}")
    
    def analyze_data(self, data, last_n_candles=None, silent=False, start_date=None, end_date=None):
        """Analyzes the data and returns trades and statistics"""
        if not silent:
            print("\nCalculating signals...")
            pbar = tqdm(total=100, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}')

        # Get dates from strategy_params if not provided
        if start_date is None:
            start_date = self.strategy_params.get('start_date')
        if end_date is None:
            end_date = self.strategy_params.get('end_date')
            
        print(f"TradeAnalyzer - analyze_data - start_date: {start_date}, end_date: {end_date}")
        
        # Filtere die Daten nach Datum, falls angegeben
        if start_date and end_date:
            data = data[start_date:end_date].copy()
        elif start_date:
            data = data[start_date:].copy()
        elif end_date:
            data = data[:end_date].copy()
        else:
            data = data.copy()
            
        # Normalisiere Preise falls nötig
        if self.price_multiplier > 1:
            price_columns = ['price_open', 'price_high', 'price_low', 'price_close']
            for col in price_columns:
                if col in data.columns:
                    data[col] = data[col] * self.price_multiplier
        
        # Initialize strategy and calculate signals
        analyze_data = self.strategy.prepare_signals(data)
        if not silent:
            pbar.update(50)
        
        # Trim data for analysis if specified
        if last_n_candles and last_n_candles > 0:
            analyze_data = analyze_data.tail(last_n_candles)
        if not silent:
            pbar.update(50)
            pbar.close()
        
        # Generate trades
        trades = self._generate_trades(analyze_data)
        
        if not silent:
            self._print_trade_statistics(trades)
        
        return trades, analyze_data
    
    def _generate_trades(self, data):
        """Generates the trade list"""
        trades = []
        in_position = False
        entry_time = None
        entry_price = None
        current_equity = float(self.initial_equity)
        
        for i in range(len(data)):
            current_time = data.index[i]
            current_price = float(data['price_close'].iloc[i])
            
            if hasattr(data, 'short_entry'):
                if not in_position and data['short_entry'].iloc[i]:
                    entry_time = current_time
                    entry_price = float(current_price)
                    position_size = current_equity
                    entry_fee = position_size * (self.fee_pct / 100)
                    in_position = True
                    trade_type = "SHORT"
                elif in_position and data['short_exit'].iloc[i]:
                    exit_time = current_time
                    exit_price = float(current_price)
                    exit_reason = data['exit_reason'].iloc[i]
                    
                    contracts = position_size / entry_price
                    price_change = entry_price - exit_price
                    gross_profit = float(contracts * price_change)
                    
                    exit_position_size = position_size + gross_profit
                    exit_fee = exit_position_size * (self.fee_pct / 100)
                    
                    net_profit = gross_profit - entry_fee - exit_fee
                    
                    trades.append((
                        entry_time,
                        exit_time,
                        entry_price,
                        exit_price,
                        net_profit,
                        gross_profit,
                        entry_fee + exit_fee,
                        trade_type,
                        exit_reason
                    ))
                    
                    current_equity += net_profit
                    in_position = False
            
            if hasattr(data, 'long_entry'):
                if not in_position and data['long_entry'].iloc[i]:
                    entry_time = current_time
                    entry_price = float(current_price)
                    position_size = current_equity
                    entry_fee = position_size * (self.fee_pct / 100)
                    in_position = True
                    trade_type = "LONG"
                elif in_position and data['long_exit'].iloc[i]:
                    exit_time = current_time
                    exit_price = float(current_price)
                    exit_reason = data['exit_reason'].iloc[i]
                    
                    contracts = position_size / entry_price
                    price_change = exit_price - entry_price
                    gross_profit = float(contracts * price_change)
                    
                    exit_position_size = position_size + gross_profit
                    exit_fee = exit_position_size * (self.fee_pct / 100)
                    
                    net_profit = gross_profit - entry_fee - exit_fee
                    
                    trades.append((
                        entry_time,
                        exit_time,
                        entry_price,
                        exit_price,
                        net_profit,
                        gross_profit,
                        entry_fee + exit_fee,
                        trade_type,
                        exit_reason
                    ))
                    
                    current_equity += net_profit
                    in_position = False
        
        return trades
    
    def _print_trade_statistics(self, trades):
        """Prints the trade statistics"""
        if not trades:
            print("\nNo trades found!")
            return
        
        # Print header
        print("\nTrade List:")
        print("=" * 140)
        print(f"{'#':3} | {'Type':<6} | {'Entry Time':<16} | {'Exit Time':<16} | {'Entry':<10} | {'Exit':<10} | "
              f"{'Gross P/L':<10} | {'Fees':<8} | {'Net P/L':<10} | {'%':<8} | {'Exit Reason':<25}")
        print("=" * 140)
        
        current_equity = self.initial_equity
        for i, trade in enumerate(trades, 1):
            # Handle both old and new trade tuple formats
            if len(trade) >= 9:  # New format with exit_reason
                entry_time, exit_time, entry_price, exit_price, net_profit, gross_profit, fees, trade_type, exit_reason = trade
            else:  # Old format without exit_reason
                entry_time, exit_time, entry_price, exit_price, net_profit, gross_profit, fees, trade_type = trade
                exit_reason = "N/A"
            
            profit_pct = (net_profit / current_equity) * 100 if net_profit is not None else 0
            
            print(f"{i:3d} | {trade_type:<6} | {entry_time.strftime('%Y-%m-%d %H:%M')} | "
                  f"{exit_time.strftime('%Y-%m-%d %H:%M')} | "
                  f"${entry_price:10.2f} | ${exit_price:10.2f} | "
                  f"${gross_profit:10.2f} | ${fees:8.2f} | ${net_profit:10.2f} | {profit_pct:8.2f}% | {exit_reason:<25}")
            
            if net_profit is not None:
                current_equity += net_profit
        
        print("=" * 140)
        
        # Calculate overall statistics
        total_profit = sum(trade[4] for trade in trades if trade[4] is not None)  # net_profit
        total_gross = sum(trade[5] for trade in trades if trade[5] is not None)   # gross_profit
        total_fees = sum(trade[6] for trade in trades if trade[6] is not None)    # fees
        winning_trades = sum(1 for trade in trades if trade[4] is not None and trade[4] > 0)
        completed_trades = sum(1 for trade in trades if trade[4] is not None)
        win_rate = (winning_trades / completed_trades) * 100 if completed_trades > 0 else 0
        
        print(f"\nOverall Statistics:")
        print(f"Gross Profit: ${total_gross:.2f}")
        print(f"Total Fees: ${total_fees:.2f}")
        print(f"Net Profit: ${total_profit:.2f}")
        print(f"Winning Trades: {winning_trades} of {completed_trades} ({win_rate:.2f}%)")

    def calculate_metrics(self, equity_curve, trades):
        """Calculate various financial metrics from an equity curve and trades."""
        # Initialize variables
        max_equity = equity_curve[0]
        max_drawdown = 0
        max_drawdown_usd = 0
        total_fees = 0
        total_gross_profit = 0
        total_net_profit = 0
        total_trade_duration = 0
        winning_trades = 0
        total_profit = 0
        total_loss = 0

        # Calculate drawdown using the provided formula
        for equity in equity_curve:
            if equity > max_equity:
                max_equity = equity
            current_drawdown = (max_equity - equity) / max_equity
            current_drawdown_usd = max_equity - equity
            if current_drawdown > max_drawdown:
                max_drawdown = current_drawdown
                max_drawdown_usd = current_drawdown_usd
        max_drawdown_pct = max_drawdown * 100

        # Calculate returns
        if len(equity_curve) > 1:
            returns = np.diff(equity_curve) / equity_curve[:-1]
            if len(returns) > 1:
                returns_mean = np.mean(returns)
                returns_std = np.std(returns, ddof=1) if len(returns) > 1 else 0
                sharpe_ratio = returns_mean / returns_std if returns_std != 0 else 0
                downside_returns = returns[returns < 0]
                downside_std = np.std(downside_returns, ddof=1) if len(downside_returns) > 1 else 0
                sortino_ratio = returns_mean / downside_std if downside_std != 0 else 0
                volatility = returns_std
            else:
                sharpe_ratio = 0
                sortino_ratio = 0
                volatility = 0
        else:
            sharpe_ratio = 0
            sortino_ratio = 0
            volatility = 0

        # Calculate trade-related metrics
        for trade in trades:
            entry_time, exit_time, entry_price, exit_price, net_profit, gross_profit, fees, trade_type, exit_reason = trade
            if net_profit is not None:  # Only consider completed trades
                total_fees += fees
                total_gross_profit += gross_profit
                total_net_profit += net_profit
                total_trade_duration += (exit_time - entry_time).total_seconds()
                if net_profit > 0:
                    winning_trades += 1
                    total_profit += net_profit
                else:
                    total_loss += abs(net_profit)

        num_trades = sum(1 for trade in trades if trade[4] is not None)  # Count only completed trades
        avg_trade_profit = total_net_profit / num_trades if num_trades > 0 else 0
        avg_trade_profit_pct = (avg_trade_profit / equity_curve[0]) * 100 if equity_curve[0] != 0 else 0
        avg_trade_duration = total_trade_duration / num_trades if num_trades > 0 else 0
        profit_pct = (total_net_profit / equity_curve[0]) * 100 if equity_curve[0] != 0 else 0
        win_rate = (winning_trades / num_trades) * 100 if num_trades > 0 else 0
        profit_factor = total_profit / total_loss if total_loss != 0 else float('inf')

        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_usd': max_drawdown_usd,
            'max_drawdown_pct': max_drawdown_pct,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'volatility': volatility,
            'total_fees': total_fees,
            'total_gross_profit': total_gross_profit,
            'total_net_profit': total_net_profit,
            'avg_trade_profit': avg_trade_profit,
            'avg_trade_profit_pct': avg_trade_profit_pct,
            'avg_trade_duration': avg_trade_duration,
            'profit_pct': profit_pct,
            'win_rate': win_rate,
            'profit_factor': profit_factor
        }

    @staticmethod
    def process_chunk_base(data, trade_direction, initial_equity, fee_pct):
        """Base implementation of process_chunk that all strategies can use"""
        trades = []
        in_position = False
        entry_time = None
        entry_price = None
        current_equity = initial_equity
        current_direction = None
        
        for i in range(1, len(data)):
            current_time = data.index[i]
            
            if not in_position:
                if (trade_direction in ['long', 'both'] and data['long_entry'].iloc[i]) or \
                   (trade_direction in ['short', 'both'] and data['short_entry'].iloc[i]):
                    entry_time = current_time
                    entry_price = data['price_close'].iloc[i]
                    in_position = True
                    current_direction = 'long' if data['long_entry'].iloc[i] else 'short'
                    
            elif (current_direction == 'long' and data['long_exit'].iloc[i]) or \
                 (current_direction == 'short' and data['short_exit'].iloc[i]):
                exit_price = data['price_close'].iloc[i]
                
                # Calculate position size and contracts
                position_size = current_equity
                contracts = position_size / entry_price
                
                # Calculate profit based on trade direction
                if current_direction == "long":
                    gross_profit = contracts * (exit_price - entry_price)
                else:  # short
                    gross_profit = contracts * (entry_price - exit_price)
                
                # Calculate fees and net profit
                entry_fee = position_size * fee_pct
                exit_fee = (position_size + gross_profit) * fee_pct
                total_fee = entry_fee + exit_fee
                net_profit = gross_profit - total_fee
                
                # Update equity
                current_equity += net_profit
                
                # Add trade to list
                trade = (entry_time, current_time, entry_price, exit_price, net_profit, gross_profit, total_fee, current_direction)
                trades.append(trade)
                
                in_position = False
                entry_time = None
                entry_price = None
        
        # If the last position is still open
        if in_position:
            trade = (entry_time, data.index[-1], entry_price, data['price_close'].iloc[-1], None, None, None, current_direction)
            trades.append(trade)
        
        return trades 
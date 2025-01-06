from tqdm import tqdm
import pandas as pd
import numpy as np

class TradeAnalyzer:
    def __init__(self, strategy_class, strategy_params):
        self.strategy = strategy_class(**strategy_params)
        self.initial_equity = strategy_params.get('initial_equity', 10000)
        self.fee_pct = strategy_params.get('fee_pct', 0.04)
    
    def analyze_data(self, data, last_n_candles_analyze=None, last_n_candles_display=None, silent=False):
        """Analyzes the data and returns trades and statistics"""
        if not silent:
            print("\nCalculating signals...")
            pbar = tqdm(total=100, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}')
        
        # Initialize strategy and calculate signals for the entire analysis period
        analyze_data = self.strategy.prepare_signals(data)
        if not silent:
            pbar.update(50)
        
        # Trim data for analysis
        if last_n_candles_analyze and last_n_candles_analyze > 0:
            analyze_data = analyze_data.tail(last_n_candles_analyze)
        if not silent:
            pbar.update(50)
            pbar.close()
        
        # Generate trades
        trades = self._generate_trades(analyze_data)
        
        # Prepare data for display
        display_data = analyze_data.copy()
        if last_n_candles_display and last_n_candles_display > 0:
            display_data = display_data.tail(last_n_candles_display)
        
        if not silent:
            self._print_trade_statistics(trades)
        
        return trades, display_data
    
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
                        trade_type
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
                        trade_type
                    ))
                    
                    current_equity += net_profit
                    in_position = False
        
        return trades
    
    def _print_trade_statistics(self, trades):
        """Prints the trade statistics"""
        if not trades:
            print("\nNo trades found!")
            return
        
        print(f"\nClosed trades in analysis period: {len(trades)}")
        print("\nDetailed trade list:")
        print("=" * 120)
        print(f"{'No':>3} | {'Type':<6} | {'Entry Time':^19} | {'Exit Time':^19} | {'Entry Price':>10} | {'Exit Price':>10} | {'Gross $':>10} | {'Fees':>8} | {'Net $':>10} | {'Profit %':>8}")
        print("-" * 120)
        
        current_equity = self.initial_equity
        for i, trade in enumerate(trades, 1):
            entry_time, exit_time, entry_price, exit_price, net_profit, gross_profit, fees, trade_type = trade
            profit_pct = (net_profit / current_equity) * 100
            
            print(f"{i:3d} | {trade_type:<6} | {entry_time.strftime('%Y-%m-%d %H:%M')} | "
                  f"{exit_time.strftime('%Y-%m-%d %H:%M')} | "
                  f"${entry_price:10.2f} | ${exit_price:10.2f} | "
                  f"${gross_profit:10.2f} | ${fees:8.2f} | ${net_profit:10.2f} | {profit_pct:8.2f}%")
            
            current_equity += net_profit
        
        print("=" * 120)
        
        # Calculate overall statistics
        total_profit = sum(trade[4] for trade in trades)  # net_profit
        total_gross = sum(trade[5] for trade in trades)   # gross_profit
        total_fees = sum(trade[6] for trade in trades)    # fees
        winning_trades = sum(1 for trade in trades if trade[4] > 0)
        win_rate = (winning_trades / len(trades)) * 100
        
        print(f"\nOverall Statistics:")
        print(f"Gross Profit: ${total_gross:.2f}")
        print(f"Total Fees: ${total_fees:.2f}")
        print(f"Net Profit: ${total_profit:.2f}")
        print(f"Winning Trades: {winning_trades} of {len(trades)} ({win_rate:.2f}%)")

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
            entry_time, exit_time, entry_price, exit_price, net_profit, gross_profit, fees, trade_type = trade
            total_fees += fees
            total_gross_profit += gross_profit
            total_net_profit += net_profit
            total_trade_duration += (exit_time - entry_time).total_seconds()
            if net_profit > 0:
                winning_trades += 1
                total_profit += net_profit
            else:
                total_loss += abs(net_profit)

        num_trades = len(trades)
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
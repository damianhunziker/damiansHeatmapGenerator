from abc import ABC, abstractmethod
from classes.trade_analyzer import TradeAnalyzer

class BaseStrategy(ABC):
    def __init__(self, initial_equity=10000, fee_pct=0.04, start_date=None, end_date=None, asset=None, **kwargs):
        self.initial_equity = initial_equity
        self.fee_pct = fee_pct
        self.start_date = start_date
        self.end_date = end_date
        self.asset = asset
        self.strategy_params = {
            'initial_equity': initial_equity,
            'fee_pct': fee_pct,
            'start_date': start_date,
            'end_date': end_date,
            'asset': asset
        }
        self.strategy_params.update(kwargs)

    @staticmethod
    @abstractmethod
    def get_parameters():
        """Returns the parameters and their descriptions"""
        pass
    
    @abstractmethod
    def prepare_signals(self, data):
        """Prepares the trading signals"""
        pass
    
    def process_chunk(self, data):
        """Process data chunk and return trades using the base implementation"""
        df = self.prepare_signals(data)
        return TradeAnalyzer.process_chunk_base(df, self.trade_direction, self.initial_equity, self.fee_pct)
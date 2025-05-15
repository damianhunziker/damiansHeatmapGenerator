from abc import ABC, abstractmethod
from classes.trade_analyzer import TradeAnalyzer

class BaseStrategy(ABC):
    def __init__(self, initial_equity=10000, fee_pct=0.04):
        self.initial_equity = initial_equity
        self.fee_pct = fee_pct
        self.strategy_params = {
            'initial_equity': initial_equity,
            'fee_pct': fee_pct
        }

    @staticmethod
    @abstractmethod
    def get_parameters():
        """Gibt die Parameter und ihre Beschreibungen zurück"""
        pass
    
    @abstractmethod
    def prepare_signals(self, data):
        """Bereitet die Handelssignale vor"""
        pass
    
    def process_chunk(self, data):
        """Process data chunk and return trades using the base implementation"""
        df = self.prepare_signals(data)
        return TradeAnalyzer.process_chunk_base(df, self.trade_direction, self.initial_equity, self.fee_pct) 
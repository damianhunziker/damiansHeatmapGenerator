from abc import ABC, abstractmethod

class BaseStrategy(ABC):
    def __init__(self, initial_equity=10000, fee_pct=0.04):
        self.initial_equity = initial_equity
        self.fee_pct = fee_pct
    
    @staticmethod
    @abstractmethod
    def get_parameters():
        """Gibt die Parameter und ihre Beschreibungen zurück"""
        pass
    
    @abstractmethod
    def prepare_signals(self, data):
        """Bereitet die Handelssignale vor"""
        pass
    
    @abstractmethod
    def process_chunk(self, data):
        """Verarbeitet die Daten und gibt die Trades zurück"""
        pass 
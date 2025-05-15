class LVNBreakoutScreener:
    def __init__(self):
        self.volume_analyzer = VolumeLiquidityAnalyzer()
        self.breakout_analyzer = BreakoutAnalyzer()
        
    def scan_for_opportunities(self, symbol, timeframe='1h'):
        # Daten abrufen
        df = self.fetch_market_data(symbol, timeframe)
        orderbook = self.fetch_orderbook(symbol)
        
        # LVN/HVN Zonen erkennen
        zones = self.volume_analyzer.detect_volume_zones(df)
        
        # Orderbuch-Analyse
        bid_gaps, ask_gaps = self.volume_analyzer.analyze_orderbook(orderbook)
        
        # Aktuelle Breakouts identifizieren
        opportunities = []
        current_price = df['price_close'].iloc[-1]
        
        for zone in zones:
            if self.is_breaking_out(current_price, zone):
                analysis = self.breakout_analyzer.analyze_breakout(
                    current_price, 
                    zone,
                    self.find_next_hvn_zone(zones, current_price)
                )
                
                if analysis['score'] > 1.5:  # Mindest-Score
                    opportunities.append({
                        'symbol': symbol,
                        'zone': zone,
                        'analysis': analysis
                    })
        
        return opportunities 
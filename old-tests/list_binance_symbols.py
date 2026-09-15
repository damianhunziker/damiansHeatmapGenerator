import requests
import pandas as pd
from datetime import datetime

def get_binance_symbols():
    # Hole alle Ticker-Informationen von Binance
    ticker_url = "https://api.binance.com/api/v3/ticker/price"
    exchange_info_url = "https://api.binance.com/api/v3/exchangeInfo"
    
    try:
        # Hole aktuelle Preise
        prices = requests.get(ticker_url).json()
        # Hole Exchange Info für zusätzliche Details
        exchange_info = requests.get(exchange_info_url).json()
        
        # Erstelle Dictionary mit Symbol-Details
        symbol_details = {
            symbol['symbol']: {
                'baseAsset': symbol['baseAsset'],
                'quoteAsset': symbol['quoteAsset'],
                'status': symbol['status']
            }
            for symbol in exchange_info['symbols']
        }
        
        # Erstelle DataFrame mit allen relevanten Informationen
        df = pd.DataFrame(prices)
        df['price'] = df['price'].astype(float)
        
        # Füge Base und Quote Asset hinzu
        df['baseAsset'] = df['symbol'].map(lambda x: symbol_details.get(x, {}).get('baseAsset', 'N/A'))
        df['quoteAsset'] = df['symbol'].map(lambda x: symbol_details.get(x, {}).get('quoteAsset', 'N/A'))
        df['status'] = df['symbol'].map(lambda x: symbol_details.get(x, {}).get('status', 'N/A'))
        
        # Filtere nur aktive Symbole
        df = df[df['status'] == 'TRADING']
        
        # Sortiere nach Quote Asset und Preis
        df = df.sort_values(['quoteAsset', 'price'], ascending=[True, False])
        
        # Formatiere die Ausgabe
        print(f"\nBinance Symbols - Updated: {datetime.now()}\n")
        
        # Gruppiere nach Quote Asset
        for quote_asset in sorted(df['quoteAsset'].unique()):
            quote_df = df[df['quoteAsset'] == quote_asset]
            print(f"\n=== {quote_asset} Pairs ===")
            print(f"{'Symbol':<12} {'Base':<8} {'Price':<15}")
            print("-" * 35)
            
            for _, row in quote_df.iterrows():
                # Formatiere den Preis basierend auf seiner Größe
                if row['price'] < 0.0001:
                    price_str = f"{row['price']:.8f}"
                elif row['price'] < 0.01:
                    price_str = f"{row['price']:.6f}"
                elif row['price'] < 1:
                    price_str = f"{row['price']:.4f}"
                elif row['price'] < 100:
                    price_str = f"{row['price']:.2f}"
                else:
                    price_str = f"{row['price']:.1f}"
                
                print(f"{row['symbol']:<12} {row['baseAsset']:<8} {price_str:<15}")
        
        return df
        
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

if __name__ == "__main__":
    df = get_binance_symbols()
    
    if df is not None:
        # Speichere die Daten in einer CSV-Datei
        filename = f"binance_symbols_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(filename, index=False)
        print(f"\nData saved to {filename}") 
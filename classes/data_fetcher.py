import requests
import pandas as pd
import os
import time
from datetime import timedelta
from decimal import Decimal
import numpy as np

class OHLCFetcher:
    def __init__(self, api_key=None):
        self.cache_dir = 'ohlc_cache'
        os.makedirs(self.cache_dir, exist_ok=True)

    def fetch_data(self, asset, interval, start_date=None, limit=100000):
        """Fetches OHLC data from Binance API or cache"""
        cache_file = self._get_cache_filename(asset, interval)
        
        # Check cache first
        if os.path.exists(cache_file):
            print(f"Loading data from cache: {cache_file}")
            df = pd.read_csv(cache_file, index_col='time_period_start', parse_dates=True)
            
            # Normalisiere die Preise wenn nötig
            price_cols = ['price_open', 'price_high', 'price_low', 'price_close']
            min_price = min(df[price_cols].min())
            
            if min_price < 0.1:
                # Berechne den Multiplikator
                multiplier = 10 ** (abs(int(np.log10(min_price))) + 1)
                print(f"Normalizing prices with multiplier: {multiplier}")
                
                # Multipliziere alle Preisspalten
                for col in price_cols:
                    df[col] = df[col] * multiplier
                
                # Speichere den Multiplikator in den Metadaten
                df.attrs['price_multiplier'] = multiplier
            else:
                df.attrs['price_multiplier'] = 1
                
            return df

        # If not in cache, fetch from Binance API
        print("Fetching data from Binance API...")
        df = self._fetch_from_binance(asset, interval, limit)

        # Convert to CoinAPI format
        df = self._convert_to_coinapi_format(df)
        
        # Normalisiere die Preise wenn nötig
        price_cols = ['price_open', 'price_high', 'price_low', 'price_close']
        min_price = min(df[price_cols].min())
        
        if min_price < 0.1:
            # Berechne den Multiplikator
            multiplier = 10 ** (abs(int(np.log10(min_price))) + 1)
            print(f"Normalizing prices with multiplier: {multiplier}")
            
            # Multipliziere alle Preisspalten
            for col in price_cols:
                df[col] = df[col] * multiplier
            
            # Speichere den Multiplikator in den Metadaten
            df.attrs['price_multiplier'] = multiplier
        else:
            df.attrs['price_multiplier'] = 1

        # Save to cache
        self._save_to_cache(df, cache_file)
        return df

    def _fetch_from_binance(self, symbol, interval, total_candles):
        """Fetches OHLC data from Binance API."""
        BINANCE_API_URL = "https://api.binance.com/api/v3/klines"
        LIMIT = 3000  # Max candles per request

        def fetch_ohlc(symbol, interval, start_time, limit=3000):
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": start_time,
                "limit": limit
            }
            response = requests.get(BINANCE_API_URL, params=params)
            response.raise_for_status()
            return response.json()

        candles = []
        start_time = 0  # Start from the earliest available data

        while len(candles) < total_candles:
            data = fetch_ohlc(symbol, interval, start_time, LIMIT)
            if not data:
                break

            candles.extend(data)
            start_time = data[-1][0] + 1  # Increment start_time to avoid overlaps
            print(f"Fetched {len(candles)} candles...")

            # To avoid rate limits
            time.sleep(0.1)

        # Limit to TOTAL_CANDLES
        candles = candles[:total_candles]

        # Create a DataFrame
        columns = [
            "open_time", "open", "high", "low", "close", "volume", "close_time",
            "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume", "ignore"
        ]
        df = pd.DataFrame(candles, columns=columns)

        # Convert timestamps from milliseconds to seconds
        df["open_time"] = df["open_time"] / 1000
        df["close_time"] = df["close_time"] / 1000

        # Convert timestamp to datetime and add 8 hours
        df["open_time"] = pd.to_datetime(df["open_time"], unit='s') + timedelta(hours=8)
        df["close_time"] = pd.to_datetime(df["close_time"], unit='s') + timedelta(hours=8)

        # Set the index to open_time
        df.set_index("open_time", inplace=True)

        # Convert OHLC values to Decimal for high precision
        for column in ["open", "high", "low", "close", "volume"]:
            df[column] = df[column].apply(Decimal)

        return df

    def _convert_to_coinapi_format(self, df):
        """Converts Binance DataFrame to CoinAPI format"""
        df.rename(columns={
            "open": "price_open",
            "high": "price_high",
            "low": "price_low",
            "close": "price_close",
            "volume": "volume_traded"
        }, inplace=True)
        df.index.name = 'time_period_start'
        return df[["price_open", "price_high", "price_low", "price_close", "volume_traded"]]

    def _get_cache_filename(self, asset, interval):
        """Generates a cache filename based on asset and interval"""
        return os.path.join(self.cache_dir, f"{asset}_{interval}_ohlc.csv")

    def _save_to_cache(self, df, filename):
        """Saves the dataframe to cache"""
        print(f"Saving data to cache: {filename}")
        # Speichere den Multiplikator als separate Datei
        multiplier_file = filename.replace('.csv', '_multiplier.txt')
        with open(multiplier_file, 'w') as f:
            f.write(str(df.attrs.get('price_multiplier', 1)))
        
        df.to_csv(filename) 
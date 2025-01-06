import requests
import pandas as pd
import os
from datetime import datetime

class OHLCFetcher:
    def __init__(self, api_key="575C5ABE-B09B-41E1-8597-7D24B65373C7"):
        self.api_key = api_key
        self.cache_dir = 'ohlc_cache'
        os.makedirs(self.cache_dir, exist_ok=True)

    def fetch_data(self, asset, interval, start_date="2018-03-01T00:00:00", limit=100000):
        """Fetches OHLC data from CoinAPI or cache"""
        # Check cache first
        cache_file = self._get_cache_filename(asset, interval)
        if os.path.exists(cache_file):
            print(f"Loading data from cache: {cache_file}")
            return pd.read_csv(cache_file, index_col='time_period_start', parse_dates=True)

        # If not in cache, fetch from API
        print("Fetching data from CoinAPI...")
        url = f"https://rest.coinapi.io/v1/ohlcv/{asset}/history?period_id={interval}&time_start={start_date}&limit={limit}"
        headers = {"X-CoinAPI-Key": self.api_key}

        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            data = response.json()
            if data:
                df = pd.DataFrame(data)
                df["time_period_start"] = pd.to_datetime(df["time_period_start"])
                
                # Adjust timestamps to Hong Kong timezone (add 12 hours)
                df["time_period_start"] += pd.Timedelta(hours=12)
                
                df.set_index("time_period_start", inplace=True)
                df[["price_open", "price_high", "price_low", "price_close", "volume_traded"]] = \
                    df[["price_open", "price_high", "price_low", "price_close", "volume_traded"]].astype(float)
                
                # Save to cache with adjusted timestamps
                self._save_to_cache(df, cache_file)
                
                return df[["price_open", "price_high", "price_low", "price_close", "volume_traded"]]
            else:
                raise Exception("Empty response from CoinAPI.")
        else:
            raise Exception(f"Failed to fetch data. Status Code: {response.status_code}")

    def _get_cache_filename(self, asset, interval):
        """Generates a cache filename based on asset and interval"""
        return os.path.join(self.cache_dir, f"{asset}_{interval}_ohlc.csv")

    def _save_to_cache(self, df, filename):
        """Saves the dataframe to cache"""
        print(f"Saving data to cache: {filename}")
        df.to_csv(filename) 
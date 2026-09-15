"""IBKR historical candle fetcher for the heatmap generator.

Pulls OHLC bars from a running IB Gateway (paper: port 4002) via ib_async and
returns them in the same format the rest of the project expects:

    index:    time_period_start (naive UTC)
    columns:  price_open, price_high, price_low, price_close, volume_traded
    attrs:    price_multiplier

Asset notation (used as the "asset" input in the generator):

    IBKR_AAPL              -> Stock/ETF on SMART (auto)
    IBKR_SPX               -> Index on CBOE (known index)
    IBKR_ES                -> front-month Future on CME (known future)
    IBKR_GC                -> front-month Future on COMEX
    IBKR_STK_AAPL_NASDAQ   -> explicit type + exchange
    IBKR_IDX_SPX_CBOE
    IBKR_FUT_ES_CME

Cache files match OHLCFetcher: ohlc_cache/{asset}_{interval}_ohlc.csv
"""

import os
import random
import logging
import pandas as pd
import numpy as np

# IBKR fires "Error 162 HMDS query returned no data" when a pagination request
# reaches the start of history - expected, not a failure. Silence it.
logging.getLogger("ib_async").setLevel(logging.CRITICAL)

try:
    from ib_async import IB, Stock, Index, Future
    _IB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _IB_AVAILABLE = False


INDEX_EXCHANGES = {
    "SPX": "CBOE", "NDX": "NASDAQ", "DJI": "CBOE", "RUT": "RUSSELL",
    "VIX": "CBOE", "ESTX50": "EUREX", "DAX": "EUREX", "SMI": "EBS",
}
FUTURE_EXCHANGES = {
    "ES": "CME", "MES": "CME", "NQ": "CME", "MNQ": "CME", "RTY": "CME", "M2K": "CME",
    "YM": "CBOT", "MYM": "CBOT",
    "GC": "COMEX", "MGC": "COMEX", "SI": "COMEX", "SIL": "COMEX", "HG": "COMEX",
    "CL": "NYMEX", "MCL": "NYMEX", "NG": "NYMEX", "MNG": "NYMEX",
    "ZC": "CBOT", "ZS": "CBOT", "ZW": "CBOT", "ZM": "CBOT", "ZL": "CBOT",
    "6E": "CME", "6J": "CME", "6B": "CME", "6A": "CME",
}
# interval -> (IBKR barSize, max durationStr per request)
BAR_MAP = {
    "1m": ("1 min", "1 D"),
    "5m": ("5 mins", "5 D"),
    "15m": ("15 mins", "10 D"),
    "1h": ("1 hour", "1 Y"),
    "4h": ("4 hours", "1 Y"),
    "1d": ("1 day", "10 Y"),
}
INTERVAL_ALIASES = {"60m": "1h", "240m": "4h", "1d": "1d", "1day": "1d", "1h": "1h"}


class IBKRFetcher:
    def __init__(self, host=None, port=None, client_id=None, max_requests=None):
        self.cache_dir = "ohlc_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        default_host = "host.docker.internal" if os.environ.get("HEATMAP_IN_DOCKER") else "127.0.0.1"
        self.host = host or os.environ.get("IBKR_HOST") or default_host
        self.port = int(port or os.environ.get("IBKR_PORT", 4002))
        self.client_id = int(client_id or os.environ.get("IBKR_CLIENT_ID", random.randint(30, 99)))
        self.max_requests = int(max_requests or os.environ.get("IBKR_MAX_REQUESTS", 30))

    # ------------------------------------------------------------------
    def fetch_data(self, asset, interval, limit=100000):
        if not _IB_AVAILABLE:
            raise RuntimeError("ib_async ist nicht installiert (pip install ib_async).")
        interval = INTERVAL_ALIASES.get(str(interval).lower(), str(interval).lower())
        if interval not in BAR_MAP:
            raise ValueError(f"IBKR: Interval '{interval}' nicht unterstuetzt. Erlaubt: {', '.join(BAR_MAP)}")

        cache_file = self._get_cache_filename(asset, interval)
        if os.path.exists(cache_file):
            print(f"Loading data from cache: {cache_file}")
            df = pd.read_csv(cache_file, index_col="time_period_start", parse_dates=True)
            return self._apply_price_normalization(df)

        print(f"Fetching data from IBKR Gateway ({self.host}:{self.port})...")
        ib = IB()
        ib.connect(self.host, self.port, clientId=self.client_id, timeout=20, readonly=True)
        try:
            contract = self._resolve_contract(ib, asset)
            bars = self._fetch_bars(ib, contract, interval, limit)
        finally:
            ib.disconnect()

        if not bars:
            raise RuntimeError(f"IBKR lieferte keine Candles fuer '{asset}' ({interval}). "
                               "Marktdaten-Berechtigung/Subscription pruefen.")
        df = self._bars_to_frame(bars)
        df = self._apply_price_normalization(df)
        self._save_to_cache(df, cache_file)
        return df

    # ------------------------------------------------------------------
    def _resolve_contract(self, ib, asset):
        spec = str(asset)
        for prefix in ("IBKR_", "IBKR:"):
            if spec.upper().startswith(prefix):
                spec = spec[len(prefix):]
                break
        parts = [p for p in spec.split("_") if p]
        if not parts:
            raise ValueError(f"Ungueltige IBKR-Asset-Angabe: {asset}")

        if parts[0].upper() in ("STK", "ETF", "IDX", "FUT"):
            kind = parts[0].upper()
            symbol = parts[1].upper() if len(parts) > 1 else ""
            exchange = parts[2].upper() if len(parts) > 2 else None
        else:
            symbol = parts[0].upper()
            exchange = None
            if symbol in INDEX_EXCHANGES:
                kind = "IDX"
            elif symbol in FUTURE_EXCHANGES:
                kind = "FUT"
            else:
                kind = "STK"

        if kind in ("STK", "ETF"):
            contract = Stock(symbol, exchange or "SMART", "USD")
        elif kind == "IDX":
            contract = Index(symbol, exchange or INDEX_EXCHANGES.get(symbol, "CBOE"), "USD")
        else:  # FUT
            exchange = exchange or FUTURE_EXCHANGES.get(symbol)
            if not exchange:
                raise ValueError(f"IBKR: Futures-Exchange fuer '{symbol}' unbekannt. "
                                 f"Nutze IBKR_FUT_{symbol}_<EXCHANGE>.")
            contract = self._front_month(ib, symbol, exchange)

        qualified = ib.qualifyContracts(contract)
        resolved = qualified[0] if qualified else None
        if resolved is None or not getattr(resolved, "conId", 0):
            raise RuntimeError(
                f"IBKR: Kontrakt '{asset}' nicht gefunden (Symbol/Exchange pruefen). "
                f"Beispiel: 'IBKR_AAPL' (nicht APPL), 'IBKR_SPY', 'IBKR_SPX', 'IBKR_ES'."
            )
        return resolved

    @staticmethod
    def _front_month(ib, symbol, exchange):
        details = ib.reqContractDetails(Future(symbol=symbol, exchange=exchange, currency="USD"))
        import datetime as _dt
        today = _dt.date.today().strftime("%Y%m%d")
        valid = [d.contract for d in details
                 if d.contract.lastTradeDateOrContractMonth >= today]
        if not valid:
            raise RuntimeError(f"IBKR: kein aktiver Future fuer {symbol}@{exchange} gefunden.")
        valid.sort(key=lambda c: c.lastTradeDateOrContractMonth)
        return valid[0]

    # ------------------------------------------------------------------
    def _fetch_bars(self, ib, contract, interval, limit):
        bar_size, chunk_dur = BAR_MAP[interval]
        bars = []
        end = ""
        requests = 0
        while len(bars) < limit and requests < self.max_requests:
            chunk = ib.reqHistoricalData(
                contract, endDateTime=end, durationStr=chunk_dur,
                barSizeSetting=bar_size, whatToShow="TRADES",
                useRTH=True, formatDate=2,
            )
            if not chunk:
                break
            bars = list(chunk) + bars
            end = chunk[0].date
            requests += 1
            if len(chunk) < 2:
                break
            ib.sleep(0.4)  # IBKR pacing
        return bars[:limit]

    @staticmethod
    def _bars_to_frame(bars):
        rows = []
        for b in bars:
            rows.append({
                "time_period_start": pd.Timestamp(b.date),
                "price_open": float(b.open),
                "price_high": float(b.high),
                "price_low": float(b.low),
                "price_close": float(b.close),
                "volume_traded": max(float(b.volume), 0.0),
            })
        df = pd.DataFrame(rows).set_index("time_period_start")
        df = df[~df.index.duplicated(keep="last")].sort_index()
        # normalize to naive UTC so the CSV round-trip matches the crypto pipeline
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)
        df.index.name = "time_period_start"
        return df[["price_open", "price_high", "price_low", "price_close", "volume_traded"]]

    def _apply_price_normalization(self, df):
        price_cols = ["price_open", "price_high", "price_low", "price_close"]
        min_price = df[price_cols].min().min()
        if min_price < 0.1:
            multiplier = 10 ** (abs(int(np.log10(min_price))) + 1)
            print(f"Normalizing prices with multiplier: {multiplier}")
            for col in price_cols:
                df[col] = df[col] * multiplier
            df.attrs["price_multiplier"] = multiplier
        else:
            df.attrs["price_multiplier"] = 1
        return df

    def _get_cache_filename(self, asset, interval):
        return os.path.join(self.cache_dir, f"{asset}_{interval}_ohlc.csv")

    def _save_to_cache(self, df, filename):
        print(f"Saving data to cache: {filename}")
        with open(filename.replace(".csv", "_multiplier.txt"), "w") as fh:
            fh.write(str(df.attrs.get("price_multiplier", 1)))
        df.to_csv(filename)

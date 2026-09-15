"""JSON serialization helpers for the machine-readable API.

Converts numpy/pandas/datetime objects into plain JSON-safe Python types and
maps non-finite floats (NaN/inf) to ``None`` so ``json.dumps`` never fails.
"""

import math
from datetime import date, datetime, time
from decimal import Decimal

import numpy as np
import pandas as pd

# Trade tuples produced by TradeAnalyzer / strategies.
TRADE_FIELDS = [
    "entry_time",
    "exit_time",
    "entry_price",
    "exit_price",
    "net_profit",
    "gross_profit",
    "fees",
    "direction",
    "exit_reason",
]


def to_jsonable(obj):
    """Recursively convert ``obj`` into JSON-serializable Python primitives."""
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj

    # numpy scalars / arrays
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return _finite(float(obj))
    if isinstance(obj, np.ndarray):
        return [to_jsonable(x) for x in obj.tolist()]

    # pandas
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, pd.Timedelta):
        return obj.total_seconds()
    if isinstance(obj, pd.Series):
        return [to_jsonable(x) for x in obj.tolist()]
    if isinstance(obj, pd.Index):
        return [to_jsonable(x) for x in obj.tolist()]
    if isinstance(obj, pd.DataFrame):
        return [to_jsonable(row) for row in obj.to_dict(orient="records")]
    try:
        if obj is pd.NaT:
            return None
    except Exception:
        pass

    # stdlib
    if isinstance(obj, float):
        return _finite(obj)
    if isinstance(obj, Decimal):
        return _finite(float(obj))
    if isinstance(obj, (datetime, date, time)):
        return obj.isoformat()

    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(x) for x in obj]

    # numpy void / generic fallback
    if hasattr(obj, "item"):
        try:
            return to_jsonable(obj.item())
        except Exception:
            pass

    return str(obj)


def _finite(value):
    if value is None:
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def trades_to_records(trades):
    """Convert trade tuples/lists into a list of named dicts."""
    records = []
    for trade in trades:
        if not isinstance(trade, (list, tuple)):
            records.append(to_jsonable(trade))
            continue
        record = {}
        for idx, field in enumerate(TRADE_FIELDS):
            record[field] = to_jsonable(trade[idx]) if idx < len(trade) else None
        if len(trade) > len(TRADE_FIELDS):
            record["extra"] = to_jsonable(list(trade[len(TRADE_FIELDS):]))
        records.append(record)
    return records

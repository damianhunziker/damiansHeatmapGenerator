"""Tests for core.serialize (JSON conversion of numpy/pandas/trades)."""

import numpy as np
import pandas as pd

from core.serialize import to_jsonable, trades_to_records


def test_primitives_pass_through():
    assert to_jsonable(None) is None
    assert to_jsonable(True) is True
    assert to_jsonable(3) == 3
    assert to_jsonable("x") == "x"


def test_numpy_scalars_become_python():
    assert to_jsonable(np.int64(5)) == 5
    assert isinstance(to_jsonable(np.int64(5)), int)
    assert to_jsonable(np.float64(1.5)) == 1.5
    assert isinstance(to_jsonable(np.float64(1.5)), float)
    assert to_jsonable(np.bool_(True)) is True


def test_non_finite_floats_map_to_none():
    assert to_jsonable(float("inf")) is None
    assert to_jsonable(float("-inf")) is None
    assert to_jsonable(float("nan")) is None
    assert to_jsonable(np.float64("inf")) is None


def test_arrays_and_series():
    assert to_jsonable(np.array([1, 2, 3])) == [1, 2, 3]
    assert to_jsonable(pd.Series([1, 2])) == [1, 2]


def test_timestamp_is_iso():
    assert to_jsonable(pd.Timestamp("2024-01-02T03:04:05")) == "2024-01-02T03:04:05"


def test_nested_structures():
    out = to_jsonable({"a": np.array([1.0, float("nan")]), "b": (np.int32(2),)})
    assert out == {"a": [1.0, None], "b": [2]}


def test_trades_to_records_full_tuple():
    trades = [("2024-01-01", "2024-01-02", 1.0, 2.0, 1.0, 1.1, 0.1, "LONG", "KAMA exit signal")]
    recs = trades_to_records(trades)
    assert recs[0]["direction"] == "LONG"
    assert recs[0]["exit_reason"] == "KAMA exit signal"
    assert recs[0]["net_profit"] == 1.0


def test_trades_to_records_short_tuple_has_no_exit_reason():
    recs = trades_to_records([(1, 2, 3, 4, 5, 6, 7, "SHORT")])
    assert recs[0]["exit_reason"] is None
    assert recs[0]["direction"] == "SHORT"


def test_trades_to_records_keeps_extra_fields():
    recs = trades_to_records([(1, 2, 3, 4, 5, 6, 7, "LONG", "reason", "EXTRA")])
    assert recs[0]["extra"] == ["EXTRA"]

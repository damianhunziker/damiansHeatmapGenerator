"""Machine-readable API layer for the trading-strategy tools.

This module is the single source of truth for the ``--api`` JSON contract used
by ``test.py`` and by the MCP server.  It normalises parameters, loads the OHLC
data, dispatches to the existing tool functions (``pnl``, ``chart_analysis``,
``heatmap``, ``automator``, ``fetcher``), and returns a JSON-serializable
envelope.

All tool output (including ``print`` calls from worker processes) is redirected
away from stdout while a tool runs, so ``--api`` prints exactly one JSON object.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

from core.serialize import to_jsonable, trades_to_records
from core.kama_scaling import normalize_derived
from core.params import (
    applicable_virtual_params,
    describe_resolvers,
    load_resolver_config,
    resolver_applies,
)

VALID_INTERVALS = ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"]
TOOLS = [
    "pnl", "chart_analysis", "heatmap", "automator", "fetcher", "series",
    "plateaus", "validate_plateau", "plot_plateau",
]

# Keys that are API/tool control parameters, not strategy constructor args.
_CONTROL_KEYS = {
    "asset", "interval", "start_date", "end_date", "strategy", "tool",
    "initial_equity", "fee_pct", "direction", "trade_direction",
    "param_ranges", "derived_params", "workers", "no_images", "image", "max_combos",
    "columns", "tail", "format", "inline_max_cells", "pairs", "higher_tf",
    "cache_dir", "output", "resolvers", "resolver_config", "use_config_ranges",
    "plateau_config", "grid", "grid_path", "region", "windows", "x_param", "y_param",
    "title",
}

DEFAULT_SERIES_COLUMNS = [
    "price_open", "price_high", "price_low", "price_close", "volume_traded",
    "long_entry", "short_entry", "long_exit", "short_exit", "exit_reason",
]


# ---------------------------------------------------------------------------
# stdout handling
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def suppress_stdout(redirect_to_stderr=True):
    """Redirect the process stdout (fd level) for the duration of the block.

    Operating on the file descriptor (not just ``sys.stdout``) also captures
    output from ``multiprocessing`` child processes, which is essential for the
    heatmap tool.  When ``redirect_to_stderr`` is False the output is discarded.
    """
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass

    saved_fd = os.dup(1)
    target_fd = 2 if redirect_to_stderr else os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(target_fd, 1)
        yield
    finally:
        try:
            sys.stdout.flush()
        except Exception:
            pass
        os.dup2(saved_fd, 1)
        os.close(saved_fd)
        if not redirect_to_stderr:
            os.close(target_fd)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _strategy_by_name(name):
    from core.strategy_utils import get_available_strategies

    if not name:
        raise ValueError("Parameter 'strategy' is required")
    for _, (sname, sclass) in get_available_strategies().items():
        if sname == name:
            return sclass
    raise ValueError(f"Unknown strategy: {name}")


def _load_data(asset, interval, cache_dir="ohlc_cache"):
    from core.strategy_utils import fetch_data

    os.makedirs(cache_dir, exist_ok=True)
    data = fetch_data(asset, interval)
    return data


def _timeframe_data(asset, interval):
    data = _load_data(asset, interval)
    return {"primary": {"interval": interval, "data": data}}


def _build_strategy_params(strategy_class, params):
    """Merge strategy defaults with all user-provided parameters."""
    strategy_params = {}
    if hasattr(strategy_class, "get_parameters"):
        for pname, (default, _desc) in strategy_class.get_parameters().items():
            strategy_params[pname] = params.get(pname, default)

    # pass through any extra strategy constructor parameters (e.g. debug_mode)
    for key, value in params.items():
        if key in _CONTROL_KEYS:
            continue
        strategy_params.setdefault(key, value)

    strategy_params.update({
        "initial_equity": params.get("initial_equity", 10000),
        "fee_pct": params.get("fee_pct", 0.04),
        "start_date": params.get("start_date"),
        "end_date": params.get("end_date"),
        "asset": params.get("asset"),
        "trade_direction": params.get("trade_direction") or params.get("direction") or "both",
    })
    return strategy_params


def _clean_range(values, precision=6):
    out = []
    for v in np.asarray(values).tolist():
        if isinstance(v, float):
            out.append(round(v, precision))
        else:
            out.append(v)
    return out


_CONFIG_RANGE_TOKENS = {"config", "@config", "default", "@default"}


def _configured_range(name, virtual_defaults):
    meta = (virtual_defaults or {}).get(name)
    if not meta or "range" not in meta:
        raise ValueError(
            f"'{name}' asked for its configured range, but the resolver config defines none")
    return meta["range"]


def _spec_to_values(name, spec, virtual_defaults=None):
    """Turn a range spec into a numpy array of values.

    Accepts a list, ``{"min","max","step"}`` / ``{"min","max","count"}`` /
    ``{"values": [...]}``, or the tokens ``"config"`` / ``{"$config": true}``
    to use the range declared in the resolver config for that parameter.
    """
    if isinstance(spec, str) and spec.strip().lower() in _CONFIG_RANGE_TOKENS:
        spec = _configured_range(name, virtual_defaults)
    if isinstance(spec, dict):
        if spec.get("$config"):
            spec = _configured_range(name, virtual_defaults)
        elif "values" in spec:
            return np.asarray(spec["values"])
        else:
            start = spec.get("min", spec.get("start"))
            stop = spec.get("max", spec.get("stop"))
            if start is None or stop is None:
                raise ValueError(f"Range for '{name}' needs 'min' and 'max'")
            step = spec.get("step")
            if step in (None, 0):
                count = int(spec.get("count", spec.get("num", 5)))
                return np.linspace(start, stop, count)
            return np.arange(start, stop + step, step)
    if isinstance(spec, (list, tuple)):
        return np.asarray(spec)
    return np.asarray([spec])


def _parse_param_ranges(param_ranges, strategy_class, max_combos,
                        virtual_defaults=None, use_config_ranges=False):
    if param_ranges:
        ranges = {
            name: _spec_to_values(name, spec, virtual_defaults)
            for name, spec in param_ranges.items()
        }
    else:
        default = strategy_class.get_parameter_ranges() or {}
        keys = list(default.keys())[:2]
        if not keys and not (use_config_ranges and virtual_defaults):
            raise ValueError("Strategy defines no parameter ranges; pass 'param_ranges'")
        ranges = {k: np.asarray(default[k]) for k in keys}

    # Optionally inject configured ranges for virtual params not swept explicitly.
    if use_config_ranges and virtual_defaults:
        for name, meta in virtual_defaults.items():
            if name not in ranges and "range" in meta:
                ranges[name] = _spec_to_values(name, meta["range"], virtual_defaults)

    total = 1
    for values in ranges.values():
        total *= max(1, len(values))
    if total > max_combos:
        raise ValueError(
            f"Parameter grid has {total} combinations which exceeds max_combos={max_combos}. "
            f"Pass a smaller 'param_ranges' or raise 'max_combos'."
        )
    return ranges


def _virtual_alias_map(resolver_cfg, strategy_name):
    """Map resolver ids -> canonical virtual input, for resolvers that apply."""
    alias = {}
    for resolver in resolver_cfg["resolvers"]:
        if resolver["id"] != resolver["input"] and resolver_applies(resolver, strategy_name):
            alias[resolver["id"]] = resolver["input"]
    return alias


def _normalize_virtual_ranges(ranges, alias_map, resolver_cfg, strategy_name):
    """Rename resolver-id aliases to the canonical virtual input and reject
    virtual params that no resolver would resolve for this strategy."""
    normalized = {}
    conversions = {}
    for key, values in ranges.items():
        canonical = alias_map.get(key, key)
        if canonical != key:
            if canonical in ranges:
                raise ValueError(
                    f"'{key}' is an alias of '{canonical}'; sweep only one of them")
            conversions[key] = canonical
        if canonical in normalized:
            raise ValueError(f"duplicate parameter '{canonical}'")
        normalized[canonical] = values

    virtual_inputs = {r["input"] for r in resolver_cfg["resolvers"]}
    applicable = {r["input"] for r in resolver_cfg["resolvers"]
                  if resolver_applies(r, strategy_name)}
    for key in normalized:
        if key in virtual_inputs and key not in applicable:
            raise ValueError(
                f"'{key}' is a virtual parameter, but no resolver applies to strategy "
                f"'{strategy_name}'. Add a resolver with a matching 'when', or use a "
                f"strategy that has one.")
    return normalized, conversions


def _normalize_derived_names(derived_params, alias_map):
    """Rewrite derived params (target names and 'scale' keys) through the alias map."""
    if not derived_params:
        return derived_params
    out = {}
    for name, spec in derived_params.items():
        spec = dict(spec)
        if spec.get("scale") in alias_map:
            spec["scale"] = alias_map[spec["scale"]]
        out[alias_map.get(name, name)] = spec
    return out


def prepare_heatmap_params(params, strategy_class, max_combos):
    """Load resolvers, parse/normalize the grid and derive params cleanly.

    Resolver-id aliases are rewritten to their canonical virtual input (both in
    ``param_ranges`` and in ``derived_params``), so a caller may sweep the
    friendly id (e.g. ``slow_kama_normalized_length``) and it is converted to
    the virtual input (``kama_normalized_length``).  Virtual parameters that no
    resolver would resolve for this strategy raise a clear error instead of
    silently producing identical grid cells.
    """
    resolver_cfg = load_resolver_config(
        path=params.get("resolver_config"), inline=params.get("resolvers"))
    name = strategy_class.__name__
    virtual_defaults = applicable_virtual_params(
        resolver_cfg["virtual_params"], resolver_cfg["resolvers"], name)
    ranges = _parse_param_ranges(
        params.get("param_ranges"), strategy_class, max_combos,
        virtual_defaults=virtual_defaults,
        use_config_ranges=bool(params.get("use_config_ranges", False)))
    alias_map = _virtual_alias_map(resolver_cfg, name)
    ranges, conversions = _normalize_virtual_ranges(
        ranges, alias_map, resolver_cfg, name)
    derived = normalize_derived(
        _normalize_derived_names(params.get("derived_params"), alias_map), ranges)
    return {
        "resolver_cfg": resolver_cfg,
        "ranges": ranges,
        "derived": derived,
        "conversions": conversions,
    }


def _artifact(path):
    if not path:
        return None
    abspath = os.path.abspath(path)
    url = None
    try:
        from core.html_viewer import url_for

        url = url_for(abspath)
    except Exception:
        pass
    return {"path": abspath, "url": url}


def _lookback_candles(data, start_date, end_date):
    """Return (candles_in_range, candles_up_to_end) using the data index."""
    if start_date is None and end_date is None:
        return None, None
    index = data.index
    start_ts = pd.to_datetime(start_date) if start_date else index.min()
    end_ts = pd.to_datetime(end_date) if end_date else index.max()
    end_lookback = int((index <= end_ts).sum())
    start_lookback = int((index < start_ts).sum())
    lookback = end_lookback - start_lookback
    return lookback, end_lookback


def _clean_direction(result):
    if not result:
        return None
    cleaned = {
        "direction": result.get("direction"),
        "metrics": to_jsonable(result.get("metrics")),
        "trades": trades_to_records(result.get("trades") or []),
        "equity_curve": to_jsonable(result.get("equity_curve")),
        "equity_curve_timestamps": to_jsonable(result.get("equity_curve_timestamps")),
        "pnl_performance": to_jsonable(result.get("pnl_performance")),
        "buy_hold": to_jsonable(result.get("buy_hold")),
        "drawdown": to_jsonable(result.get("drawdown")),
    }
    return cleaned


# ---------------------------------------------------------------------------
# tool dispatch
# ---------------------------------------------------------------------------

def _run_pnl(params, options):
    from pnl import create_interactive_chart

    strategy_class = _strategy_by_name(params.get("strategy"))
    strategy_params = _build_strategy_params(strategy_class, params)
    timeframe_data = _timeframe_data(params["asset"], params.get("interval", "4h"))

    result = create_interactive_chart(
        timeframe_data=timeframe_data,
        strategy_class=strategy_class,
        strategy_params=strategy_params,
        last_n_candles_analyze=None,
        last_n_candles_display=None,
    )

    directions = {
        name: _clean_direction(res)
        for name, res in (result.get("directions") or {}).items()
    }
    return {
        "strategy": strategy_class.__name__,
        "asset": params["asset"],
        "interval": params.get("interval", "4h"),
        "selected_direction": result.get("selected_direction"),
        "directions": directions,
    }, [_artifact(result.get("artifact"))]


def _run_chart_analysis(params, options):
    from chart_analysis import create_interactive_chart

    strategy_class = _strategy_by_name(params.get("strategy"))
    strategy_params = _build_strategy_params(strategy_class, params)
    timeframe_data = _timeframe_data(params["asset"], params.get("interval", "4h"))

    result = create_interactive_chart(
        timeframe_data=timeframe_data,
        strategy_class=strategy_class,
        strategy_params=strategy_params,
        last_n_candles_analyze=None,
        last_n_candles_display=None,
    )

    data = {
        "strategy": strategy_class.__name__,
        "asset": params["asset"],
        "interval": params.get("interval", "4h"),
        "summary": to_jsonable(result.get("summary")),
        "trades": trades_to_records(result.get("trades") or []),
        "divergence_indicators": to_jsonable(result.get("divergence_indicators")),
    }
    return data, [_artifact(result.get("artifact"))]


def _run_heatmap(params, options):
    from heatmap import create_heatmap

    strategy_class = _strategy_by_name(params.get("strategy"))
    timeframe_data = _timeframe_data(params["asset"], params.get("interval", "4h"))
    data = timeframe_data["primary"]["data"]

    max_combos = int(params.get("max_combos", options.get("max_combos", 400)))
    prep = prepare_heatmap_params(params, strategy_class, max_combos)
    resolver_cfg = prep["resolver_cfg"]
    ranges = prep["ranges"]
    derived = prep["derived"]

    lookback, end_lookback = _lookback_candles(
        data, params.get("start_date"), params.get("end_date")
    )

    result = create_heatmap(
        timeframe_data=timeframe_data,
        strategy_class=strategy_class,
        param_ranges=ranges,
        initial_equity=params.get("initial_equity", 10000),
        fee_pct=params.get("fee_pct", 0.04),
        last_n_candles_analyze=lookback,
        last_n_candles_display=end_lookback,
        interval=params.get("interval", "4h"),
        asset=params["asset"],
        strategy_name=strategy_class.__name__,
        start_date=params.get("start_date"),
        end_date=params.get("end_date"),
        workers=params.get("workers", options.get("workers")),
        derived_params=derived,
        resolvers=resolver_cfg["resolvers"],
        resolver_snippets=resolver_cfg["snippets"],
        plateau_config=params.get("plateau_config"),
    )

    artifacts = [_artifact(result.get("artifact"))]
    sidecar = _artifact(result.get("sidecar"))
    if sidecar:
        artifacts.append(sidecar)

    return {
        "strategy": strategy_class.__name__,
        "asset": params["asset"],
        "interval": params.get("interval", "4h"),
        "x_param": result.get("x_param"),
        "y_param": result.get("y_param"),
        "param_ranges": to_jsonable(result.get("param_ranges")),
        "param_aliases": to_jsonable(prep["conversions"]),
        "derived_params": to_jsonable(derived),
        "resolvers": describe_resolvers(
            resolver_cfg["resolvers"], resolver_cfg["snippets"], resolver_cfg["virtual_params"]),
        "grid": to_jsonable(result.get("grid")),
        "best": to_jsonable(result.get("best")),
        "robustness": to_jsonable(result.get("robustness")),
        "plateaus": to_jsonable(result.get("plateaus")),
        "sidecar": result.get("sidecar"),
        "fixed_params": to_jsonable(result.get("fixed_params")),
        "warnings": to_jsonable(result.get("warnings") or []),
    }, artifacts


def _ranges_from_grid(grid, x_name="x", y_name="y"):
    """Derive axis values from a bare grid (fallback when no ranges are given)."""
    xs, ys = [], []
    for row in grid or []:
        if row.get("x") is not None and row["x"] not in xs:
            xs.append(row["x"])
        if row.get("y") is not None and row["y"] not in ys:
            ys.append(row["y"])

    def _sorted(values):
        try:
            return sorted(values)
        except TypeError:
            return values

    return {x_name: _sorted(xs), y_name: _sorted(ys)}


def _plateau_inputs(params, options):
    """Resolve ``(grid, param_ranges, source)`` from inline grid / sidecar / heatmap."""
    grid = params.get("grid")
    param_ranges = params.get("param_ranges")
    source = "inline-grid"
    if grid is None and params.get("grid_path"):
        path = params["grid_path"]
        if not os.path.isabs(path):
            path = os.path.join(os.getcwd(), path)
        with open(path) as handle:
            payload = json.load(handle)
        grid = payload.get("grid")
        param_ranges = param_ranges or payload.get("param_ranges")
        source = path
    if grid is None:
        data, _artifacts = _run_heatmap(params, options)
        grid = data.get("grid")
        param_ranges = param_ranges or data.get("param_ranges")
        source = "heatmap"
    if not param_ranges:
        param_ranges = _ranges_from_grid(
            grid, params.get("x_param", "x"), params.get("y_param", "y"))
    return grid, param_ranges, source


def _run_plateaus(params, options):
    """Algorithmic plateau detection over an existing grid (no re-backtest)."""
    from core import plateaus as plateau_analysis

    grid, param_ranges, source = _plateau_inputs(params, options)
    result = plateau_analysis.detect_plateaus(
        grid, param_ranges, params.get("plateau_config"))
    result["source"] = source
    return result, []


def _normalize_region(region):
    """Accept either a param-ranges spec or a ``best_region`` block."""
    if (isinstance(region, dict) and isinstance(region.get("x"), dict)
            and "param" in region["x"] and isinstance(region.get("y"), dict)):
        x, y = region["x"], region["y"]
        return {x["param"]: {"min": x["min"], "max": x["max"]},
                y["param"]: {"min": y["min"], "max": y["max"]}}
    return region


def _run_validate_plateau(params, options):
    """Re-run the heatmap on walk-forward windows restricted to a region."""
    region = params.get("region") or params.get("param_ranges")
    if not region:
        raise ValueError("validate_plateau needs 'region' or 'param_ranges'")
    region = _normalize_region(region)
    start = params.get("start_date")
    end = params.get("end_date")
    if not start or not end:
        raise ValueError("validate_plateau needs 'start_date' and 'end_date'")
    windows = max(1, int(params.get("windows", 3)))
    config = params.get("plateau_config")

    edges = np.linspace(np.datetime64(start).astype("int64"),
                        np.datetime64(end).astype("int64"), windows + 1)
    results, representatives = [], []
    for i in range(windows):
        w0 = str(np.datetime64(int(edges[i]), "D"))
        w1 = str(np.datetime64(int(edges[i + 1]), "D"))
        entry = {"window": [w0, w1]}
        try:
            data, _artifacts = _run_heatmap({
                **params, "start_date": w0, "end_date": w1,
                "param_ranges": region, "plateau_config": config,
            }, options)
            analysis = data.get("plateaus") or {}
            representative = analysis.get("representative")
            entry.update({
                "ok": True,
                "region_count": analysis.get("region_count"),
                "representative": representative,
                "region_score": (analysis.get("best_region") or {}).get("region_score"),
            })
            if representative and representative.get("x") is not None:
                representatives.append(representative)
        except Exception as exc:  # noqa: BLE001 - reported per window
            entry.update({"ok": False, "error": f"{exc.__class__.__name__}: {exc}"})
        results.append(entry)

    xs = [r["x"] for r in representatives]
    ys = [r["y"] for r in representatives]
    summary = {
        "windows": windows,
        "windows_with_plateau": len(representatives),
        "consistency": (len(representatives) / windows) if windows else 0.0,
        "representative_mean": {"x": float(np.mean(xs)), "y": float(np.mean(ys))}
        if representatives else None,
        "representative_std": {"x": float(np.std(xs)), "y": float(np.std(ys))}
        if representatives else None,
    }
    return {"region": to_jsonable(region), "results": to_jsonable(results),
            "summary": to_jsonable(summary)}, []


def _run_plot_plateau(params, options):
    """Render a score matrix with region outlines as a presentation PNG."""
    from core import plateaus as plateau_analysis

    grid, param_ranges, source = _plateau_inputs(params, options)
    os.makedirs("html_cache", exist_ok=True)
    path = params.get("output") or os.path.join("html_cache", "plateau_overlay.png")
    path = path if os.path.isabs(path) else os.path.abspath(path)
    plateau_analysis.plot_overlay(
        grid, param_ranges, params.get("plateau_config"), path=path,
        title=params.get("title") or "Plateau regions")
    keys = list(param_ranges.keys())
    return {"source": source, "path": path, "x_param": keys[0],
            "y_param": keys[1] if len(keys) > 1 else None}, [_artifact(path)]


def _run_automator(params, options):
    from automator import run_heatmap_for_pairs

    pairs = params.get("pairs")
    if isinstance(pairs, str):
        pairs = [p.strip() for p in pairs.replace(",", " ").split() if p.strip()]

    result = run_heatmap_for_pairs({
        "strategy": params.get("strategy"),
        "pairs": pairs,
        "interval": params.get("interval", "4h"),
        "higher_tf": params.get("higher_tf", "1d"),
        "initial_equity": params.get("initial_equity", 1000),
        "fee_pct": params.get("fee_pct", 0.04),
        "start_date": params.get("start_date"),
        "end_date": params.get("end_date"),
        "param_ranges": params.get("param_ranges"),
        "derived_params": params.get("derived_params"),
        "resolvers": params.get("resolvers"),
        "resolver_config": params.get("resolver_config"),
        "use_config_ranges": params.get("use_config_ranges"),
        "max_combos": params.get("max_combos", options.get("max_combos", 400)),
        "workers": params.get("workers", options.get("workers")),
    })

    artifacts = [_artifact(result.get("log_file"))]
    for run in result.get("runs", []):
        if run.get("artifact"):
            artifacts.append(_artifact(run["artifact"]))
    return {
        "output_dir": result.get("output_dir"),
        "runs": to_jsonable(result.get("runs")),
    }, artifacts


def _run_fetcher(params, options):
    asset = params["asset"]
    interval = params.get("interval", "4h")
    data = _load_data(asset, interval)
    if data is None or len(data) == 0:
        raise ValueError(f"No data returned for {asset} {interval}")
    cache_file = os.path.abspath(os.path.join("ohlc_cache", f"{asset}_{interval}_ohlc.csv"))
    return {
        "asset": asset,
        "interval": interval,
        "rows": int(len(data)),
        "start": to_jsonable(data.index.min()),
        "end": to_jsonable(data.index.max()),
        "cache_path": cache_file,
    }, []


def _run_series(params, options):
    return get_series(params, options)


_DISPATCH = {
    "pnl": _run_pnl,
    "chart_analysis": _run_chart_analysis,
    "heatmap": _run_heatmap,
    "automator": _run_automator,
    "fetcher": _run_fetcher,
    "series": _run_series,
    "plateaus": _run_plateaus,
    "validate_plateau": _run_validate_plateau,
    "plot_plateau": _run_plot_plateau,
}


def run_tool(tool, params=None, options=None, suppress=True):
    """Run a tool and return the JSON-serializable envelope dict."""
    params = dict(params or {})
    options = dict(options or {})
    started = time.time()

    envelope = {
        "ok": False,
        "tool": tool,
        "params": to_jsonable(params),
        "data": None,
        "artifacts": [],
        "warnings": [],
        "error": None,
        "duration_ms": 0,
    }

    if tool not in _DISPATCH:
        envelope["error"] = f"Unknown tool: {tool}. Valid tools: {', '.join(TOOLS)}"
        envelope["duration_ms"] = int((time.time() - started) * 1000)
        return envelope

    try:
        runner = _DISPATCH[tool]
        if suppress:
            with suppress_stdout():
                data, artifacts = runner(params, options)
        else:
            data, artifacts = runner(params, options)
        envelope["data"] = to_jsonable(data)
        envelope["artifacts"] = [a for a in artifacts if a]
        if isinstance(data, dict) and data.get("warnings"):
            envelope["warnings"] = to_jsonable(data["warnings"])
        envelope["ok"] = True
    except Exception as exc:  # noqa: BLE001 - surfaced to the caller as JSON
        import traceback

        envelope["error"] = f"{exc.__class__.__name__}: {exc}"
        envelope["traceback"] = traceback.format_exc()

    envelope["duration_ms"] = int((time.time() - started) * 1000)
    return envelope


# ---------------------------------------------------------------------------
# series / runtime debugging
# ---------------------------------------------------------------------------

def get_series(params, options=None):
    """Return per-candle indicator/signal series for runtime debugging."""
    options = dict(options or {})
    strategy_class = _strategy_by_name(params.get("strategy"))
    strategy_params = _build_strategy_params(strategy_class, params)
    interval = params.get("interval", "4h")
    timeframe_data = _timeframe_data(params["asset"], interval)
    data = timeframe_data["primary"]["data"]

    from core.strategy_utils import instantiate_strategy

    strategy = instantiate_strategy(strategy_class, strategy_params)
    strategy.timeframe_data = timeframe_data
    if hasattr(strategy, "divergence_detector"):
        strategy.divergence_detector.set_date_range(
            params.get("start_date"), params.get("end_date")
        )

    prepared = strategy.prepare_signals(data)

    start_date = params.get("start_date")
    end_date = params.get("end_date")
    if start_date:
        prepared = prepared[prepared.index >= pd.to_datetime(start_date)]
    if end_date:
        prepared = prepared[prepared.index <= pd.to_datetime(end_date)]

    tail = params.get("tail")
    if tail:
        prepared = prepared.tail(int(tail))

    requested = params.get("columns")
    if requested:
        if isinstance(requested, str):
            requested = [c.strip() for c in requested.split(",") if c.strip()]
        columns = [c for c in requested if c in prepared.columns]
        missing = [c for c in requested if c not in prepared.columns]
    else:
        columns = [c for c in DEFAULT_SERIES_COLUMNS if c in prepared.columns]
        missing = []

    frame = prepared[columns].copy()
    index = [to_jsonable(ts) for ts in frame.index]

    fmt = params.get("format", options.get("format", "json"))
    inline_max_cells = int(params.get("inline_max_cells", options.get("inline_max_cells", 5000)))
    cell_count = len(frame) * max(1, len(columns))

    data_out = {
        "asset": params["asset"],
        "interval": interval,
        "strategy": strategy_class.__name__,
        "count": int(len(frame)),
        "columns": columns,
        "missing_columns": missing,
        "index": index,
        "rows": None,
        "format": fmt,
        "path": None,
    }

    if fmt == "parquet" or cell_count > inline_max_cells:
        path, used_format = _write_series_file(frame, params, interval)
        data_out["path"] = path
        data_out["format"] = used_format
        data_out["rows_omitted"] = True
        if cell_count <= inline_max_cells:
            data_out["rows"] = to_jsonable(frame.to_dict(orient="records"))
            data_out["rows_omitted"] = False
    else:
        data_out["rows"] = to_jsonable(frame.to_dict(orient="records"))

    artifacts = []
    if data_out["path"]:
        artifacts.append(_artifact(data_out["path"]))
    return data_out, artifacts


def _write_series_file(frame, params, interval):
    os.makedirs("series_cache", exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{params['asset']}_{interval}_{params.get('strategy')}_{stamp}"
    parquet_path = os.path.abspath(os.path.join("series_cache", base + ".parquet"))
    try:
        frame.to_parquet(parquet_path)
        return parquet_path, "parquet"
    except Exception:
        csv_path = os.path.abspath(os.path.join("series_cache", base + ".csv"))
        frame.to_csv(csv_path)
        return csv_path, "csv"


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------

def _param_meta(signature_params):
    import inspect

    meta = {}
    for name, param in signature_params.items():
        if name == "self":
            continue
        default = None if param.default is inspect.Parameter.empty else to_jsonable(param.default)
        meta[name] = {
            "default": default,
            "required": param.default is inspect.Parameter.empty,
            "type": None if param.annotation is inspect.Parameter.empty else str(param.annotation),
        }
    return meta


def get_schema():
    """Return a machine-readable schema of all tools and strategies."""
    import inspect

    common_params = {
        "asset": {"type": "string", "required": True, "description": "Trading pair, e.g. BTCUSDT or IBKR_SPY"},
        "interval": {"type": "string", "enum": VALID_INTERVALS, "default": "4h"},
        "start_date": {"type": "string", "format": "YYYY-MM-DD"},
        "end_date": {"type": "string", "format": "YYYY-MM-DD"},
        "strategy": {"type": "string", "required": True, "description": "Strategy class name"},
        "initial_equity": {"type": "number", "default": 10000},
        "fee_pct": {"type": "number", "default": 0.04},
    }

    tools = {
        "pnl": {"description": "PnL analysis for both/long/short directions", "params": {**common_params, "direction": {"enum": ["both", "long", "short"], "default": "both"}}},
        "chart_analysis": {"description": "Interactive chart + trade list + summary", "params": dict(common_params)},
        "heatmap": {"description": "Parameter grid sweep producing a heatmap", "params": {**common_params, "param_ranges": {"type": "object"}, "derived_params": {"type": "object", "description": "derive params from a swept scale, e.g. KAMA lengths"}, "resolvers": {"type": "object", "description": "generic virtual->concrete parameter resolvers (inline)"}, "resolver_config": {"type": "string", "description": "path to a JSON resolver config"}, "use_config_ranges": {"type": "boolean", "default": False, "description": "inject configured ranges for virtual params not swept explicitly"}, "plateau_config": {"type": "object", "description": "plateau detection tuning (weights, min_trades, threshold, min_area, representative, ...)"}, "max_combos": {"type": "integer", "default": 400}, "workers": {"type": "integer"}}},
        "plateaus": {"description": "Algorithmic plateau/region detection over a grid (inline 'grid', 'grid_path' sidecar, or a fresh heatmap)", "params": {**common_params, "grid": {"type": "array", "description": "grid rows from a previous heatmap"}, "grid_path": {"type": "string", "description": "heatmap sidecar JSON (grid + plateaus)"}, "param_ranges": {"type": "object"}, "plateau_config": {"type": "object"}, "x_param": {"type": "string"}, "y_param": {"type": "string"}}},
        "validate_plateau": {"description": "Walk-forward validation of a parameter region across N time windows", "params": {**common_params, "region": {"type": "object", "description": "param_ranges spec or a heatmap 'best_region' block"}, "param_ranges": {"type": "object"}, "windows": {"type": "integer", "default": 3}, "plateau_config": {"type": "object"}, "max_combos": {"type": "integer", "default": 400}, "workers": {"type": "integer"}}},
        "plot_plateau": {"description": "Render the score matrix with region outlines to a PNG", "params": {**common_params, "grid": {"type": "array"}, "grid_path": {"type": "string"}, "param_ranges": {"type": "object"}, "plateau_config": {"type": "object"}, "output": {"type": "string"}}},
        "automator": {"description": "Run heatmaps across multiple pairs", "params": {**common_params, "pairs": {"type": "array"}, "higher_tf": {"type": "string", "default": "1d"}, "param_ranges": {"type": "object"}, "derived_params": {"type": "object"}, "resolvers": {"type": "object"}, "resolver_config": {"type": "string"}, "use_config_ranges": {"type": "boolean", "default": False}}},
        "fetcher": {"description": "Ensure/refresh OHLC cache for an asset", "params": {"asset": common_params["asset"], "interval": common_params["interval"]}},
        "series": {"description": "Per-candle indicator/signal series for debugging", "params": {**common_params, "columns": {"type": "array"}, "tail": {"type": "integer"}, "format": {"enum": ["json", "parquet"], "default": "json"}, "inline_max_cells": {"type": "integer", "default": 5000}}},
    }

    strategies = {}
    try:
        from core.strategy_utils import get_available_strategies

        for _, (name, sclass) in get_available_strategies().items():
            entry = {"parameters": {}, "parameter_ranges": {}, "required_timeframes": {}}
            try:
                entry["parameters"] = {
                    pname: {"default": to_jsonable(default), "description": desc}
                    for pname, (default, desc) in sclass.get_parameters().items()
                }
            except Exception:
                pass
            try:
                entry["parameter_ranges"] = {
                    pname: _clean_range(values)
                    for pname, values in (sclass.get_parameter_ranges() or {}).items()
                }
            except Exception:
                pass
            try:
                entry["required_timeframes"] = sclass.get_required_timeframes()
            except Exception:
                pass
            try:
                entry["init_signature"] = _param_meta(inspect.signature(sclass.__init__).parameters)
            except Exception:
                pass
            strategies[name] = entry
    except Exception as exc:  # noqa: BLE001
        strategies = {"_error": str(exc)}

    # Default parameter-resolution config: which virtual parameters exist and
    # what they resolve into (so agents can discover e.g. kama_normalized_length).
    try:
        cfg = load_resolver_config(use_default=True)
        resolvers_info = describe_resolvers(
            cfg["resolvers"], cfg["snippets"], cfg["virtual_params"])
    except Exception as exc:  # noqa: BLE001
        resolvers_info = {"_error": str(exc)}

    return {
        "tools": tools,
        "intervals": VALID_INTERVALS,
        "strategies": strategies,
        "resolvers": resolvers_info,
        "envelope": {
            "ok": "boolean",
            "tool": "string",
            "params": "object",
            "data": "object",
            "artifacts": "array of {path, url}",
            "warnings": "array",
            "error": "string|null",
            "duration_ms": "integer",
        },
    }


def dumps(envelope, indent=None):
    return json.dumps(to_jsonable(envelope), indent=indent, ensure_ascii=False)

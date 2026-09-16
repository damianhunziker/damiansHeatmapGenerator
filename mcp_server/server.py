#!/usr/bin/env python3
"""MCP server for Damian's Heatmap Generator.

Exposes the four strategy tools (``pnl``, ``chart_analysis``, ``heatmap``,
``automator``), the data fetcher and the ``series`` debugging endpoint over the
Model Context Protocol (stdio transport).

Every tool call runs ``python test.py <tool> ... --api`` inside the project
directory and returns the resulting JSON envelope.  Running the CLI as a
subprocess keeps the MCP stdout channel (used for JSON-RPC) clean, since the
tool code prints heavily and uses multiprocessing.

Run locally (inside the project container):

    python mcp_server/server.py

The process speaks MCP on stdin/stdout; it opens no network ports.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

try:
    # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server
except ImportError:  # pragma: no cover - mcp 2.x renamed FastMCP -> MCPServer
    try:
        from mcp.server.mcpserver import MCPServer as _Server
    except ImportError as exc:  # pragma: no cover - dependency guard
        sys.stderr.write(
            "The 'mcp' package is required to run this server.\n"
            "Install it with:  pip install mcp\n"
            f"Import error: {exc}\n"
        )
        raise

PROJECT_ROOT = os.environ.get(
    "HEATMAP_PROJECT_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

PYTHON = os.environ.get("HEATMAP_PYTHON", sys.executable)
TEST_PY = os.path.join(PROJECT_ROOT, "test.py")
DEFAULT_TIMEOUT = int(os.environ.get("HEATMAP_MCP_TIMEOUT", "1500"))
ARTIFACT_DIRS = ["html_cache", "automator_html", "series_cache", "pnl_cache"]

# Tools that are offloaded to the home server instead of running in this
# container.  The container has no ssh/rsync, so the job is handed to a
# host-side watcher (scripts/remote_watcher.py) through the bind-mounted
# .remote_jobs directory; the watcher runs scripts/remote.py and returns the
# JSON envelope.
REMOTE_TOOLS = {
    t.strip()
    for t in os.environ.get("HEATMAP_MCP_REMOTE_TOOLS", "heatmap,automator").split(",")
    if t.strip()
}
JOBS_DIR = os.environ.get("HEATMAP_JOBS_DIR", os.path.join(PROJECT_ROOT, ".remote_jobs"))

mcp = _Server("damians-heatmap")


# ---------------------------------------------------------------------------
# subprocess bridge
# ---------------------------------------------------------------------------

def _fmt_arg(key, value):
    if isinstance(value, bool):
        return f"--{key}={'true' if value else 'false'}"
    if isinstance(value, (dict, list)):
        return f"--{key}={json.dumps(value)}"
    return f"--{key}={value}"


def _call_remote(tool, params, timeout=None):
    """Offload a tool to the home server via the host-side job watcher.

    Writes a job file into the bind-mounted ``.remote_jobs/incoming`` directory
    and polls ``.remote_jobs/done/<id>/result.json`` for the JSON envelope.
    """
    import time
    import uuid

    incoming = os.path.join(JOBS_DIR, "incoming")
    done = os.path.join(JOBS_DIR, "done")
    os.makedirs(incoming, exist_ok=True)
    os.makedirs(done, exist_ok=True)

    job_id = f"{tool}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    args = [_fmt_arg(key, value) for key, value in params.items()]
    job = {"id": job_id, "tool": tool, "args": args, "created_at": time.time()}

    tmp = os.path.join(incoming, job_id + ".json.tmp")
    with open(tmp, "w") as handle:
        json.dump(job, handle)
    os.replace(tmp, os.path.join(incoming, job_id + ".json"))

    limit = timeout or DEFAULT_TIMEOUT
    deadline = time.time() + limit
    result_path = os.path.join(done, job_id, "result.json")
    while time.time() < deadline:
        if os.path.exists(result_path):
            try:
                with open(result_path) as handle:
                    return json.load(handle)
            except (json.JSONDecodeError, OSError):
                pass
        time.sleep(0.5)

    return {
        "ok": False,
        "tool": tool,
        "error": (
            f"remote job {job_id} timed out after {limit}s "
            "(is scripts/remote_watcher.py running on the host?)"
        ),
    }


def _call(tool, params, timeout=None):
    """Run ``test.py <tool> ... --api`` and return the parsed JSON envelope."""
    clean = {k: v for k, v in (params or {}).items() if v is not None}
    if tool != "schema" and tool in REMOTE_TOOLS:
        return _call_remote(tool, clean, timeout)
    if tool == "schema":
        args = [PYTHON, TEST_PY, "--schema", "--api"]
    else:
        args = [PYTHON, TEST_PY, tool]
        for key, value in clean.items():
            args.append(_fmt_arg(key, value))
        args.append("--api")

    try:
        proc = subprocess.run(
            args,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout or DEFAULT_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "tool": tool, "error": f"timeout after {timeout or DEFAULT_TIMEOUT}s"}

    stdout = (proc.stdout or "").strip()
    if not stdout:
        return {
            "ok": False,
            "tool": tool,
            "error": f"no output from test.py (exit {proc.returncode})",
            "stderr_tail": (proc.stderr or "")[-2000:],
        }
    try:
        envelope = json.loads(stdout)
    except json.JSONDecodeError:
        return {
            "ok": False,
            "tool": tool,
            "error": "test.py did not return valid JSON",
            "stdout_tail": stdout[-2000:],
            "stderr_tail": (proc.stderr or "")[-2000:],
        }
    envelope.setdefault("stderr_tail", (proc.stderr or "")[-1000:])
    return envelope


def _merge(extra, **kwargs):
    base = {k: v for k, v in kwargs.items() if v is not None}
    if extra:
        base.update(extra)
    return base


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------

@mcp.tool()
def get_schema() -> dict:
    """Return the full tool/strategy schema: parameters, ranges and intervals."""
    return _call("schema", {}, timeout=300)


@mcp.tool()
def fetch_data(asset: str, interval: str = "4h") -> dict:
    """Ensure the OHLC cache for an asset/interval exists and report its range."""
    return _call("fetcher", {"asset": asset, "interval": interval})


@mcp.tool()
def run_pnl(asset: str, strategy: str, start_date: str = None, end_date: str = None,
            interval: str = "4h", direction: str = "both",
            initial_equity: float = 10000, fee_pct: float = 0.04,
            params: dict = None, timeout: int = None) -> dict:
    """PnL analysis for both/long/short: metrics, trades, equity curve, drawdown."""
    return _call("pnl", _merge(
        params, asset=asset, strategy=strategy, start_date=start_date, end_date=end_date,
        interval=interval, trade_direction=direction,
        initial_equity=initial_equity, fee_pct=fee_pct,
    ), timeout=timeout)


@mcp.tool()
def run_chart_analysis(asset: str, strategy: str, start_date: str = None,
                       end_date: str = None, interval: str = "4h",
                       initial_equity: float = 10000, fee_pct: float = 0.04,
                       params: dict = None, timeout: int = None) -> dict:
    """Interactive chart + trade list + summary; returns the HTML artifact URL."""
    return _call("chart_analysis", _merge(
        params, asset=asset, strategy=strategy, start_date=start_date, end_date=end_date,
        interval=interval, initial_equity=initial_equity, fee_pct=fee_pct,
    ), timeout=timeout)


@mcp.tool()
def run_heatmap(asset: str, strategy: str, param_ranges: dict = None,
                derived_params: dict = None,
                resolvers: dict = None, resolver_config: str = None,
                use_config_ranges: bool = False,
                start_date: str = None, end_date: str = None, interval: str = "4h",
                initial_equity: float = 10000, fee_pct: float = 0.04,
                max_combos: int = 400, params: dict = None,
                timeout: int = None) -> dict:
    """Sweep a parameter grid. Returns grid, best cells, robustness and HTML URL.

    ``param_ranges`` maps a parameter to a list of values or a spec like
    ``{"min": 0.1, "max": 1.0, "step": 0.1}`` (or ``{"count": 5}``).

    **Virtual parameters (recommended for KAMA).**  The default resolver config
    (``configs/param_resolvers.json``) exposes the virtual axis
    ``kama_normalized_length``, resolved with the *all_exact* rule
    (``length' = k*L``, ``fast'/slow' = k*(P+1)-1``) into
    ``entry_kama_length/fast/slow``, ``kama2_*``, ``exit_kama_*``
    (LiveKAMASSLStrategy) or ``slow_kama_length/fast/slow`` (DMXStrategy).  Sweep
    it like any parameter::

        run_heatmap(
            asset="BTCUSDT", strategy="LiveKAMASSLStrategy",
            start_date="2024-01-01", end_date="2024-06-01",
            param_ranges={
                "kama_normalized_length": {"min": 1, "max": 8, "step": 1},
                "entry_filter": {"min": 0.5, "max": 1.5, "step": 0.25},
            },
        )

    Use the range declared in the config with ``{"kama_normalized_length":
    "config"}``, or auto-inject configured virtual ranges with
    ``use_config_ranges=True``.  ``resolvers`` overrides the config inline and
    ``resolver_config`` points at a different JSON file.  Applied resolvers are
    returned in ``data.resolvers``; see ``docs/parameter-resolution-suite.md``.

    ``derived_params`` is the older KAMA-only variant, e.g.
    ``{"entry_kama_length": {"scale": "kama_scale", "base": 16, "kind": "window"}}``.
    """
    return _call("heatmap", _merge(
        params, asset=asset, strategy=strategy, param_ranges=param_ranges,
        derived_params=derived_params, resolvers=resolvers,
        resolver_config=resolver_config, use_config_ranges=use_config_ranges,
        start_date=start_date, end_date=end_date, interval=interval,
        initial_equity=initial_equity, fee_pct=fee_pct,
        max_combos=max_combos,
    ), timeout=timeout)


@mcp.tool()
def analyze_plateaus(grid: list = None, grid_path: str = None,
                     asset: str = None, strategy: str = None,
                     param_ranges: dict = None, plateau_config: dict = None,
                     x_param: str = None, y_param: str = None,
                     start_date: str = None, end_date: str = None,
                     interval: str = "4h", initial_equity: float = 10000,
                     fee_pct: float = 0.04, max_combos: int = 400,
                     params: dict = None, timeout: int = None) -> dict:
    """Find robust parameter *plateaus* algorithmically (no vision).

    Operates on the underlying data matrix, never on a rendered image:
    composite score -> smoothing -> local stability -> threshold -> morphology
    -> connected regions -> region scoring -> representative cell.

    Provide the grid in one of three ways (cheapest first):
    - ``grid``: rows from a previous ``run_heatmap`` (or any list of records
      with ``x``, ``y`` and metric columns);
    - ``grid_path``: the JSON sidecar written next to a heatmap HTML
      (``data.sidecar`` from ``run_heatmap``);
    - ``asset`` + ``strategy`` + ``param_ranges``: runs a fresh heatmap first.

    ``plateau_config`` tunes the analysis without re-running the backtest, e.g.::

        analyze_plateaus(grid=..., plateau_config={
            "min_trades": 30,
            "weights": {"sharpe_ratio": 1.0, "drawdown_pct": -0.7},
            "threshold_k": 0.5,
            "min_area": 6,
            "representative": "medoid",
        })

    Returns ``regions`` (ranked, with parameter ranges, score stats, area, cv,
    boundary penalty and a representative cell), ``best_region``,
    ``largest_rectangle`` and the config used.
    """
    return _call("plateaus", _merge(
        params, grid=grid, grid_path=grid_path, asset=asset, strategy=strategy,
        param_ranges=param_ranges, plateau_config=plateau_config,
        x_param=x_param, y_param=y_param, start_date=start_date,
        end_date=end_date, interval=interval, initial_equity=initial_equity,
        fee_pct=fee_pct, max_combos=max_combos,
    ), timeout=timeout)


@mcp.tool()
def validate_plateau(region: dict = None, param_ranges: dict = None,
                     asset: str = None, strategy: str = None,
                     start_date: str = None, end_date: str = None,
                     interval: str = "4h", windows: int = 3,
                     plateau_config: dict = None, initial_equity: float = 10000,
                     fee_pct: float = 0.04, max_combos: int = 400,
                     params: dict = None, timeout: int = None) -> dict:
    """Walk-forward validation of a parameter region across ``windows`` slices.

    ``region`` may be a ``param_ranges`` spec (``{"a": {"min":..,"max":..}}``)
    or the ``best_region`` block returned by ``analyze_plateaus``.  Each window
    is a fresh heatmap restricted to that region; the response reports the
    representative cell per window plus a consistency summary.  A real plateau
    keeps producing a region; an overfit spike does not.
    """
    return _call("validate_plateau", _merge(
        params, region=region, param_ranges=param_ranges, asset=asset,
        strategy=strategy, start_date=start_date, end_date=end_date,
        interval=interval, windows=windows, plateau_config=plateau_config,
        initial_equity=initial_equity, fee_pct=fee_pct, max_combos=max_combos,
    ), timeout=timeout)


@mcp.tool()
def plot_plateau(grid: list = None, grid_path: str = None, asset: str = None,
                 strategy: str = None, param_ranges: dict = None,
                 plateau_config: dict = None, output: str = None,
                 title: str = None, params: dict = None,
                 timeout: int = None) -> dict:
    """Render the score matrix with region outlines to a PNG (presentation only)."""
    return _call("plot_plateau", _merge(
        params, grid=grid, grid_path=grid_path, asset=asset, strategy=strategy,
        param_ranges=param_ranges, plateau_config=plateau_config,
        output=output, title=title,
    ), timeout=timeout)


@mcp.tool()
def run_automator(strategy: str, pairs: list = None, param_ranges: dict = None,
                  derived_params: dict = None,
                  resolvers: dict = None, resolver_config: str = None,
                  use_config_ranges: bool = False,
                  start_date: str = None, end_date: str = None,
                  interval: str = "4h", higher_tf: str = "1d",
                  initial_equity: float = 1000, fee_pct: float = 0.04,
                  max_combos: int = 400, params: dict = None,
                  timeout: int = None) -> dict:
    """Run the heatmap across multiple pairs. Returns per-pair results."""
    return _call("automator", _merge(
        params, strategy=strategy, pairs=pairs, param_ranges=param_ranges,
        derived_params=derived_params, resolvers=resolvers,
        resolver_config=resolver_config, use_config_ranges=use_config_ranges,
        start_date=start_date, end_date=end_date, interval=interval,
        higher_tf=higher_tf, initial_equity=initial_equity, fee_pct=fee_pct,
        max_combos=max_combos,
    ), timeout=timeout)


@mcp.tool()
def get_series(asset: str, strategy: str, start_date: str = None,
               end_date: str = None, interval: str = "4h", columns: list = None,
               tail: int = None, format: str = "json", inline_max_cells: int = 5000,
               params: dict = None, timeout: int = None) -> dict:
    """Per-candle indicator/signal series for runtime debugging.

    Returns inline rows for small results; for large results writes a Parquet
    (or CSV) file and returns its path instead (query it with DuckDB).
    """
    return _call("series", _merge(
        params, asset=asset, strategy=strategy, start_date=start_date,
        end_date=end_date, interval=interval, columns=columns, tail=tail,
        format=format, inline_max_cells=inline_max_cells,
    ), timeout=timeout)


@mcp.tool()
def run_tool(tool: str, params: dict = None, timeout: int = None) -> dict:
    """Generic escape hatch: run any tool ('pnl', 'heatmap', 'series', ...)."""
    return _call(tool, params or {}, timeout=timeout)


@mcp.tool()
def list_artifacts() -> dict:
    """List generated report/series files with their viewer URLs."""
    from core.html_viewer import url_for  # local import: works inside the project

    items = []
    for directory in ARTIFACT_DIRS:
        abs_dir = os.path.join(PROJECT_ROOT, directory)
        if not os.path.isdir(abs_dir):
            continue
        for root, _dirs, files in os.walk(abs_dir):
            for name in files:
                full = os.path.join(root, name)
                rel = os.path.relpath(full, PROJECT_ROOT).replace(os.sep, "/")
                try:
                    url = url_for(full)
                except Exception:
                    url = None
                items.append({
                    "path": full,
                    "rel": rel,
                    "size": os.path.getsize(full),
                    "mtime": os.path.getmtime(full),
                    "url": url,
                })
    items.sort(key=lambda item: item["mtime"], reverse=True)
    return {"ok": True, "count": len(items), "artifacts": items}


@mcp.tool()
def read_artifact(path: str, max_bytes: int = 200000) -> dict:
    """Read a generated text artifact (html/json/csv). Parquet is not read raw."""
    full = path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)
    if not os.path.exists(full):
        return {"ok": False, "error": f"not found: {full}"}
    if full.endswith(".parquet"):
        return {
            "ok": True,
            "path": full,
            "note": "binary Parquet; query it with DuckDB instead of reading raw",
            "size": os.path.getsize(full),
        }
    with open(full, "r", encoding="utf-8", errors="replace") as handle:
        content = handle.read(max_bytes)
    truncated = os.path.getsize(full) > max_bytes
    return {"ok": True, "path": full, "truncated": truncated, "content": content}


if __name__ == "__main__":
    mcp.run()

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


def _call(tool, params, timeout=None):
    """Run ``test.py <tool> ... --api`` and return the parsed JSON envelope."""
    clean = {k: v for k, v in (params or {}).items() if v is not None}
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
                start_date: str = None, end_date: str = None, interval: str = "4h",
                initial_equity: float = 10000, fee_pct: float = 0.04,
                workers: int = None, no_images: bool = True,
                max_combos: int = 400, params: dict = None,
                timeout: int = None) -> dict:
    """Sweep a parameter grid. Returns grid, best cells, robustness and HTML URL.

    ``param_ranges`` maps a parameter to either a list of values or a spec like
    ``{"min": 0.1, "max": 1.0, "step": 0.1}`` (or ``{"count": 5}``).
    """
    return _call("heatmap", _merge(
        params, asset=asset, strategy=strategy, param_ranges=param_ranges,
        start_date=start_date, end_date=end_date, interval=interval,
        initial_equity=initial_equity, fee_pct=fee_pct,
        workers=workers, no_images=no_images, max_combos=max_combos,
    ), timeout=timeout)


@mcp.tool()
def run_automator(strategy: str, pairs: list = None, param_ranges: dict = None,
                  start_date: str = None, end_date: str = None,
                  interval: str = "4h", higher_tf: str = "1d",
                  initial_equity: float = 1000, fee_pct: float = 0.04,
                  workers: int = None, no_images: bool = True,
                  max_combos: int = 400, params: dict = None,
                  timeout: int = None) -> dict:
    """Run the heatmap across multiple pairs. Returns per-pair results."""
    return _call("automator", _merge(
        params, strategy=strategy, pairs=pairs, param_ranges=param_ranges,
        start_date=start_date, end_date=end_date, interval=interval,
        higher_tf=higher_tf, initial_equity=initial_equity, fee_pct=fee_pct,
        workers=workers, no_images=no_images, max_combos=max_combos,
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

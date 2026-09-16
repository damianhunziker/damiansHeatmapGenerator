#!/usr/bin/env python3
"""Coarse-to-fine heatmap optimization, generic over strategy and parameters.

Runs a multi-resolution search for a *stable plateau* (not a single point) and
varies both the grid resolution and the evaluation window / range, so the result
is validated instead of overfit.

Stages
------
1. ``coarse``  - wide ranges, few values per axis (fast orientation).
2. ``medium``  - zoom around the best robust cell, more values.
3. ``fine``    - zoom again, highest resolution (resolve the plateau centre).
4. ``validate``- re-run the final ranges on several walk-forward windows and on
   shifted/widened ranges; a real plateau survives all of them.

The robust score per cell prefers high profit that is *consistent with its
neighbours* (``min(profit, neighbor_min)``), requires enough trades and a full
neighbourhood.

Usage (from the host)::

    docker compose exec app python scripts/coarse_to_fine_heatmap.py \
        --asset ETHUSDT --strategy DMXStrategy \
        --start 2019-01-01 --end 2024-01-01 \
        --param-ranges '{"squeeze_limit":{"min":100,"max":400},"kama_normalized_length":{"min":1,"max":8}}' \
        --stage-counts 4,6,10 --zoom 0.5 --windows 3

Exactly two axes are required (the heatmap is 2D). A virtual parameter (resolver
input) counts as one axis; see the ``heatmap-optimization`` skill.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from core import api


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asset", required=True)
    ap.add_argument("--strategy", required=True)
    ap.add_argument("--interval", default="4h")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--param-ranges", required=True,
                    help='JSON: {"axis1":{"min":..,"max":..},"axis2":{"min":..,"max":..}}')
    ap.add_argument("--stage-counts", required=True,
                    help="values per axis for coarse,medium,fine as 'c,m,f' - "
                         "the caller decides the resolution (no default)")
    ap.add_argument("--zoom", type=float, default=0.5,
                    help="each stage keeps this fraction of the half-range (default 0.5)")
    ap.add_argument("--min-trades", type=int, default=10)
    ap.add_argument("--windows", type=int, default=3,
                    help="walk-forward slices for validation (default 3)")
    ap.add_argument("--shift", type=float, default=0.15,
                    help="range shift for validation (fraction, default 0.15)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max-combos", type=int, default=400)
    ap.add_argument("--equity", type=float, default=10000.0)
    ap.add_argument("--fee", type=float, default=0.04)
    ap.add_argument("--resolvers", default=None, help="inline resolvers JSON (optional)")
    ap.add_argument("--resolver-config", default=None)
    ap.add_argument("--out", default=None, help="write the full report as JSON")
    return ap.parse_args()


def run_heatmap(args, ranges, start, end):
    """Run one heatmap via the API and return its data dict (or raise)."""
    params = {
        "asset": args.asset,
        "strategy": args.strategy,
        "interval": args.interval,
        "start_date": start,
        "end_date": end,
        "param_ranges": ranges,
        "workers": args.workers,
        "max_combos": args.max_combos,
        "initial_equity": args.equity,
        "fee_pct": args.fee,
    }
    if args.resolvers:
        params["resolvers"] = json.loads(args.resolvers)
    if args.resolver_config:
        params["resolver_config"] = args.resolver_config
    env = api.run_tool("heatmap", params)
    if not env.get("ok"):
        raise RuntimeError(f"heatmap failed: {env.get('error')}")
    return env["data"]


def _robustness_index(data):
    idx = {}
    for row in data.get("robustness") or []:
        idx[(row.get("x"), row.get("y"))] = row
    return idx


def robust_score(cell, rob, min_trades):
    """Higher is better. Rewards profit consistent with the neighbourhood."""
    if cell.get("num_trades", 0) < min_trades:
        return float("-inf")
    profit = cell.get("profit")
    if profit is None:
        return float("-inf")
    r = rob.get((cell.get("x"), cell.get("y")))
    if not r or r.get("neighbor_count", 0) < 3:
        return float(profit)
    neighbor_min = r.get("neighbor_min")
    neighbor_mean = r.get("neighbor_mean")
    floor = neighbor_min if neighbor_min is not None else profit
    # balance absolute level and neighbourhood consistency
    return min(float(profit), float(floor)) * 0.7 + float(profit) * 0.3


def pick_centre(data, min_trades, top_k=3):
    """Return the robust centre: mean coordinates of the top-K scoring cells."""
    rob = _robustness_index(data)
    scored = sorted(
        ((robust_score(c, rob, min_trades), c) for c in data.get("grid") or []),
        key=lambda t: t[0], reverse=True)
    good = [(s, c) for s, c in scored if np.isfinite(s)]
    if not good:
        raise RuntimeError("no cell met the trade/robustness criteria")
    top = good[:top_k]
    xs = [c["x"] for _, c in top]
    ys = [c["y"] for _, c in top]
    return {"x": float(np.mean(xs)), "y": float(np.mean(ys)),
            "score": float(np.mean([s for s, _ in top])),
            "num_trades": int(np.median([c["num_trades"] for _, c in top]))}


def make_ranges(base, count, axes, int_axes):
    """Build a stage grid from min/max ranges and a value count per axis."""
    out = {}
    for axis in axes:
        lo, hi = base[axis]["min"], base[axis]["max"]
        if int_axes[axis]:
            lo, hi = int(round(lo)), int(round(hi))
            out[axis] = sorted({int(round(v)) for v in np.linspace(lo, hi, count)})
        else:
            out[axis] = [float(v) for v in np.linspace(lo, hi, count)]
    return out


def zoom_ranges(base, centre, zoom, int_axes):
    """Shrink each axis to ``zoom`` of its half-range around ``centre``."""
    out = {}
    for axis, c in zip(base.keys(), (centre["x"], centre["y"])):
        lo, hi = base[axis]["min"], base[axis]["max"]
        half = max((hi - lo) / 2.0 * zoom, 1e-9)
        new_lo, new_hi = max(lo, c - half), min(hi, c + half)
        if int_axes[axis]:
            new_lo, new_hi = int(round(new_lo)), int(round(new_hi))
            if new_hi <= new_lo:
                new_hi = new_lo + 1
        out[axis] = {"min": new_lo, "max": new_hi}
    return out


def main():
    args = parse_args()
    base = json.loads(args.param_ranges)
    axes = list(base.keys())
    if len(axes) != 2:
        raise SystemExit("exactly two axes are required (got %d)" % len(axes))
    int_axes = {
        a: (float(base[a]["min"]).is_integer() and float(base[a]["max"]).is_integer())
        for a in axes
    }
    counts = [int(x) for x in str(args.stage_counts).split(",")]
    if len(counts) != 3:
        raise SystemExit("--stage-counts must be 'coarse,medium,fine'")

    report = {"asset": args.asset, "strategy": args.strategy,
              "window": [args.start, args.end], "axes": axes, "stages": []}

    ranges = {a: {"min": base[a]["min"], "max": base[a]["max"]} for a in axes}
    centre = None
    for name, count in zip(("coarse", "medium", "fine"), counts):
        grid = make_ranges(ranges, count, axes, int_axes)
        n = int(np.prod([len(grid[a]) for a in axes]))
        print(f"\n=== {name.upper()}: {n} combinations, ranges={grid}")
        data = run_heatmap(args, grid, args.start, args.end)
        if data.get("warnings"):
            print("  warnings:", data["warnings"])
        centre = pick_centre(data, args.min_trades)
        print(f"  robust centre: {axes[0]}={centre['x']:.4g}  {axes[1]}={centre['y']:.4g}"
              f"  (score={centre['score']:.2f}, ~{centre['num_trades']} trades)")
        report["stages"].append({"stage": name, "counts": count, "ranges": grid,
                                 "centre": centre})
        if name != "fine":
            ranges = zoom_ranges(ranges, centre, args.zoom, int_axes)

    # Final (fine) ranges for validation.
    final_ranges = report["stages"][-1]["ranges"]

    # --- Validation: walk-forward windows -----------------------------------
    start = np.datetime64(args.start)
    end = np.datetime64(args.end)
    edges = np.linspace(start.astype("int64"), end.astype("int64"), args.windows + 1)
    print(f"\n=== VALIDATION: {args.windows} walk-forward window(s)")
    window_results = []
    for i in range(args.windows):
        w0 = np.datetime64(int(edges[i]), "D")
        w1 = np.datetime64(int(edges[i + 1]), "D")
        try:
            data = run_heatmap(args, final_ranges, str(w0), str(w1))
        except Exception as exc:  # noqa: BLE001
            print(f"  window {i + 1}: skipped ({exc})")
            continue
        c = pick_centre(data, max(1, args.min_trades // 3))
        window_results.append({"window": [str(w0), str(w1)], "centre": c})
        print(f"  window {i + 1} {w0}..{w1}: centre "
              f"{axes[0]}={c['x']:.4g} {axes[1]}={c['y']:.4g} (score={c['score']:.2f})")

    # --- Validation: shifted / widened range --------------------------------
    shifted = {}
    for a in axes:
        lo, hi = final_ranges[a][0], final_ranges[a][-1]
        pad = (hi - lo) * args.shift
        shifted[a] = {"min": lo - pad, "max": hi + pad}
    print(f"\n=== VALIDATION: shifted/widened range (+/-{args.shift:.0%})")
    try:
        data = run_heatmap(args, make_ranges(shifted, counts[2], axes, int_axes), args.start, args.end)
        c = pick_centre(data, args.min_trades)
        print(f"  centre {axes[0]}={c['x']:.4g} {axes[1]}={c['y']:.4g} (score={c['score']:.2f})")
        report["shift_validation"] = {"ranges": shifted, "centre": c}
    except Exception as exc:  # noqa: BLE001
        print(f"  skipped ({exc})")

    report["windows_validation"] = window_results
    report["recommended"] = {
        axes[0]: centre["x"], axes[1]: centre["y"],
        "_note": "centre of the fine-stage plateau; confirm it holds across the validation windows",
    }

    print("\n=== RECOMMENDED (plateau centre)")
    print(json.dumps(report["recommended"], indent=2))

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2, default=str)
        print(f"\nreport written: {args.out}")


if __name__ == "__main__":
    main()

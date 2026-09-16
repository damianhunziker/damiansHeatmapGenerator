"""Algorithmic plateau / region detection for heatmap parameter grids.

The heatmap already owns the underlying data matrix (one row of metrics per
parameter combination), so plateaus are found **numerically** - never by reading
pixels of a rendered image.  A *plateau* is a connected area of high score that
barely degrades when a parameter is nudged::

    grid -> composite score matrix -> smoothing -> local stability
         -> threshold -> morphology -> connected components
         -> region scoring -> representative cell

The result is a JSON-safe payload (``detect_plateaus``) that an LLM can rank,
compare and report on.  Vision is only ever a fallback for images that have no
underlying matrix.

Dependencies: numpy, scipy.ndimage, scikit-image, matplotlib (overlay only).
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage
from skimage import measure, morphology

METHOD = "algorithmic-plateau-v1"

DEFAULT_CONFIG = {
    # Composite score (z-scored per metric; sign = direction of "good").
    "score_metric": "composite",
    "weights": {
        "sharpe_ratio": 1.0,
        "profit_factor": 0.3,
        "drawdown_pct": -0.7,
        "num_trades": -0.5,
    },
    # Hard filter: fewer trades than this makes a cell invalid.
    "min_trades": 10,
    # Smoothing of the score matrix.
    "smooth_size": 3,
    "smooth_kind": "median",  # median | mean | gaussian
    # Local stability = local_mean - std_penalty * local_std.
    "stability_size": 3,
    "std_penalty": 0.8,
    # Threshold: None -> auto (mean + threshold_k * std of the stability map).
    "threshold": None,
    "threshold_k": 1.0,
    # Optional absolute floor on the smoothed score.
    "min_score": None,
    # Drop regions smaller than this many cells.
    "min_area": 4,
    # Optional opening (in cells) to cut thin bridges; 0 disables it.
    "opening_size": 0,
    # Representative cell: medoid | maximin | center_of_mass | rectangle_center | best.
    "representative": "medoid",
    # Region ranking weights.
    "region_weights": {
        "mean": 1.0,
        "area": 0.5,
        "std": -1.0,
        "drawdown": -0.7,
        "consistency": 0.8,
        "boundary": -0.5,
        "duration": -0.3,
    },
}


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _key(value):
    """Normalise a parameter value for use as a dict key (float when numeric)."""
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _f(value):
    """JSON-safe float (NaN/inf -> None)."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _i(value):
    return None if value is None else int(value)


def _merge_config(config=None):
    cfg = dict(DEFAULT_CONFIG)
    cfg["weights"] = dict(DEFAULT_CONFIG["weights"])
    cfg["region_weights"] = dict(DEFAULT_CONFIG["region_weights"])
    for key, value in (config or {}).items():
        if key in ("weights", "region_weights") and isinstance(value, dict):
            cfg[key] = dict(value)
        else:
            cfg[key] = value
    return cfg


def _robust_z(matrix):
    """Median/MAD z-score; NaN stays NaN; constant input -> zeros."""
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        return np.zeros_like(matrix)
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    scale = 1.4826 * mad
    if scale < 1e-12:
        scale = float(finite.std())
    if scale < 1e-12:
        scale = 1.0
    return (matrix - median) / scale


def _nan_smooth(matrix, size, kind):
    """NaN-aware smoothing; masked cells stay masked."""
    if not size or int(size) <= 1:
        return matrix.astype(float, copy=True)
    size = int(size)
    nan_mask = ~np.isfinite(matrix)
    if not nan_mask.any():
        filled = matrix.astype(float, copy=True)
    else:
        finite = matrix[np.isfinite(matrix)]
        fill = float(np.median(finite)) if finite.size else 0.0
        filled = np.where(nan_mask, fill, matrix).astype(float)
    if kind == "gaussian":
        out = ndimage.gaussian_filter(filled, sigma=size / 2.0, mode="nearest")
    elif kind == "mean":
        out = ndimage.uniform_filter(filled, size=size, mode="nearest")
    else:
        out = ndimage.median_filter(filled, size=size, mode="nearest")
    out[nan_mask] = np.nan
    return out


def _local_stats(matrix, size):
    """NaN-aware local mean / std over a ``size`` x ``size`` window."""
    size = max(1, int(size))
    valid = np.isfinite(matrix).astype(float)
    filled = np.where(valid > 0, matrix, 0.0)
    count = ndimage.uniform_filter(valid, size=size, mode="nearest")
    s1 = ndimage.uniform_filter(filled, size=size, mode="nearest")
    s2 = ndimage.uniform_filter(filled * filled, size=size, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, s1 / np.maximum(count, 1e-12), np.nan)
        var = np.where(count > 0, s2 / np.maximum(count, 1e-12) - mean ** 2, np.nan)
    std = np.sqrt(np.maximum(var, 0.0))
    return mean, std


# ---------------------------------------------------------------------------
# score matrix
# ---------------------------------------------------------------------------

class ScoreMatrix:
    """The grid reshaped into a 2D matrix plus per-metric matrices."""

    def __init__(self, x_key, y_key, x_values, y_values, score, metrics, trades,
                 lookup, valid, meta):
        self.x_key = x_key
        self.y_key = y_key
        self.x_values = x_values
        self.y_values = y_values
        self.score = score
        self.metrics = metrics
        self.trades = trades
        self.lookup = lookup
        self.valid = valid
        self.meta = meta

    @property
    def shape(self):
        return self.score.shape


def build_score_matrix(grid, param_ranges, config=None):
    """Reshape ``grid`` into matrices and build the composite score.

    ``score = sum(weight_i * z(metric_i))`` with robust (median/MAD) z-scoring.
    Cells with ``num_trades < min_trades`` (or a missing metric) become invalid
    (NaN) and never enter a plateau.
    """
    cfg = _merge_config(config)
    keys = list(param_ranges.keys())
    if len(keys) < 2:
        raise ValueError("plateau detection needs exactly two parameter axes")
    x_key, y_key = keys[0], keys[1]
    x_values = [_key(v) for v in np.asarray(param_ranges[x_key]).tolist()]
    y_values = [_key(v) for v in np.asarray(param_ranges[y_key]).tolist()]
    x_index = {v: i for i, v in enumerate(x_values)}
    y_index = {v: i for i, v in enumerate(y_values)}
    ny, nx = len(y_values), len(x_values)

    metric_names = list(cfg["weights"].keys())
    metrics = {name: np.full((ny, nx), np.nan) for name in metric_names}
    trades = np.full((ny, nx), np.nan)
    lookup = {}

    for row in grid or []:
        ix = x_index.get(_key(row.get("x")))
        iy = y_index.get(_key(row.get("y")))
        if ix is None or iy is None:
            continue
        lookup[(x_values[ix], y_values[iy])] = row
        if row.get("num_trades") is not None:
            trades[iy, ix] = float(row["num_trades"])
        for name in metric_names:
            value = row.get(name)
            if value is not None:
                try:
                    metrics[name][iy, ix] = float(value)
                except (TypeError, ValueError):
                    pass

    used, missing = [], []
    score = np.zeros((ny, nx))
    for name, weight in cfg["weights"].items():
        matrix = metrics[name]
        if np.isfinite(matrix).any():
            score += float(weight) * _robust_z(matrix)
            used.append(name)
        else:
            missing.append(name)

    min_trades = float(cfg["min_trades"])
    valid = np.isfinite(trades) & (trades >= min_trades)
    score = np.where(valid, score, np.nan)

    meta = {
        "metrics_used": used,
        "metrics_missing": missing,
        "min_trades": int(min_trades),
        "valid_cells": int(valid.sum()),
        "total_cells": int(ny * nx),
    }
    return ScoreMatrix(x_key, y_key, x_values, y_values, score, metrics, trades,
                       lookup, valid, meta)


def stability_map(score, size=3, std_penalty=0.8):
    """High local mean with low local spread -> high stability."""
    mean, std = _local_stats(score, size)
    return mean - float(std_penalty) * std


def threshold_mask(stability, score, threshold=None, threshold_k=1.0,
                   min_score=None, valid=None):
    """Binary mask of cells that are stable enough and score high enough."""
    finite = stability[np.isfinite(stability)]
    if threshold is None:
        if finite.size == 0:
            threshold = 0.0
        else:
            threshold = float(np.mean(finite) + float(threshold_k) * np.std(finite))
    threshold = float(threshold)
    mask = np.isfinite(stability) & (stability >= threshold)
    if valid is not None:
        mask &= valid
    if min_score is not None:
        mask &= np.isfinite(score) & (score >= float(min_score))
    return mask, threshold


def clean_mask(mask, min_area=1, opening_size=0):
    """Close small holes, optionally cut thin bridges, drop tiny specks.

    Opening is deliberately *off* by default: a full 3x3 opening erodes small
    (but genuine) plateaus to nothing.  ``min_area`` removes specks instead, and
    callers that really need thin-bridge removal can pass ``opening_size``.
    """
    footprint = np.ones((3, 3), dtype=bool)
    closed = morphology.closing(mask, footprint=footprint)
    if opening_size and int(opening_size) > 1:
        size = int(opening_size)
        closed = morphology.opening(closed, footprint=np.ones((size, size), dtype=bool))
    if min_area and int(min_area) > 1:
        labels, count = ndimage.label(closed, structure=np.ones((3, 3), dtype=int))
        if count:
            sizes = np.bincount(labels.ravel())
            keep = sizes >= int(min_area)
            keep[0] = False
            closed = keep[labels]
    return closed


def largest_rectangle(mask):
    """Largest all-True axis-aligned rectangle in a 2D boolean mask.

    Returns ``(x0, y0, x1, y1)`` inclusive index bounds and its area, or
    ``None`` when the mask is empty.  Standard histogram/monotonic-stack
    algorithm, O(rows * cols).
    """
    if mask is None or not mask.any():
        return None
    heights = np.zeros(mask.shape[1], dtype=int)
    best = (0, 0, -1, -1, 0)
    for row in range(mask.shape[0]):
        heights = np.where(mask[row], heights + 1, 0)
        stack = []
        for col in range(mask.shape[1] + 1):
            h = heights[col] if col < mask.shape[1] else 0
            start = col
            while stack and stack[-1][1] >= h:
                idx, height = stack.pop()
                area = height * (col - idx)
                if area > best[4]:
                    best = (idx, row - height + 1, col - 1, row, area)
                start = idx
            stack.append((start, h))
    x0, y0, x1, y1, area = best
    if area <= 0:
        return None
    return {"x0": int(x0), "y0": int(y0), "x1": int(x1), "y1": int(y1),
            "area": int(area)}


# ---------------------------------------------------------------------------
# region scoring
# ---------------------------------------------------------------------------

def _boundary_penalty(coords, shape):
    """1.0 at the search-space edge, -> 0 towards the centre."""
    ny, nx = shape
    rows = [c[0] for c in coords]
    cols = [c[1] for c in coords]
    distance = min(min(cols), min(rows), (nx - 1) - max(cols), (ny - 1) - max(rows))
    return 1.0 / (1.0 + max(distance, 0))


def _representative(sm, coords, method):
    """Pick a robust cell from a region (default: the medoid)."""
    scores = np.array([sm.score[r, c] for r, c in coords], dtype=float)
    finite = np.isfinite(scores)
    coords = [c for c, ok in zip(coords, finite) if ok]
    scores = scores[finite]
    if not coords:
        return None
    if method == "best":
        idx = int(np.argmax(scores))
    elif method == "maximin":
        best_idx, best_val = 0, -np.inf
        for i, (r, c) in enumerate(coords):
            neighbours = []
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = r + dr, c + dc
                    if (dr or dc) and 0 <= rr < sm.shape[0] and 0 <= cc < sm.shape[1]:
                        value = sm.score[rr, cc]
                        if np.isfinite(value):
                            neighbours.append(value)
            worst = min(neighbours) if neighbours else scores[i]
            if worst > best_val:
                best_idx, best_val = i, worst
        idx = best_idx
    elif method == "center_of_mass":
        rows = np.array([c[0] for c in coords], dtype=float)
        cols = np.array([c[1] for c in coords], dtype=float)
        cr, cc = rows.mean(), cols.mean()
        idx = int(np.argmin((rows - cr) ** 2 + (cols - cc) ** 2))
    elif method == "rectangle_center":
        rows = [c[0] for c in coords]
        cols = [c[1] for c in coords]
        cr, cc = (min(rows) + max(rows)) / 2.0, (min(cols) + max(cols)) / 2.0
        idx = int(np.argmin([(r - cr) ** 2 + (c - cc) ** 2 for r, c in coords]))
    else:  # medoid
        mean = float(scores.mean())
        idx = int(np.argmin(np.abs(scores - mean)))
    r, c = coords[idx]
    row = sm.lookup.get((sm.x_values[c], sm.y_values[r])) or {}
    return {
        "x": sm.x_values[c],
        "y": sm.y_values[r],
        "method": method,
        "score": _f(sm.score[r, c]),
        "num_trades": _i(row.get("num_trades")),
        "profit": _f(row.get("profit")),
        "sharpe_ratio": _f(row.get("sharpe_ratio")),
        "drawdown_pct": _f(row.get("drawdown_pct")),
    }


def score_region(sm, region_id, coords, cfg):
    """Aggregate robustness metrics for one connected region."""
    rows = np.array([c[0] for c in coords])
    cols = np.array([c[1] for c in coords])
    scores = np.array([sm.score[r, c] for r, c in coords], dtype=float)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        return None
    trades = np.array([sm.trades[r, c] for r, c in coords], dtype=float)
    trades = trades[np.isfinite(trades)]

    drawdowns = []
    for r, c in coords:
        row = sm.lookup.get((sm.x_values[c], sm.y_values[r])) or {}
        value = row.get("drawdown_pct")
        if value is not None:
            drawdowns.append(abs(float(value)))
    mean_dd = float(np.mean(drawdowns)) if drawdowns else 0.0
    duration = []
    for r, c in coords:
        row = sm.lookup.get((sm.x_values[c], sm.y_values[r])) or {}
        value = row.get("avg_trade_duration")
        if value is not None:
            duration.append(float(value))
    duration_penalty = (float(np.mean(duration)) / 100.0) if duration else 0.0

    mean_score = float(scores.mean())
    std_score = float(scores.std())
    cv = (std_score / abs(mean_score)) if abs(mean_score) > 1e-12 else 0.0
    # Cross-period consistency is established by ``validate_plateau`` (walk
    # forward); it is not part of a single grid, so it stays neutral here.
    consistency = None
    boundary = _boundary_penalty(coords, sm.shape)

    weights = cfg["region_weights"]
    region_score = (
        weights["mean"] * mean_score
        + weights["area"] * math.log(max(len(coords), 1))
        + weights["std"] * std_score
        + weights["drawdown"] * (mean_dd / 100.0)
        + weights["consistency"] * (consistency or 0.0)
        + weights["boundary"] * boundary
        + weights["duration"] * duration_penalty
    )

    x_vals = [sm.x_values[c] for c in cols]
    y_vals = [sm.y_values[r] for r in rows]
    representative = _representative(sm, list(coords), cfg["representative"])

    cells = []
    for r, c in coords:
        row = sm.lookup.get((sm.x_values[c], sm.y_values[r])) or {}
        cells.append({
            "x": sm.x_values[c],
            "y": sm.y_values[r],
            "score": _f(sm.score[r, c]),
            "num_trades": _i(row.get("num_trades")),
            "profit": _f(row.get("profit")),
        })

    return {
        "region_id": int(region_id),
        "area": int(len(coords)),
        "x": {"param": sm.x_key, "min": min(x_vals), "max": max(x_vals)},
        "y": {"param": sm.y_key, "min": min(y_vals), "max": max(y_vals)},
        "score": {
            "mean": _f(mean_score),
            "median": _f(np.median(scores)),
            "min": _f(scores.min()),
            "max": _f(scores.max()),
            "p10": _f(np.percentile(scores, 10)),
            "std": _f(std_score),
        },
        "mean_score": _f(mean_score),
        "min_score": _f(scores.min()),
        "cv": _f(cv),
        "subperiod_consistency": _f(consistency),
        "avg_trades": _f(np.mean(trades)) if trades.size else None,
        "mean_max_dd": _f(mean_dd),
        "boundary_penalty": _f(boundary),
        "touches_edge": bool(_boundary_penalty(coords, sm.shape) >= 1.0),
        "region_score": _f(region_score),
        "representative": representative,
        "cells": cells,
    }


def rank_regions(regions):
    return sorted(regions, key=lambda r: (r.get("region_score") or -np.inf),
                  reverse=True)


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def detect_plateaus(grid, param_ranges, config=None):
    """Full pipeline: score -> smooth -> stability -> mask -> regions -> rank.

    Returns a JSON-safe dict with ``regions`` (ranked), ``best_region``, the
    representative cell, the largest rectangle and the config that was used.
    """
    cfg = _merge_config(config)
    sm = build_score_matrix(grid, param_ranges, cfg)

    smoothed = _nan_smooth(sm.score, cfg["smooth_size"], cfg["smooth_kind"])
    stability = stability_map(sm.score, cfg["stability_size"], cfg["std_penalty"])
    mask, threshold = threshold_mask(
        stability, smoothed, threshold=cfg["threshold"],
        threshold_k=cfg["threshold_k"], min_score=cfg["min_score"], valid=sm.valid)
    mask = clean_mask(mask, cfg["min_area"], cfg.get("opening_size", 0))

    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    regions = []
    for region_id in range(1, count + 1):
        coords = list(zip(*np.where(labels == region_id)))
        if len(coords) < int(cfg["min_area"]):
            continue
        region = score_region(sm, region_id, coords, cfg)
        if region:
            regions.append(region)
    regions = rank_regions(regions)

    rectangle = largest_rectangle(mask)
    if rectangle:
        rectangle = {
            "x": {"param": sm.x_key,
                  "min": sm.x_values[rectangle["x0"]],
                  "max": sm.x_values[rectangle["x1"]]},
            "y": {"param": sm.y_key,
                  "min": sm.y_values[rectangle["y0"]],
                  "max": sm.y_values[rectangle["y1"]]},
            "area": rectangle["area"],
        }

    best = regions[0] if regions else None
    return {
        "method": METHOD,
        "x_param": sm.x_key,
        "y_param": sm.y_key,
        "config": _json_config(cfg),
        "score": sm.meta,
        "threshold": _f(threshold),
        "mask_cells": int(mask.sum()),
        "region_count": len(regions),
        "regions": regions,
        "best_region": best,
        "representative": (best or {}).get("representative"),
        "largest_rectangle": rectangle,
    }


def _json_config(cfg):
    return {
        "score_metric": cfg["score_metric"],
        "weights": {k: _f(v) for k, v in cfg["weights"].items()},
        "min_trades": int(cfg["min_trades"]),
        "smooth_size": int(cfg["smooth_size"]),
        "smooth_kind": cfg["smooth_kind"],
        "stability_size": int(cfg["stability_size"]),
        "std_penalty": _f(cfg["std_penalty"]),
        "threshold": _f(cfg["threshold"]),
        "threshold_k": _f(cfg["threshold_k"]),
        "min_score": _f(cfg["min_score"]),
        "min_area": int(cfg["min_area"]),
        "opening_size": int(cfg.get("opening_size", 0)),
        "representative": cfg["representative"],
        "region_weights": {k: _f(v) for k, v in cfg["region_weights"].items()},
    }


# ---------------------------------------------------------------------------
# presentation overlay
# ---------------------------------------------------------------------------

def plot_overlay(grid, param_ranges, config=None, path="plateau_overlay.png",
                 title="Plateau regions"):
    """Render the score matrix with region outlines to a PNG (presentation only)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    cfg = _merge_config(config)
    sm = build_score_matrix(grid, param_ranges, cfg)
    smoothed = _nan_smooth(sm.score, cfg["smooth_size"], cfg["smooth_kind"])
    stability = stability_map(sm.score, cfg["stability_size"], cfg["std_penalty"])
    mask, _threshold = threshold_mask(
        stability, smoothed, threshold=cfg["threshold"],
        threshold_k=cfg["threshold_k"], min_score=cfg["min_score"], valid=sm.valid)
    mask = clean_mask(mask, cfg["min_area"], cfg.get("opening_size", 0))

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(np.ma.masked_invalid(sm.score), origin="lower", aspect="auto",
                      cmap="viridis")
    fig.colorbar(image, ax=ax, label="composite score")
    ax.contour(mask.astype(float), levels=[0.5], colors="red", linewidths=1.5)

    def _index(values, target):
        for i, value in enumerate(values):
            if value == target:
                return i
        return 0

    for region in detect_plateaus(grid, param_ranges, cfg)["regions"]:
        rep = region.get("representative") or {}
        if rep.get("x") is None or rep.get("y") is None:
            continue
        ax.plot(_index(sm.x_values, _key(rep["x"])),
                _index(sm.y_values, _key(rep["y"])),
                marker="*", color="white", markersize=12, markeredgecolor="black")
    ax.set_xlabel(sm.x_key)
    ax.set_ylabel(sm.y_key)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path

"""Tests for algorithmic plateau detection on synthetic parameter grids."""

import json

import numpy as np
import pytest

from core import plateaus


def _grid(shape=(10, 10), base=None, plateau=None, spike=None):
    """Build a synthetic grid.

    ``plateau`` / ``spike`` are dicts mapping (ix, iy) -> metric overrides.
    ``base`` is the metric dict for all other cells.
    """
    base = base or {
        "sharpe_ratio": 0.5, "profit_factor": 1.0,
        "drawdown_pct": 30.0, "num_trades": 50, "profit": 5.0,
    }
    overrides = dict(plateau or {})
    overrides.update(spike or {})
    nx, ny = shape
    rows = []
    for ix in range(nx):
        for iy in range(ny):
            metrics = dict(base)
            metrics.update(overrides.get((ix, iy), {}))
            rows.append({"x": ix, "y": iy, **metrics})
    param_ranges = {"x": list(range(nx)), "y": list(range(ny))}
    return rows, param_ranges


def _plateau_cells(x0, x1, y0, y1, overrides):
    return {(ix, iy): overrides for ix in range(x0, x1 + 1) for iy in range(y0, y1 + 1)}


GOOD = {"sharpe_ratio": 2.0, "profit_factor": 2.0, "drawdown_pct": 10.0,
        "num_trades": 100, "profit": 30.0}


def test_detects_broad_plateau_not_spike():
    grid, ranges = _grid(
        plateau=_plateau_cells(3, 6, 3, 6, GOOD),
        spike={(0, 0): {"sharpe_ratio": 9.0, "profit_factor": 3.0,
                        "drawdown_pct": 2.0, "num_trades": 200, "profit": 90.0}},
    )
    result = plateaus.detect_plateaus(grid, ranges)
    assert result["region_count"] >= 1
    best = result["best_region"]
    assert best["area"] >= 9
    rep = best["representative"]
    assert 3 <= rep["x"] <= 6 and 3 <= rep["y"] <= 6
    assert (0, 0) not in {(c["x"], c["y"]) for c in best["cells"]}


def test_min_trades_filter_removes_cells():
    grid, ranges = _grid(plateau=_plateau_cells(3, 6, 3, 6, GOOD))
    # starve the plateau of trades -> nothing valid there
    for row in grid:
        if 3 <= row["x"] <= 6 and 3 <= row["y"] <= 6:
            row["num_trades"] = 1
    result = plateaus.detect_plateaus(grid, ranges, {"min_trades": 10})
    for region in result["regions"]:
        for cell in region["cells"]:
            assert cell["num_trades"] >= 10


def test_threshold_is_reported_and_masks_cells():
    grid, ranges = _grid(plateau=_plateau_cells(3, 6, 3, 6, GOOD))
    result = plateaus.detect_plateaus(grid, ranges)
    assert isinstance(result["threshold"], float)
    assert result["mask_cells"] > 0


def test_largest_rectangle_finds_block():
    mask = np.zeros((5, 6), dtype=bool)
    mask[1:3, 2:5] = True  # 2x3
    rect = plateaus.largest_rectangle(mask)
    assert rect["area"] == 6
    assert rect["x0"] == 2 and rect["x1"] == 4
    assert rect["y0"] == 1 and rect["y1"] == 2


def test_largest_rectangle_empty():
    assert plateaus.largest_rectangle(np.zeros((3, 3), dtype=bool)) is None


def test_stability_prefers_flat_plateau_over_spike():
    flat = np.full((9, 9), 2.0)
    flat[4, 4] = 2.0
    spiky = np.full((9, 9), -1.0)
    spiky[4, 4] = 10.0
    flat_stability = plateaus.stability_map(flat, 3, 0.8)
    spike_stability = plateaus.stability_map(spiky, 3, 0.8)
    assert flat_stability[4, 4] > spike_stability[4, 4]


def test_representative_methods_stay_in_region():
    grid, ranges = _grid(plateau=_plateau_cells(2, 6, 2, 6, GOOD))
    base = plateaus.detect_plateaus(grid, ranges)
    region_id = base["best_region"]["region_id"]
    for method in ("medoid", "maximin", "center_of_mass", "rectangle_center", "best"):
        result = plateaus.detect_plateaus(
            grid, ranges, {"representative": method, "min_area": 1})
        region = next(r for r in result["regions"] if r["region_id"] == region_id)
        rep = region["representative"]
        assert 2 <= rep["x"] <= 6 and 2 <= rep["y"] <= 6
        assert rep["method"] == method


def test_output_is_json_safe():
    grid, ranges = _grid(plateau=_plateau_cells(3, 6, 3, 6, GOOD),
                         spike={(0, 0): {"sharpe_ratio": 9.0}})
    result = plateaus.detect_plateaus(grid, ranges)
    encoded = json.dumps(result)  # must not raise on NaN/inf
    assert "regions" in json.loads(encoded)


def test_empty_grid_yields_no_regions():
    result = plateaus.detect_plateaus([], {"x": [1, 2], "y": [1, 2]})
    assert result["region_count"] == 0
    assert result["best_region"] is None
    assert result["representative"] is None


def test_requires_two_axes():
    with pytest.raises(ValueError):
        plateaus.detect_plateaus([], {"x": [1, 2]})


def test_custom_weights_change_ranking():
    grid, ranges = _grid(plateau=_plateau_cells(3, 6, 3, 6, GOOD))
    sharpe_only = plateaus.detect_plateaus(
        grid, ranges, {"weights": {"sharpe_ratio": 1.0}})
    assert sharpe_only["score"]["metrics_used"] == ["sharpe_ratio"]
    assert "profit_factor" not in sharpe_only["score"]["metrics_used"]


def test_plot_overlay_writes_png(tmp_path):
    grid, ranges = _grid(plateau=_plateau_cells(3, 6, 3, 6, GOOD))
    out = tmp_path / "overlay.png"
    path = plateaus.plot_overlay(grid, ranges, path=str(out))
    assert out.exists() and out.stat().st_size > 0
    assert path == str(out)

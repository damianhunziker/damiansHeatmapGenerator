# Algorithmic plateau detection

The heatmap owns the underlying **data matrix** (one row of metrics per
parameter combination).  Plateaus are therefore found numerically — a vision
model is never asked to read numbers off a rendered image.  A *plateau* is a
connected area of high score that barely degrades when a parameter is nudged;
it is the robust thing to trade, unlike a single lucky peak.

```
grid -> composite score matrix -> smoothing -> local stability
     -> threshold -> morphology -> connected regions
     -> region scoring -> representative cell
```

Implementation: `core/plateaus.py` (numpy, `scipy.ndimage`, `scikit-image`).
The overlay renderer uses matplotlib.  No image/OCR step is involved.

## Where it runs

- **Automatically** in every `run_heatmap` / `test.py heatmap --api` call: the
  response gains a `plateaus` block (and a JSON `sidecar` next to the HTML with
  the grid + plateaus, so later tool calls do not need to re-run the backtest).
- **On demand** via the `plateaus` tool (API + MCP `analyze_plateaus`), which
  works on an inline `grid`, a `grid_path` sidecar, or a fresh heatmap.
- **Validation** via `validate_plateau` (MCP `validate_plateau`).
- **Presentation** via `plot_plateau` (MCP `plot_plateau`).

## 1. Composite score

Each metric is robustly z-scored (median/MAD, falling back to std for constant
input), then combined with weights.  Signs encode the direction of "good":

```
score = 1.0*z(sharpe_ratio) + 0.3*z(profit_factor)
      - 0.7*z(drawdown_pct) - 0.5*z(num_trades)      # default
```

`num_trades` is a turnover/cost proxy.  A hard filter drops cells with
`num_trades < min_trades` (default `10`) before any region is formed.  Override
the weights per call — the analysis is cheap, so re-ranking never needs a
re-run:

```json
{"weights": {"sharpe_ratio": 1.0, "drawdown_pct": -0.8, "num_trades": -0.4}}
```

## 2. Smoothing, stability, mask

- **Smoothing** (`smooth_size`, default 3; `smooth_kind`: `median`|`mean`|`gaussian`)
  removes single-cell noise.  Masked (invalid) cells stay masked.
- **Local stability**: `stability = local_mean - std_penalty * local_std` over a
  `stability_size` window.  High mean with low spread wins.
- **Threshold**: explicit `threshold`, or auto `mean + threshold_k * std` of the
  stability map (`threshold_k` default 1.0).  Optional `min_score` floor.
- **Morphology**: binary closing, optional opening (`opening_size`, default 0 —
  a full 3×3 opening erodes small genuine plateaus to nothing), then removal of
  specks smaller than `min_area` (default 4).

## 3. Regions and representative cells

Connected components (`scipy.ndimage.label`, 8-connectivity) become candidate
plateaus.  Each region reports:

| Field | Meaning |
|-------|---------|
| `area` | number of cells |
| `x` / `y` | `{param, min, max}` parameter span |
| `score` | `mean`, `median`, `min`, `max`, `p10`, `std` of the cell score |
| `cv` | coefficient of variation (`std / |mean|`) — lower is more robust |
| `avg_trades` / `mean_max_dd` | average trades and mean max drawdown |
| `boundary_penalty` / `touches_edge` | closeness to the search-space edge |
| `region_score` | ranking score (below) |
| `representative` | the cell to actually use |
| `cells` | every cell (x, y, score, trades, profit) |

Ranking:

```
region_score = 1.0*mean_score + 0.5*log(area) - 1.0*std_score
             - 0.7*|mean_max_dd|/100 + 0.8*consistency
             - 0.5*boundary_penalty - 0.3*duration_penalty
```

Representative methods (`representative`):

| Method | Picks |
|--------|-------|
| `medoid` (default) | cell closest to the region mean — the robust centre |
| `maximin` | cell that maximises its worst neighbour |
| `center_of_mass` | cell nearest the region centroid |
| `rectangle_center` | centre of the region's bounding box |
| `best` | the single highest-scoring cell (least robust) |

`largest_rectangle` additionally returns the best axis-aligned rectangular
parameter block in the mask.

## 4. Validation (walk-forward)

`validate_plateau` re-runs the heatmap on `windows` consecutive time slices,
restricted to the region's parameter range, and reports the representative cell
per window plus a consistency summary.  A genuine plateau keeps producing a
region; an overfit spike does not.

```json
{
  "region": {"a": {"min": 10, "max": 14}, "b": {"min": 40, "max": 55}},
  "windows": 3, "summary": {"consistency": 1.0, "representative_std": {"x": 0.5, "y": 1.2}}
}
```

`region` accepts a `param_ranges` spec or the `best_region` block returned by
`analyze_plateaus` (the `x`/`y` `{param,min,max}` form is converted back).

## 5. Tools

| API tool | MCP tool | Purpose |
|----------|----------|---------|
| `plateaus` | `analyze_plateaus` | detect/rank regions on a grid |
| `validate_plateau` | `validate_plateau` | walk-forward validation |
| `plot_plateau` | `plot_plateau` | score matrix + outlines PNG |

All three also work from the human CLI without `--api`:

```bash
python test.py plateaus --grid_path=html_cache/ETHUSDT_DMXStrategy_4h_....json
python test.py plateaus --api \
  '--plateau_config={"min_trades":30,"threshold_k":0.5,"representative":"maximin"}'
```

## 6. Pitfalls

- **Multiple testing**: 1000 cells produce good peaks by chance.  Plateaus
  reduce this; validation is still mandatory.
- **Overfitting**: the global maximum is usually unstable — prefer the robust
  centre (`medoid`).
- **Costs/slippage**: a score without fees is worthless (`fee_pct` is already in
  the metrics; `num_trades` penalises churn).
- **Regime changes**: always run `validate_plateau` across windows.
- **Edge plateaus**: a region touching the border (`touches_edge`) means the
  search space should be widened.
- **Non-linear parameters**: periods belong on a log scale; sweep them so.
- **Never read numbers from an image**: if you only have a picture, reconstruct
  the matrix first (colormap + colorbar OCR + grid detection), then run the same
  algorithmic detection.  That fallback is intentionally **not** implemented
  here and would add opencv/OCR dependencies.

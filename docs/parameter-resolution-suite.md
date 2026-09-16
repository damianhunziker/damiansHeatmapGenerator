# Parameter Resolution Suite

Generic, config-driven resolution of **virtual** parameters into **concrete**
strategy parameters.  A virtual parameter (e.g. `kama_normalized_length`) is
swept on a heatmap axis; a declarative resolver splits it into the real
parameters the strategy consumes (e.g. `entry_kama_length`,
`entry_kama_fast`, `entry_kama_slow`).  It works for **any** parameter, not just
KAMA, and is extended through **hooks**, **middlewares** and **snippets**.

This supersedes the KAMA-specific `derived_params` feature, which remains
supported as a special case.

---

## 1. Motivation

Before this feature, scaling KAMA to a single heatmap axis was hard-coded:

```python
# core/kama_scaling.py
{"entry_kama_length": {"scale": "kama_scale", "base": 16, "kind": "window"}}
```

Problems:

* KAMA-only (`kind` is `window`/`alpha`).
* One swept value drives exactly one formula per target.
* No way to express arbitrary parameter relationships (risk profiles, BB
  lengths, position sizing, ...).
* Only wired into `heatmap` / `automator`.

The suite generalises this: **a virtual parameter resolves into any number of
concrete parameters via whitelisted, declarative operations**, configured in
JSON and/or inline.

---

## 2. Concepts

| Concept | Meaning |
|---|---|
| **Virtual parameter** | The swept axis name (e.g. `kama_normalized_length`). |
| **Concrete parameter** | A real strategy constructor argument. |
| **Resolver** | `{id, input, when, outputs}`; maps one virtual param to concrete params. |
| **Op-tree** | A declarative formula built from whitelisted ops (no `eval`). |
| **Snippet** | A named, reusable op-tree (config) or a registered op (code). |
| **Hook** | A callback at a pipeline stage (`pre_resolve`, `post_resolve`). |
| **Middleware** | Wraps the resolve step (`mw(next, params, ctx)`). |
| **Context (`ctx`)** | Per-run scratch space (`strategy`, `records`). |

---

## 3. Module layout

```
core/params/
  __init__.py     # public API
  ops.py          # op catalogue, eval_node, validate_node, register_op
  config.py       # JSON load/merge/validate
  suite.py        # ResolverSuite, hooks, middlewares, resolve_params, warmup
configs/
  param_resolvers.json   # default resolver config
```

Public API:

```python
from core.params import (
    load_resolver_config, resolve_params, resolver_max_period,
    hook, middleware, register_op, describe_resolvers,
)
```

---

## 4. Config schema (JSON)

```jsonc
{
  "version": 1,
  "virtual_params": {
    "<virtual param>": {
      "description": "...",                                  // optional
      "range": {"min": 1, "max": 8, "step": 1}               // optional sweep default
    }
  },
  "snippets": {
    "<name>": <op-tree>            // reusable formula
  },
  "resolvers": [
    {
      "id": "<unique id>",
      "input": "<swept virtual parameter>",
      "when": {"strategy": "<Name>" | ["<Name>", "..."]},   // optional
      "warmup_targets": ["<concrete period param>", "..."], // optional
      "range": {"min": 1, "max": 8, "step": 1},             // optional (input default)
      "outputs": {
        "<concrete parameter>": <op-tree>
      }
    }
  ]
}
```

* `when` scopes a resolver to a strategy class name (`strategy_class.__name__`).
* `warmup_targets` names the outputs that are *periods* (used for the KAMA
  warmup warning).  If omitted, names containing `length` / `period` / `window`
  are used.
* `range` (top-level `virtual_params` or on a resolver) declares a **sweep
  default** for a virtual parameter: `{"min","max","step"}`, `{"min","max","count"}`,
  `{"values": [...]}` or a plain list.  A resolver-level `range` applies to its
  `input`; a top-level `virtual_params` entry wins.
* A resolver only fires when its `input` is present in the parameter dict
  (i.e. it was swept or fixed).
* The name you sweep is the resolver's **`input`** (reported as `sweep` in
  `data.resolvers`).  The resolver **`id`** is also accepted as an **alias**, so
  sweeping either `kama_normalized_length` (input) or
  `slow_kama_normalized_length` (DMX resolver id) fires the same resolver.

### Sweeping a virtual parameter

Three equivalent ways to feed the virtual axis with `min/max/step`:

```bash
# 1) explicit in param_ranges (works like any other parameter)
'--param_ranges={"kama_normalized_length":{"min":1,"max":8,"step":1},"entry_filter":{"min":0.5,"max":1.5,"step":0.25}}'

# 2) use the range declared in the resolver config ("config" token)
'--param_ranges={"kama_normalized_length":"config","entry_filter":{"min":0.5,"max":1.5,"step":0.25}}'

# 3) auto-inject configured ranges for virtual params not swept explicitly
--param_ranges='{"entry_filter":{"min":0.5,"max":1.5,"step":0.25}}' --use_config_ranges=true
```

`use_config_ranges` only injects virtual params whose resolver applies to the
running strategy, so unrelated strategies are unaffected.

### Clean conversion (aliases & validation)

At the API/MCP boundary (`prepare_heatmap_params`) the grid is normalised before
it runs:

* A swept **resolver `id`** is rewritten to its canonical virtual **`input`** —
  both in `param_ranges` and in `derived_params` (`target` names and `scale`
  keys).  Sweeping `slow_kama_normalized_length` (DMX resolver id) is therefore
  converted to `kama_normalized_length`; the mapping is reported in
  `data.param_aliases`.
* Sweeping both an alias and its input is rejected (`... is an alias of ...`).
* A swept **virtual parameter that no resolver resolves for the running
  strategy** raises a clear error instead of silently producing identical grid
  cells:

  ```
  ValueError: 'kama_normalized_length' is a virtual parameter, but no resolver
  applies to strategy 'SMACrossoverStrategy'. Add a resolver with a matching
  'when', or use a strategy that has one.
  ```



### Op catalogue (whitelist, no code execution)

| Kind | Node | Notes |
|---|---|---|
| literal | `5`, `"x"` | constant |
| param ref | `{"$param": "risk_pct"}` | value of a parameter in scope |
| swept input | `{"$input": true}` | the resolver's `input` value |
| snippet | `{"$snippet": "name", "args": {...}}` | reusable formula |
| arithmetic | `{"op":"add","args":[a,b]}`, `sub`, `mul`, `div`, `pow`, `neg` | |
| bounds | `round`, `floor`, `ceil`, `abs`, `min`, `max`, `clamp` | |
| scaling | `scale_window(base,k)`, `scale_alpha(base,k)`, `scale_naive(base,k)` | |

`scale_window` / `scale_alpha` implement the **all_exact** rule
(`L'=k·L`, `P'=k·(P+1)−1`); `scale_naive` is `L'≈k·L`.

### Example — KAMA "all_exact" via one virtual axis

```json
{
  "version": 1,
  "virtual_params": {
    "kama_normalized_length": {"range": {"min": 1, "max": 8, "step": 1}}
  },
  "snippets": {
    "all_exact_window": {"op": "scale_window", "base": {"$param": "base"}, "k": {"$param": "k"}},
    "all_exact_alpha":  {"op": "scale_alpha",  "base": {"$param": "base"}, "k": {"$param": "k"}}
  },
  "resolvers": [
    {
      "id": "kama_normalized_length",
      "when": {"strategy": ["LiveKAMASSLStrategy"]},
      "input": "kama_normalized_length",
      "warmup_targets": ["entry_kama_slow", "kama2_slow", "exit_kama_slow"],
      "outputs": {
        "entry_kama_length": {"$snippet": "all_exact_window", "args": {"base": 16, "k": {"$input": true}}},
        "entry_kama_fast":   {"$snippet": "all_exact_alpha",  "args": {"base": 4,  "k": {"$input": true}}},
        "entry_kama_slow":   {"$snippet": "all_exact_alpha",  "args": {"base": 24, "k": {"$input": true}}}
      }
    }
  ]
}
```

Resolved values for `kama_normalized_length = k`:

| k | entry length | entry fast | entry slow |
|---|---|---|---|
| 1 | 16 | 4 | 24 |
| 2 | 32 | 9 | 49 |
| 3 | 48 | 14 | 74 |
| 4 | 64 | 19 | 99 |

### Example — generic (not KAMA)

```json
{
  "resolvers": [{
    "id": "risk_profile",
    "input": "risk_pct",
    "outputs": {
      "stop_loss_mult_long": {"op": "sub", "args": [1.0, {"op": "div", "args": [{"$input": true}, 100]}]},
      "position_size_value": {"op": "div", "args": [{"$param": "risk_pct"}, 100]}
    }
  }]
}
```

---

## 5. Hooks, middlewares, snippets

```python
from core.params import hook, middleware, register_op

@hook("pre_resolve", priority=-10)          # lower priority runs first
def inject_defaults(params, ctx):
    return params or None                   # None -> keep params

@hook("post_resolve", priority=0)
def audit(params, ctx):
    ctx["records"].append({"id": "audit", "resolved": {"n": len(params)}})

@middleware(priority=0)                      # wraps the resolve step
def cache(next_, params, ctx):
    return next_(params, ctx)

@register_op("scale_custom")                 # a code "snippet" usable from config
def scale_custom(base, k):
    return max(1, round(base * k ** 0.75))
```

Pipeline order:

```
pre_resolve hooks -> [middlewares (resolve)] -> post_resolve hooks
```

The resolve step is stateless and config is plain data, so the normalized
resolver list is picklable and safe to pass into `multiprocessing` workers.

---

## 6. Data flow into the heatmap

```
param_ranges {"kama_normalized_length": [1,2,4], "squeeze_limit": [...]}
        │
        ▼  itertools.product
  grid cell = {kama_normalized_length: 2, squeeze_limit: 150, initial_equity, fee_pct}
        │
        ▼  resolve_params(cell, resolvers, strategy=<class name>, snippets)
  resolved  = {entry_kama_length: 32, entry_kama_fast: 9, entry_kama_slow: 49, ...}
        │
        ▼  instantiate_strategy(strategy_class, resolved)
  strategy execution
```

* The heatmap axes stay the **virtual** parameters (readable labels).
* The **concrete** values are recorded per resolver and returned in the API
  envelope (`data.resolvers`) and per-cell `resolved` records.
* Warmup: `resolver_max_period` evaluates the period outputs across the swept
  values; if the available history is too short a `warnings` entry is added.

---

## 7. Usage

### CLI / `--api` (heatmap)

```bash
python test.py heatmap --asset=BTCUSDT --strategy=LiveKAMASSLStrategy \
  --start_date=2024-01-01 --end_date=2024-06-01 --workers=1 \
  '--param_ranges={"kama_normalized_length":[1,2,3,4,6,8],"entry_filter":{"min":0.5,"max":1.5,"step":0.25}}' \
  --api
```

The default config (`configs/param_resolvers.json`) is loaded automatically.
Inline resolvers override by `id`:

```bash
python test.py heatmap --asset=BTCUSDT --strategy=LiveKAMASSLStrategy \
  --param_ranges='{"kama_normalized_length":[1,2,4],"squeeze_limit":[150]}' \
  '--resolvers={"resolvers":[{"id":"kama_normalized_length","input":"kama_normalized_length","outputs":{"entry_kama_length":{"op":"scale_window","base":16,"k":{"$input":true}}}}]}' \
  --api
```

`--resolver_config=/path/to/config.json` loads a different config file.

### MCP

The MCP tools `run_heatmap` and `run_automator` accept the same `param_ranges`,
plus `resolvers`, `resolver_config` and `use_config_ranges`.  The default config
is loaded automatically, so `kama_normalized_length` can be swept directly:

```python
run_heatmap(
    asset="BTCUSDT", strategy="LiveKAMASSLStrategy",
    start_date="2024-01-01", end_date="2024-06-01",
    param_ranges={
        "kama_normalized_length": {"min": 1, "max": 8, "step": 1},
        "entry_filter": {"min": 0.5, "max": 1.5, "step": 0.25},
    },
)
# -> data.param_ranges shows kama_normalized_length, data.resolvers lists the
#    applied resolvers and their resolved targets.

# use the range declared in the config:
run_heatmap(..., param_ranges={"kama_normalized_length": "config", "entry_filter": {...}})
# or auto-inject configured virtual ranges:
run_heatmap(..., param_ranges={"entry_filter": {...}}, use_config_ranges=True)
# override the config inline / point at another file:
run_heatmap(..., resolvers={"resolvers": [...]}, resolver_config="my.json")
```

Sweeping the resolver **id** also works (`slow_kama_normalized_length` for DMX),
because the id is an alias for the virtual `input` — but prefer the `sweep`
name returned by `get_schema` / `data.resolvers`.

`get_schema` (MCP) also returns a `resolvers` section listing the default
virtual parameters, their ranges and their resolved targets, so agents can
discover `kama_normalized_length` without reading the config file.

---

## 8. Backward compatibility

* `derived_params` still works unchanged (KAMA `window`/`alpha`) and is applied
  after the resolvers.
* `expand_derived` / `normalize_derived` in `core/kama_scaling.py` are untouched.
* Existing heatmaps without virtual parameters behave exactly as before.

---

## 9. Integration points

| File | Change |
|---|---|
| `core/params/*` | new suite (ops, config, hooks, pipeline) |
| `configs/param_resolvers.json` | default resolvers (KAMA all_exact) |
| `heatmap.py:15` `analyze_strategy` | applies resolvers before `expand_derived` |
| `heatmap.py:218` `create_heatmap` | threads `resolvers`/`resolver_snippets`, warmup |
| `core/api.py:280` `_run_heatmap` | loads config, returns `data.resolvers` |
| `core/api.py:329` `_run_automator` | forwards `resolvers`/`resolver_config` |
| `automator.py:83` | loads config per run |
| `mcp_server/server.py` | `run_heatmap` / `run_automator` gain `resolvers`, `resolver_config` |
| `core/api.py` `_CONTROL_KEYS`, `get_schema` | expose the new params |

---

## 10. Testing

`tests/test_param_resolvers.py` covers:

* op evaluation, arithmetic, bounds, snippets, `$param`/`$input`;
* `validate_node` rejects unknown ops/snippets;
* config validation (duplicate id, empty outputs, bad version);
* default config loads; inline overrides by id;
* strategy-scoped resolution (LiveKAMA vs DMX vs unrelated);
* no-op when the input is absent; resolution records;
* hooks + middlewares run; `resolver_max_period`.

```bash
docker compose exec app python -m pytest tests/test_param_resolvers.py -q
```

---

## 11. Rollout

* **Phase 1 (done):** suite + config + heatmap/automator + API/MCP.
* **Phase 2:** apply resolvers in `pnl`, `chart_analysis`, `series` (so a single
  virtual parameter can be used outside heatmaps).
* **Phase 3:** per-strategy config files, config discovery/validation command,
  richer ops (piecewise, lookup tables) if needed.

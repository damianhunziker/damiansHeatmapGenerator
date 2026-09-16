"""Load, merge and validate resolver configs (JSON only, no extra deps).

A config is::

    {
      "version": 1,
      "snippets": {"<name>": <op-tree>, ...},          # reusable formulas
      "resolvers": [
        {
          "id": "<unique id>",
          "input": "<swept virtual parameter>",
          "when": {"strategy": "<Name>" | ["<Name>", ...]},   # optional
          "warmup_targets": ["entry_kama_slow", ...],         # optional
          "outputs": {"<concrete param>": <op-tree>, ...}
        }
      ]
    }

``load_resolver_config`` merges an optional default file with inline config
(from the API / MCP) and returns the normalized structure.
"""

from __future__ import annotations

import json
import os

from core.params.ops import ResolutionError, validate_node

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "param_resolvers.json")

SUPPORTED_VERSIONS = (1,)


def _validate_range(spec, name):
    """Validate a sweep spec: list, ``{"values": [...]}`` or ``{"min","max",...}``."""
    if isinstance(spec, (list, tuple)):
        if not spec:
            raise ResolutionError(f"range for '{name}' must not be empty")
        return list(spec)
    if isinstance(spec, dict):
        if "values" in spec:
            values = list(spec["values"])
            if not values:
                raise ResolutionError(f"range for '{name}' has empty 'values'")
            return {"values": values}
        start = spec.get("min", spec.get("start"))
        stop = spec.get("max", spec.get("stop"))
        if start is None or stop is None:
            raise ResolutionError(f"range for '{name}' needs 'min' and 'max'")
        out = {"min": start, "max": stop}
        if spec.get("step") is not None:
            out["step"] = spec["step"]
        if spec.get("count") is not None:
            out["count"] = spec["count"]
        return out
    raise ResolutionError(
        f"range for '{name}' must be a list or an object with min/max/step/count")


def _normalize_virtual_params(raw):
    raw = raw or {}
    if not isinstance(raw, dict):
        raise ResolutionError("'virtual_params' must be an object")
    out = {}
    for name, meta in raw.items():
        if isinstance(meta, (list, tuple)):
            meta = {"range": list(meta)}
        if not isinstance(meta, dict):
            raise ResolutionError(f"virtual_params['{name}'] must be an object or list")
        entry = {}
        if "range" in meta:
            entry["range"] = _validate_range(meta["range"], name)
        if "description" in meta:
            entry["description"] = str(meta["description"])
        out[name] = entry
    return out



def _read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        raise ResolutionError(f"resolver config not found: {path}")
    except json.JSONDecodeError as exc:
        raise ResolutionError(f"resolver config is not valid JSON ({path}): {exc}")


def _as_resolver_list(inline):
    if inline is None:
        return []
    if isinstance(inline, dict):
        return list(inline.get("resolvers") or [])
    if isinstance(inline, list):
        return list(inline)
    raise ResolutionError("'resolvers' must be a list or an object with 'resolvers'")


def _merge_resolvers(base, extra):
    """Merge by id: an inline resolver with the same id replaces the base one."""
    by_id = {r.get("id"): r for r in base if isinstance(r, dict)}
    order = [r.get("id") for r in base if isinstance(r, dict)]
    for resolver in extra:
        rid = resolver.get("id") if isinstance(resolver, dict) else None
        if rid in by_id:
            by_id[rid] = resolver
        else:
            by_id[rid] = resolver
            order.append(rid)
    return [by_id[rid] for rid in order if rid in by_id]


def normalize_config(config):
    """Validate and normalize a raw config dict.  Raises ResolutionError."""
    config = dict(config or {})
    version = int(config.get("version", 1))
    if version not in SUPPORTED_VERSIONS:
        raise ResolutionError(
            f"unsupported resolver config version {version}; supported: {SUPPORTED_VERSIONS}")

    snippets = dict(config.get("snippets") or {})
    for name, node in snippets.items():
        validate_node(node, snippets, path=f"snippets.{name}")

    virtual_params = _normalize_virtual_params(config.get("virtual_params"))

    resolvers = []
    seen_ids = set()
    for index, raw in enumerate(config.get("resolvers") or []):
        if not isinstance(raw, dict):
            raise ResolutionError(f"resolvers[{index}] must be an object")
        rid = raw.get("id")
        inp = raw.get("input")
        outputs = raw.get("outputs")
        if not rid or not isinstance(rid, str):
            raise ResolutionError(f"resolvers[{index}] needs a string 'id'")
        if rid in seen_ids:
            raise ResolutionError(f"duplicate resolver id '{rid}'")
        seen_ids.add(rid)
        if not inp or not isinstance(inp, str):
            raise ResolutionError(f"resolver '{rid}' needs a string 'input'")
        if not isinstance(outputs, dict) or not outputs:
            raise ResolutionError(f"resolver '{rid}' needs a non-empty 'outputs' object")

        for target, node in outputs.items():
            validate_node(node, snippets, path=f"resolver '{rid}'.outputs.{target}")

        when = raw.get("when") or {}
        if when and not isinstance(when, dict):
            raise ResolutionError(f"resolver '{rid}'.when must be an object")

        resolver_range = None
        if raw.get("range") is not None:
            resolver_range = _validate_range(raw["range"], inp)

        resolvers.append({
            "id": rid,
            "input": inp,
            "when": when,
            "warmup_targets": list(raw.get("warmup_targets") or []),
            "outputs": outputs,
            "range": resolver_range,
        })

    # A resolver-level range is a default for its swept input; top-level
    # virtual_params take precedence.
    for resolver in resolvers:
        if resolver.get("range") is not None:
            entry = virtual_params.setdefault(resolver["input"], {})
            entry.setdefault("range", resolver["range"])

    return {
        "version": version,
        "snippets": snippets,
        "resolvers": resolvers,
        "virtual_params": virtual_params,
    }


def load_resolver_config(path=None, inline=None, use_default=True):
    """Load the default config (unless ``path`` is given) and merge inline config."""
    if path:
        base = _read_json(path)
    elif use_default and os.path.exists(DEFAULT_CONFIG_PATH):
        base = _read_json(DEFAULT_CONFIG_PATH)
    else:
        base = {}

    inline = inline or {}
    inline_resolvers = _as_resolver_list(inline)
    inline_snippets = dict(inline.get("snippets") or {}) if isinstance(inline, dict) else {}
    inline_virtual = dict(inline.get("virtual_params") or {}) if isinstance(inline, dict) else {}

    merged = {
        "version": base.get("version", 1),
        "snippets": {**(base.get("snippets") or {}), **inline_snippets},
        "resolvers": _merge_resolvers(list(base.get("resolvers") or []), inline_resolvers),
        "virtual_params": {**(base.get("virtual_params") or {}), **inline_virtual},
    }
    return normalize_config(merged)

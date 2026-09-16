"""Resolver suite: hooks, middlewares and the resolve pipeline.

The suite turns *virtual* parameters (swept on a heatmap axis) into *concrete*
strategy parameters via declarative resolver configs (see ``core.params.config``).

Extension points
----------------
* ``@hook(stage, priority)``      - run at a pipeline stage
      - ``pre_resolve``  (params, ctx) -> params|None
      - ``post_resolve`` (params, ctx) -> params|None
* ``@middleware(priority)``       - wrap the resolve step: ``mw(next, params, ctx)``
* ``register_op(name, fn)``       - add a custom op ("snippet") for configs

The pipeline is stateless and config is plain data, so it is safe to pass the
normalized resolver list into ``multiprocessing`` workers.
"""

from __future__ import annotations

from core.params.ops import ResolutionError, eval_node, register_op  # noqa: F401 (re-export)

_HOOKS = {}
_MIDDLEWARES = []


def hook(stage, priority=0):
    """Register a pipeline hook.  Lower priority runs first."""
    def decorator(fn):
        _HOOKS.setdefault(stage, []).append((priority, fn))
        _HOOKS[stage].sort(key=lambda item: item[0])
        return fn
    return decorator


def middleware(priority=0):
    """Register a middleware wrapping the resolve step."""
    def decorator(fn):
        _MIDDLEWARES.append((priority, fn))
        _MIDDLEWARES.sort(key=lambda item: item[0])
        return fn
    return decorator


def _when_matches(when, strategy):
    if not when:
        return True
    names = when.get("strategy")
    if names is None:
        return True
    if isinstance(names, str):
        names = [names]
    return strategy in names


def resolver_applies(resolver, strategy):
    """Public: does ``resolver`` apply to the given strategy class name?"""
    return _when_matches((resolver or {}).get("when"), strategy)


class ResolverSuite:
    """Applies a normalized resolver list to a parameter dict."""

    def __init__(self, resolvers=None, snippets=None):
        self.resolvers = list(resolvers or [])
        self.snippets = dict(snippets or {})

    @staticmethod
    def sweep_key(resolver, params):
        """The key a resolver is driven by: its ``input`` or (alias) its ``id``."""
        if resolver["input"] in params:
            return resolver["input"]
        rid = resolver.get("id")
        if rid and rid in params:
            return rid
        return None

    def applicable(self, params, strategy=None):
        for resolver in self.resolvers:
            if self.sweep_key(resolver, params) is None:
                continue
            if not _when_matches(resolver.get("when"), strategy):
                continue
            yield resolver

    def resolve(self, params, strategy=None):
        """Return ``(resolved_params, records)``."""
        out = dict(params)
        records = []
        for resolver in self.applicable(out, strategy):
            key = self.sweep_key(resolver, out)
            produced = {}
            for target, node in resolver["outputs"].items():
                produced[target] = eval_node(node, out, key, self.snippets)
            out.update(produced)
            records.append({
                "id": resolver["id"],
                "input": resolver["input"],
                "swept": key,
                "value": out.get(key),
                "resolved": produced,
            })
        return out, records


def _bind(mw, nxt):
    def wrapped(params, ctx):
        return mw(nxt, params, ctx)
    return wrapped


def resolve_params(params, resolvers, strategy=None, snippets=None, records=None):
    """Run the full pipeline (hooks -> middlewares -> resolve -> hooks).

    ``records`` (a list) is extended with one entry per applied resolver so
    callers can surface how a virtual parameter was split into concrete ones.
    """
    suite = ResolverSuite(resolvers, snippets)
    ctx = {"strategy": strategy, "records": []}

    for _, fn in _HOOKS.get("pre_resolve", []):
        params = fn(params, ctx) or params

    def core(current, context):
        out, applied = suite.resolve(current, strategy)
        context["records"].extend(applied)
        return out

    chain = core
    for _, mw in reversed(_MIDDLEWARES):
        chain = _bind(mw, chain)
    params = chain(params, ctx)

    for _, fn in _HOOKS.get("post_resolve", []):
        params = fn(params, ctx) or params

    if records is not None:
        records.extend(ctx["records"])
    return params


def _is_period_name(name):
    lowered = name.lower()
    return "length" in lowered or "period" in lowered or "window" in lowered


def resolver_max_period(param_ranges, resolvers, name_filter=None, snippets=None):
    """Largest derived period across the swept input values (for warmup checks)."""
    snippets = dict(snippets or {})
    base_scope = {k: (v[0] if len(v) else 0) for k, v in (param_ranges or {}).items()}
    maximum = 0
    for resolver in resolvers or []:
        key = resolver["input"]
        if key not in (param_ranges or {}):
            rid = resolver.get("id")
            if rid and rid in (param_ranges or {}):
                key = rid
            else:
                continue
        values = (param_ranges or {}).get(key)
        if values is None:
            continue
        targets = resolver.get("warmup_targets") or [
            t for t in resolver["outputs"] if _is_period_name(t)
        ]
        if name_filter is not None:
            # an explicit filter is authoritative over warmup_targets
            targets = [t for t in resolver["outputs"] if name_filter(t)]
        if not targets:
            continue
        for value in values:
            scope = dict(base_scope)
            scope[key] = value
            for target in targets:
                node = resolver["outputs"].get(target)
                if node is None:
                    continue
                try:
                    resolved = eval_node(node, scope, key, snippets)
                except ResolutionError:
                    continue
                if isinstance(resolved, (int, float)) and not isinstance(resolved, bool):
                    maximum = max(maximum, int(resolved))
    return maximum


def applicable_virtual_params(virtual_params, resolvers, strategy):
    """Subset of ``virtual_params`` whose input is used by a resolver that
    applies to ``strategy`` (used to inject configured ranges safely)."""
    inputs = {
        r["input"] for r in (resolvers or [])
        if _when_matches(r.get("when"), strategy)
    }
    return {
        name: meta for name, meta in (virtual_params or {}).items()
        if name in inputs
    }


def describe_resolvers(resolvers, snippets=None, virtual_params=None):
    """JSON-friendly description of the active resolvers (for schema output)."""
    return {
        "version": 1,
        "snippets": sorted((snippets or {}).keys()),
        "virtual_params": virtual_params or {},
        "resolvers": [
            {
                "id": r["id"],
                "input": r["input"],
                "sweep": r["input"],
                "sweep_aliases": [r["id"]] if r["id"] != r["input"] else [],
                "when": r.get("when") or {},
                "targets": sorted(r["outputs"].keys()),
                "range": r.get("range"),
            }
            for r in (resolvers or [])
        ],
    }

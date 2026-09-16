"""Generic, config-driven parameter resolution for strategies.

Virtual parameters (e.g. ``kama_normalized_length``) are swept on a heatmap
axis and *resolved* into concrete strategy parameters via declarative op-trees
defined in JSON (``configs/param_resolvers.json``) or inline (API / MCP).

Public API::

    from core.params import load_resolver_config, resolve_params, resolver_max_period

    cfg = load_resolver_config(path=None, inline={"resolvers": [...]})
    resolved = resolve_params(params, cfg["resolvers"], strategy="LiveKAMASSLStrategy",
                              snippets=cfg["snippets"])

See ``docs/parameter-resolution-suite.md`` for the full design.
"""

from core.params.config import (
    DEFAULT_CONFIG_PATH,
    load_resolver_config,
    normalize_config,
)
from core.params.ops import OPS, ResolutionError, register_op
from core.params.suite import (
    ResolverSuite,
    applicable_virtual_params,
    describe_resolvers,
    hook,
    middleware,
    resolve_params,
    resolver_applies,
    resolver_max_period,
)

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "OPS",
    "ResolutionError",
    "ResolverSuite",
    "applicable_virtual_params",
    "describe_resolvers",
    "hook",
    "load_resolver_config",
    "middleware",
    "normalize_config",
    "register_op",
    "resolve_params",
    "resolver_applies",
    "resolver_max_period",
]

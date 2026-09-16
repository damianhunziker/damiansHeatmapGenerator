"""Exact KAMA parameter scaling for the heatmap derived-parameter axis.

KAMA has three bar-based parameters: the Efficiency-Ratio window ``length`` and
the two EMA smoothing bounds ``fast`` / ``slow``.  Scaling KAMA to a ``k``-times
longer horizon means making it behave like the same indicator on ``k``-times
longer bars.  Because the EMA alpha is ``2 / (period + 1)``, the exact rule is:

    length' = k * length
    period' = k * (period + 1) - 1        (for fast / slow)

The naive approximation ``period' ~= k * period`` is only correct for large
periods and visibly distorts the fast bound (e.g. base 4, k=6 -> 24 vs exact 29).

A heatmap axis can therefore sweep a single ``scale`` parameter, while the KAMA
lengths are *derived* from it via this module.
"""

from __future__ import annotations

VALID_KINDS = ("window", "alpha")


def scale_window(base, k):
    """Scale a plain rolling-window length (e.g. the ER length)."""
    return max(1, int(round(float(base) * float(k))))


def scale_alpha_period(base, k):
    """Scale an EMA smoothing period exactly (alpha' = alpha / k)."""
    return max(2, int(round(float(k) * (float(base) + 1.0) - 1.0)))


def expand_derived(strategy_params, derived_specs):
    """Return a copy of ``strategy_params`` with the derived params applied.

    ``derived_specs`` maps a parameter name to a normalized spec
    ``{"scale": <swept param name>, "base": <number>, "kind": "window"|"alpha"}``.
    """
    out = dict(strategy_params)
    for name, spec in (derived_specs or {}).items():
        scale_key = spec["scale"]
        if scale_key not in out:
            continue
        k = out[scale_key]
        if spec["kind"] == "alpha":
            out[name] = scale_alpha_period(spec["base"], k)
        else:
            out[name] = scale_window(spec["base"], k)
    return out


def normalize_derived(derived_params, param_ranges):
    """Validate a ``derived_params`` mapping and return normalized specs.

    Raises ``ValueError`` with a clear message on invalid input so the caller
    surfaces it in the API envelope instead of silently doing nothing.
    """
    specs = {}
    for name, spec in (derived_params or {}).items():
        if not isinstance(spec, dict):
            raise ValueError(
                f"derived_params['{name}'] must be an object like "
                f"{{\"scale\": \"kama_scale\", \"base\": 16, \"kind\": \"window\"}}"
            )
        scale_key = spec.get("scale")
        base = spec.get("base")
        kind = spec.get("kind", "window")
        if scale_key not in (param_ranges or {}):
            raise ValueError(
                f"derived_params['{name}'] scale '{scale_key}' is not a swept parameter "
                f"(available: {list((param_ranges or {}).keys())})"
            )
        if base is None:
            raise ValueError(f"derived_params['{name}'] needs a 'base' value")
        if kind not in VALID_KINDS:
            raise ValueError(
                f"derived_params['{name}'] kind must be one of {VALID_KINDS}, got '{kind}'"
            )
        specs[name] = {"scale": scale_key, "base": float(base), "kind": kind}
    return specs


def max_effective_period(param_ranges, derived_specs, name_filter=None):
    """Largest derived period across the swept scale values (for warmup checks)."""
    maximum = 0
    for name, spec in (derived_specs or {}).items():
        if name_filter and not name_filter(name):
            continue
        values = (param_ranges or {}).get(spec["scale"])
        if values is None:
            continue
        for k in values:
            if spec["kind"] == "alpha":
                value = scale_alpha_period(spec["base"], k)
            else:
                value = scale_window(spec["base"], k)
            maximum = max(maximum, value)
    return maximum

"""Declarative op primitives for generic parameter resolution.

Everything here is **data-driven and side-effect free**: a "formula" is a small
op tree built from the whitelisted ops below.  There is no ``eval`` and no code
execution - only a fixed catalogue of numeric primitives, which keeps resolver
configs safe to accept from the API / MCP.

Node grammar
------------
* literal number / string          -> constant
* ``{"$param": "name"}``           -> value of a parameter in scope
* ``{"$input": true}``             -> value of the resolver's swept input param
* ``{"$snippet": "name", "args": {...}}`` -> evaluate a named snippet
* ``{"op": "<name>", ...}``        -> call an op; positional ``args`` list or
                                      keyword args (other keys / ``args`` dict)
"""

from __future__ import annotations

import math

from core.kama_scaling import scale_window, scale_alpha_period


class ResolutionError(ValueError):
    """Raised for invalid resolver configs or unresolvable expressions."""


def _num(value, where):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResolutionError(f"{where}: expected a number, got {type(value).__name__}")
    return float(value)


OPS = {
    # constants / references handled by eval_node, kept for validation clarity
    "const": lambda value: value,
    # arithmetic
    "add": lambda *args: sum(_num(a, "add") for a in args),
    "sub": lambda a, b: _num(a, "sub") - _num(b, "sub"),
    "mul": lambda *args: float(math.prod(_num(a, "mul") for a in args)),
    "div": lambda a, b: _num(a, "div") / _num(b, "div"),
    "pow": lambda a, b: _num(a, "pow") ** _num(b, "pow"),
    "neg": lambda a: -_num(a, "neg"),
    # rounding / bounds
    "round": lambda value, ndigits=0: round(_num(value, "round"), int(ndigits)),
    "floor": lambda value: float(math.floor(_num(value, "floor"))),
    "ceil": lambda value: float(math.ceil(_num(value, "ceil"))),
    "abs": lambda value: abs(_num(value, "abs")),
    "min": lambda *args: min(_num(a, "min") for a in args),
    "max": lambda *args: max(_num(a, "max") for a in args),
    "clamp": lambda value, lo, hi: max(_num(lo, "clamp"), min(_num(hi, "clamp"), _num(value, "clamp"))),
    # scaling primitives (KAMA "all_exact" is scale_window + scale_alpha)
    "scale_window": lambda base, k: scale_window(base, k),
    "scale_alpha": lambda base, k: scale_alpha_period(base, k),
    "scale_naive": lambda base, k: max(1, int(round(float(base) * float(k)))),
}


def register_op(name, fn):
    """Register a custom op (a Python "snippet") usable from resolver configs."""
    if not name or not callable(fn):
        raise ResolutionError("register_op needs a name and a callable")
    OPS[name] = fn
    return fn


def validate_node(node, snippets, path="node"):
    """Validate an op tree against the whitelist.  Raises ResolutionError."""
    if node is None or isinstance(node, (int, float, str, bool)):
        return
    if isinstance(node, dict):
        if "$param" in node:
            if not isinstance(node["$param"], str):
                raise ResolutionError(f"{path}: $param must be a string")
            return
        if "$input" in node:
            return
        if "$snippet" in node:
            name = node["$snippet"]
            if name not in snippets:
                raise ResolutionError(f"{path}: unknown snippet '{name}'")
            for key, value in (node.get("args") or {}).items():
                validate_node(value, snippets, f"{path}.{key}")
            return
        if "op" in node:
            op = node["op"]
            if op not in OPS:
                raise ResolutionError(f"{path}: unknown op '{op}'")
            args = node.get("args")
            if isinstance(args, list):
                for i, value in enumerate(args):
                    validate_node(value, snippets, f"{path}.args[{i}]")
            elif isinstance(args, dict):
                for key, value in args.items():
                    validate_node(value, snippets, f"{path}.args.{key}")
            elif args is not None:
                raise ResolutionError(f"{path}: 'args' must be a list or object")
            for key, value in node.items():
                if key in ("op", "args"):
                    continue
                validate_node(value, snippets, f"{path}.{key}")
            return
    raise ResolutionError(f"{path}: unsupported node {node!r}")


def eval_node(node, scope, input_name, snippets):
    """Evaluate an op tree against ``scope`` (a dict of parameters)."""
    if node is None or isinstance(node, (int, float, str, bool)):
        return node
    if not isinstance(node, dict):
        raise ResolutionError(f"unsupported node {node!r}")

    if "$param" in node:
        name = node["$param"]
        if name not in scope:
            raise ResolutionError(f"unknown parameter '{name}' referenced in resolver")
        return scope[name]

    if "$input" in node:
        if input_name is None or input_name not in scope:
            raise ResolutionError("$input referenced but no swept input is in scope")
        return scope[input_name]

    if "$snippet" in node:
        name = node["$snippet"]
        if name not in snippets:
            raise ResolutionError(f"unknown snippet '{name}'")
        local = dict(scope)
        for key, value in (node.get("args") or {}).items():
            local[key] = eval_node(value, scope, input_name, snippets)
        return eval_node(snippets[name], local, input_name, snippets)

    if "op" in node:
        op = node["op"]
        if op not in OPS:
            raise ResolutionError(f"unknown op '{op}'")
        fn = OPS[op]
        args = node.get("args")
        if isinstance(args, list):
            return fn(*[eval_node(a, scope, input_name, snippets) for a in args])
        kwargs = {
            key: eval_node(value, scope, input_name, snippets)
            for key, value in node.items() if key not in ("op", "args")
        }
        if isinstance(args, dict):
            kwargs.update({
                key: eval_node(value, scope, input_name, snippets)
                for key, value in args.items()
            })
        return fn(**kwargs)

    raise ResolutionError(f"unsupported node {node!r}")

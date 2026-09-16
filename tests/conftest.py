"""Shared fixtures for the API / tool test suite."""

import importlib.util
import json
import os
import subprocess
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CACHED_ASSET = ("BTCUSDT", "4h")


@pytest.fixture(scope="session")
def project_root():
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def cli():
    """Load the repository's ``test.py`` as a module (avoids the stdlib ``test``)."""
    path = os.path.join(PROJECT_ROOT, "test.py")
    spec = importlib.util.spec_from_file_location("heatmap_cli_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def cached_asset():
    """A cached (asset, interval) pair, or skip if the cache is missing."""
    asset, interval = CACHED_ASSET
    cache = os.path.join(PROJECT_ROOT, "ohlc_cache", f"{asset}_{interval}_ohlc.csv")
    if not os.path.exists(cache):
        pytest.skip(f"cached OHLC data not available: {cache}")
    return CACHED_ASSET


@pytest.fixture(scope="session")
def api_call():
    """Run ``python test.py <tool> ... --api`` and return (returncode, envelope)."""

    def _call(tool, params=None, timeout=900):
        args = [sys.executable, os.path.join(PROJECT_ROOT, "test.py")]
        if tool == "schema":
            args += ["--schema", "--api"]
        else:
            args.append(tool)
            for key, value in (params or {}).items():
                if isinstance(value, bool):
                    args.append(f"--{key}={'true' if value else 'false'}")
                elif isinstance(value, (dict, list)):
                    args.append(f"--{key}={json.dumps(value)}")
                else:
                    args.append(f"--{key}={value}")
            args.append("--api")
        proc = subprocess.run(
            args, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=timeout
        )
        envelope = json.loads(proc.stdout) if proc.stdout.strip() else None
        return proc.returncode, envelope

    return _call

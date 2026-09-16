#!/usr/bin/env python3
"""Offload heavy tools to the home server while developing locally.

Local development stays on this machine (git, editor, tests).  Only the heavy
compute runs on the remote host::

    ./scripts/remote.py setup          # one-time: build image + start container
    ./scripts/remote.py sync           # push code + mirror the OHLC cache
    ./scripts/remote.py run heatmap --asset=BTCUSDT --strategy=... --api
    ./scripts/remote.py shell          # shell inside the remote container
    ./scripts/remote.py status         # container status
    ./scripts/remote.py logs           # last container logs

The remote run reuses the existing ``test.py <tool> ... --api`` contract: the
JSON envelope goes to stdout, all tool chatter to stderr.  Generated artifacts
(HTML reports, series files) are pulled back into the local tree afterwards, so
the local viewer (127.0.0.1:8900) serves them unchanged.

The remote host runs the container with rootless ``podman`` directly (no
``docker compose`` socket needed).  The container publishes **no ports** - the
host is used strictly as compute.

Configuration (environment variables)::

    HEATMAP_REMOTE_HOST       default "gmktec"
    HEATMAP_REMOTE_DIR        default "/data/home/projects/heatmap"
    HEATMAP_REMOTE_ENGINE     default "podman"
    HEATMAP_REMOTE_CONTAINER  default "damians-heatmap-dev"
    HEATMAP_REMOTE_IMAGE      default "damians-heatmap-generator:dev"
    HEATMAP_REMOTE_WORKERS    optional override; default = remote CPU count (nproc)
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HOST = os.environ.get("HEATMAP_REMOTE_HOST", "gmktec")
REMOTE_DIR = os.environ.get("HEATMAP_REMOTE_DIR", "/data/home/projects/heatmap")
ENGINE = os.environ.get("HEATMAP_REMOTE_ENGINE", "podman")
CONTAINER = os.environ.get("HEATMAP_REMOTE_CONTAINER", "damians-heatmap-dev")
IMAGE = os.environ.get("HEATMAP_REMOTE_IMAGE", "damians-heatmap-generator:dev")
# Optional override for the multiprocessing pool size on the home server.  When
# unset, the remote host's CPU count (``nproc``) is used.
WORKERS_OVERRIDE = os.environ.get("HEATMAP_REMOTE_WORKERS")
_WORKERS_CACHE = None

CONTAINER_ROOT = "/app"

# Directories that are bind-mounted from ./.docker-data on the host.
OVERMOUNT = ["ohlc_cache", "data_cache", "html_cache", "pnl_cache", "automator_html"]

# Output directories: never pushed, always pulled back (remote rel -> local rel).
OUTPUT_DIRS = [
    (".docker-data/html_cache", "html_cache"),
    (".docker-data/pnl_cache", "pnl_cache"),
    (".docker-data/automator_html", "automator_html"),
    ("series_cache", "series_cache"),
]

CODE_EXCLUDES = [
    ".git",
    "__pycache__",
    "*.pyc",
    ".pytest_cache",
    ".mypy_cache",
    ".DS_Store",
    ".docker-data",
    ".remote_jobs",
    ".devcontainer",
    "ohlc_cache",
    "data_cache",
    "html_cache",
    "html_cache_temp",
    "pnl_cache",
    "automator_html",
    "series_cache",
    "*.egg-info",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def info(msg: str) -> None:
    print(f"[remote] {msg}", file=sys.stderr)


def die(msg: str, code: int = 1) -> None:
    print(f"[remote] ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def ssh(command: str, *, capture: bool = False, check: bool = False):
    """Run a shell command string on the remote host over SSH."""
    argv = ["ssh", HOST, command]
    if capture:
        return subprocess.run(argv, capture_output=True, text=True)
    proc = subprocess.run(argv, check=False)
    if check and proc.returncode != 0:
        die(f"remote command failed (exit {proc.returncode}): {command}")
    return proc


def ssh_out(command: str) -> str:
    proc = ssh(command, capture=True)
    return (proc.stdout or "").strip()


def container_to_local(path: str) -> str:
    """Map an in-container artifact path to the equivalent local path."""
    rel = os.path.relpath(path, CONTAINER_ROOT).replace(os.sep, "/")
    first = rel.split("/", 1)[0]
    if first in OVERMOUNT:
        return os.path.join(PROJECT_ROOT, ".docker-data", rel)
    return os.path.join(PROJECT_ROOT, rel)


def _run_args() -> list[str]:
    """Build the ``podman run`` argument list (rootless, no published ports)."""
    args = [
        ENGINE, "run", "-d", "--name", CONTAINER, "--init",
        "--userns=keep-id",
        "--workdir", "/app",
        "-e", "PYTHONUNBUFFERED=1",
        "-e", "MPLBACKEND=Agg",
        "-e", "HEATMAP_IN_DOCKER=1",
        "-e", "HEATMAP_VIEWER_PORT=8900",
        "-v", f"{REMOTE_DIR}:/app",
    ]
    for name in OVERMOUNT:
        args += ["-v", f"{REMOTE_DIR}/.docker-data/{name}:/app/{name}"]
    args += [IMAGE, "sleep", "infinity"]
    return args


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

def cmd_build() -> int:
    """Build the container image on the remote host."""
    uid = ssh_out("id -u") or "1000"
    gid = ssh_out("id -g") or "1000"
    info(f"building {IMAGE} on {HOST} (uid={uid} gid={gid}) ...")
    cmd = "cd {} && {} build -t {} --build-arg DEV_UID={} --build-arg DEV_GID={} " \
          "--build-arg DEV_USER=dev .".format(
              shlex.quote(REMOTE_DIR), ENGINE, shlex.quote(IMAGE), uid, gid)
    proc = ssh(cmd)
    if proc.returncode != 0:
        die(f"image build failed (exit {proc.returncode})")
    info("image built")
    return 0


def cmd_up() -> int:
    """(Re)create and start the compute container."""
    ssh(f"{ENGINE} rm -f {shlex.quote(CONTAINER)} >/dev/null 2>&1", check=False)
    cmd = " ".join(shlex.quote(a) for a in _run_args())
    proc = ssh(cmd, capture=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr or "")
        die(f"container start failed (exit {proc.returncode})")
    info(f"container {CONTAINER} started")
    return cmd_status()


def cmd_setup() -> int:
    """One-time setup: dirs, initial sync, build and start."""
    info(f"host={HOST} dir={REMOTE_DIR} engine={ENGINE}")

    dirs = " ".join(
        f"{shlex.quote(REMOTE_DIR)}/.docker-data/{name}" for name in OVERMOUNT
    )
    ssh(f"mkdir -p {dirs} {shlex.quote(REMOTE_DIR)}/series_cache", check=True)

    cmd_sync()
    cmd_sync_ohlc()
    if cmd_build() != 0:
        return 1
    return cmd_up()


def cmd_sync() -> int:
    """Push the working tree (code) to the remote host.

    ``--delete`` mirrors deletions, but excluded patterns (output dirs,
    ``.docker-data``, caches) are protected by rsync and left untouched.
    """
    args = ["rsync", "-az", "--delete"]
    for pattern in CODE_EXCLUDES:
        args += ["--exclude", pattern]
    args += [f"{PROJECT_ROOT}/", f"{HOST}:{REMOTE_DIR}/"]
    proc = subprocess.run(args)
    if proc.returncode != 0:
        die(f"code sync failed (exit {proc.returncode})")
    info("code synced")
    return 0


def cmd_sync_ohlc() -> int:
    """Mirror the local OHLC cache to the remote host (reproducible runs)."""
    local = os.path.join(PROJECT_ROOT, ".docker-data", "ohlc_cache") + "/"
    if not os.path.isdir(local):
        info("no local OHLC cache; skipping")
        return 0
    proc = subprocess.run([
        "rsync", "-az", "--delete",
        local,
        f"{HOST}:{REMOTE_DIR}/.docker-data/ohlc_cache/",
    ])
    if proc.returncode != 0:
        die(f"OHLC mirror failed (exit {proc.returncode})")
    info("OHLC cache mirrored")
    return 0


def _remote_workers():
    """Worker count for remote heatmap/automator runs.

    ``HEATMAP_REMOTE_WORKERS`` overrides; otherwise the remote host's CPU count
    (``nproc``) is used, detected once per process.
    """
    global _WORKERS_CACHE
    if WORKERS_OVERRIDE:
        try:
            return max(1, int(WORKERS_OVERRIDE))
        except ValueError:
            pass
    if _WORKERS_CACHE is None:
        try:
            _WORKERS_CACHE = max(1, int((ssh_out("nproc") or "").strip()))
        except (TypeError, ValueError):
            _WORKERS_CACHE = 4
    return _WORKERS_CACHE


def _requested_workers(args):
    for i, arg in enumerate(args):
        if arg.startswith("--workers="):
            return arg.split("=", 1)[1]
        if arg == "--workers" and i + 1 < len(args):
            return args[i + 1]
    return None


def _strip_workers(args):
    out = []
    i = 0
    while i < len(args):
        if args[i].startswith("--workers="):
            i += 1
            continue
        if args[i] == "--workers" and i + 1 < len(args):
            i += 2
            continue
        out.append(args[i])
        i += 1
    return out


def _inject_workers(tool: str, args: list[str]) -> list[str]:
    """Force ``--workers`` to the remote CPU count for heatmap/automator.

    A client-provided value is ignored on purpose: the home server decides how
    many CPUs to use, so a low value from a caller cannot leave most cores idle.
    """
    if tool not in ("heatmap", "automator"):
        return args

    workers = _remote_workers()
    requested = _requested_workers(args)
    if requested is not None and requested != str(workers):
        info(f"workers forced {requested} -> {workers} (remote CPU count)")

    out = _strip_workers(list(args))
    out.append(f"--workers={workers}")
    return out


def cmd_run(argv: list[str]) -> int:
    """Sync, run a tool remotely and pull the artifacts back."""
    if not argv:
        die("usage: remote.py run <tool> [args...]")
    tool, tool_args = argv[0], argv[1:]
    tool_args = _inject_workers(tool, tool_args)

    no_sync = "--no-sync" in tool_args
    tool_args = [a for a in tool_args if a != "--no-sync"]
    if "--api" not in tool_args:
        tool_args = tool_args + ["--api"]

    if not no_sync:
        cmd_sync()

    remote = "{} exec {} python test.py {}".format(
        ENGINE,
        shlex.quote(CONTAINER),
        " ".join(shlex.quote(a) for a in [tool] + tool_args),
    )
    proc = ssh(remote, capture=True)

    envelope = _extract_envelope(proc.stdout or "")
    if envelope is None:
        sys.stderr.write(proc.stderr or "")
        sys.stderr.write(proc.stdout or "")
        die("could not find a JSON envelope in the remote output")

    _pull_artifacts(envelope)
    print(json.dumps(envelope, ensure_ascii=False))
    return 0 if envelope.get("ok") else 1


def _extract_envelope(stdout: str):
    """Return the JSON envelope dict from the remote stdout."""
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def _pull_artifacts(envelope) -> None:
    """Rsync the generated artifact directories back into the local tree."""
    if not isinstance(envelope, dict):
        return

    for remote_rel, _local_rel in OUTPUT_DIRS:
        remote_path = f"{REMOTE_DIR}/{remote_rel}/"
        local_path = os.path.join(PROJECT_ROOT, remote_rel) + "/"
        os.makedirs(local_path, exist_ok=True)
        proc = subprocess.run(
            ["rsync", "-az", f"{HOST}:{remote_path}", local_path],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            info(f"artifact pull skipped for {remote_rel} (exit {proc.returncode})")

    for artifact in (envelope.get("artifacts") or []):
        path = artifact.get("path") if isinstance(artifact, dict) else None
        if path and path.startswith(CONTAINER_ROOT):
            info(f"artifact -> {os.path.relpath(container_to_local(path), PROJECT_ROOT)}")


def cmd_shell() -> int:
    return ssh(f"{ENGINE} exec -it {shlex.quote(CONTAINER)} bash").returncode


def cmd_status() -> int:
    proc = ssh(
        f"{ENGINE} ps --filter name={shlex.quote(CONTAINER)} "
        "--format '{{.Names}}\t{{.Status}}\t{{.Image}}'",
        capture=True,
    )
    sys.stderr.write(proc.stdout or "")
    sys.stderr.write(proc.stderr or "")
    return proc.returncode


def cmd_logs() -> int:
    proc = ssh(f"{ENGINE} logs --tail 50 {shlex.quote(CONTAINER)}", capture=True)
    sys.stderr.write(proc.stdout or "")
    return proc.returncode


def usage() -> None:
    print(__doc__.strip())


def main() -> int:
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help", "help"):
        usage()
        return 0

    cmd, rest = argv[0], argv[1:]
    dispatch = {
        "setup": cmd_setup,
        "build": cmd_build,
        "up": cmd_up,
        "sync": cmd_sync,
        "sync-ohlc": cmd_sync_ohlc,
        "run": lambda: cmd_run(rest),
        "shell": cmd_shell,
        "status": cmd_status,
        "logs": cmd_logs,
    }
    if cmd not in dispatch:
        die(f"unknown command: {cmd}")
    return dispatch[cmd]()


if __name__ == "__main__":
    sys.exit(main())

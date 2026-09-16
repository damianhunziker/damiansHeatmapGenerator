#!/usr/bin/env python3
"""Host-side watcher for remote-offload jobs submitted by the MCP server.

The MCP server runs *inside* the Docker container, which has neither ``ssh``
nor ``rsync``.  For tools that should be offloaded to the home server it writes
a job file into ``<project>/.remote_jobs/incoming/`` (a bind-mounted directory
visible from both sides).  This watcher runs on the **host**, picks the job up,
executes::

    python scripts/remote.py run <tool> <args...>

and writes the resulting JSON envelope to
``<project>/.remote_jobs/done/<id>/result.json`` where the MCP server collects
it.

Run it in the foreground for debugging::

    python scripts/remote_watcher.py

or in the background (the ``scripts/mcp_launch.sh`` helper does this
automatically)::

    nohup python scripts/remote_watcher.py >> .remote_jobs/watcher.log 2>&1 &
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOBS_DIR = os.environ.get("HEATMAP_JOBS_DIR", os.path.join(PROJECT_ROOT, ".remote_jobs"))
INCOMING = os.path.join(JOBS_DIR, "incoming")
RUNNING = os.path.join(JOBS_DIR, "running")
DONE = os.path.join(JOBS_DIR, "done")

REMOTE_PY = os.path.join(PROJECT_ROOT, "scripts", "remote.py")
POLL_SECONDS = float(os.environ.get("HEATMAP_WATCHER_POLL", "0.5"))
JOB_TIMEOUT = int(os.environ.get("HEATMAP_WATCHER_TIMEOUT", "3600"))


def log(message: str) -> None:
    print(f"[watcher] {message}", flush=True)


def ensure_dirs() -> None:
    for directory in (INCOMING, RUNNING, DONE):
        os.makedirs(directory, exist_ok=True)


def _write_result(job_id: str, envelope: dict) -> None:
    outdir = os.path.join(DONE, job_id)
    os.makedirs(outdir, exist_ok=True)
    tmp = os.path.join(outdir, "result.json.tmp")
    with open(tmp, "w") as handle:
        json.dump(envelope, handle)
    os.replace(tmp, os.path.join(outdir, "result.json"))


def _prune_done(keep: int = 20) -> None:
    """Keep only the most recent ``keep`` result directories."""
    try:
        entries = sorted(
            (os.path.getmtime(os.path.join(DONE, name)), name)
            for name in os.listdir(DONE)
            if os.path.isdir(os.path.join(DONE, name))
        )
        for _mtime, name in entries[:-keep]:
            shutil.rmtree(os.path.join(DONE, name), ignore_errors=True)
    except Exception:  # noqa: BLE001 - pruning is best effort
        pass


def run_job(path: str) -> None:
    with open(path) as handle:
        job = json.load(handle)

    job_id = job.get("id") or os.path.splitext(os.path.basename(path))[0]
    tool = job["tool"]
    args = job.get("args", [])

    try:
        os.replace(path, os.path.join(RUNNING, os.path.basename(path)))
    except OSError:
        pass

    log(f"job {job_id}: {tool} {' '.join(args)[:140]}")
    started = time.time()

    try:
        proc = subprocess.run(
            [sys.executable, REMOTE_PY, "run", tool] + args,
            capture_output=True, text=True, timeout=JOB_TIMEOUT,
        )
        stdout = (proc.stdout or "").strip()
        try:
            envelope = json.loads(stdout)
        except json.JSONDecodeError:
            envelope = {
                "ok": False, "tool": tool,
                "error": f"remote.py produced no JSON envelope (exit {proc.returncode})",
                "stdout_tail": stdout[-2000:],
                "stderr_tail": (proc.stderr or "")[-2000:],
            }
    except subprocess.TimeoutExpired:
        envelope = {"ok": False, "tool": tool,
                    "error": f"remote job timed out after {JOB_TIMEOUT}s"}
    except Exception as exc:  # noqa: BLE001 - reported back to the client
        envelope = {"ok": False, "tool": tool, "error": f"{exc.__class__.__name__}: {exc}"}

    _write_result(job_id, envelope)
    _prune_done()
    try:
        os.remove(os.path.join(RUNNING, os.path.basename(path)))
    except OSError:
        pass
    log(f"job {job_id}: done ok={envelope.get('ok')} in {time.time() - started:.1f}s")


def main() -> None:
    ensure_dirs()
    log(f"watching {INCOMING}")
    while True:
        try:
            for name in sorted(os.listdir(INCOMING)):
                if name.endswith(".json"):
                    run_job(os.path.join(INCOMING, name))
        except Exception as exc:  # noqa: BLE001 - keep the watcher alive
            log(f"loop error: {exc.__class__.__name__}: {exc}")
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()

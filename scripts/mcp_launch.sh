#!/bin/bash
# Launch the Damian's Heatmap Generator MCP server.
#
# The server itself keeps running *inside* the container (as before).  This
# wrapper additionally makes sure the host-side remote-job watcher is alive, so
# that heatmap/automator calls submitted by the MCP server are offloaded to the
# home server (see scripts/remote_watcher.py).
set -eu

DIR="$(cd "$(dirname "$0")/.." && pwd)"
JOBS="$DIR/.remote_jobs"
mkdir -p "$JOBS"

PIDFILE="$JOBS/watcher.pid"
if ! kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null; then
    nohup python3 "$DIR/scripts/remote_watcher.py" >>"$JOBS/watcher.log" 2>&1 &
    echo $! > "$PIDFILE"
fi

exec docker exec -i damians-heatmap-dev python mcp_server/server.py

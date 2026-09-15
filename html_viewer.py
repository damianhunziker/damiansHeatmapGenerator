"""Central HTML output handling for Damian's Heatmap Generator.

All tools write their generated charts / reports to disk under the project
tree.  This module turns those files into clickable links that work no matter
where the tool runs:

* On the host the file is opened directly (``file://``) and an optional
  in-process HTTP server is started so the printed URL works too.
* Inside the Docker container there is no browser, so the tools only print the
  URL of the persistent ``viewer`` service (see ``docker-compose.yml``), which
  serves the whole project directory on ``HEATMAP_VIEWER_PORT``.

The module also maintains a small ``html_cache/viewer.html`` index page that
links to every generated report.
"""

from __future__ import annotations

import os
import threading
import webbrowser
from datetime import datetime
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DEFAULT_PORT = int(os.environ.get("HEATMAP_VIEWER_PORT", "8900"))
PUBLIC_HOST = os.environ.get("HEATMAP_VIEWER_HOST", "localhost")
BIND_HOST = os.environ.get("HEATMAP_VIEWER_BIND", "127.0.0.1")
PUBLIC_BASE_URL = os.environ.get("HEATMAP_VIEWER_BASE_URL", "").rstrip("/")

INDEX_PATH = os.path.join(PROJECT_ROOT, "html_cache", "viewer.html")

_server_lock = threading.Lock()
_server: ThreadingHTTPServer | None = None


class _QuietHandler(SimpleHTTPRequestHandler):
    """Static file handler that does not spam stdout with request logs."""

    def log_message(self, *args, **kwargs):  # noqa: D102 - silence logging
        pass


def is_container() -> bool:
    """Best-effort detection of a Docker/container environment."""
    if os.environ.get("HEATMAP_IN_DOCKER") == "1":
        return True
    return os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv")


def public_base_url(port: int | None = None) -> str:
    if PUBLIC_BASE_URL:
        return PUBLIC_BASE_URL
    return f"http://{PUBLIC_HOST}:{port or DEFAULT_PORT}"


def start_server(port: int | None = None) -> ThreadingHTTPServer | None:
    """Start a background static server for the project tree (host only)."""
    global _server
    port = port or DEFAULT_PORT
    with _server_lock:
        if _server is not None:
            return _server
        try:
            httpd = ThreadingHTTPServer(
                (BIND_HOST, port),
                partial(_QuietHandler, directory=PROJECT_ROOT),
            )
        except OSError:
            # Port already in use (e.g. the docker "viewer" service) - that is
            # fine, the printed URL is served there.
            return None
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        _server = httpd
        return _server


def url_for(file_path: str, port: int | None = None) -> str:
    """Return the public viewer URL for a file inside the project tree."""
    rel = os.path.relpath(os.path.abspath(file_path), PROJECT_ROOT)
    rel = rel.replace(os.sep, "/")
    return f"{public_base_url(port)}/{rel}"


def _relative_url(file_path: str) -> str:
    return os.path.relpath(os.path.abspath(file_path), PROJECT_ROOT).replace(os.sep, "/")


def _open_on_host(file_path: str, url: str) -> None:
    start_server()
    for target in (url, Path(file_path).as_uri()):
        try:
            if webbrowser.open(target):
                return
        except Exception:
            continue


def publish_file(file_path: str, label: str | None = None) -> str:
    """Announce a generated file and return its viewer URL.

    On the host the file is opened in the default browser; inside a container
    only the URL is printed so it can be opened from the host.
    """
    path = os.path.abspath(file_path)
    rel = _relative_url(path)
    url = f"{public_base_url()}/{rel}"
    title = label or os.path.basename(path)

    print("\n" + "=" * 72)
    print(f"  {title}")
    print(f"  {url}")
    print("=" * 72)

    if not os.path.exists(path):
        print(f"[viewer] Warning: file not found: {file_path}")

    try:
        write_index()
    except Exception as exc:  # pragma: no cover - index is best effort
        print(f"[viewer] Could not update index: {exc}")

    if not is_container():
        _open_on_host(path, url)

    return url


def publish_figure(fig, file_path: str, label: str | None = None,
                   include_plotlyjs=True) -> str:
    """Write a Plotly figure to ``file_path`` and publish it."""
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    fig.write_html(file_path, include_plotlyjs=include_plotlyjs, full_html=True)
    return publish_file(file_path, label=label)


def print_viewer_info() -> None:
    """Print the viewer entry point so users know where to find the reports."""
    base = public_base_url()
    try:
        write_index()
    except Exception:
        pass
    print("\n" + "-" * 72)
    print(f"  HTML viewer:   {base}/html_cache/viewer.html")
    print(f"  Browse files:  {base}/")
    print("  (Every generated report is linked there after each run.)")
    print("-" * 72)


def _collect_reports() -> list[tuple[str, str, float]]:
    """Return (title, relative path, mtime) for every generated report."""
    reports: list[tuple[str, str, float]] = []
    search_dirs = ["html_cache", "automator_html", "pnl_cache"]
    fragment_names = {"chart_both.html", "chart_long.html", "chart_short.html"}
    skip_names = {"viewer.html"}

    for directory in search_dirs:
        abs_dir = os.path.join(PROJECT_ROOT, directory)
        if not os.path.isdir(abs_dir):
            continue
        for root, _dirs, files in os.walk(abs_dir):
            for name in files:
                if not name.endswith(".html") or name in skip_names:
                    continue
                if directory == "html_cache" and name in fragment_names:
                    continue
                full = os.path.join(root, name)
                rel = _relative_url(full)
                title = os.path.relpath(full, PROJECT_ROOT)
                reports.append((title, rel, os.path.getmtime(full)))

    reports.sort(key=lambda item: item[2], reverse=True)
    return reports


def write_index() -> str:
    """(Re)generate the HTML index page linking all generated reports."""
    reports = _collect_reports()
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)

    rows = []
    for title, rel, mtime in reports:
        # Links are relative to html_cache/ where the index lives.
        href = os.path.relpath(os.path.join(PROJECT_ROOT, rel), os.path.dirname(INDEX_PATH))
        href = href.replace(os.sep, "/")
        stamp = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        rows.append(
            f'<li><a href="{href}">{title}</a>'
            f'<span class="ts">{stamp}</span></li>'
        )
    items = "\n".join(rows) or '<li class="empty">No reports generated yet.</li>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Damian's Heatmap Generator - Reports</title>
<style>
  body {{ font-family: -apple-system, Arial, sans-serif; margin: 40px auto; max-width: 900px;
          background: #f5f5f5; color: #222; }}
  h1 {{ border-bottom: 2px solid #2196F3; padding-bottom: 8px; }}
  ul {{ list-style: none; padding: 0; }}
  li {{ background: #fff; margin: 8px 0; padding: 12px 16px; border-radius: 6px;
        box-shadow: 0 1px 3px rgba(0,0,0,.08); display: flex; justify-content: space-between; }}
  a {{ color: #1565c0; text-decoration: none; font-weight: 600; }}
  a:hover {{ text-decoration: underline; }}
  .ts {{ color: #888; font-size: .85em; }}
  .empty {{ color: #888; }}
</style>
</head>
<body>
  <h1>Damian's Heatmap Generator &mdash; Reports</h1>
  <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
  <ul>
{items}
  </ul>
</body>
</html>
"""
    with open(INDEX_PATH, "w", encoding="utf-8") as handle:
        handle.write(html)
    return INDEX_PATH

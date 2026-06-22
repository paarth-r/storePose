"""Background localhost HTTP server for the live dashboard (stdlib only).

Serves three things:

- ``GET /metrics`` — the JSON metrics payload (polled ~1Hz by the dashboard).
- ``GET /stream`` — the annotated browser feed as Server-Sent Events (frame +
  overlay per event), when a :class:`~storepose.dashboard.stream.StreamHub` is
  attached. 404 otherwise.
- everything else — static files from ``static_dir`` (the built Next.js export)
  when one is given, else the self-contained legacy :data:`PAGE_HTML`.
"""
from __future__ import annotations

import json
import mimetypes
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from . import metrics
from .page import PAGE_HTML
from .state import DashboardState
from .stream import StreamHub


def _resolve_static(static_dir: Path, path: str) -> Path | None:
    """Map a URL path to a file under ``static_dir``, or ``None`` if unsafe/missing.

    Strips the query, maps ``/`` (and directories) to ``index.html``, and refuses
    any path that escapes ``static_dir`` (traversal).
    """
    rel = unquote(urlparse(path).path).lstrip("/")
    base = static_dir.resolve()
    target = (base / rel).resolve()
    if target != base and base not in target.parents:
        return None
    if target.is_dir():
        target = target / "index.html"
    return target if target.is_file() else None


def _make_handler(state: DashboardState, hub: StreamHub | None, static_dir: Path | None):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep the console quiet
            pass

        def _send(self, body: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_metrics(self) -> None:
            payload = metrics.build_payload(
                state.snapshot(), state.busy_snapshot(), state.debug_snapshot(),
                num_mashgins=state.num_mashgins)
            self._send(json.dumps(payload).encode("utf-8"), "application/json")

        def _send_stream(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Accel-Buffering", "no")  # disable proxy buffering
            self.end_headers()
            gen = hub.events()
            try:
                for chunk in gen:
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass  # browser disconnected
            finally:
                gen.close()  # releases the subscriber slot

        def _send_static(self) -> bool:
            if static_dir is None:
                return False
            target = _resolve_static(static_dir, self.path)
            if target is None:
                return False
            ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
            body = target.read_bytes()
            if ctype.startswith("text/") or ctype in ("application/javascript",):
                ctype += "; charset=utf-8"
            self._send(body, ctype)
            return True

        def do_GET(self):
            route = urlparse(self.path).path
            if route.startswith("/metrics"):
                self._send_metrics()
            elif route == "/stream":
                if hub is None:
                    self.send_response(404)
                    self.end_headers()
                else:
                    self._send_stream()
            elif self._send_static():
                return
            elif route == "/" or route.startswith("/index"):
                self._send(PAGE_HTML.encode("utf-8"), "text/html; charset=utf-8")
            else:
                self.send_response(404)
                self.end_headers()

    return Handler


class DashboardServer:
    """Serves the dashboard on a daemon thread; ``stop()`` on shutdown."""

    def __init__(self, state: DashboardState, port: int = 8000, host: str = "127.0.0.1",
                 hub: StreamHub | None = None, static_dir: Path | None = None):
        self._hub = hub
        self._httpd = ThreadingHTTPServer(
            (host, port), _make_handler(state, hub, static_dir))
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        return self._httpd.server_address[1]

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/"

    def start(self) -> None:
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._hub is not None:
            self._hub.close()  # wake SSE generators so their threads exit
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread:
            self._thread.join(timeout=2.0)

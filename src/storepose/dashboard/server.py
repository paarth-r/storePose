"""Background localhost HTTP server for the live dashboard (stdlib only)."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import metrics
from .page import PAGE_HTML
from .state import DashboardState


def _make_handler(state: DashboardState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep the console quiet
            pass

        def _send(self, body: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/" or self.path.startswith("/index"):
                self._send(PAGE_HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif self.path.startswith("/metrics"):
                payload = metrics.build_payload(
                    state.snapshot(), state.busy_snapshot(), state.debug_snapshot())
                body = json.dumps(payload).encode("utf-8")
                self._send(body, "application/json")
            else:
                self.send_response(404)
                self.end_headers()

    return Handler


class DashboardServer:
    """Serves the dashboard on a daemon thread; ``stop()`` on shutdown."""

    def __init__(self, state: DashboardState, port: int = 8000, host: str = "127.0.0.1"):
        self._httpd = ThreadingHTTPServer((host, port), _make_handler(state))
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
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread:
            self._thread.join(timeout=2.0)

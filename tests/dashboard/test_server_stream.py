import time
import urllib.request
from pathlib import Path

import numpy as np

from storepose.dashboard.server import DashboardServer, _resolve_static
from storepose.dashboard.state import DashboardState
from storepose.dashboard.stream import StreamHub


def _get(url):
    with urllib.request.urlopen(url, timeout=2.0) as r:
        return r.status, r.read(), r.headers.get_content_type()


def test_resolve_static_maps_root_to_index(tmp_path):
    (tmp_path / "index.html").write_text("x")
    assert _resolve_static(tmp_path, "/") == tmp_path / "index.html"
    assert _resolve_static(tmp_path, "/index.html") == tmp_path / "index.html"


def test_resolve_static_rejects_traversal(tmp_path):
    (tmp_path / "index.html").write_text("x")
    assert _resolve_static(tmp_path, "/../../etc/passwd") is None


def test_resolve_static_missing_file_is_none(tmp_path):
    assert _resolve_static(tmp_path, "/nope.js") is None


def test_static_dir_served_when_present(tmp_path):
    (tmp_path / "index.html").write_text("<h1>NEXT APP</h1>")
    (tmp_path / "app.js").write_text("console.log(1)")
    server = DashboardServer(DashboardState(), port=0, static_dir=tmp_path)
    server.start()
    try:
        status, body, ctype = _get(server.url)
        assert status == 200 and b"NEXT APP" in body and ctype == "text/html"
        status, body, ctype = _get(server.url + "app.js")
        assert status == 200 and "javascript" in ctype
    finally:
        server.stop()


def test_falls_back_to_page_html_without_static_dir():
    server = DashboardServer(DashboardState(), port=0)
    server.start()
    try:
        status, body, ctype = _get(server.url)
        assert status == 200 and b"Mashgin" in body
    finally:
        server.stop()


def test_stream_endpoint_emits_event():
    hub = StreamHub()
    server = DashboardServer(DashboardState(), port=0, hub=hub)
    server.start()
    try:
        conn = urllib.request.urlopen(server.url + "stream", timeout=3.0)
        assert conn.headers.get_content_type() == "text/event-stream"
        for _ in range(200):
            if hub.active:
                break
            time.sleep(0.005)
        assert hub.active is True
        hub.publish(np.zeros((48, 64, 3), dtype=np.uint8),
                    {"w": 64, "h": 48, "people": [], "zones": {}, "busy": None})
        line = conn.readline()
        assert line.startswith(b"data: ")
        conn.close()
    finally:
        server.stop()


def test_stream_404_without_hub():
    server = DashboardServer(DashboardState(), port=0)
    server.start()
    try:
        try:
            urllib.request.urlopen(server.url + "stream", timeout=2.0)
            assert False, "expected 404"
        except urllib.error.HTTPError as e:
            assert e.code == 404
    finally:
        server.stop()

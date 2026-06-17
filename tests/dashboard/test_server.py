import json
import urllib.request

from storepose.dashboard.server import DashboardServer
from storepose.dashboard.state import DashboardState


def _get(url):
    with urllib.request.urlopen(url, timeout=2.0) as r:
        return r.status, r.read().decode("utf-8")


def test_server_serves_page_and_metrics():
    state = DashboardState(occ_interval=0.0)
    state.observe(0.0, 2, 1)
    state.add_visit(0.0, 5.0, 2.0, "served")
    server = DashboardServer(state, port=0)  # ephemeral port
    server.start()
    try:
        status, body = _get(server.url)
        assert status == 200
        assert "Mashgin" in body and "chart-occ" in body and "echarts" in body
        status, body = _get(server.url + "metrics")
        assert status == 200
        payload = json.loads(body)
        assert "occupancy" in payload and "throughput" in payload
    finally:
        server.stop()

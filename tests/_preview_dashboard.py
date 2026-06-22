"""Throwaway preview server: serves the dashboard page with a synthetic
/metrics payload so the redesign can be eyeballed without the CV pipeline.

    uv run python tests/_preview_dashboard.py   # http://127.0.0.1:8077/
"""
import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from storepose.dashboard.page import PAGE_HTML

# a realistic-ish 6 minutes of occupancy + a handful of completed visits
T = list(range(0, 360))
waiting = [max(0, int(2.4 + 2.2 * math.sin(t / 38) + 0.6 * math.sin(t / 7))) for t in T]
serving = [1 if (t // 9) % 2 else 0 for t in T]


def ma(xs, w=30):
    out, j = [], 0
    for i in range(len(xs)):
        if i - j > w:
            j = i - w
        seg = xs[j:i + 1]
        out.append(sum(seg) / len(seg))
    return out


occ_t = [float(t) for t in T]
served_visits = [(70 + i * 11, 35 + (i % 5) * 9, 13 + (i % 4) * 2) for i in range(22)]
PAYLOAD = {
    "now": 360.0,
    "summary": {"in_line": waiting[-1], "at_pos": serving[-1],
                "avg_line_s": 41.6, "avg_pos_s": 14.2, "avg_total_s": 55.8,
                "served_count": len(served_visits)},
    "busy": {"current": {"level": "Medium", "value": 2.4}},
    "checkouts": {
        "mashgin_avg": 14.2, "mashgin_n": 22, "other_avg": 44.0, "other_n": 9,
        "delta": 29.8,
        "series": {"t_mashgin": [v[0] for v in served_visits],
                   "mashgin_ma": ma([v[2] for v in served_visits], 6),
                   "t_other": [80 + i * 28 for i in range(9)],
                   "other_ma": ma([42 + (i % 3) * 4 for i in range(9)], 4)}},
    "occupancy": {"t": occ_t, "waiting": waiting, "serving": serving,
                  "waiting_ma": ma([float(x) for x in waiting]),
                  "serving_ma": ma([float(x) for x in serving])},
    "wait_serve": {"t": [v[0] for v in served_visits],
                   "wait_ma": ma([v[1] for v in served_visits], 4),
                   "serve_ma": ma([v[2] for v in served_visits], 4)},
    "throughput": {"t": [i * 60 for i in range(6)],
                   "served_per_min": [2, 4, 5, 3, 6, 4]},
    "debug": {"frame": None, "rows": []},
}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/metrics"):
            body = json.dumps(PAYLOAD).encode()
            ct = "application/json"
        else:
            body = PAGE_HTML.encode()
            ct = "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    print("preview at http://127.0.0.1:8077/")
    ThreadingHTTPServer(("127.0.0.1", 8077), H).serve_forever()

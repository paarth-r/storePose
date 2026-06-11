# Live Web Dashboard (+ video-time clock) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Auto-launch a localhost web dashboard (stdlib only) showing live occupancy, wait/serve moving-average, and throughput charts while the pipeline runs; and fix the runner clock so file sources advance in video time.

**Architecture:** A `dashboard/` package: a thread-safe `DashboardState` the runner pushes into, pure `metrics` series builders, a stdlib `ThreadingHTTPServer` (`DashboardServer`) serving a self-contained Chart.js HTML page (`/`) and a JSON snapshot (`/metrics`). The runner starts the server, opens the browser, and feeds state each frame; a video-time `dt` makes file analysis use content time.

**Tech Stack:** Python 3.12 stdlib (`http.server`, `threading`, `webbrowser`, `json`), Chart.js via CDN, pytest, `uv`.

**Spec:** `docs/superpowers/specs/2026-06-11-live-dashboard-design.md`

---

## File Structure
- Create: `src/storepose/dashboard/__init__.py`, `state.py`, `metrics.py`, `server.py`, `page.py`.
- Modify: `src/storepose/config.py`, `src/storepose/runner.py`.
- Modify: `README.md`, `docs/usage.md`.
- Tests: `tests/dashboard/__init__.py`, `test_state.py`, `test_metrics.py`, `test_server.py`, plus `tests/test_config.py`.

---

## Task 1: `DashboardState` (thread-safe buffers)

**Files:** Create `src/storepose/dashboard/__init__.py` (empty), `src/storepose/dashboard/state.py`; Create `tests/dashboard/__init__.py` (empty), `tests/dashboard/test_state.py`.

- [ ] **Step 1: Write the failing test** — `tests/dashboard/test_state.py`:

```python
from storepose.dashboard.state import DashboardState, Visit


def test_observe_downsamples_to_interval():
    s = DashboardState(occ_interval=1.0)
    s.observe(0.0, 2, 1)
    s.observe(0.5, 3, 0)   # within interval -> dropped
    s.observe(1.0, 4, 2)   # >= interval -> kept
    occ, _ = s.snapshot()
    assert occ == [(0.0, 2, 1), (1.0, 4, 2)]


def test_add_visit_records():
    s = DashboardState()
    s.add_visit(5.0, 7.0, 3.0, "served")
    _, visits = s.snapshot()
    assert visits == [Visit(5.0, 7.0, 3.0, "served")]


def test_snapshot_is_independent_copy():
    s = DashboardState(occ_interval=0.0)
    s.observe(0.0, 1, 0)
    occ, visits = s.snapshot()
    occ.append((9.0, 9, 9))
    occ2, _ = s.snapshot()
    assert len(occ2) == 1   # mutating the snapshot didn't touch state


def test_buffers_are_bounded():
    s = DashboardState(occ_interval=0.0, max_samples=3, max_visits=2)
    for i in range(10):
        s.observe(float(i), i, 0)
        s.add_visit(float(i), 1.0, 1.0, "served")
    occ, visits = s.snapshot()
    assert len(occ) == 3 and len(visits) == 2
```

- [ ] **Step 2: Run** `uv run pytest tests/dashboard/test_state.py -v` — FAIL (no module).

- [ ] **Step 3: Implement** `src/storepose/dashboard/state.py`:

```python
"""Thread-safe live metric buffers the runner pushes into for the dashboard."""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass


@dataclass
class Visit:
    t: float
    wait_seconds: float
    serving_seconds: float
    outcome: str


class DashboardState:
    """Bounded, lock-guarded occupancy samples + completed-visit events."""

    def __init__(self, occ_interval: float = 1.0, max_samples: int = 3600,
                 max_visits: int = 5000):
        self._lock = threading.Lock()
        self._occ: deque = deque(maxlen=max_samples)      # (t, waiting, serving)
        self._visits: deque = deque(maxlen=max_visits)    # Visit
        self._occ_interval = occ_interval
        self._last_occ_t: float | None = None

    def observe(self, t: float, waiting: int, serving: int) -> None:
        with self._lock:
            if self._last_occ_t is None or (t - self._last_occ_t) >= self._occ_interval:
                self._occ.append((float(t), int(waiting), int(serving)))
                self._last_occ_t = t

    def add_visit(self, t: float, wait_seconds: float, serving_seconds: float,
                  outcome: str) -> None:
        with self._lock:
            self._visits.append(
                Visit(float(t), float(wait_seconds), float(serving_seconds), str(outcome))
            )

    def snapshot(self) -> tuple[list, list]:
        with self._lock:
            return list(self._occ), list(self._visits)
```

Also create the two empty `__init__.py` files (`src/storepose/dashboard/__init__.py`,
`tests/dashboard/__init__.py`).

- [ ] **Step 4: Run** `uv run pytest tests/dashboard/test_state.py -v` — PASS (4).

- [ ] **Step 5: Commit**

```bash
git add src/storepose/dashboard/__init__.py src/storepose/dashboard/state.py tests/dashboard/__init__.py tests/dashboard/test_state.py
git commit -m "feat: thread-safe DashboardState occupancy/visit buffers"
```

---

## Task 2: `metrics.py` (pure series builders)

**Files:** Create `src/storepose/dashboard/metrics.py`; Test `tests/dashboard/test_metrics.py`.

- [ ] **Step 1: Write the failing test** — `tests/dashboard/test_metrics.py`:

```python
from storepose.dashboard.metrics import (
    moving_average, occupancy_series, wait_serve_series, throughput_series, build_payload,
)
from storepose.dashboard.state import Visit


def test_moving_average_trailing_window():
    t = [0.0, 1.0, 2.0, 3.0]
    v = [0.0, 2.0, 4.0, 6.0]
    # window 1.5: at t=3 average of t in (1.5, 3] -> values [4,6] -> 5
    ma = moving_average(t, v, 1.5)
    assert ma[0] == 0.0
    assert ma[-1] == 5.0


def test_occupancy_series_shape():
    occ = [(0.0, 2, 1), (1.0, 3, 0)]
    s = occupancy_series(occ, ma_window=10.0)
    assert s["t"] == [0.0, 1.0]
    assert s["waiting"] == [2, 3] and s["serving"] == [1, 0]
    assert len(s["waiting_ma"]) == 2 and len(s["serving_ma"]) == 2


def test_wait_serve_series_only_served():
    visits = [Visit(1.0, 4.0, 2.0, "served"), Visit(2.0, 6.0, 4.0, "abandoned")]
    s = wait_serve_series(visits, window=100.0)
    assert s["t"] == [1.0]              # abandoned excluded
    assert s["wait_ma"] == [4.0] and s["serve_ma"] == [2.0]


def test_throughput_per_minute_buckets():
    # three served in the first 60s bucket, one in the second
    visits = [Visit(t, 1.0, 1.0, "served") for t in (0.0, 10.0, 50.0, 70.0)]
    s = throughput_series(visits, bucket=60.0)
    assert s["served_per_min"] == [3.0, 1.0]


def test_build_payload_keys():
    occ = [(0.0, 1, 0)]
    visits = [Visit(0.0, 2.0, 1.0, "served")]
    p = build_payload((occ, visits))
    assert set(p) >= {"occupancy", "wait_serve", "throughput"}
```

- [ ] **Step 2: Run** `uv run pytest tests/dashboard/test_metrics.py -v` — FAIL.

- [ ] **Step 3: Implement** `src/storepose/dashboard/metrics.py`:

```python
"""Pure series builders turning DashboardState snapshots into the JSON payload."""
from __future__ import annotations


def moving_average(times: list[float], values: list[float], window: float) -> list[float]:
    """Trailing-``window`` mean of ``values`` aligned to ``times`` (same length)."""
    out: list[float] = []
    j = 0
    for i in range(len(times)):
        while times[j] < times[i] - window:
            j += 1
        seg = values[j:i + 1]
        out.append(sum(seg) / len(seg) if seg else 0.0)
    return out


def occupancy_series(occ: list, ma_window: float = 30.0) -> dict:
    t = [s[0] for s in occ]
    waiting = [s[1] for s in occ]
    serving = [s[2] for s in occ]
    return {
        "t": t, "waiting": waiting, "serving": serving,
        "waiting_ma": moving_average(t, [float(w) for w in waiting], ma_window),
        "serving_ma": moving_average(t, [float(s) for s in serving], ma_window),
    }


def wait_serve_series(visits: list, window: float = 120.0) -> dict:
    served = [v for v in visits if v.outcome == "served"]
    t = [v.t for v in served]
    return {
        "t": t,
        "wait_ma": moving_average(t, [v.wait_seconds for v in served], window),
        "serve_ma": moving_average(t, [v.serving_seconds for v in served], window),
    }


def throughput_series(visits: list, bucket: float = 60.0) -> dict:
    served = sorted(v.t for v in visits if v.outcome == "served")
    if not served:
        return {"t": [], "served_per_min": []}
    start, end = served[0], served[-1]
    counts: dict[int, int] = {}
    for tt in served:
        counts[int((tt - start) // bucket)] = counts.get(int((tt - start) // bucket), 0) + 1
    n = int((end - start) // bucket) + 1
    return {
        "t": [start + b * bucket for b in range(n)],
        "served_per_min": [counts.get(b, 0) * (60.0 / bucket) for b in range(n)],
    }


def build_payload(snapshot: tuple[list, list]) -> dict:
    occ, visits = snapshot
    now = occ[-1][0] if occ else 0.0
    return {
        "now": now,
        "occupancy": occupancy_series(occ),
        "wait_serve": wait_serve_series(visits),
        "throughput": throughput_series(visits),
    }
```

- [ ] **Step 4: Run** `uv run pytest tests/dashboard/test_metrics.py -v` — PASS (5).

- [ ] **Step 5: Commit**

```bash
git add src/storepose/dashboard/metrics.py tests/dashboard/test_metrics.py
git commit -m "feat: dashboard metrics series (occupancy MA, wait/serve MA, throughput)"
```

---

## Task 3: `DashboardServer` + HTML page

**Files:** Create `src/storepose/dashboard/page.py`, `src/storepose/dashboard/server.py`; Test `tests/dashboard/test_server.py`.

- [ ] **Step 1: Write the failing test** — `tests/dashboard/test_server.py`:

```python
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
        assert status == 200 and "<canvas" in body
        status, body = _get(server.url + "metrics")
        assert status == 200
        payload = json.loads(body)
        assert "occupancy" in payload and "throughput" in payload
    finally:
        server.stop()
```

- [ ] **Step 2: Run** `uv run pytest tests/dashboard/test_server.py -v` — FAIL.

- [ ] **Step 3a: Implement** `src/storepose/dashboard/page.py`:

```python
"""Self-contained dashboard HTML page (Chart.js via CDN, polls /metrics)."""

PAGE_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>storePose dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  body { font-family: system-ui, sans-serif; margin: 16px; background:#111; color:#eee; }
  h1 { font-size: 18px; } .chart { max-width: 900px; margin-bottom: 28px; }
</style></head><body>
<h1>storePose &mdash; live</h1>
<div class="chart"><canvas id="occ"></canvas></div>
<div class="chart"><canvas id="ws"></canvas></div>
<div class="chart"><canvas id="tp"></canvas></div>
<script>
function mk(id, title, series) {
  return new Chart(document.getElementById(id), {
    type: 'line',
    data: { datasets: series.map(s => ({label: s.label, data: [], borderColor: s.color,
            borderWidth: 2, pointRadius: 0, tension: 0.2})) },
    options: { animation: false, responsive: true,
      plugins: { title: { display: true, text: title, color: '#eee' },
                 legend: { labels: { color: '#eee' } } },
      scales: { x: { type: 'linear', title: { display: true, text: 'seconds', color:'#aaa' },
                     ticks:{color:'#aaa'} }, y: { beginAtZero: true, ticks:{color:'#aaa'} } } }
  });
}
const occ = mk('occ', 'Occupancy', [
  {label:'in line', color:'#ffb300'}, {label:'at POS', color:'#00c8ff'},
  {label:'in line (avg)', color:'#ffe082'}, {label:'at POS (avg)', color:'#80deea'}]);
const ws = mk('ws', 'Wait & serve (moving avg, s)', [
  {label:'wait', color:'#ff5252'}, {label:'serve', color:'#69f0ae'}]);
const tp = mk('tp', 'Throughput (served/min)', [{label:'served/min', color:'#b388ff'}]);
function xy(t, v){ return t.map((tt,i)=>({x:tt, y:v[i]})); }
async function poll(){
  try {
    const r = await fetch('metrics'); const d = await r.json();
    const o = d.occupancy;
    occ.data.datasets[0].data = xy(o.t, o.waiting);
    occ.data.datasets[1].data = xy(o.t, o.serving);
    occ.data.datasets[2].data = xy(o.t, o.waiting_ma);
    occ.data.datasets[3].data = xy(o.t, o.serving_ma);
    occ.update();
    const w = d.wait_serve;
    ws.data.datasets[0].data = xy(w.t, w.wait_ma);
    ws.data.datasets[1].data = xy(w.t, w.serve_ma);
    ws.update();
    const p = d.throughput;
    tp.data.datasets[0].data = xy(p.t, p.served_per_min);
    tp.update();
  } catch (e) {}
}
poll(); setInterval(poll, 1000);
</script></body></html>
"""
```

- [ ] **Step 3b: Implement** `src/storepose/dashboard/server.py`:

```python
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
                body = json.dumps(metrics.build_payload(state.snapshot())).encode("utf-8")
                self._send(body, "application/json")
            else:
                self.send_response(404)
                self.end_headers()

    return Handler


class DashboardServer:
    """Serves the dashboard on a daemon thread; stop() on shutdown."""

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
```

- [ ] **Step 4: Run** `uv run pytest tests/dashboard/test_server.py -v` — PASS.

- [ ] **Step 5: Commit**

```bash
git add src/storepose/dashboard/page.py src/storepose/dashboard/server.py tests/dashboard/test_server.py
git commit -m "feat: stdlib dashboard server + Chart.js page"
```

---

## Task 4: Config flags

**Files:** Modify `src/storepose/config.py`; Test `tests/test_config.py`.

- [ ] **Step 1: Write the failing test** — append to `tests/test_config.py`:

```python
def test_dashboard_defaults_on():
    cfg = from_args([])
    assert cfg.dashboard is True
    assert cfg.dashboard_port == 8000


def test_no_dashboard_and_port():
    assert from_args(["--no-dashboard"]).dashboard is False
    assert from_args(["--dashboard-port", "9001"]).dashboard_port == 9001


def test_dashboard_port_range_validated():
    with pytest.raises(ValueError):
        AppConfig(dashboard_port=70000)
```

- [ ] **Step 2: Run** `uv run pytest tests/test_config.py -k dashboard -v` — FAIL.

- [ ] **Step 3: Implement** in `src/storepose/config.py`:

(a) Dataclass fields (place after the `busy_hysteresis` field at the end of the field list):

```python
    dashboard: bool = True
    dashboard_port: int = 8000
```

(b) Docstring Attributes (after the `busy_hysteresis` entry):

```python
        dashboard: Serve the live web dashboard during the run.
        dashboard_port: Port for the dashboard HTTP server.
```

(c) Validation in `__post_init__` (after the `busy_hysteresis` check):

```python
        if not 1 <= self.dashboard_port <= 65535:
            raise ValueError(f"dashboard_port must be in [1, 65535], got {self.dashboard_port}")
```

(d) CLI args in `_build_parser` (after the `--busy-hysteresis` argument):

```python
    parser.add_argument(
        "--no-dashboard", dest="dashboard", action="store_false",
        help="Disable the live web dashboard.",
    )
    parser.add_argument(
        "--dashboard-port", type=int, default=8000,
        help="Port for the live dashboard HTTP server (default: 8000).",
    )
```

(e) Wire into `from_args` (after `busy_hysteresis=args.busy_hysteresis,`):

```python
        dashboard=args.dashboard,
        dashboard_port=args.dashboard_port,
```

- [ ] **Step 4: Run** `uv run pytest tests/test_config.py -v` — PASS.

- [ ] **Step 5: Commit**

```bash
git add src/storepose/config.py tests/test_config.py
git commit -m "feat: --no-dashboard / --dashboard-port config flags"
```

---

## Task 5: Runner — video-time dt + dashboard wiring

**Files:** Modify `src/storepose/runner.py`.

- [ ] **Step 1: Imports.** Add near the top of `src/storepose/runner.py`:

```python
import webbrowser

from .dashboard.server import DashboardServer
from .dashboard.state import DashboardState
```

- [ ] **Step 2: Start the dashboard before the loop.** Inside `run()`'s `ExitStack`,
after the analyzer/zone block and before the wait-log/`cv2.namedWindow` setup, add:

```python
                dash_state = None
                if config.dashboard:
                    dash_state = DashboardState()
                    dash_server = DashboardServer(dash_state, port=config.dashboard_port)
                    dash_server.start()
                    stack.callback(dash_server.stop)
                    print(f"Dashboard: {dash_server.url}")
                    try:
                        webbrowser.open(dash_server.url)
                    except Exception:
                        pass
```

- [ ] **Step 3: Video-time dt + single clock.** Replace the per-frame `dt`/`clock`
handling. The loop currently computes `dt` from `perf_counter` and does `clock += dt`
inside the busy block. Change the top of the loop so the clock ticks every frame and
files use video time:

```python
                is_file = isinstance(config.source, str)
                base_dt = 1.0 / (source.fps or _DEFAULT_FPS)
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                prev = None
                clock = 0.0
                for frame in source:
                    if is_file:
                        dt = base_dt                       # video time
                    else:
                        now = time.perf_counter()          # webcam: real time
                        dt = (now - prev) if prev else base_dt
                        prev = now
                    clock += dt
```

Then delete the now-duplicated `clock = 0.0` initializer above the loop and the
`clock += dt` line inside the busy block (busy keeps using `clock`).

- [ ] **Step 4: Feed the dashboard each frame.** In the `if analyzer is not None:`
branch, after `qresult = analyzer.update(people, dt)` and the completed-visit loop,
add the dashboard observe + visits; and observe zeros when there is no analyzer.
Concretely, right after the wait-log writer loop (still inside `if analyzer is not None:`):

```python
                            if dash_state is not None:
                                dash_state.observe(clock, qresult.count, qresult.serving_count)
                                for c in qresult.completed:
                                    dash_state.add_visit(clock, c.wait_seconds,
                                                         c.serving_seconds, c.outcome)
```

And, so the dashboard still updates when there is no zone, after the
`if tracker is not None:` / `else:` annotate split (i.e. once `canvas` is set for the
frame, near where `sink.write` happens), add:

```python
                    if dash_state is not None and analyzer is None:
                        dash_state.observe(clock, 0, 0)
```

- [ ] **Step 4b: Verify nothing broke + dt is video-time.**

Run: `uv run pytest -q` → full suite green.
Run: `uv run python -c "from storepose.runner import Runner; print('ok')"` → `ok`.

Smoke that a file run now produces video-time busy windows (the 10-min clip should
yield ~5 windows of 120 s instead of ~2). One line, no backslashes:

Run: `uv run python main.py --source "videos/2706969 - Rock Hill, SC _SCO 1 _2026-05-07 12_09_21_370.mp4" --zone "zones/2706969 - Rock Hill, SC _SCO 1 _2026-05-07 12_09_21_370.json" --busy --busy-log /tmp/vt_busy.csv --busy-window 120 --no-dashboard`
Expected: `Wrote busy report (5 window(s))` (≈, vs 2 before).

- [ ] **Step 5: Commit**

```bash
git add src/storepose/runner.py
git commit -m "feat: video-time dt for files + live dashboard wiring in runner"
```

---

## Task 6: Documentation

**Files:** Modify `README.md`, `docs/usage.md`.

- [ ] **Step 1: README** — add flag rows to the flags table (after the `--busy-hysteresis` row):

```markdown
| `--no-dashboard` | —       | Disable the live web dashboard.                   |
| `--dashboard-port` | `8000` | Port for the live dashboard server.             |
```

- [ ] **Step 2: README** — add a short "Live dashboard" subsection before "## Performance":

```markdown
## Live dashboard

Every run auto-starts a localhost web dashboard (no extra dependencies) and opens
it in your browser, with three live charts: **occupancy** (in line / at POS with
moving averages), **wait & serve** moving averages, and **throughput**
(served/min). Disable with `--no-dashboard`; change the port with
`--dashboard-port`. On a **file** source the timeline is video time; on a **webcam**
it is real time.
```

- [ ] **Step 3: usage.md** — add a "Dashboard" note in Section 2 listing
`--no-dashboard` and `--dashboard-port` and that the page is at
`http://127.0.0.1:<port>/`.

```markdown
### Live dashboard

Auto-starts at `http://127.0.0.1:8000/` (override `--dashboard-port`; disable with
`--no-dashboard`). Three live charts — occupancy, wait/serve moving averages, and
throughput — fed by the running pipeline. Most useful with `--zone` / `--pos-zone`.
```

- [ ] **Step 4: Commit**

```bash
git add README.md docs/usage.md
git commit -m "docs: document the live dashboard + flags"
```

---

## Task 7: Full verification

- [ ] **Step 1:** `uv run pytest -q` → all green.
- [ ] **Step 2:** Live smoke — run on the clip with the dashboard, confirm the server
answers (the window also opens; here we just check HTTP from another process while it
runs is out of scope — instead confirm the page/metrics endpoints via the unit test in
Task 3 and that the run prints `Dashboard: http://127.0.0.1:8000/`).

Run: `uv run python main.py --source /tmp/sco_clip.mp4 --zone "zones/2706969 - Rock Hill, SC _SCO 1 _2026-05-07 12_09_21_370.json" --pos-zone "zones/2706969 - Rock Hill, SC _SCO 1 _2026-05-07 12_09_21_370_pos.json"`
Expected: prints `Dashboard: http://127.0.0.1:8000/`, the browser opens to three charts that populate as the clip plays, and the run exits cleanly.

---

## Self-Review Notes
- **Spec coverage:** video-time dt (Task 5) ✓; `DashboardState` thread-safe bounded buffers (Task 1) ✓; pure `metrics` series incl. MA/throughput (Task 2) ✓; stdlib `DashboardServer` `/` + `/metrics` + page (Task 3) ✓; config on-by-default + port (Task 4) ✓; runner start/feed/stop + auto-open browser (Task 5) ✓; three panels in the page (Task 3) ✓; docs (Task 6) ✓; tests for state/metrics/server (Tasks 1–3) ✓.
- **Type consistency:** `DashboardState(occ_interval, max_samples, max_visits)` with `observe(t,waiting,serving)` / `add_visit(t,wait_seconds,serving_seconds,outcome)` / `snapshot()->(occ_list,visits_list)`; `Visit(t,wait_seconds,serving_seconds,outcome)`; `metrics.build_payload((occ,visits))`; `DashboardServer(state, port, host)` with `.url`/`.start()`/`.stop()`. Used consistently across tasks.
- **Deps:** stdlib only (`http.server`, `threading`, `webbrowser`, `json`); Chart.js from CDN in the page. No `pyproject` change.
- **Back-compat:** dashboard defaults on but is purely additive; `--no-dashboard` restores prior behavior; the dt change only affects file sources (intended accuracy fix).
```

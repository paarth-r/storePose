# Live web dashboard (+ video-time clock fix)

**Date:** 2026-06-11
**Status:** Approved design, pre-implementation
**Branch:** feat/realtime-pose
**Sub-project 2 of 2** (consumes the waiting/serving/throughput signals from sub-project 1).

## Goal

While the pipeline runs, auto-start a localhost web dashboard (zero new
dependencies) showing three live, moving-average charts — occupancy, wait/serve
times, and throughput. Also fix the runner clock so a **file** source advances in
**video time** (`1/fps`) rather than wall-clock processing time, making both the
dashboard axis and the existing wait/serve/busy numbers correct for recorded video.

## Component 0: video-time clock (runner)

The frame loop currently computes `dt` from `time.perf_counter()` (wall-clock).
Change it so:
- **File source** (`config.source` is a `str`): `dt = 1.0 / (source.fps or _DEFAULT_FPS)` — fixed video-frame interval, so the clock tracks content time.
- **Webcam** (`config.source` is an `int`): keep the wall-clock `perf_counter` delta (real time).

This corrects wait/serve timers and busy windows for files (a 10-min clip → ~600 s
of analysis time) and gives the dashboard a true time axis in both modes. A single
`clock` accumulator ticks every frame (independent of `--busy`).

## Component package: `src/storepose/dashboard/`

### `state.py` — `DashboardState` (thread-safe)
The runner pushes into this each frame; the server reads snapshots. Uses a lock.
- `observe(t, waiting, serving)`: append an occupancy sample, but at most once per
  `occ_interval` (default 1.0 s) to bound size; keep a bounded deque (last
  `max_samples`, default 3600 ≈ 1 h at 1/s).
- `add_visit(t, wait_seconds, serving_seconds, outcome)`: append a completed-visit
  event to a bounded deque (last `max_visits`, default 5000).
- `snapshot()`: return copies of both buffers under the lock.

### `metrics.py` — pure series builders (the unit-tested core)
No server/threads. Given a state snapshot and `now`:
- `moving_average(times, values, window)` → values smoothed by a trailing-`window`
  mean, aligned to `times`.
- occupancy series: `{t, waiting, serving, waiting_ma, serving_ma}` (MA window
  default 30 s).
- wait/serve series: for each served visit time, the trailing-`window` (default
  120 s) mean of `wait_seconds` and `serving_seconds` → `{t, wait_ma, serve_ma}`.
- throughput: served visits bucketed per 60 s → `{t, served_per_min}`.
- `build_payload(snapshot, now)` → the JSON-able dict the page consumes.

### `server.py` — `DashboardServer`
Wraps `http.server.ThreadingHTTPServer` (stdlib). A handler closed over the
`DashboardState`:
- `GET /` → the HTML page (`page.PAGE_HTML`).
- `GET /metrics` → `json.dumps(metrics.build_payload(state.snapshot(), now))`.
- anything else → 404.
`start()` binds `127.0.0.1:port` and runs `serve_forever` on a daemon thread;
`url` → `http://127.0.0.1:<port>/`; `stop()` calls `shutdown()`. Binding to port 0
is supported (tests use it).

### `page.py` — `PAGE_HTML`
A self-contained HTML string: Chart.js from a CDN, three `<canvas>` charts, and JS
that `fetch('/metrics')` every ~1 s and updates the charts (occupancy lines +
MA overlay; wait/serve MA lines; throughput bars/line). Static asset — verified
manually in the browser, not unit-tested.

## Config (`config.py`)
- `dashboard: bool = True` + `--no-dashboard` (`store_false`).
- `dashboard_port: int = 8000` + `--dashboard-port`; validate `1 <= port <= 65535`.

## Runner integration
During setup (after models load, before the capture loop): if `config.dashboard`,
build `DashboardState` + `DashboardServer(state, config.dashboard_port)`, `start()`
it, `webbrowser.open(server.url)` (failure is non-fatal — wrap in try/except),
print the URL. Register `server.stop()` in the `ExitStack`/`finally` so it stops on
exit. Each frame: `clock += dt`; when an analyzer is present,
`state.observe(clock, qresult.count, qresult.serving_count)` and
`state.add_visit(clock, c.wait_seconds, c.serving_seconds, c.outcome)` for each
completed visit. With no analyzer (no `--zone`), still observe
`state.observe(clock, 0, 0)` so the server runs (occupancy flat) — the dashboard is
most useful with a zone.

## Panels (v1)
1. **Occupancy over time** — `in line` (waiting) and `at POS` (serving) lines + MA overlay.
2. **Wait & serve moving averages** — trailing-window mean of completed `wait_seconds` / `serving_seconds`.
3. **Throughput** — served visits per minute.
(Abandonment/summary cards deferred.)

## Testing
- `metrics.py`: `moving_average` on a known series; occupancy payload shape;
  wait/serve windowed mean over fixed visits; per-minute throughput bucketing.
- `DashboardState`: `observe` downsamples to `occ_interval`; deques bound size;
  `snapshot` returns independent copies.
- `server.py`: bind port 0 on a thread, `GET /` returns HTML (200, contains a
  `<canvas>`), `GET /metrics` returns parseable JSON with the expected keys, then
  `stop()`. (Uses `urllib.request`.)
- HTML/JS: manual browser check.

## Out of scope
Abandonment chart + summary cards; offline CSV replay; persistence/history across
runs; auth (localhost only).

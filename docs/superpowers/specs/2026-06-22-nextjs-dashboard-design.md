# Polished Next.js Dashboard + Browser Annotated Feed

Date: 2026-06-22
Status: Approved

## Goal

A polished, hand-designed (Linear-like, not templated) web dashboard for storePose,
built in Next.js, that connects to the running process and shows:

1. A **live annotated video feed** that looks clean and understandable — not OpenCV
   debug output. Overlays (person boxes, skeletons, state labels, zones) are rendered
   **client-side as crisp SVG** on top of the raw camera frame.
2. The existing rich **metrics** (occupancy, wait/serve times, throughput, busy level,
   Mashgin-vs-staffed comparison) as clean cards and charts.

## Non-goals / constraints

- **Do not change the existing `cv2.imshow` window** or `drawing.py`. That local
  annotated window is the "current stream" and stays exactly as is.
- The new browser feed is **additive** and gated by the **existing `--dashboard` flag**.
  No new CLI flag: the stream is rendered only when the dashboard is enabled **and** a
  browser is actively subscribed (lazy — zero cost when nobody is watching).
- Face blurring stays on: the streamed frame is the raw camera frame with face-blur
  applied and **nothing else drawn**.
- Backend stays stdlib-only (`http.server`); no new Python dependencies.

## Architecture & data flow

Two same-origin channels from the Python process to the browser:

```
Runner loop ──┬─ GET /metrics  (existing JSON, polled @1Hz)  → cards + charts
              └─ GET /stream   (NEW, SSE @~12-15fps)          → live feed + SVG overlay
```

Each `/stream` Server-Sent Event carries the frame **and** its overlay data together,
so overlays are always pixel-aligned to the frame (MJPEG can't carry a sidecar — that
is why combined-SSE is used):

```json
{
  "seq": 142,
  "t": 88.4,
  "w": 1280, "h": 720,
  "jpeg": "data:image/jpeg;base64,...",   // raw, face-blurred, downscaled
  "people": [
    {"id": 5, "box": [x1,y1,x2,y2],
     "kpts": [[x,y,score], ... 17],
     "state": "waiting|serving|serving_other|candidate|out",
     "wait": 12.3, "serve": 0.0, "progress": 1.0}
  ],
  "zones": {"line": [[[x,y],...]], "pos": [[[x,y],...]], "alt": [[[x,y],...]]},
  "busy": {"level": "Low|Medium|High|null", "value": 1.4}
}
```

Coordinates are in **original frame space** (`w`,`h`); the JPEG may be downscaled. The
SVG overlay uses `viewBox="0 0 w h"` so it scales with the displayed image and stays
crisp at any size, regardless of JPEG resolution.

## Backend (Python — additive)

New files under `src/storepose/dashboard/`:

- **`stream_payload.py`** — pure builder `build_overlay(people, qresult, zones, busy,
  width, height) -> dict` producing the `people`/`zones`/`busy`/`w`/`h` portion from the
  objects the runner already has (`TrackedPerson`, `QueueResult.statuses`, `Zone`). No
  cv2, fully unit-testable.
- **`stream.py`** — `StreamHub`: thread-safe holder of the latest `(seq, jpeg_b64,
  overlay)`. `active` property (subscriber count > 0). `publish(frame_bgr, overlay)`
  downscales (max width ~960), JPEG-encodes (q≈72), base64s, bumps seq, notifies. A
  `stream()` generator yields SSE `data: ...\n\n` lines, blocking on a `Condition` until
  a new seq arrives; increments/decrements subscriber count around its lifetime.

Edits:

- **`server.py`** — add `GET /stream` (`text/event-stream`, no-cache, streams
  `hub.stream()`); serve static files from `web/out/` at `/` and other non-API paths
  **when that directory exists**, else fall back to today's `PAGE_HTML` so
  `uv run python main.py` always works without a Node build. `/metrics` unchanged.
- **`runner.py`** — when the dashboard is on, construct a `StreamHub`, pass it to the
  server. Each frame, **only when `hub.active`**: build a face-blurred copy of the raw
  `frame` (reusing `faces.blur_faces`), build the overlay via `stream_payload`, and
  `hub.publish(...)`. The existing `canvas`/`annotate_*`/`imshow` path is untouched.

## Frontend (`web/` — Next.js + TypeScript + Tailwind)

Single dashboard route, hand-built components (no generic UI kit):

- `web/next.config.*` — `output: 'export'`; dev `rewrites` proxy `/metrics` and
  `/stream` to `http://127.0.0.1:8000` (no CORS). In prod the static export is served by
  the Python server, so relative URLs are same-origin.
- `lib/useStream.ts` — `EventSource('/stream')` hook → latest frame + overlay.
- `lib/useMetrics.ts` — 1Hz `fetch('/metrics')` poll.
- `components/LiveFeed.tsx` — `<img>` of the streamed frame + absolutely-positioned
  `<svg viewBox="0 0 w h">` overlay: rounded-rect person boxes (color encodes **state**,
  not identity), thin low-opacity tapered skeleton (COCO-17 edges), small rounded state
  pills in a real web font, soft translucent zone polygons. Keypoints **lerp** between
  SSE frames via `requestAnimationFrame` for buttery motion at low fps.
- `components/StatCards.tsx` — In line · At checkout · Avg wait · Avg total · Served,
  large tabular-figure numerals, hairline dividers.
- `components/BusyPill.tsx` — Low/Med/High + value, calm color coding.
- `components/Charts.tsx` — occupancy, wait/serve, throughput, Mashgin-vs-staffed delta.
  Lightweight (custom SVG sparklines for tiles; a small chart lib if a full chart is
  warranted), restrained styling.
- `components/DebugTable.tsx` — per-person reasoning; shown only when `metrics.debug.rows`
  is populated (process running `--debug`).

Aesthetic: neutral light surface, subtle borders/soft shadows, tight modern sans,
tabular numerals for live figures, smooth micro-interactions. Deliberate type scale and
restrained palette so it reads as a crafted product, not generated output.

## Run / build workflow

- Dev: `uv run python main.py …` (process, :8000) + `cd web && npm run dev` (:3000,
  proxied). Edit/iterate with hot reload.
- Integrated: `cd web && npm run build` → static `web/out/` → Python serves the polished
  dashboard at the dashboard port. One URL, no Node running.

## Testing

- TDD (pytest) for `stream_payload.build_overlay` (shape + values from synthetic
  `TrackedPerson`/`PersonStatus`), `StreamHub` (lazy encode only when active, seq bump,
  SSE framing, subscriber count), and `server.py` (`/stream` headers; static-vs-fallback
  serving at `/`).
- Frontend: production `npm run build` must succeed; pure overlay-scaling/lerp helpers
  unit-testable and extracted from components.

## File summary

```
web/                                  NEW  Next.js app (app/, components/, lib/, config)
src/storepose/dashboard/
  stream.py            NEW  SSE StreamHub (lazy latest-frame holder)
  stream_payload.py    NEW  per-frame people/zones/busy overlay builder
  server.py            EDIT +GET /stream, +static serving (fallback to page.py)
  runner.py            EDIT publish raw frame + overlay to the hub when active
docs/superpowers/specs/2026-06-22-nextjs-dashboard-design.md  NEW  this doc
```

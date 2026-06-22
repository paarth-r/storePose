# storePose dashboard (`web/`)

The live dashboard UI for storePose: a **Next.js 15** app (React 19, TypeScript,
Tailwind v4) that renders a live annotated video feed plus occupancy, wait/serve,
throughput, and Mashgin-vs-staffed charts.

It is **statically exported** (`next build` → `web/out/`) and served by the
storePose Python process — there is **no Node runtime at display time**. The
Python server (`src/storepose/dashboard/server.py`) serves `web/out` for any
non-API path and exposes two endpoints the UI consumes:

- `GET /metrics` — JSON time series, polled once per second (`lib/useMetrics`).
- `GET /stream` — Server-Sent Events; each event carries the latest JPEG frame +
  an overlay (people, keypoints, zones, busy) (`lib/useStream`).

## Build & serve

```bash
npm install          # first time
npm run build        # → web/out  (the Python server serves this)
```

Then run storePose normally; the dashboard opens at `http://127.0.0.1:8000/`. If
`web/out` is absent the Python server falls back to a self-contained legacy HTML
page. `../video-run.sh` runs this build automatically when it is stale.

## Develop (hot reload)

```bash
# terminal 1: the pipeline (serves /metrics and /stream on :8000)
uv run python ../main.py --source ../videos/clip.mp4 --zone ../zones/clip.json

# terminal 2: the Next.js dev server on :3000
npm run dev
```

`next.config.mjs` proxies `/metrics` and `/stream` from :3000 to
`http://127.0.0.1:8000`, so relative fetches work without CORS. Browse to
`http://localhost:3000/`.

## Layout

```
app/         page.tsx (single route) + global styles
components/   LiveFeed, RightNow, CheckoutEdge, StatStrip,
             OccupancyChart, WaitServeChart, ThroughputChart, DebugTable
lib/         useMetrics, useStream, useAnimatedOverlay (keypoint lerp), types
```

Charts are hand-rolled SVG — no third-party chart dependency.

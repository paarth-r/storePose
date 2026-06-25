# Store Busyness Dashboard Refactor — Design

Date: 2026-06-25
Branch: `feat/busyness-dashboard-refactor` (git worktree)
Requested by: Dobromir (via Paarth)

## Goal

Refactor the storePose dashboard into a clean, intuitive "Store Busyness" view that
makes the Mashgin throughput story legible at a glance: a per-timestamp histogram of
store occupancy, a headline "one cashier becomes X" efficiency metric, a now-vs-earlier
time comparison, the existing average graphs, a default-off per-person debug view, and a
top slider to switch between Mashgin / non-Mashgin / side-by-side stats. Match the
current dashboard aesthetic and add Mashgin branding.

## Constraints & Context

- **Stack:** Next.js 15 / React 19 / Tailwind v4 static export in `web/`, served by the
  stdlib Python server (`src/storepose/dashboard/server.py`). Charts are hand-rolled SVG.
- **Data is already sufficient — frontend-only change.** The `/metrics` JSON payload
  (`build_payload` in `dashboard/metrics.py`, polled ~1Hz via `useMetrics`) already
  exposes everything needed. **No backend change.**
- **Aesthetic to preserve:** warm-neutral light theme — bg `#fafaf9`, white cards, 14px
  radius, hairline borders `#e9e9e6`, Geist Sans/Mono, semantic state colors
  (go/green `#15803d`, wait/amber `#b45309`, busy/red `#be123c`).
- **Mashgin brand (fetched from mashgin.com):** colorful hexagonal-cube network mark
  (`Logo.png` 472×97; favicon 32×32). Palette: signature **lime `#E4F222`**, green
  `#63E14F`, teal `#45C7AB`, near-black `#151515`. The wordmark text is white (built for
  dark bg); the hex *mark* reads fine on the light theme.

## Decisions (confirmed with user)

1. **Logo:** fetch from mashgin.com (done — saved to `web/public/`).
2. **"In store" count:** `in_line + at_pos` (no new tracking).
3. **Histogram vs existing OccupancyChart:** keep both.
4. **Slider empty state:** disable unavailable modes with a tooltip until data exists.
5. **Time comparison:** "Now vs Earlier" — recent window vs session baseline (confirmed).

## Relevant existing data shapes (`Metrics`, `web/lib/types.ts`)

```ts
summary:   { in_line, at_pos, avg_line_s, avg_pos_s, avg_total_s, served_count }
busy:      { current:{level,value}, t[], level_idx[], value[] }
checkouts: { mashgin_avg, mashgin_avg_eff, num_mashgins, mashgin_n,
             other_avg, other_n, delta, series:{t_mashgin,mashgin_ma,t_other,other_ma} }
occupancy: { t[], waiting[], serving[], waiting_ma[], serving_ma[] }   // waiting=at line, serving=at POS
wait_serve:{ t[], wait_ma[], serve_ma[] }
throughput:{ t[], served_per_min[] }
debug:     { frame, rows:[{id,state,wait,serve,speed,line,pos,reg,transit}] }  // already streamed every frame
```

## Components

All work in `web/`. New components in `web/components/`, helpers in `web/lib/`.

### 1. Branding — `Header.tsx` (edit) + `web/public/` assets
- Save `mashgin-logo.png` + favicon to `web/public/`.
- Header left: Mashgin hex mark + title **"Store Busyness"**, with a small
  `storePose · Line Monitor` subtag (co-brand, don't erase the existing identity).
- Keep session timer + connection pulse on the right.
- Add a default-OFF **debug toggle** control on the right (see §6).
- Add lime `--color-mashgin: #E4F222` (+ a darker readable variant for text on light bg,
  e.g. `#848C14`) to `globals.css` design tokens; use it as the accent for Mashgin data.

### 2. Occupancy histogram (centerpiece) — `Histogram.tsx` (new)
- Per-timestamp **stacked bar** chart from `occupancy` series. To keep bars readable,
  bucket/downsample `occupancy.{t,waiting,serving}` to ~40–60 bars across the visible
  window (averaged per bucket, rounded), not one bar per raw sample.
- Each bar height = **total in store** = `waiting + serving`; bottom segment = **at line**
  (amber), top segment = **at POS** (lime/Mashgin). Hover tooltip shows all three numbers
  + timestamp. Reuses the SVG idioms from `Charts.tsx` (axis, gridlines, tnum labels).
- Placed as a full-width section above the existing trends row.

### 3. "Turn one cashier into X" — `CashierMultiplier.tsx` (new)
- Hero stat card. **X = `other_avg / mashgin_avg_eff`** (how many staffed lanes one
  Mashgin kiosk's throughput replaces), computed client-side. Display big: **"1 kiosk = X
  cashiers"** in the Mashgin lime accent.
- Supporting line: per-customer seconds at Mashgin (`mashgin_avg_eff`) vs staffed
  (`other_avg`), plus the existing `delta` ("Xs saved per customer").
- Empty/degenerate guard: if `other_n === 0` or `mashgin_avg_eff <= 0`, show
  "awaiting staffed-lane data" instead of a divide-by-zero / Infinity.

### 4. View slider — `ViewSlider.tsx` (new) + `lib/useViewMode.ts`
- Segmented control pinned at the top of the stats area: **Mashgin | Non-Mashgin |
  Side-by-side**. Holds `viewMode` state lifted into `page.tsx`.
- Filters which checkout-oriented stats render (`CheckoutEdge`, `CashierMultiplier`,
  checkout series in charts): Mashgin-only, staffed-only, or two columns.
- **Disabled + tooltip** on Non-Mashgin and Side-by-side while `checkouts.other_n === 0`
  ("no staffed-lane data yet"); auto-enable when data appears. Mashgin is always enabled
  and is the default.

### 5. Averages graphs — `Charts.tsx` (light edit)
- Keep `OccupancyChart` (line, you chose keep-both), `WaitServeChart`, `ThroughputChart`.
- Restyle minimally for consistency with the new histogram + Mashgin accent. No new charts.

### 6. Per-person reasoning → in-UI debug toggle — `DebugTable.tsx` (edit) + `page.tsx`
- Data already streams every frame; **no backend change**. Convert from "render if rows
  exist" to a header-controlled toggle, **default OFF**.
- `page.tsx` owns a `showDebug` boolean (default `false`); Header renders the toggle;
  `DebugTable` renders only when `showDebug && rows.length`.
- Optional: if launched with `--debug`, the toggle may start expanded (nice-to-have, not
  required; the flag no longer governs data flow).

### 7. Time comparison — `TimeCompare.tsx` (new) + `lib/compareWindows.ts`
- Compact "Now vs Earlier" card. Define **recent window** = last ~120s of each series;
  **baseline** = session-so-far (or the prior equal-length window if enough data).
- Compare three figures with up/down delta arrows + color: **busyness** (mean
  `busy.value` recent vs baseline), **throughput** (mean `served_per_min`), **avg total
  time** (from `wait_serve` / `summary`). Pure client-side derivation from existing series.

## Layout (page.tsx, top → bottom)

1. Header (Mashgin brand + Store Busyness + timer/connection + debug toggle)
2. **ViewSlider** (Mashgin / Non-Mashgin / Side-by-side)
3. Hero row: LiveFeed | (RightNow + CashierMultiplier + CheckoutEdge, view-filtered)
4. **Occupancy histogram** (full width)
5. **TimeCompare** (Now vs Earlier) + StatStrip
6. Trends row: OccupancyChart · WaitServeChart · ThroughputChart
7. DebugTable (only when toggle ON)

## Data flow

Unchanged backend → `/metrics` JSON (1Hz) → `useMetrics` → `page.tsx` holds `viewMode` +
`showDebug` state → passes `metrics` + view state to components. All new metrics
(multiplier, histogram buckets, time comparison) are **pure client-side derivations** of
the existing payload, kept in small `lib/` helpers so they're unit-testable in isolation.

## Testing

- `lib/` derivation helpers (`cashierMultiplier`, `histogramBuckets`, `compareWindows`)
  are pure functions with unit tests (edge cases: empty series, `other_n===0`,
  single-sample, divide-by-zero).
- `npm run build` (typecheck + static export) must pass — the Python server serves
  `web/out/`.
- Manual smoke: run `uv run python main.py --source <clip> --busy` and visually confirm
  the histogram, slider disable/enable, multiplier, and debug toggle behave.

## Out of scope

- No backend/Python data changes (data already sufficient).
- No new tracking/metrics (e.g. a true total-in-frame count) — "in store" = line + POS.
- No dark-mode rebrand; the dashboard stays light, Mashgin mark used as accent.
- No changes to the live video overlay / stream pipeline.

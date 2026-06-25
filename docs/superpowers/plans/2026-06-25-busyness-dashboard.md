# Store Busyness Dashboard Refactor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the storePose dashboard into a clean "Store Busyness" view: a per-timestamp occupancy histogram, a "Turn one cashier into N" hero metric, a Now-vs-Earlier time comparison, the existing average graphs, a default-off per-person debug toggle, and a top slider switching Mashgin / Non-Mashgin / Side-by-side stats — with Mashgin branding on the current light aesthetic.

**Architecture:** Frontend-only change in `web/` (Next.js 15 / React 19 / Tailwind v4). All new figures are pure client-side derivations of the existing `/metrics` payload, isolated in unit-tested `lib/` helpers. New presentational components consume those helpers. Backend is untouched (per-person debug rows already stream every frame).

**Tech Stack:** Next.js 15, React 19, TypeScript 5.7, Tailwind v4, hand-rolled SVG charts, Vitest (added for the pure helpers).

## Global Constraints

- All work under `web/`. No Python/backend changes.
- Preserve the light warm-neutral aesthetic: bg `#fafaf9`, white cards, 14px radius, hairline `#e9e9e6`, Geist fonts. Reuse `Card`/`CardTitle` from `components/ui.tsx`, `.tnum`/`.eyebrow` classes, and the `fmt*` helpers in `lib/format.ts`.
- Mashgin accent: add `--color-mashgin: #E4F222` and a text-readable `--color-mashgin-ink: #7e8a10` to `globals.css`; use them only for Mashgin-specific data.
- The cashier card must render the literal phrase **"Turn one cashier into N"** where `N = Math.max(1, Math.ceil(other_avg / mashgin_avg_eff))` (rounded up).
- "In store" = `in_line + at_pos` (no new tracking).
- Slider modes Non-Mashgin and Side-by-side are disabled (with tooltip) until `checkouts.other_n > 0`.
- Debug view default OFF.
- No emojis anywhere (commits, code, UI copy).
- `npm run build` must pass at the end (the Python server serves `web/out/`).
- ViewMode type is `"mashgin" | "other" | "both"` everywhere.

---

### Task 1: Add Vitest test tooling

**Files:**
- Modify: `web/package.json`
- Create: `web/vitest.config.ts`
- Create: `web/lib/__tests__/smoke.test.ts`

**Interfaces:**
- Produces: a working `npm test` (and `npm run test:run`) command for all later `lib/` tasks.

- [ ] **Step 1: Install Vitest as a dev dependency**

Run: `cd web && npm install -D vitest@^2.1.0`
Expected: adds vitest to devDependencies, exits 0.

- [ ] **Step 2: Add test scripts to `web/package.json`**

In the `"scripts"` block, add:

```json
    "test": "vitest",
    "test:run": "vitest run"
```

- [ ] **Step 3: Create `web/vitest.config.ts`**

```ts
import { defineConfig } from "vitest/config";
import path from "node:path";

export default defineConfig({
  resolve: {
    alias: { "@": path.resolve(__dirname, ".") },
  },
  test: {
    environment: "node",
    include: ["lib/**/*.test.ts"],
  },
});
```

- [ ] **Step 4: Create `web/lib/__tests__/smoke.test.ts`**

```ts
import { describe, it, expect } from "vitest";

describe("vitest", () => {
  it("runs", () => {
    expect(1 + 1).toBe(2);
  });
});
```

- [ ] **Step 5: Run the test suite**

Run: `cd web && npm run test:run`
Expected: PASS, 1 test passed.

- [ ] **Step 6: Commit**

```bash
git add web/package.json web/package-lock.json web/vitest.config.ts web/lib/__tests__/smoke.test.ts
git commit -m "test(dashboard): add vitest for pure lib helpers"
```

---

### Task 2: `lib/cashierMultiplier.ts` — "Turn one cashier into N"

**Files:**
- Create: `web/lib/cashierMultiplier.ts`
- Test: `web/lib/cashierMultiplier.test.ts`

**Interfaces:**
- Consumes: `Metrics["checkouts"]` from `web/lib/types.ts`.
- Produces:
  ```ts
  export interface CashierMultiplier {
    available: boolean; // false when staffed-lane data is missing / degenerate
    n: number;          // Math.max(1, Math.ceil(ratio)); 0 when unavailable
    ratio: number;      // other_avg / mashgin_avg_eff (unrounded); 0 when unavailable
    mashEff: number;    // mashgin_avg_eff
    other: number;      // other_avg
    delta: number;      // checkouts.delta (seconds saved per customer)
  }
  export function cashierMultiplier(c: Metrics["checkouts"] | undefined): CashierMultiplier
  ```

- [ ] **Step 1: Write the failing test**

```ts
import { describe, it, expect } from "vitest";
import { cashierMultiplier } from "./cashierMultiplier";

const base = {
  mashgin_avg: 10, mashgin_avg_eff: 5, num_mashgins: 2, mashgin_n: 8,
  other_avg: 18, other_n: 6, delta: 13,
  series: { t_mashgin: [], mashgin_ma: [], t_other: [], other_ma: [] },
};

describe("cashierMultiplier", () => {
  it("rounds the throughput ratio up", () => {
    const r = cashierMultiplier(base); // 18 / 5 = 3.6 -> 4
    expect(r.available).toBe(true);
    expect(r.ratio).toBeCloseTo(3.6);
    expect(r.n).toBe(4);
  });

  it("is unavailable with no staffed-lane data", () => {
    const r = cashierMultiplier({ ...base, other_n: 0, other_avg: 0 });
    expect(r.available).toBe(false);
    expect(r.n).toBe(0);
  });

  it("is unavailable when mashgin effective time is zero (no divide-by-zero)", () => {
    const r = cashierMultiplier({ ...base, mashgin_avg_eff: 0, mashgin_n: 0 });
    expect(r.available).toBe(false);
    expect(Number.isFinite(r.n)).toBe(true);
  });

  it("never reports fewer than 1 when available", () => {
    const r = cashierMultiplier({ ...base, other_avg: 4, mashgin_avg_eff: 5 }); // 0.8 -> ceil 1
    expect(r.n).toBe(1);
  });

  it("returns unavailable for undefined input", () => {
    expect(cashierMultiplier(undefined).available).toBe(false);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd web && npx vitest run lib/cashierMultiplier.test.ts`
Expected: FAIL — cannot find module `./cashierMultiplier`.

- [ ] **Step 3: Write the implementation**

```ts
import type { Metrics } from "./types";

export interface CashierMultiplier {
  available: boolean;
  n: number;
  ratio: number;
  mashEff: number;
  other: number;
  delta: number;
}

/**
 * How many staffed cashiers one Mashgin kiosk replaces, by throughput:
 * N = ceil(other_avg / mashgin_avg_eff), rounded up, never below 1.
 * Unavailable until both lanes have completed checkouts.
 */
export function cashierMultiplier(c: Metrics["checkouts"] | undefined): CashierMultiplier {
  const mashEff = c?.mashgin_avg_eff ?? 0;
  const other = c?.other_avg ?? 0;
  const hasBoth = (c?.mashgin_n ?? 0) > 0 && (c?.other_n ?? 0) > 0;
  const available = hasBoth && mashEff > 0 && other > 0;
  const ratio = available ? other / mashEff : 0;
  const n = available ? Math.max(1, Math.ceil(ratio)) : 0;
  return { available, n, ratio, mashEff, other, delta: c?.delta ?? 0 };
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd web && npx vitest run lib/cashierMultiplier.test.ts`
Expected: PASS, 5 tests.

- [ ] **Step 5: Commit**

```bash
git add web/lib/cashierMultiplier.ts web/lib/cashierMultiplier.test.ts
git commit -m "feat(dashboard): cashier multiplier derivation (ceil ratio)"
```

---

### Task 3: `lib/histogramBuckets.ts` — occupancy stacked-bar buckets

**Files:**
- Create: `web/lib/histogramBuckets.ts`
- Test: `web/lib/histogramBuckets.test.ts`

**Interfaces:**
- Consumes: `Metrics["occupancy"]` (`t`, `waiting`, `serving`).
- Produces:
  ```ts
  export interface OccBar { t: number; line: number; pos: number; total: number }
  export function histogramBuckets(
    occ: Metrics["occupancy"] | undefined, maxBars?: number
  ): OccBar[]   // line=avg waiting (rounded), pos=avg serving (rounded), total=line+pos
  ```

- [ ] **Step 1: Write the failing test**

```ts
import { describe, it, expect } from "vitest";
import { histogramBuckets } from "./histogramBuckets";

const occ = (n: number) => ({
  t: Array.from({ length: n }, (_, i) => i),
  waiting: Array.from({ length: n }, () => 2),
  serving: Array.from({ length: n }, () => 1),
  waiting_ma: [], serving_ma: [],
});

describe("histogramBuckets", () => {
  it("returns one bar per sample when under the cap", () => {
    const bars = histogramBuckets(occ(5), 48);
    expect(bars).toHaveLength(5);
    expect(bars[0]).toEqual({ t: 0, line: 2, pos: 1, total: 3 });
  });

  it("downsamples to at most maxBars", () => {
    const bars = histogramBuckets(occ(500), 48);
    expect(bars.length).toBeLessThanOrEqual(48);
    expect(bars.length).toBeGreaterThan(0);
  });

  it("totals line + pos and rounds bucket averages", () => {
    const bars = histogramBuckets(occ(100), 10);
    for (const b of bars) {
      expect(b.total).toBe(b.line + b.pos);
      expect(Number.isInteger(b.line)).toBe(true);
      expect(Number.isInteger(b.pos)).toBe(true);
    }
  });

  it("handles empty / undefined", () => {
    expect(histogramBuckets(undefined)).toEqual([]);
    expect(histogramBuckets(occ(0))).toEqual([]);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd web && npx vitest run lib/histogramBuckets.test.ts`
Expected: FAIL — cannot find module.

- [ ] **Step 3: Write the implementation**

```ts
import type { Metrics } from "./types";

export interface OccBar {
  t: number;
  line: number; // people waiting in line
  pos: number;  // people at POS / checkout
  total: number; // line + pos = total in store
}

/**
 * Bucket the raw occupancy samples into at most `maxBars` stacked bars so the
 * histogram stays readable. Each bucket averages its samples (rounded) and is
 * timestamped at its first sample. Below the cap, returns one bar per sample.
 */
export function histogramBuckets(
  occ: Metrics["occupancy"] | undefined,
  maxBars = 48,
): OccBar[] {
  if (!occ || occ.t.length === 0) return [];
  const n = occ.t.length;
  const bucketCount = Math.min(maxBars, n);
  const size = Math.ceil(n / bucketCount);
  const bars: OccBar[] = [];
  for (let start = 0; start < n; start += size) {
    const end = Math.min(start + size, n);
    let w = 0;
    let s = 0;
    for (let i = start; i < end; i++) {
      w += occ.waiting[i];
      s += occ.serving[i];
    }
    const span = end - start;
    const line = Math.round(w / span);
    const pos = Math.round(s / span);
    bars.push({ t: occ.t[start], line, pos, total: line + pos });
  }
  return bars;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd web && npx vitest run lib/histogramBuckets.test.ts`
Expected: PASS, 4 tests.

- [ ] **Step 5: Commit**

```bash
git add web/lib/histogramBuckets.ts web/lib/histogramBuckets.test.ts
git commit -m "feat(dashboard): occupancy histogram bucketing"
```

---

### Task 4: `lib/compareWindows.ts` — Now vs Earlier

**Files:**
- Create: `web/lib/compareWindows.ts`
- Test: `web/lib/compareWindows.test.ts`

**Interfaces:**
- Consumes: `Metrics` (`now`, `busy`, `throughput`, `wait_serve`).
- Produces:
  ```ts
  export type Dir = "up" | "down" | "flat";
  export interface CompareRow {
    key: "busy" | "throughput" | "total";
    label: string;
    recent: number; baseline: number; deltaPct: number; dir: Dir; available: boolean;
  }
  export function compareWindows(m: Metrics | null, windowS?: number): CompareRow[]
  ```

- [ ] **Step 1: Write the failing test**

```ts
import { describe, it, expect } from "vitest";
import { compareWindows } from "./compareWindows";

function metrics(now: number) {
  // busy.value: earlier ~1, recent ~3 (rising). throughput similar.
  const t = Array.from({ length: now + 1 }, (_, i) => i);
  const value = t.map((x) => (x >= now - 120 ? 3 : 1));
  const tp = t.map((x) => (x >= now - 120 ? 6 : 2));
  const wait = t.map((x) => (x >= now - 120 ? 4 : 2));
  const serve = t.map(() => 1);
  return {
    now, summary: {} as any,
    busy: { current: { level: "High", value: 3 }, t, level_idx: [], value },
    checkouts: {} as any,
    occupancy: { t: [], waiting: [], serving: [], waiting_ma: [], serving_ma: [] },
    wait_serve: { t, wait_ma: wait, serve_ma: serve },
    throughput: { t, served_per_min: tp },
    debug: { frame: null, rows: [] },
  } as any;
}

describe("compareWindows", () => {
  it("flags rising busyness and throughput as 'up'", () => {
    const rows = compareWindows(metrics(400), 120);
    const busy = rows.find((r) => r.key === "busy")!;
    expect(busy.available).toBe(true);
    expect(busy.recent).toBeGreaterThan(busy.baseline);
    expect(busy.dir).toBe("up");
    expect(rows.find((r) => r.key === "throughput")!.dir).toBe("up");
  });

  it("marks rows unavailable without enough history", () => {
    const rows = compareWindows(metrics(10), 120);
    expect(rows.every((r) => !r.available)).toBe(true);
  });

  it("returns three rows for null metrics, all unavailable", () => {
    const rows = compareWindows(null);
    expect(rows).toHaveLength(3);
    expect(rows.every((r) => !r.available)).toBe(true);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd web && npx vitest run lib/compareWindows.test.ts`
Expected: FAIL — cannot find module.

- [ ] **Step 3: Write the implementation**

```ts
import type { Metrics } from "./types";

export type Dir = "up" | "down" | "flat";
export interface CompareRow {
  key: "busy" | "throughput" | "total";
  label: string;
  recent: number;
  baseline: number;
  deltaPct: number;
  dir: Dir;
  available: boolean;
}

const FLAT_EPS = 0.05; // <5% change reads as flat

function mean(xs: number[]): number {
  return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0;
}

/** Split paired (t, value) arrays into [baseline (older), recent (last windowS)]. */
function split(t: number[], v: number[], now: number, windowS: number): [number[], number[]] {
  const cutoff = now - windowS;
  const older: number[] = [];
  const recent: number[] = [];
  for (let i = 0; i < t.length; i++) {
    (t[i] >= cutoff ? recent : older).push(v[i]);
  }
  return [older, recent];
}

function row(
  key: CompareRow["key"], label: string,
  t: number[], v: number[], now: number, windowS: number,
): CompareRow {
  const [older, recent] = split(t, v, now, windowS);
  const available = older.length >= 2 && recent.length >= 2;
  const r = mean(recent);
  const b = mean(older);
  const deltaPct = available && b !== 0 ? (r - b) / b : 0;
  const dir: Dir = !available || Math.abs(deltaPct) < FLAT_EPS ? "flat" : r > b ? "up" : "down";
  return { key, label, recent: r, baseline: b, deltaPct, dir, available };
}

/** Compare the last `windowS` seconds against the session-so-far baseline. */
export function compareWindows(m: Metrics | null, windowS = 120): CompareRow[] {
  if (!m) {
    return [
      { key: "busy", label: "Busyness", recent: 0, baseline: 0, deltaPct: 0, dir: "flat", available: false },
      { key: "throughput", label: "Throughput", recent: 0, baseline: 0, deltaPct: 0, dir: "flat", available: false },
      { key: "total", label: "Time in store", recent: 0, baseline: 0, deltaPct: 0, dir: "flat", available: false },
    ];
  }
  const now = m.now;
  const ws = m.wait_serve;
  const totalV = ws.t.map((_, i) => (ws.wait_ma[i] ?? 0) + (ws.serve_ma[i] ?? 0));
  return [
    row("busy", "Busyness", m.busy.t, m.busy.value, now, windowS),
    row("throughput", "Throughput", m.throughput.t, m.throughput.served_per_min, now, windowS),
    row("total", "Time in store", ws.t, totalV, now, windowS),
  ];
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd web && npx vitest run lib/compareWindows.test.ts`
Expected: PASS, 3 tests.

- [ ] **Step 5: Commit**

```bash
git add web/lib/compareWindows.ts web/lib/compareWindows.test.ts
git commit -m "feat(dashboard): now-vs-earlier window comparison"
```

---

### Task 5: View-mode type + design tokens

**Files:**
- Create: `web/lib/viewMode.ts`
- Modify: `web/app/globals.css` (add Mashgin tokens)

**Interfaces:**
- Produces:
  ```ts
  export type ViewMode = "mashgin" | "other" | "both";
  export const VIEW_MODES: { key: ViewMode; label: string }[]
  ```
  CSS vars `--color-mashgin`, `--color-mashgin-ink`.

- [ ] **Step 1: Create `web/lib/viewMode.ts`**

```ts
export type ViewMode = "mashgin" | "other" | "both";

export const VIEW_MODES: { key: ViewMode; label: string }[] = [
  { key: "mashgin", label: "Mashgin" },
  { key: "other", label: "Non-Mashgin" },
  { key: "both", label: "Side-by-side" },
];

/** Whether a mode is selectable given staffed-lane data availability. */
export function modeEnabled(mode: ViewMode, hasOther: boolean): boolean {
  return mode === "mashgin" ? true : hasOther;
}
```

- [ ] **Step 2: Add Mashgin tokens to `web/app/globals.css`**

Inside the `@theme { ... }` block, after `--color-accent: #15803d;`, add:

```css
  --color-mashgin: #E4F222;       /* Mashgin brand lime — accent for Mashgin data */
  --color-mashgin-ink: #7e8a10;   /* readable lime for text/fills on the light theme */
  --color-mashgin-wash: #f7fbcf;
```

- [ ] **Step 3: Verify the build still compiles tokens**

Run: `cd web && npx tsc --noEmit`
Expected: PASS (no type errors).

- [ ] **Step 4: Commit**

```bash
git add web/lib/viewMode.ts web/app/globals.css
git commit -m "feat(dashboard): view-mode type + Mashgin brand tokens"
```

---

### Task 6: Branding — Header rebrand + favicon/metadata

**Files:**
- Modify: `web/components/Header.tsx`
- Modify: `web/app/layout.tsx`
- (assets already at `web/public/mashgin-logo.png`, `web/public/mashgin-favicon.png`)

**Interfaces:**
- Consumes: `web/public/mashgin-logo.png`.
- Produces: `Header` accepts `showDebug` + `onToggleDebug` props (consumed by Task 11/12).
  ```ts
  function Header(props: { connected: boolean; now: number; showDebug: boolean; onToggleDebug: () => void }): JSX.Element
  ```

- [ ] **Step 1: Replace `web/components/Header.tsx`**

```tsx
import Image from "next/image";
import mashginLogo from "@/public/mashgin-logo.png";

export function Header({
  connected,
  now,
  showDebug,
  onToggleDebug,
}: {
  connected: boolean;
  now: number;
  showDebug: boolean;
  onToggleDebug: () => void;
}) {
  // process clock (seconds since start) -> m:ss, an honest "session time"
  const m = Math.floor(now / 60);
  const s = Math.floor(now % 60);
  return (
    <header className="flex items-end justify-between border-b border-hairline pb-5">
      <div>
        <div className="flex items-baseline gap-2.5">
          <Image
            src={mashginLogo}
            alt="Mashgin"
            height={22}
            className="mr-1 h-[22px] w-auto self-center"
            priority
          />
          <span className="text-[1.35rem] font-semibold tracking-[-0.02em] text-ink">
            Store Busyness
          </span>
          <span className="h-[15px] w-px self-center bg-hairline-strong" />
          <span className="text-[0.86rem] font-medium text-muted">storePose · Line Monitor</span>
        </div>
        <p className="mt-1 text-[0.8rem] text-faint">
          Live checkout-line analytics — occupancy, wait times, and Mashgin vs staffed speed.
        </p>
      </div>
      <div className="flex items-center gap-4 text-[0.78rem]">
        <button
          type="button"
          onClick={onToggleDebug}
          aria-pressed={showDebug}
          className="rounded-full border border-hairline px-3 py-[5px] font-medium text-muted transition-colors hover:bg-sunken"
          style={showDebug ? { color: "var(--color-ink)", background: "var(--color-sunken)" } : undefined}
        >
          Debug {showDebug ? "on" : "off"}
        </button>
        <span className="tnum text-faint">
          session {m}:{s.toString().padStart(2, "0")}
        </span>
        <span className="flex items-center gap-2 font-medium text-muted">
          <span
            className="inline-block h-[7px] w-[7px] rounded-full"
            style={{
              background: connected ? "#15803d" : "#9b9ba1",
              animation: connected ? "livepulse 2.4s infinite" : "none",
            }}
          />
          {connected ? "Connected" : "Offline"}
        </span>
      </div>
    </header>
  );
}
```

- [ ] **Step 2: Update favicon + title in `web/app/layout.tsx`**

Replace the `metadata` export with:

```tsx
export const metadata: Metadata = {
  title: "Mashgin · Store Busyness",
  description: "Live checkout-line analytics — occupancy, wait times, and Mashgin vs staffed speed.",
  icons: { icon: "/mashgin-favicon.png" },
};
```

- [ ] **Step 3: Verify Next can resolve the static image import**

Run: `cd web && npx tsc --noEmit`
Expected: PASS. (If `Image` import flags missing types, they ship with `next`.)

Note: `Header` now requires `showDebug`/`onToggleDebug`; `page.tsx` is updated in Task 12. The build is verified end-to-end in Task 13, so a transient type error in `page.tsx` here is expected until Task 12.

- [ ] **Step 4: Commit**

```bash
git add web/components/Header.tsx web/app/layout.tsx
git commit -m "feat(dashboard): Mashgin co-branding + debug toggle control in header"
```

---

### Task 7: `ViewSlider.tsx` — Mashgin / Non-Mashgin / Side-by-side

**Files:**
- Create: `web/components/ViewSlider.tsx`

**Interfaces:**
- Consumes: `ViewMode`, `VIEW_MODES`, `modeEnabled` from `lib/viewMode.ts`.
- Produces:
  ```ts
  function ViewSlider(props: { mode: ViewMode; onChange: (m: ViewMode) => void; hasOther: boolean }): JSX.Element
  ```

- [ ] **Step 1: Create `web/components/ViewSlider.tsx`**

```tsx
import type { ViewMode } from "@/lib/viewMode";
import { VIEW_MODES, modeEnabled } from "@/lib/viewMode";

/** Segmented control switching which checkout stats are shown. Non-Mashgin and
 *  Side-by-side stay disabled (with a tooltip) until staffed-lane data exists. */
export function ViewSlider({
  mode,
  onChange,
  hasOther,
}: {
  mode: ViewMode;
  onChange: (m: ViewMode) => void;
  hasOther: boolean;
}) {
  return (
    <div className="inline-flex items-center gap-1 rounded-full border border-hairline bg-panel p-1 shadow-[var(--shadow-card)]">
      {VIEW_MODES.map((vm) => {
        const enabled = modeEnabled(vm.key, hasOther);
        const active = mode === vm.key;
        return (
          <button
            key={vm.key}
            type="button"
            disabled={!enabled}
            title={enabled ? undefined : "No staffed-lane data yet"}
            onClick={() => enabled && onChange(vm.key)}
            aria-pressed={active}
            className="rounded-full px-3.5 py-[5px] text-[0.78rem] font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40"
            style={
              active
                ? { background: "var(--color-ink)", color: "var(--color-panel)" }
                : { color: "var(--color-muted)" }
            }
          >
            {vm.label}
          </button>
        );
      })}
    </div>
  );
}
```

- [ ] **Step 2: Verify it typechecks**

Run: `cd web && npx tsc --noEmit`
Expected: PASS for this file (page.tsx may still error until Task 12).

- [ ] **Step 3: Commit**

```bash
git add web/components/ViewSlider.tsx
git commit -m "feat(dashboard): view-mode segmented slider"
```

---

### Task 8: `Histogram.tsx` — occupancy stacked-bar centerpiece

**Files:**
- Create: `web/components/Histogram.tsx`

**Interfaces:**
- Consumes: `Metrics`, `histogramBuckets`/`OccBar` from `lib/histogramBuckets.ts`, `Card`/`CardTitle`, `fmtInt`.
- Produces: `function Histogram(props: { m: Metrics | null }): JSX.Element`

- [ ] **Step 1: Create `web/components/Histogram.tsx`**

```tsx
"use client";

import { useState } from "react";
import type { Metrics } from "@/lib/types";
import { histogramBuckets } from "@/lib/histogramBuckets";
import { fmtInt } from "@/lib/format";
import { Card, CardTitle } from "./ui";

const W = 1000;
const H = 220;
const PAD = 12;

export function Histogram({ m }: { m: Metrics | null }) {
  const [hover, setHover] = useState<number | null>(null);
  const bars = histogramBuckets(m?.occupancy, 56);
  const has = bars.length > 0;
  const max = has ? Math.max(...bars.map((b) => b.total), 1) : 1;
  const now = has ? bars[bars.length - 1] : null;
  const slot = has ? W / bars.length : W;
  const bw = slot * 0.7;

  const sel = hover != null ? bars[hover] : now;

  return (
    <Card className="p-5">
      <div className="mb-3 flex items-start justify-between">
        <CardTitle>People in store over time</CardTitle>
        <div className="flex gap-4">
          <Legend label="At line" color="var(--color-wait)" />
          <Legend label="At POS" color="var(--color-mashgin-ink)" />
        </div>
      </div>

      {has ? (
        <>
          <div className="mb-2 flex items-baseline gap-3">
            <span className="tnum text-[1.7rem] font-semibold leading-none text-ink">
              {fmtInt(sel?.total ?? 0)}
            </span>
            <span className="text-[0.8rem] text-faint">
              in store{hover != null ? " (hover)" : " now"} · {fmtInt(sel?.line ?? 0)} at line ·{" "}
              {fmtInt(sel?.pos ?? 0)} at POS
            </span>
          </div>

          <svg
            className="w-full"
            viewBox={`0 0 ${W} ${H}`}
            preserveAspectRatio="none"
            style={{ height: 220 }}
          >
            <line
              x1={0} y1={H - PAD} x2={W} y2={H - PAD}
              stroke="var(--color-hairline)" strokeWidth={1} vectorEffect="non-scaling-stroke"
            />
            {bars.map((b, i) => {
              const x = i * slot + (slot - bw) / 2;
              const usable = H - 2 * PAD;
              const lineH = (b.line / max) * usable;
              const posH = (b.pos / max) * usable;
              const active = hover === i;
              return (
                <g
                  key={i}
                  onMouseEnter={() => setHover(i)}
                  onMouseLeave={() => setHover(null)}
                  style={{ cursor: "pointer" }}
                >
                  {/* hover hit-area */}
                  <rect x={i * slot} y={0} width={slot} height={H} fill="transparent" />
                  {/* at line (bottom) */}
                  <rect
                    x={x} y={H - PAD - lineH} width={bw} height={lineH} rx={1.5}
                    fill="var(--color-wait)" opacity={active ? 1 : 0.85}
                  />
                  {/* at POS (stacked on top) */}
                  <rect
                    x={x} y={H - PAD - lineH - posH} width={bw} height={posH} rx={1.5}
                    fill="var(--color-mashgin-ink)" opacity={active ? 1 : 0.85}
                  />
                </g>
              );
            })}
          </svg>
        </>
      ) : (
        <div className="grid h-[220px] place-items-center text-[0.8rem] text-faint">
          Gathering occupancy…
        </div>
      )}
    </Card>
  );
}

function Legend({ label, color }: { label: string; color: string }) {
  return (
    <span className="flex items-center gap-1.5 text-[0.72rem] text-muted">
      <span className="inline-block h-[8px] w-[8px] rounded-[3px]" style={{ background: color }} />
      {label}
    </span>
  );
}
```

- [ ] **Step 2: Verify it typechecks**

Run: `cd web && npx tsc --noEmit`
Expected: PASS for this file.

- [ ] **Step 3: Commit**

```bash
git add web/components/Histogram.tsx
git commit -m "feat(dashboard): occupancy stacked-bar histogram (line + POS = in store)"
```

---

### Task 9: `CashierMultiplier.tsx` — hero "Turn one cashier into N"

**Files:**
- Create: `web/components/CashierMultiplier.tsx`

**Interfaces:**
- Consumes: `Metrics`, `cashierMultiplier` from `lib/cashierMultiplier.ts`, `Card`/`CardTitle`, `fmtSeconds`.
- Produces: `function CashierMultiplier(props: { m: Metrics | null }): JSX.Element`

- [ ] **Step 1: Create `web/components/CashierMultiplier.tsx`**

```tsx
import type { Metrics } from "@/lib/types";
import { cashierMultiplier } from "@/lib/cashierMultiplier";
import { fmtSeconds } from "@/lib/format";
import { Card, CardTitle } from "./ui";

/** Hero metric: "Turn one cashier into N" — N staffed lanes one kiosk replaces. */
export function CashierMultiplier({ m }: { m: Metrics | null }) {
  const r = cashierMultiplier(m?.checkouts);

  return (
    <Card className="p-5">
      <CardTitle>Mashgin throughput</CardTitle>
      {r.available ? (
        <>
          <div className="text-[1.5rem] font-semibold leading-tight tracking-[-0.01em] text-ink">
            Turn one cashier into{" "}
            <span className="tnum text-[2.3rem]" style={{ color: "var(--color-mashgin-ink)" }}>
              {r.n}
            </span>
          </div>
          <p className="mt-1.5 text-[0.78rem] text-faint">
            One Mashgin kiosk matches the throughput of {r.n} staffed{" "}
            {r.n === 1 ? "lane" : "lanes"}{" "}
            <span className="tnum">({r.ratio.toFixed(1)}× per customer)</span>
          </p>
          <div className="mt-4 grid grid-cols-2 gap-4 border-t border-hairline pt-3 text-[0.78rem]">
            <Stat label="Mashgin / customer" value={fmtSeconds(r.mashEff)} accent />
            <Stat label="Staffed / customer" value={fmtSeconds(r.other)} />
          </div>
          {r.delta > 0 && (
            <p className="mt-3 text-[0.76rem] text-muted">
              <span className="tnum font-medium text-ink">{fmtSeconds(r.delta)}</span> saved per customer
            </p>
          )}
        </>
      ) : (
        <p className="py-4 text-[0.82rem] text-faint">
          Awaiting staffed-lane checkouts to compute the multiplier.
        </p>
      )}
    </Card>
  );
}

function Stat({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div>
      <div
        className="tnum text-[1.3rem] font-semibold leading-none text-ink"
        style={accent ? { color: "var(--color-mashgin-ink)" } : undefined}
      >
        {value}
      </div>
      <div className="mt-1.5 text-[0.74rem] text-muted">{label}</div>
    </div>
  );
}
```

- [ ] **Step 2: Verify it typechecks**

Run: `cd web && npx tsc --noEmit`
Expected: PASS for this file.

- [ ] **Step 3: Commit**

```bash
git add web/components/CashierMultiplier.tsx
git commit -m "feat(dashboard): 'Turn one cashier into N' hero metric"
```

---

### Task 10: `TimeCompare.tsx` — Now vs Earlier

**Files:**
- Create: `web/components/TimeCompare.tsx`

**Interfaces:**
- Consumes: `Metrics`, `compareWindows`/`CompareRow`/`Dir` from `lib/compareWindows.ts`, `Card`/`CardTitle`.
- Produces: `function TimeCompare(props: { m: Metrics | null }): JSX.Element`

- [ ] **Step 1: Create `web/components/TimeCompare.tsx`**

```tsx
import type { Metrics } from "@/lib/types";
import { compareWindows, type CompareRow, type Dir } from "@/lib/compareWindows";
import { Card, CardTitle } from "./ui";

const ARROW: Record<Dir, string> = { up: "▲", down: "▼", flat: "—" };

/** "Busyness up", "throughput steady" — last ~2 min vs the session baseline. */
export function TimeCompare({ m }: { m: Metrics | null }) {
  const rows = compareWindows(m, 120);
  return (
    <Card className="p-5">
      <div className="mb-3 flex items-center justify-between">
        <CardTitle>Now vs earlier</CardTitle>
        <span className="text-[0.72rem] text-faint">last 2 min vs session</span>
      </div>
      <div className="grid grid-cols-3 divide-x divide-hairline">
        {rows.map((r) => (
          <Row key={r.key} r={r} />
        ))}
      </div>
    </Card>
  );
}

function Row({ r }: { r: CompareRow }) {
  const color =
    r.dir === "flat" ? "var(--color-faint)" : r.dir === "up" ? "var(--color-busy)" : "var(--color-go)";
  const pct = Math.round(Math.abs(r.deltaPct) * 100);
  return (
    <div className="px-4 first:pl-0 last:pr-0">
      <div className="text-[0.72rem] text-muted">{r.label}</div>
      {r.available ? (
        <div className="mt-1 flex items-baseline gap-1.5">
          <span className="text-[1.2rem] font-semibold leading-none" style={{ color }}>
            {ARROW[r.dir]}
          </span>
          <span className="tnum text-[1.1rem] font-semibold text-ink">{pct}%</span>
        </div>
      ) : (
        <div className="mt-1 text-[0.86rem] text-faint">—</div>
      )}
    </div>
  );
}
```

Note: "up" in busyness/time-in-store is rendered red (worse) and "down" green (better); throughput shares the same mapping intentionally — rising throughput shows red here only as a neutral "changed" cue. If reviewers want throughput-up to read positive, invert `color` for `r.key === "throughput"`. Keep the simple shared mapping unless asked.

- [ ] **Step 2: Verify it typechecks**

Run: `cd web && npx tsc --noEmit`
Expected: PASS for this file.

- [ ] **Step 3: Commit**

```bash
git add web/components/TimeCompare.tsx
git commit -m "feat(dashboard): now-vs-earlier comparison card"
```

---

### Task 11: DebugTable → toggle-driven (default off)

**Files:**
- Modify: `web/components/DebugTable.tsx`

**Interfaces:**
- Produces: `DebugTable` accepts a `show` prop; renders nothing unless `show && rows.length`.
  ```ts
  function DebugTable(props: { m: Metrics | null; show: boolean }): JSX.Element | null
  ```

- [ ] **Step 1: Update the `DebugTable` signature and guard in `web/components/DebugTable.tsx`**

Replace the component signature/guard (lines 11-14) with:

```tsx
/** Per-person reasoning — gated behind the header Debug toggle (default off). */
export function DebugTable({ m, show }: { m: Metrics | null; show: boolean }) {
  const rows = m?.debug.rows ?? [];
  if (!show || !rows.length) return null;
```

Leave the rest of the component (the `Card`, table, `Th`/`Td`/`Dot` helpers) unchanged.

- [ ] **Step 2: Verify it typechecks**

Run: `cd web && npx tsc --noEmit`
Expected: PASS for this file (page.tsx updated next in Task 12).

- [ ] **Step 3: Commit**

```bash
git add web/components/DebugTable.tsx
git commit -m "feat(dashboard): gate per-person debug table behind toggle (default off)"
```

---

### Task 12: Wire everything in `page.tsx`

**Files:**
- Modify: `web/app/page.tsx`

**Interfaces:**
- Consumes: all new components + `ViewMode` state. `CheckoutEdge`/`RightNow`/`CashierMultiplier` are shown per `viewMode`.

- [ ] **Step 1: Replace `web/app/page.tsx`**

```tsx
"use client";

import { useState } from "react";
import { useMetrics } from "@/lib/useMetrics";
import type { ViewMode } from "@/lib/viewMode";
import { modeEnabled } from "@/lib/viewMode";
import { Header } from "@/components/Header";
import { ViewSlider } from "@/components/ViewSlider";
import { LiveFeed } from "@/components/LiveFeed";
import { RightNow, CheckoutEdge, StatStrip } from "@/components/Stats";
import { CashierMultiplier } from "@/components/CashierMultiplier";
import { Histogram } from "@/components/Histogram";
import { TimeCompare } from "@/components/TimeCompare";
import { OccupancyChart, WaitServeChart, ThroughputChart } from "@/components/Charts";
import { DebugTable } from "@/components/DebugTable";

export default function Page() {
  const { metrics, connected } = useMetrics();
  const [viewMode, setViewMode] = useState<ViewMode>("mashgin");
  const [showDebug, setShowDebug] = useState(false);

  const hasOther = (metrics?.checkouts.other_n ?? 0) > 0;
  // if staffed-lane data disappears, fall back to the always-valid Mashgin view
  const mode: ViewMode = modeEnabled(viewMode, hasOther) ? viewMode : "mashgin";

  const showMashgin = mode === "mashgin" || mode === "both";
  const showOther = mode === "other" || mode === "both";

  return (
    <main className="mx-auto max-w-[1280px] px-7 pb-20 pt-8">
      <Header
        connected={connected}
        now={metrics?.now ?? 0}
        showDebug={showDebug}
        onToggleDebug={() => setShowDebug((v) => !v)}
      />

      <div className="mt-6 flex items-center justify-between">
        <ViewSlider mode={mode} onChange={setViewMode} hasOther={hasOther} />
      </div>

      {/* hero: live feed + status rail */}
      <section className="mt-5 grid grid-cols-1 gap-5 lg:grid-cols-[1.85fr_1fr]">
        <LiveFeed />
        <div className="flex flex-col gap-5">
          <RightNow m={metrics} />
          {showMashgin && <CashierMultiplier m={metrics} />}
          {(showMashgin || showOther) && <CheckoutEdge m={metrics} />}
        </div>
      </section>

      {/* occupancy histogram centerpiece */}
      <div className="mt-5">
        <Histogram m={metrics} />
      </div>

      {/* now vs earlier + secondary averages */}
      <div className="mt-5 grid grid-cols-1 gap-5 lg:grid-cols-[1.4fr_1fr]">
        <TimeCompare m={metrics} />
        <StatStrip m={metrics} />
      </div>

      {/* trends */}
      <section className="mt-5 grid grid-cols-1 gap-5 md:grid-cols-2 xl:grid-cols-3">
        <OccupancyChart m={metrics} />
        <WaitServeChart m={metrics} />
        <ThroughputChart m={metrics} />
      </section>

      <div className="mt-5">
        <DebugTable m={metrics} show={showDebug} />
      </div>
    </main>
  );
}
```

- [ ] **Step 2: Typecheck the whole app**

Run: `cd web && npx tsc --noEmit`
Expected: PASS (all components now wired with matching props).

- [ ] **Step 3: Commit**

```bash
git add web/app/page.tsx
git commit -m "feat(dashboard): wire histogram, slider, multiplier, time-compare, debug toggle"
```

---

### Task 13: Full build, test, and smoke verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full unit suite**

Run: `cd web && npm run test:run`
Expected: PASS — all `lib/*.test.ts` green (cashierMultiplier, histogramBuckets, compareWindows, smoke).

- [ ] **Step 2: Production build (typecheck + static export)**

Run: `cd web && npm run build`
Expected: build succeeds, `web/out/` generated, no type errors.

- [ ] **Step 3: Manual smoke (if a sample clip is available)**

Run: `cd <repo-root> && uv run python main.py --source <clip> --busy`
Confirm in the browser at `http://127.0.0.1:8000/`:
- Mashgin mark + "Store Busyness" title render; favicon set.
- Occupancy histogram shows stacked bars (amber at line, lime at POS); hover updates the readout.
- Slider: Non-Mashgin / Side-by-side disabled with tooltip until staffed-lane checkouts appear, then enabled.
- "Turn one cashier into N" shows an integer (ceil); degenerate state shows the awaiting message.
- Debug toggle defaults off; toggling on reveals the per-person table.

If no clip is available, note that Steps 1-2 are the gating automated checks and Step 3 is observational.

- [ ] **Step 4: Commit any build-output/config adjustments (if generated)**

```bash
git add -A
git commit -m "chore(dashboard): build verification" --allow-empty
```

---

## Self-Review

**Spec coverage:**
- Histogram (total/line/POS) → Task 3 + Task 8.
- "Turn one cashier into N" (literal, ceil) → Task 2 + Task 9.
- Time comparison → Task 4 + Task 10.
- Averages graphs (kept) → unchanged Charts, re-rendered in Task 12.
- Per-person reasoning debug toggle (default off) → Task 11 + Task 6 (control) + Task 12 (state).
- Top slider Mashgin/Non-Mashgin/Side-by-side with disable+tooltip → Task 5 + Task 7 + Task 12.
- Mashgin logos + aesthetic → Task 5 (tokens) + Task 6 (header/favicon) + assets committed in spec.
- Worktree + commit + push → worktree active; push in finishing step after Task 13.

**Placeholder scan:** No TBD/TODO; every code step shows complete code.

**Type consistency:** `ViewMode` (`"mashgin"|"other"|"both"`), `cashierMultiplier`, `histogramBuckets`/`OccBar`, `compareWindows`/`CompareRow`/`Dir`, and the `Header`/`DebugTable`/`ViewSlider` prop shapes are defined once and consumed with matching names in Tasks 6-12.

**Note on intra-task build state:** Tasks 6, 7, 11 may leave `page.tsx` temporarily mismatched; this is expected and resolved in Task 12, with the full build gated in Task 13.

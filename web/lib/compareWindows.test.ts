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

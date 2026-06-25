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

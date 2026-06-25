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

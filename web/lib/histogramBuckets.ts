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

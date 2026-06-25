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

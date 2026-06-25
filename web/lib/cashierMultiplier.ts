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

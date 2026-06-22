import type { PersonState } from "./types";

/** Compact clock-style duration: 8s, 47s, 1:23, 12:05. */
export function fmtDuration(seconds: number): string {
  if (!isFinite(seconds) || seconds <= 0) return "0s";
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

/** One-decimal seconds for averages: "4.2s". */
export function fmtSeconds(seconds: number): string {
  if (!isFinite(seconds) || seconds <= 0) return "—";
  return `${seconds.toFixed(1)}s`;
}

export function fmtInt(n: number): string {
  return Math.round(n).toString();
}

/** Normalize a busy level (handles "—"/null/casing) to Low|Medium|High|null. */
export function normLevel(level: string | null | undefined): "Low" | "Medium" | "High" | null {
  if (!level) return null;
  const l = level.toLowerCase();
  if (l.startsWith("low")) return "Low";
  if (l.startsWith("med")) return "Medium";
  if (l.startsWith("high")) return "High";
  return null;
}

// Human-facing label for each overlay person state.
export const STATE_LABEL: Record<PersonState, string> = {
  tracked: "Tracked",
  candidate: "Joining",
  waiting: "Waiting",
  serving: "Checkout",
  serving_other: "Staffed lane",
  out: "Passing",
};

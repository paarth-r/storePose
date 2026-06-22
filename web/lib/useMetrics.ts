"use client";

import { useEffect, useState } from "react";
import type { Metrics } from "./types";

export interface MetricsState {
  metrics: Metrics | null;
  connected: boolean;
}

/** Poll `/metrics` once a second. Lightweight; charts read from this. */
export function useMetrics(intervalMs = 1000): MetricsState {
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    let alive = true;
    let timer: ReturnType<typeof setTimeout>;

    const tick = async () => {
      try {
        const res = await fetch("/metrics", { cache: "no-store" });
        const data = (await res.json()) as Metrics;
        if (!alive) return;
        setMetrics(data);
        setConnected(true);
      } catch {
        if (alive) setConnected(false);
      } finally {
        if (alive) timer = setTimeout(tick, intervalMs);
      }
    };
    tick();
    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, [intervalMs]);

  return { metrics, connected };
}

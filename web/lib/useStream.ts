"use client";

import { useEffect, useRef, useState } from "react";
import type { StreamEvent } from "./types";

export interface StreamState {
  event: StreamEvent | null;
  connected: boolean;
}

/**
 * Subscribe to the SSE annotated feed at `/stream`.
 *
 * Each event carries the JPEG frame and its overlay data together, so consumers
 * always render aligned data. Reconnects automatically (EventSource default).
 */
export function useStream(): StreamState {
  const [event, setEvent] = useState<StreamEvent | null>(null);
  const [connected, setConnected] = useState(false);
  const lastSeq = useRef(0);

  useEffect(() => {
    const es = new EventSource("/stream");
    es.onopen = () => setConnected(true);
    es.onerror = () => setConnected(false);
    es.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data) as StreamEvent;
        if (data.seq <= lastSeq.current) return;
        lastSeq.current = data.seq;
        setConnected(true);
        setEvent(data);
      } catch {
        /* ignore malformed frame */
      }
    };
    return () => es.close();
  }, []);

  return { event, connected };
}

"use client";

import { useEffect, useReducer, useRef } from "react";
import type { OverlayPerson, StreamEvent } from "./types";

/** A person whose box + keypoints are smoothly interpolated between frames. */
export type AnimatedPerson = OverlayPerson;

function clone(p: OverlayPerson): AnimatedPerson {
  return { ...p, box: [...p.box] as OverlayPerson["box"], kpts: p.kpts.map((k) => [...k] as [number, number, number]) };
}

/**
 * Interpolate person boxes/keypoints toward each new stream frame so motion is
 * smooth at the stream's ~12fps. New ids snap in; departed ids are dropped.
 * Honors `prefers-reduced-motion` by snapping instead of easing.
 */
export function useAnimatedOverlay(event: StreamEvent | null): AnimatedPerson[] {
  const animated = useRef<Map<number, AnimatedPerson>>(new Map());
  const targets = useRef<Map<number, OverlayPerson>>(new Map());
  const [, force] = useReducer((x: number) => x + 1, 0);

  useEffect(() => {
    if (!event) return;
    const next = new Map<number, OverlayPerson>();
    for (const p of event.people) next.set(p.id, p);
    targets.current = next;
    for (const p of event.people) {
      if (!animated.current.has(p.id)) animated.current.set(p.id, clone(p));
    }
    for (const id of [...animated.current.keys()]) {
      if (!next.has(id)) animated.current.delete(id);
    }
  }, [event]);

  useEffect(() => {
    const reduce =
      typeof window !== "undefined" &&
      window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    const k = reduce ? 1 : 0.32;
    let raf = 0;
    const step = () => {
      for (const [id, target] of targets.current) {
        const cur = animated.current.get(id);
        if (!cur) {
          animated.current.set(id, clone(target));
          continue;
        }
        cur.box = cur.box.map((v, i) => v + (target.box[i] - v) * k) as OverlayPerson["box"];
        if (target.kpts.length === cur.kpts.length) {
          cur.kpts = target.kpts.map((tp, i) => {
            const cp = cur.kpts[i];
            return [cp[0] + (tp[0] - cp[0]) * k, cp[1] + (tp[1] - cp[1]) * k, tp[2]] as [
              number,
              number,
              number,
            ];
          });
        } else {
          cur.kpts = target.kpts.map((p) => [...p] as [number, number, number]);
        }
        cur.state = target.state;
        cur.wait = target.wait;
        cur.serve = target.serve;
        cur.progress = target.progress;
      }
      force();
      raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, []);

  return [...animated.current.values()].sort((a, b) => a.id - b.id);
}

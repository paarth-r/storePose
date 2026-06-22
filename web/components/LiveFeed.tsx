"use client";

import { useEffect, useRef, useState } from "react";
import { useStream } from "@/lib/useStream";
import { useAnimatedOverlay } from "@/lib/useAnimatedOverlay";
import { SKELETON_EDGES, JOINT_INDICES, stateColor } from "@/lib/skeleton";
import { STATE_LABEL, fmtDuration, normLevel } from "@/lib/format";
import type { OverlayZones } from "@/lib/types";

const KPT_THR = 0.3;

const ZONE_STYLES: Record<keyof OverlayZones, { stroke: string; fill: string; label: string }> = {
  line: { stroke: "#fbbf24", fill: "rgba(251,191,36,0.08)", label: "Line" },
  pos: { stroke: "#34d399", fill: "rgba(52,211,153,0.10)", label: "Checkout" },
  alt: { stroke: "#60a5fa", fill: "rgba(96,165,250,0.10)", label: "Staffed" },
};

export function LiveFeed() {
  const { event, connected } = useStream();
  const people = useAnimatedOverlay(event);
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const [width, setWidth] = useState(0);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => setWidth(entries[0].contentRect.width));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const fw = event?.w ?? 16;
  const fh = event?.h ?? 9;
  const scale = width > 0 ? width / fw : 0;
  // sizes expressed in frame units but tuned to a constant on-screen pixel size
  const px = (screenPx: number) => (scale > 0 ? screenPx / scale : screenPx);
  const stroke = px(1.6);
  const jointR = px(2.2);
  const corner = px(10);
  const busy = normLevel(event?.busy?.level);

  return (
    <div className="overflow-hidden rounded-[14px] border border-hairline bg-[#0b0c0f] shadow-[var(--shadow-raised)]">
      {/* feed header */}
      <div className="flex items-center justify-between border-b border-white/[0.06] px-4 py-2.5">
        <div className="flex items-center gap-2.5">
          <span
            className="inline-block h-[7px] w-[7px] rounded-full"
            style={{
              background: connected ? "#34d399" : "#71717a",
              animation: connected ? "livepulse 2.4s infinite" : "none",
            }}
          />
          <span className="text-[0.74rem] font-semibold tracking-wide text-white/85">
            {connected ? "Live feed" : "Reconnecting…"}
          </span>
          <span className="text-[0.7rem] text-white/35">·</span>
          <span className="tnum text-[0.7rem] text-white/45">{people.length} tracked</span>
        </div>
        {busy && (
          <span
            className="rounded-full px-2.5 py-[3px] text-[0.66rem] font-semibold uppercase tracking-[0.08em]"
            style={{
              color: busy === "Low" ? "#34d399" : busy === "Medium" ? "#fbbf24" : "#fb7185",
              background:
                busy === "Low"
                  ? "rgba(52,211,153,0.12)"
                  : busy === "Medium"
                    ? "rgba(251,191,36,0.12)"
                    : "rgba(251,113,133,0.12)",
            }}
          >
            {busy} traffic
          </span>
        )}
      </div>

      {/* feed surface */}
      <div ref={wrapRef} className="relative w-full" style={{ aspectRatio: `${fw} / ${fh}` }}>
        {event ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={event.jpeg} alt="Live store feed" className="absolute inset-0 h-full w-full object-cover" />
        ) : (
          <div className="absolute inset-0 grid place-items-center text-[0.8rem] text-white/40">
            Waiting for the feed…
          </div>
        )}

        {/* vector overlay — scales perfectly with the frame */}
        {event && (
          <svg
            className="absolute inset-0 h-full w-full"
            viewBox={`0 0 ${fw} ${fh}`}
            preserveAspectRatio="none"
          >
            {/* zones */}
            {(Object.keys(ZONE_STYLES) as (keyof OverlayZones)[]).map((key) => {
              const polys = event.zones[key];
              if (!polys) return null;
              const s = ZONE_STYLES[key];
              return polys.map((poly, i) => (
                <polygon
                  key={`${key}-${i}`}
                  points={poly.map((p) => p.join(",")).join(" ")}
                  fill={s.fill}
                  stroke={s.stroke}
                  strokeWidth={px(1)}
                  strokeDasharray={`${px(7)} ${px(5)}`}
                  strokeLinejoin="round"
                  opacity={0.85}
                />
              ));
            })}

            {/* people */}
            {people.map((p) => {
              const color = stateColor(p.state);
              const [x1, y1, x2, y2] = p.box;
              const showPose = p.kpts.length === 17;
              return (
                <g key={p.id}>
                  <rect
                    x={x1}
                    y={y1}
                    width={Math.max(0, x2 - x1)}
                    height={Math.max(0, y2 - y1)}
                    rx={corner}
                    ry={corner}
                    fill="none"
                    stroke={color}
                    strokeWidth={stroke}
                    opacity={0.92}
                  />
                  {showPose &&
                    SKELETON_EDGES.map(([a, b], i) => {
                      const ka = p.kpts[a];
                      const kb = p.kpts[b];
                      if (ka[2] < KPT_THR || kb[2] < KPT_THR) return null;
                      return (
                        <line
                          key={i}
                          x1={ka[0]}
                          y1={ka[1]}
                          x2={kb[0]}
                          y2={kb[1]}
                          stroke={color}
                          strokeWidth={px(1.4)}
                          strokeLinecap="round"
                          opacity={0.5}
                        />
                      );
                    })}
                  {showPose &&
                    JOINT_INDICES.map((idx) => {
                      const k = p.kpts[idx];
                      if (k[2] < KPT_THR) return null;
                      return <circle key={idx} cx={k[0]} cy={k[1]} r={jointR} fill={color} opacity={0.85} />;
                    })}
                </g>
              );
            })}
          </svg>
        )}

        {/* crisp HTML state pills (fixed px type, positioned by % of frame) */}
        {event &&
          people.map((p) => {
            const [x1, y1] = p.box;
            const color = stateColor(p.state);
            const t = p.state === "serving" || p.state === "serving_other" ? p.serve : p.wait;
            const showTime = p.state !== "tracked" && p.state !== "out" && t > 0.5;
            // keep the pill on-screen: sit above the box, but drop just inside it
            // when the box hugs the top edge so the label never clips.
            const above = y1 / fh > 0.07;
            return (
              <div
                key={p.id}
                className="pointer-events-none absolute flex items-center gap-1.5 whitespace-nowrap rounded-md px-2 py-[3px] text-[11px] font-medium tabular-nums backdrop-blur-sm"
                style={{
                  left: `${(x1 / fw) * 100}%`,
                  top: `${(y1 / fh) * 100}%`,
                  transform: above ? "translateY(calc(-100% - 5px))" : "translateY(5px)",
                  background: "rgba(11,12,15,0.72)",
                  color: "#f4f4f5",
                  boxShadow: `inset 0 0 0 1px ${color}55`,
                }}
              >
                <span className="inline-block h-[6px] w-[6px] rounded-full" style={{ background: color }} />
                {STATE_LABEL[p.state]}
                {showTime && <span className="text-white/55">{fmtDuration(t)}</span>}
              </div>
            );
          })}
      </div>
    </div>
  );
}

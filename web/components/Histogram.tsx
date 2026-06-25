"use client";

import { useState } from "react";
import type { Metrics } from "@/lib/types";
import { histogramBuckets } from "@/lib/histogramBuckets";
import { fmtInt } from "@/lib/format";
import { Card, CardTitle } from "./ui";

const W = 1000;
const H = 220;
const PAD = 12;

export function Histogram({ m }: { m: Metrics | null }) {
  const [hover, setHover] = useState<number | null>(null);
  const bars = histogramBuckets(m?.occupancy, 56);
  const has = bars.length > 0;
  const max = has ? Math.max(...bars.map((b) => b.total), 1) : 1;
  const now = has ? bars[bars.length - 1] : null;
  const slot = has ? W / bars.length : W;
  const bw = slot * 0.7;

  const sel = hover != null ? bars[hover] : now;

  return (
    <Card className="p-5">
      <div className="mb-3 flex items-start justify-between">
        <CardTitle>People in store over time</CardTitle>
        <div className="flex gap-4">
          <Legend label="At line" color="var(--color-wait)" />
          <Legend label="At POS" color="var(--color-mashgin-ink)" />
        </div>
      </div>

      {has ? (
        <>
          <div className="mb-2 flex items-baseline gap-3">
            <span className="tnum text-[1.7rem] font-semibold leading-none text-ink">
              {fmtInt(sel?.total ?? 0)}
            </span>
            <span className="text-[0.8rem] text-faint">
              in store{hover != null ? " (hover)" : " now"} · {fmtInt(sel?.line ?? 0)} at line ·{" "}
              {fmtInt(sel?.pos ?? 0)} at POS
            </span>
          </div>

          <svg
            className="w-full"
            viewBox={`0 0 ${W} ${H}`}
            preserveAspectRatio="none"
            style={{ height: 220 }}
          >
            <line
              x1={0} y1={H - PAD} x2={W} y2={H - PAD}
              stroke="var(--color-hairline)" strokeWidth={1} vectorEffect="non-scaling-stroke"
            />
            {bars.map((b, i) => {
              const x = i * slot + (slot - bw) / 2;
              const usable = H - 2 * PAD;
              const lineH = (b.line / max) * usable;
              const posH = (b.pos / max) * usable;
              const active = hover === i;
              return (
                <g
                  key={i}
                  onMouseEnter={() => setHover(i)}
                  onMouseLeave={() => setHover(null)}
                  style={{ cursor: "pointer" }}
                >
                  {/* hover hit-area */}
                  <rect x={i * slot} y={0} width={slot} height={H} fill="transparent" />
                  {/* at line (bottom) */}
                  <rect
                    x={x} y={H - PAD - lineH} width={bw} height={lineH} rx={1.5}
                    fill="var(--color-wait)" opacity={active ? 1 : 0.85}
                  />
                  {/* at POS (stacked on top) */}
                  <rect
                    x={x} y={H - PAD - lineH - posH} width={bw} height={posH} rx={1.5}
                    fill="var(--color-mashgin-ink)" opacity={active ? 1 : 0.85}
                  />
                </g>
              );
            })}
          </svg>
        </>
      ) : (
        <div className="grid h-[220px] place-items-center text-[0.8rem] text-faint">
          Gathering occupancy…
        </div>
      )}
    </Card>
  );
}

function Legend({ label, color }: { label: string; color: string }) {
  return (
    <span className="flex items-center gap-1.5 text-[0.72rem] text-muted">
      <span className="inline-block h-[8px] w-[8px] rounded-[3px]" style={{ background: color }} />
      {label}
    </span>
  );
}

import type { Metrics } from "@/lib/types";
import { fmtInt, fmtSeconds } from "@/lib/format";
import { Card, CardTitle } from "./ui";

const W = 600;
const H = 132;
const PAD = 8;

interface Series {
  xs: number[];
  ys: number[];
  color: string;
  area?: boolean;
}

function paths(s: Series, domX: [number, number], domY: [number, number]) {
  const [x0, x1] = domX;
  const [y0, y1] = domY;
  const sx = (x: number) => (x1 === x0 ? 0 : ((x - x0) / (x1 - x0)) * W);
  const sy = (y: number) => {
    const t = y1 === y0 ? 0 : (y - y0) / (y1 - y0);
    return H - PAD - t * (H - 2 * PAD);
  };
  const pts = s.xs.map((x, i) => [sx(x), sy(s.ys[i])]);
  if (!pts.length) return { line: "", area: "" };
  const line = pts.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" ");
  const area = `${line} L${pts[pts.length - 1][0].toFixed(1)},${H} L${pts[0][0].toFixed(1)},${H} Z`;
  return { line, area };
}

function TrendChart({ series, domX, domY }: { series: Series[]; domX: [number, number]; domY: [number, number] }) {
  return (
    <svg className="w-full" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" style={{ height: 132 }}>
      {/* baseline */}
      <line x1={0} y1={H - PAD} x2={W} y2={H - PAD} stroke="var(--color-hairline)" strokeWidth={1} vectorEffect="non-scaling-stroke" />
      {series.map((s, i) => {
        const { line, area } = paths(s, domX, domY);
        if (!line) return null;
        return (
          <g key={i}>
            {s.area && <path d={area} fill={s.color} opacity={0.08} />}
            <path
              d={line}
              fill="none"
              stroke={s.color}
              strokeWidth={1.9}
              strokeLinejoin="round"
              strokeLinecap="round"
              vectorEffect="non-scaling-stroke"
            />
          </g>
        );
      })}
    </svg>
  );
}

function Empty() {
  return <div className="grid h-[132px] place-items-center text-[0.78rem] text-faint">Gathering data…</div>;
}

function Legend({ items }: { items: { label: string; color: string }[] }) {
  return (
    <div className="flex gap-4">
      {items.map((it) => (
        <span key={it.label} className="flex items-center gap-1.5 text-[0.72rem] text-muted">
          <span className="inline-block h-[8px] w-[8px] rounded-[3px]" style={{ background: it.color }} />
          {it.label}
        </span>
      ))}
    </div>
  );
}

function domain(values: number[][], floor = 1): [number, number] {
  const all = values.flat();
  const max = all.length ? Math.max(...all) : floor;
  return [0, Math.max(max, floor)];
}

export function OccupancyChart({ m }: { m: Metrics | null }) {
  const o = m?.occupancy;
  const has = !!o && o.t.length > 1;
  const last = (a?: number[]) => (a && a.length ? a[a.length - 1] : 0);
  return (
    <Card className="p-5">
      <div className="mb-3 flex items-start justify-between">
        <CardTitle>Occupancy</CardTitle>
        <Legend items={[{ label: "In line", color: "var(--color-wait)" }, { label: "At checkout", color: "var(--color-go)" }]} />
      </div>
      {has ? (
        <>
          <div className="mb-1 tnum text-[1.4rem] font-semibold leading-none text-ink">
            {fmtInt(last(o!.waiting))}
            <span className="ml-1 text-[0.8rem] font-normal text-faint">in line</span>
          </div>
          <TrendChart
            domX={[o!.t[0], o!.t[o!.t.length - 1]]}
            domY={domain([o!.waiting_ma, o!.serving_ma], 2)}
            series={[
              { xs: o!.t, ys: o!.waiting_ma, color: "var(--color-wait)", area: true },
              { xs: o!.t, ys: o!.serving_ma, color: "var(--color-go)" },
            ]}
          />
        </>
      ) : (
        <Empty />
      )}
    </Card>
  );
}

export function WaitServeChart({ m }: { m: Metrics | null }) {
  const w = m?.wait_serve;
  const has = !!w && w.t.length > 1;
  const last = (a?: number[]) => (a && a.length ? a[a.length - 1] : 0);
  return (
    <Card className="p-5">
      <div className="mb-3 flex items-start justify-between">
        <CardTitle>Time per customer</CardTitle>
        <Legend items={[{ label: "Wait", color: "var(--color-wait)" }, { label: "Checkout", color: "var(--color-go)" }]} />
      </div>
      {has ? (
        <>
          <div className="mb-1 tnum text-[1.4rem] font-semibold leading-none text-ink">
            {fmtSeconds(last(w!.wait_ma))}
            <span className="ml-1 text-[0.8rem] font-normal text-faint">avg wait</span>
          </div>
          <TrendChart
            domX={[w!.t[0], w!.t[w!.t.length - 1]]}
            domY={domain([w!.wait_ma, w!.serve_ma], 5)}
            series={[
              { xs: w!.t, ys: w!.wait_ma, color: "var(--color-wait)", area: true },
              { xs: w!.t, ys: w!.serve_ma, color: "var(--color-go)" },
            ]}
          />
        </>
      ) : (
        <Empty />
      )}
    </Card>
  );
}

export function ThroughputChart({ m }: { m: Metrics | null }) {
  const tp = m?.throughput;
  const has = !!tp && tp.served_per_min.length > 0;
  const max = has ? Math.max(...tp!.served_per_min, 1) : 1;
  const last = has ? tp!.served_per_min[tp!.served_per_min.length - 1] : 0;
  return (
    <Card className="p-5">
      <CardTitle>Throughput</CardTitle>
      {has ? (
        <>
          <div className="mb-1 tnum text-[1.4rem] font-semibold leading-none text-ink">
            {fmtInt(last)}
            <span className="ml-1 text-[0.8rem] font-normal text-faint">served / min</span>
          </div>
          <svg className="w-full" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" style={{ height: 132 }}>
            <line x1={0} y1={H - PAD} x2={W} y2={H - PAD} stroke="var(--color-hairline)" strokeWidth={1} vectorEffect="non-scaling-stroke" />
            {tp!.served_per_min.map((v, i) => {
              const n = tp!.served_per_min.length;
              const bw = (W / n) * 0.6;
              const gap = (W / n) * 0.4;
              const x = i * (W / n) + gap / 2;
              const h = (v / max) * (H - 2 * PAD);
              return <rect key={i} x={x} y={H - PAD - h} width={bw} height={h} rx={2} fill="var(--color-go)" opacity={0.85} />;
            })}
          </svg>
        </>
      ) : (
        <Empty />
      )}
    </Card>
  );
}

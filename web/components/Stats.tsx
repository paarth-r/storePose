import type { Metrics } from "@/lib/types";
import { fmtInt, fmtSeconds, normLevel } from "@/lib/format";
import { Card, CardTitle } from "./ui";

const LEVEL_COLOR: Record<string, string> = {
  Low: "var(--color-go)",
  Medium: "var(--color-wait)",
  High: "var(--color-busy)",
};
const LEVEL_WASH: Record<string, string> = {
  Low: "var(--color-go-wash)",
  Medium: "var(--color-wait-wash)",
  High: "var(--color-busy-wash)",
};

/** The headline "right now" card: current traffic + live counts. */
export function RightNow({ m }: { m: Metrics | null }) {
  const level = normLevel(m?.busy.current.level);
  const color = level ? LEVEL_COLOR[level] : "var(--color-faint)";
  const inLine = m?.summary.in_line ?? 0;
  const atPos = m?.summary.at_pos ?? 0;

  return (
    <Card className="relative overflow-hidden p-5">
      <span className="absolute inset-y-0 left-0 w-[4px]" style={{ background: color }} />
      <div className="flex items-center justify-between">
        <CardTitle>Right now</CardTitle>
        {level && (
          <span
            className="rounded-full px-2.5 py-[3px] text-[0.66rem] font-semibold uppercase tracking-[0.08em]"
            style={{ color, background: LEVEL_WASH[level] }}
          >
            {level} traffic
          </span>
        )}
      </div>
      <div className="mt-1 grid grid-cols-2 gap-4">
        <Figure value={fmtInt(inLine)} label="In line" />
        <Figure value={fmtInt(atPos)} label="At checkout" />
      </div>
      <p className="mt-4 border-t border-hairline pt-3 text-[0.78rem] text-muted">
        Avg time in store{" "}
        <span className="tnum font-medium text-ink">{fmtSeconds(m?.summary.avg_total_s ?? 0)}</span>
      </p>
    </Card>
  );
}

function Figure({ value, label }: { value: string; label: string }) {
  return (
    <div>
      <div className="tnum text-[2.6rem] font-semibold leading-none tracking-[-0.02em] text-ink">
        {value}
      </div>
      <div className="mt-1.5 text-[0.78rem] text-muted">{label}</div>
    </div>
  );
}

/** The signature stat: seconds saved per customer by self-checkout. */
export function CheckoutEdge({ m }: { m: Metrics | null }) {
  const c = m?.checkouts;
  const delta = c?.delta ?? 0;
  const mashEff = c?.mashgin_avg_eff ?? 0;
  const other = c?.other_avg ?? 0;
  const hasBoth = (c?.mashgin_n ?? 0) > 0 && (c?.other_n ?? 0) > 0;
  const max = Math.max(mashEff, other, 0.001);
  const faster = delta > 0;
  // "×N faster": staffed time ÷ self-checkout time, rounded up — one staffed
  // cashier's worth of throughput becomes N self-checkout customers.
  const ratio = mashEff > 0 ? other / mashEff : 0;
  const times = Math.max(1, Math.ceil(ratio));

  return (
    <Card className="p-5">
      <CardTitle>Checkout edge</CardTitle>
      {hasBoth ? (
        <>
          {faster ? (
            <>
              <div className="text-[1.7rem] font-semibold leading-tight tracking-[-0.01em] text-ink">
                Turns one cashier into{" "}
                <span className="tnum" style={{ color: "var(--color-go)" }}>
                  {times}
                </span>
              </div>
              <p className="mt-1.5 text-[0.78rem] text-faint">{times}× faster than a staffed lane</p>
            </>
          ) : (
            <>
              <div className="flex items-baseline gap-2">
                <span className="tnum text-[2.3rem] font-semibold leading-none tracking-[-0.02em] text-ink">
                  {fmtSeconds(Math.abs(delta))}
                </span>
                <span className="text-[0.82rem] font-medium" style={{ color: "var(--color-busy)" }}>
                  slower / customer
                </span>
              </div>
              <p className="mt-1 text-[0.76rem] text-faint">
                Self-checkout{(c?.num_mashgins ?? 1) > 1 ? ` ×${c?.num_mashgins}` : ""} vs staffed lane
              </p>
            </>
          )}
          <div className="mt-4 space-y-2.5">
            <Bar label="Self-checkout" value={mashEff} max={max} color="var(--color-go)" />
            <Bar label="Staffed lane" value={other} max={max} color="var(--color-faint)" />
          </div>
        </>
      ) : (
        <p className="py-4 text-[0.82rem] text-faint">
          Waiting for completed checkouts on both lanes to compare speed.
        </p>
      )}
    </Card>
  );
}

function Bar({ label, value, max, color }: { label: string; value: number; max: number; color: string }) {
  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between text-[0.76rem]">
        <span className="text-muted">{label}</span>
        <span className="tnum font-medium text-ink">{fmtSeconds(value)}</span>
      </div>
      <div className="h-[6px] overflow-hidden rounded-full bg-sunken">
        <div
          className="h-full rounded-full transition-[width] duration-500"
          style={{ width: `${Math.max(2, (value / max) * 100)}%`, background: color }}
        />
      </div>
    </div>
  );
}

/** A strip of secondary averages below the hero. */
export function StatStrip({ m }: { m: Metrics | null }) {
  const items = [
    { label: "Avg wait in line", value: fmtSeconds(m?.summary.avg_line_s ?? 0) },
    { label: "Avg checkout time", value: fmtSeconds(m?.summary.avg_pos_s ?? 0) },
    { label: "Served", value: fmtInt(m?.summary.served_count ?? 0) },
  ];
  return (
    <div className="grid grid-cols-3 divide-x divide-hairline rounded-[14px] border border-hairline bg-panel shadow-[var(--shadow-card)]">
      {items.map((it) => (
        <div key={it.label} className="px-5 py-4">
          <div className="tnum text-[1.5rem] font-semibold leading-none tracking-[-0.01em] text-ink">
            {it.value}
          </div>
          <div className="mt-1.5 text-[0.76rem] text-muted">{it.label}</div>
        </div>
      ))}
    </div>
  );
}

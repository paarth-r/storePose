import type { Metrics } from "@/lib/types";
import { cashierMultiplier } from "@/lib/cashierMultiplier";
import { fmtSeconds } from "@/lib/format";
import { Card, CardTitle } from "./ui";

/** Hero metric: "Turn one cashier into N" — N staffed lanes one kiosk replaces. */
export function CashierMultiplier({ m }: { m: Metrics | null }) {
  const r = cashierMultiplier(m?.checkouts);

  return (
    <Card className="p-5">
      <CardTitle>Mashgin throughput</CardTitle>
      {r.available ? (
        <>
          <div className="text-[1.5rem] font-semibold leading-tight tracking-[-0.01em] text-ink">
            Turn one cashier into{" "}
            <span className="tnum text-[2.3rem]" style={{ color: "var(--color-mashgin-ink)" }}>
              {r.n}
            </span>
          </div>
          <p className="mt-1.5 text-[0.78rem] text-faint">
            One Mashgin kiosk matches the throughput of {r.n} staffed{" "}
            {r.n === 1 ? "lane" : "lanes"}{" "}
            <span className="tnum">({r.ratio.toFixed(1)}× per customer)</span>
          </p>
          <div className="mt-4 grid grid-cols-2 gap-4 border-t border-hairline pt-3 text-[0.78rem]">
            <Stat label="Mashgin / customer" value={fmtSeconds(r.mashEff)} accent />
            <Stat label="Staffed / customer" value={fmtSeconds(r.other)} />
          </div>
          {r.delta > 0 && (
            <p className="mt-3 text-[0.76rem] text-muted">
              <span className="tnum font-medium text-ink">{fmtSeconds(r.delta)}</span> saved per customer
            </p>
          )}
        </>
      ) : (
        <p className="py-4 text-[0.82rem] text-faint">
          Awaiting staffed-lane checkouts to compute the multiplier.
        </p>
      )}
    </Card>
  );
}

function Stat({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div>
      <div
        className="tnum text-[1.3rem] font-semibold leading-none text-ink"
        style={accent ? { color: "var(--color-mashgin-ink)" } : undefined}
      >
        {value}
      </div>
      <div className="mt-1.5 text-[0.74rem] text-muted">{label}</div>
    </div>
  );
}

import type { Metrics } from "@/lib/types";
import { compareWindows, type CompareRow, type Dir } from "@/lib/compareWindows";
import { Card, CardTitle } from "./ui";

const ARROW: Record<Dir, string> = { up: "▲", down: "▼", flat: "—" };

/** "Busyness up", "throughput steady" — last ~2 min vs the session baseline. */
export function TimeCompare({ m }: { m: Metrics | null }) {
  const rows = compareWindows(m, 120);
  return (
    <Card className="p-5">
      <div className="mb-3 flex items-center justify-between">
        <CardTitle>Now vs earlier</CardTitle>
        <span className="text-[0.72rem] text-faint">last 2 min vs session</span>
      </div>
      <div className="grid grid-cols-3 divide-x divide-hairline">
        {rows.map((r) => (
          <Row key={r.key} r={r} />
        ))}
      </div>
    </Card>
  );
}

function Row({ r }: { r: CompareRow }) {
  const color =
    r.dir === "flat" ? "var(--color-faint)" : r.dir === "up" ? "var(--color-busy)" : "var(--color-go)";
  const pct = Math.round(Math.abs(r.deltaPct) * 100);
  return (
    <div className="px-4 first:pl-0 last:pr-0">
      <div className="text-[0.72rem] text-muted">{r.label}</div>
      {r.available ? (
        <div className="mt-1 flex items-baseline gap-1.5">
          <span className="text-[1.2rem] font-semibold leading-none" style={{ color }}>
            {ARROW[r.dir]}
          </span>
          <span className="tnum text-[1.1rem] font-semibold text-ink">{pct}%</span>
        </div>
      ) : (
        <div className="mt-1 text-[0.86rem] text-faint">—</div>
      )}
    </div>
  );
}

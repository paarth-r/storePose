import type { Metrics } from "@/lib/types";
import { Card, CardTitle } from "./ui";

const FLAGS: { key: "line" | "pos" | "reg" | "transit"; label: string }[] = [
  { key: "line", label: "Line" },
  { key: "pos", label: "POS" },
  { key: "reg", label: "Reg" },
  { key: "transit", label: "Transit" },
];

/** Per-person reasoning — gated behind the header Debug toggle (default off). */
export function DebugTable({ m, show }: { m: Metrics | null; show: boolean }) {
  const rows = m?.debug.rows ?? [];
  if (!show || !rows.length) return null;

  return (
    <Card className="p-5">
      <div className="mb-3 flex items-center justify-between">
        <CardTitle>Per-person reasoning</CardTitle>
        <span className="tnum text-[0.72rem] text-faint">frame {m?.debug.frame ?? "—"}</span>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full border-collapse text-[0.78rem]">
          <thead>
            <tr className="text-left text-faint">
              <Th>ID</Th>
              <Th>State</Th>
              <Th right>Wait</Th>
              <Th right>Checkout</Th>
              <Th right>Speed</Th>
              {FLAGS.map((f) => (
                <Th key={f.key} center>
                  {f.label}
                </Th>
              ))}
            </tr>
          </thead>
          <tbody className="tnum">
            {rows.map((r) => (
              <tr key={r.id} className="border-t border-hairline">
                <Td className="font-medium text-ink">{r.id}</Td>
                <Td className="text-muted">{r.state}</Td>
                <Td right>{r.wait.toFixed(1)}s</Td>
                <Td right>{r.serve.toFixed(1)}s</Td>
                <Td right>{r.speed.toFixed(0)}</Td>
                {FLAGS.map((f) => (
                  <Td key={f.key} center>
                    <Dot on={r[f.key]} />
                  </Td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

function Th({ children, right, center }: { children: React.ReactNode; right?: boolean; center?: boolean }) {
  return (
    <th
      className={`pb-2 font-medium ${right ? "text-right" : center ? "text-center" : "text-left"}`}
    >
      {children}
    </th>
  );
}

function Td({
  children,
  right,
  center,
  className = "",
}: {
  children: React.ReactNode;
  right?: boolean;
  center?: boolean;
  className?: string;
}) {
  return (
    <td className={`py-1.5 ${right ? "text-right" : center ? "text-center" : "text-left"} ${className}`}>
      {children}
    </td>
  );
}

function Dot({ on }: { on: boolean }) {
  return (
    <span
      className="inline-block h-[7px] w-[7px] rounded-full"
      style={{ background: on ? "var(--color-go)" : "var(--color-hairline-strong)" }}
    />
  );
}

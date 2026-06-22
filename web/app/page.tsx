"use client";

import { useMetrics } from "@/lib/useMetrics";
import { Header } from "@/components/Header";
import { LiveFeed } from "@/components/LiveFeed";
import { RightNow, CheckoutEdge, StatStrip } from "@/components/Stats";
import { OccupancyChart, WaitServeChart, ThroughputChart } from "@/components/Charts";
import { DebugTable } from "@/components/DebugTable";

export default function Page() {
  const { metrics, connected } = useMetrics();

  return (
    <main className="mx-auto max-w-[1280px] px-7 pb-20 pt-8">
      <Header connected={connected} now={metrics?.now ?? 0} />

      {/* hero: live feed + status rail */}
      <section className="mt-6 grid grid-cols-1 gap-5 lg:grid-cols-[1.85fr_1fr]">
        <LiveFeed />
        <div className="flex flex-col gap-5">
          <RightNow m={metrics} />
          <CheckoutEdge m={metrics} />
        </div>
      </section>

      <div className="mt-5">
        <StatStrip m={metrics} />
      </div>

      {/* trends */}
      <section className="mt-5 grid grid-cols-1 gap-5 md:grid-cols-2 xl:grid-cols-3">
        <OccupancyChart m={metrics} />
        <WaitServeChart m={metrics} />
        <ThroughputChart m={metrics} />
      </section>

      <div className="mt-5">
        <DebugTable m={metrics} />
      </div>
    </main>
  );
}

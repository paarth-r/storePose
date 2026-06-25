"use client";

import { useState } from "react";
import { useMetrics } from "@/lib/useMetrics";
import type { ViewMode } from "@/lib/viewMode";
import { modeEnabled } from "@/lib/viewMode";
import { Header } from "@/components/Header";
import { ViewSlider } from "@/components/ViewSlider";
import { LiveFeed } from "@/components/LiveFeed";
import { RightNow, CheckoutEdge, StatStrip } from "@/components/Stats";
import { CashierMultiplier } from "@/components/CashierMultiplier";
import { Histogram } from "@/components/Histogram";
import { TimeCompare } from "@/components/TimeCompare";
import { OccupancyChart, WaitServeChart, ThroughputChart } from "@/components/Charts";
import { DebugTable } from "@/components/DebugTable";

export default function Page() {
  const { metrics, connected } = useMetrics();
  const [viewMode, setViewMode] = useState<ViewMode>("mashgin");
  const [showDebug, setShowDebug] = useState(false);

  const hasOther = (metrics?.checkouts.other_n ?? 0) > 0;
  // if staffed-lane data disappears, fall back to the always-valid Mashgin view
  const mode: ViewMode = modeEnabled(viewMode, hasOther) ? viewMode : "mashgin";

  const showMashgin = mode === "mashgin" || mode === "both";
  const showOther = mode === "other" || mode === "both";

  return (
    <main className="mx-auto max-w-[1280px] px-7 pb-20 pt-8">
      <Header
        connected={connected}
        now={metrics?.now ?? 0}
        showDebug={showDebug}
        onToggleDebug={() => setShowDebug((v) => !v)}
      />

      <div className="mt-6 flex items-center justify-between">
        <ViewSlider mode={mode} onChange={setViewMode} hasOther={hasOther} />
      </div>

      {/* hero: live feed + status rail */}
      <section className="mt-5 grid grid-cols-1 gap-5 lg:grid-cols-[1.85fr_1fr]">
        <LiveFeed />
        <div className="flex flex-col gap-5">
          <RightNow m={metrics} />
          {showMashgin && <CashierMultiplier m={metrics} />}
          {/* CheckoutEdge is the head-to-head bar comparison — shown only in the
              comparison views so it doesn't echo the Mashgin hero in mashgin-only. */}
          {showOther && <CheckoutEdge m={metrics} />}
        </div>
      </section>

      {/* occupancy histogram centerpiece */}
      <div className="mt-5">
        <Histogram m={metrics} />
      </div>

      {/* now vs earlier + secondary averages */}
      <div className="mt-5 grid grid-cols-1 gap-5 lg:grid-cols-[1.4fr_1fr]">
        <TimeCompare m={metrics} />
        <StatStrip m={metrics} />
      </div>

      {/* trends */}
      <section className="mt-5 grid grid-cols-1 gap-5 md:grid-cols-2 xl:grid-cols-3">
        <OccupancyChart m={metrics} />
        <WaitServeChart m={metrics} />
        <ThroughputChart m={metrics} />
      </section>

      <div className="mt-5">
        <DebugTable m={metrics} show={showDebug} />
      </div>
    </main>
  );
}

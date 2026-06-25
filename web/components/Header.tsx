import Image from "next/image";
import mashginLogo from "@/public/mashgin-logo.png";

export function Header({
  connected,
  now,
  showDebug,
  onToggleDebug,
}: {
  connected: boolean;
  now: number;
  showDebug: boolean;
  onToggleDebug: () => void;
}) {
  // process clock (seconds since start) → m:ss, an honest "session time"
  const m = Math.floor(now / 60);
  const s = Math.floor(now % 60);
  return (
    <header className="flex items-end justify-between border-b border-hairline pb-5">
      <div>
        <div className="flex items-baseline gap-2.5">
          {/* Mashgin's wordmark is white (built for dark UIs) — set it on a dark
              chip so the full logo reads on the light theme. */}
          <span className="mr-1 inline-flex self-center rounded-[8px] bg-[#151515] px-2.5 py-1.5">
            <Image src={mashginLogo} alt="Mashgin" height={18} className="h-[18px] w-auto" priority />
          </span>
          <span className="text-[1.35rem] font-semibold tracking-[-0.02em] text-ink">
            Store Busyness
          </span>
          <span className="h-[15px] w-px self-center bg-hairline-strong" />
          <span className="text-[0.86rem] font-medium text-muted">storePose · Line Monitor</span>
        </div>
        <p className="mt-1 text-[0.8rem] text-faint">
          Live checkout-line analytics — occupancy, wait times, and Mashgin vs staffed speed.
        </p>
      </div>
      <div className="flex items-center gap-4 text-[0.78rem]">
        <button
          type="button"
          onClick={onToggleDebug}
          aria-pressed={showDebug}
          className="rounded-full border border-hairline px-3 py-[5px] font-medium text-muted transition-colors hover:bg-sunken"
          style={showDebug ? { color: "var(--color-ink)", background: "var(--color-sunken)" } : undefined}
        >
          Debug {showDebug ? "on" : "off"}
        </button>
        <span className="tnum text-faint">
          session {m}:{s.toString().padStart(2, "0")}
        </span>
        <span className="flex items-center gap-2 font-medium text-muted">
          <span
            className="inline-block h-[7px] w-[7px] rounded-full"
            style={{
              background: connected ? "#15803d" : "#9b9ba1",
              animation: connected ? "livepulse 2.4s infinite" : "none",
            }}
          />
          {connected ? "Connected" : "Offline"}
        </span>
      </div>
    </header>
  );
}

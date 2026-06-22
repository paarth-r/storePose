export function Header({ connected, now }: { connected: boolean; now: number }) {
  // process clock (seconds since start) → m:ss, an honest "session time"
  const m = Math.floor(now / 60);
  const s = Math.floor(now % 60);
  return (
    <header className="flex items-end justify-between border-b border-hairline pb-5">
      <div>
        <div className="flex items-baseline gap-2.5">
          <span className="text-[1.35rem] font-semibold tracking-[-0.02em] text-ink">storePose</span>
          <span className="h-[15px] w-px self-center bg-hairline-strong" />
          <span className="text-[0.86rem] font-medium text-muted">Line Monitor</span>
        </div>
        <p className="mt-1 text-[0.8rem] text-faint">
          Live checkout-line analytics — occupancy, wait times, and checkout speed.
        </p>
      </div>
      <div className="flex items-center gap-4 text-[0.78rem]">
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

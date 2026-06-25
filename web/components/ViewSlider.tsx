import type { ViewMode } from "@/lib/viewMode";
import { VIEW_MODES, modeEnabled } from "@/lib/viewMode";

/** Segmented control switching which checkout stats are shown. Non-Mashgin and
 *  Side-by-side stay disabled (with a tooltip) until staffed-lane data exists. */
export function ViewSlider({
  mode,
  onChange,
  hasOther,
}: {
  mode: ViewMode;
  onChange: (m: ViewMode) => void;
  hasOther: boolean;
}) {
  return (
    <div className="inline-flex items-center gap-1 rounded-full border border-hairline bg-panel p-1 shadow-[var(--shadow-card)]">
      {VIEW_MODES.map((vm) => {
        const enabled = modeEnabled(vm.key, hasOther);
        const active = mode === vm.key;
        return (
          <button
            key={vm.key}
            type="button"
            disabled={!enabled}
            title={enabled ? undefined : "No staffed-lane data yet"}
            onClick={() => enabled && onChange(vm.key)}
            aria-pressed={active}
            className="rounded-full px-3.5 py-[5px] text-[0.78rem] font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40"
            style={
              active
                ? { background: "var(--color-ink)", color: "var(--color-panel)" }
                : { color: "var(--color-muted)" }
            }
          >
            {vm.label}
          </button>
        );
      })}
    </div>
  );
}

export type ViewMode = "mashgin" | "other" | "both";

export const VIEW_MODES: { key: ViewMode; label: string }[] = [
  { key: "mashgin", label: "Mashgin" },
  { key: "other", label: "Non-Mashgin" },
  { key: "both", label: "Side-by-side" },
];

/** Whether a mode is selectable given staffed-lane data availability. */
export function modeEnabled(mode: ViewMode, hasOther: boolean): boolean {
  return mode === "mashgin" ? true : hasOther;
}

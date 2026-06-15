# Busy calibration — design

**Date:** 2026-06-15
**Status:** approved (brainstorming) → ready for implementation plan

## Problem

Busy labels (Low / Medium / High) are driven by `BusyThresholds` cut points
(`low_max`, `medium_max`) that today are hardcoded placeholder defaults
(`occupancy_p90`, 1.0 / 3.0) passed via CLI flags — the **same numbers for every
video**. The right cut points depend on the store and camera view, so we want to
**infer them per view from the clip's own occupancy distribution** instead of
typing numbers.

A second, coupled problem surfaced while scoping this: the live badge is *not
responsive*. The runner refreshes the label every 10 s (`_BUSY_REFRESH_SECONDS`,
`runner.py:280`), but `BusyAggregator.estimate(clock)` computes the metric over
the **entire in-progress 10-minute window** (`_win(t) = t // 600`). It is a
cumulative-since-window-start statistic: once a window fills, its p90 is
dominated by accumulated history and barely moves between refreshes, so the badge
looks frozen. A relative calibration can only be judged "by feel" if the live
badge actually tracks recent activity — so the fix belongs in this work.

## Decisions (from brainstorming)

- **Anchor:** relative — bands inferred per view from that clip's occupancy
  distribution (not absolute headcount).
- **Strategies:** compute **all three** at calibration time, switch at run time:
  - `skewed` — busy is rare: `low_max = p60`, `medium_max = p85`
  - `thirds` — even terciles: `low_max = p33`, `medium_max = p66`
  - `peak` — fraction of peak occupancy: `low_max = 0.30 * peak`,
    `medium_max = 0.70 * peak`
  (percentile/fraction knobs are module constants, retunable later)
- **Workflow:** a **separate `--calibrate` command**, **headless by default**,
  `-v`/`--verbose` shows the annotated window. Calibrate once → one calib file
  holds all three strategies → flip between them at run time.
- **Out of scope (deferred):** the `store_view` filename/rename convention.
  Calibration keys off the **existing video stem**, same key as
  `zones/<stem>.json` and `runs/<stem>_*`.

## Architecture

Two pieces.

### Piece 1 — `--calibrate` command

New top-level mode handled in `main.py` alongside `--define-zone` (it short-
circuits and returns, never entering the live display loop).

```
uv run python main.py --calibrate \
    --source videos/<stem>.mp4 \
    --zone zones/<stem>.json [--pos-zone ...] \
    [--busy-metric occupancy_p90] [--busy-subwindow 30] [-v]
```

- **Headless by default:** no `cv2.namedWindow`/`imshow`, no dashboard, no sink.
  `-v`/`--verbose` enables the annotated preview window + progress printout.
- Reuses the existing construction (source → `Pipeline` → `Tracker` →
  `QueueAnalyzer` → `BusyAggregator`). To avoid duplicating that wiring, the
  build of source/pipeline/tracker/analyzer is factored into a small shared
  helper that both `Runner.run()` and the calibration loop call. The calibration
  loop is a lean frame iterator: `analyzer.update(...)` then
  `aggregator.observe(clock, qresult.count, dt)` — no annotation/encode/IO.
- After the pass, read the metric's **per-sub-window distribution** across the
  whole clip, compute all three strategies, and write `calib/<stem>.json`.

New module `src/storepose/busy/calibrate.py`:

- `collect_samples(aggregator) -> list[float]` — uses a new
  `BusyAggregator.subwindow_values()` (below) to get one metric value per
  sub-window over the whole clip (independent of the 10-min window boundaries).
- `compute_strategies(values: list[float]) -> dict[str, dict]` — returns the
  three `{low_max, medium_max}` band sets. Uses `weighted_percentile`-style
  percentiles (equal-weight is fine here; each sub-window is one sample). Guards
  the degenerate cases (all-zero clip, < a few samples) by falling back to
  ordered cut points so `medium_max >= low_max` always holds.
- `write_calib(path, ...)` / `load_calib(path)` — JSON I/O.

`BusyAggregator` gains `subwindow_values() -> list[float]`: bucket **all**
collected samples by global sub-window index (`t // sub_window_seconds`), apply
the configured `_occ_stat` metric per bucket, return the flat list. Defaults
`sub_window_seconds` to a calibration default (e.g. 30 s) when unset for this
purpose. Existing `_metric_value`/`windows()` behavior is untouched.

### Piece 2 — run-time loading + rolling live badge

**Loading.** New CLI:
- `--calib PATH` — load a calib file.
- `--busy-strategy {skewed,thirds,peak}` — default `skewed`.

When `--calib` is set, the chosen strategy's `low_max`/`medium_max` (plus the
calib file's `metric` and `subwindow`) populate `BusyThresholds`, **overriding**
the manual `--busy-low-max`/`--busy-medium-max`/`--busy-metric` flags.
Without `--calib`, behavior is exactly as today (manual flags / defaults).
`view-setup.sh` adds `--calib calib/<stem>.json` to the generated run script when
that file exists; the strategy stays a default that you flip with the optional
`--busy-strategy` flag at run time.

**Rolling live badge.** New `BusyAggregator.estimate_recent(t, lookback) ->
(BusyLevel, float)`: compute the configured metric over only the samples in
`[t - lookback, t]`, classify with no hysteresis (at-a-glance estimate, same as
`estimate()`). The runner's 10 s refresh calls `estimate_recent(clock,
config.busy_live_window)` instead of `estimate(clock)`. New CLI
`--busy-live-window` (seconds, default 30). The finalized 10-minute `windows()`
report path is **unchanged** — only the live overlay/dashboard badge becomes
trailing-window.

## Calib file format — `calib/<stem>.json`

```json
{
  "stem": "<video stem>",
  "label": "",
  "source": "videos/<stem>.mp4",
  "metric": "occupancy_p90",
  "subwindow_seconds": 30.0,
  "samples": 142,
  "generated": "2026-06-15T12:00:00",
  "strategies": {
    "skewed": {"low_max": 1.0, "medium_max": 3.0},
    "thirds": {"low_max": 1.0, "medium_max": 2.0},
    "peak":   {"low_max": 1.2, "medium_max": 2.8}
  }
}
```

`label` is an optional free-text field (e.g. store/location note); nothing is
derived from the filename.

## Data flow

```
calibrate: clip frames -> pipeline -> tracker -> analyzer.update
             -> aggregator.observe(clock, count, dt)   [whole clip]
           aggregator.subwindow_values() -> [metric per sub-window]
           compute_strategies(values) -> {skewed,thirds,peak}
           write calib/<stem>.json

run:       load calib/<stem>.json + --busy-strategy -> BusyThresholds
           per frame: aggregator.observe(...)
           every 10 s: estimate_recent(clock, live_window) -> badge
           at exit: windows() -> busy report CSV   [10-min, unchanged]
```

## Error handling / edge cases

- **`--calibrate` without `--zone`** → error and exit (busy needs a zone), same
  spirit as the runner's existing `--busy needs an active --zone` note.
- **Empty / all-zero clip** → strategies fall back to ordered, non-decreasing cut
  points; write the file with `samples` count so it's visible the clip was thin.
- **`--calib` file missing/malformed** → clear error; do not silently fall back
  to placeholder thresholds.
- **`--calib` + manual `--busy-*` flags both given** → calib wins; print a one-
  line note that manual thresholds were overridden.
- **`medium_max >= low_max`** invariant enforced in `compute_strategies` and by
  `BusyThresholds.__post_init__`.

## Testing

- `compute_strategies`: known sample set → expected cuts for all three; degenerate
  (all-zero, single-sample) → valid non-decreasing bands.
- `BusyAggregator.subwindow_values`: synthetic samples across multiple 10-min
  windows → correct per-sub-window bucketing and metric.
- `estimate_recent`: trailing-window correctness (only recent samples count),
  empty lookback → level 0, parity with `estimate` when lookback ≥ elapsed.
- calib JSON round-trip: `write_calib` → `load_calib` → `BusyThresholds`.
- Calibrate end-to-end on a short fixture clip produces a well-formed calib file.

## Out of scope

- Renaming videos / `store_view` convention (explicitly deferred).
- Absolute-headcount or hybrid anchoring (rejected in brainstorming).
- Auto-running calibration from `view-setup.sh` (calibration is its own command).
- Re-labeling/back-filling existing `runs/*_busy.csv`.

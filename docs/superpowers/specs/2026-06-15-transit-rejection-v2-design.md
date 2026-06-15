# Transit rejection v2 (windowed net displacement) — design

**Date:** 2026-06-15
**Status:** approved (brainstorming) → ready for implementation
**Supersedes the signal in:** 2026-06-11-transit-rejection-design.md

## Problem

The current walk-through filter (`analyzer.py`) flags transit from a **per-frame
velocity-vector EMA** (`α=0.3`), normalized by box height, vs `--transit-speed`
(0.4 bh/s). On the real clips (**~6 fps**) it **fails to reject clear
walk-throughs** (false positives): the EMA needs several frames to ramp, but a
walk-through crosses the zone in only a handful of frames, so during the ramp the
person reads as "slow," accrues candidate frames, and gets counted before
`ema_v` clears the threshold. A derivative-based signal is the wrong tool at low
fps.

## Decision (from brainstorming)

Replace the EMA with **windowed net displacement** (approach A). Hold a
straightness gate (B) in reserve for later if real customers get dropped; treat
edge-to-edge geometry (C) as a possible future post-hoc check, not the live
filter (it can't prevent the live occupancy count from inflating).

## Mechanism (analyzer only)

Measure transit as an **average speed over a trailing window** — same units and
knob as today, but computed from net displacement (an integral) instead of an
EMA of instantaneous velocity, so there's no ramp lag.

`_VisitState`: replace `prev_center` / `ema_v` with a trailing ring
`centers: deque[tuple[float, float, float]]` of `(t, cx, cy)`.

Each `update` frame, for each present person:
- `cx, cy` = box center; `box_h = max(1, y2 − y1)`.
- Append `(t, cx, cy)`; pop from the left while `t − centers[0].t > transit_window`.
- If the person just returned from a re-id gap (`absent_seconds > 0`), **clear
  the ring** first so the gap's displacement isn't read as a spike.
- With `(t₀, cx₀, cy₀)` = oldest in window and `elapsed = t − t₀`:
  - `net_disp = hypot(cx − cx₀, cy − cy₀)`
  - `avg_speed_norm = (net_disp / box_h) / elapsed` when `elapsed > 0` and the
    ring has ≥ 2 samples, else `0.0`.
- `transiting = transit_speed > 0 and avg_speed_norm > transit_speed`.

Gate all zone membership exactly as today: `in_line = in_zone(line) and not
transiting` (and likewise `in_pos` / `in_alt`). A transiting person is in no
zone; a counted person who starts walking off is evicted via this signal plus the
existing `wait_exit_seconds` debounce.

The debug dict's `speed` field now reports `avg_speed_norm`.

## Config (`config.py`)

- Keep `--transit-speed` (`transit_speed: float = 0.4`, body-heights/sec, same
  meaning/units; `0` disables). Validate `>= 0`.
- Add `--transit-window` (`transit_window: float = 1.0`, seconds). Validate `> 0`.

## Runner

Pass `transit_window=config.transit_window` to `QueueAnalyzer`.

## Testing (`tests/queue/test_analyzer.py`)

- **Steady mover, low-fps style** (large per-frame steps across the zone) → never
  becomes waiting (`count == 0`). This is the case the EMA missed.
- **Walk-in then stop** → becomes waiting once the trailing window decays below
  threshold (slight, expected delay).
- **Shuffler / pacer** (small oscillation, net ≈ 0) → still becomes waiting.
- **`transit_speed = 0`** → filter disabled; a moving person becomes waiting.
- **Re-id return** does not spike `avg_speed_norm` (ring cleared).
- Existing stationary-person analyzer tests stay green (default knobs don't
  disturb non-moving boxes).

## Docs

Update `docs/usage.md` and `docs/problem-definition.md` transit notes: the
discriminator is now windowed net displacement (`--transit-speed` over
`--transit-window`), still in-place movement counts.

## Out of scope (v1)

- Straightness gate (B) — added only if shufflers/real customers get dropped.
- Per-view calibrated displacement threshold (perspective handling beyond
  box-height normalization).
- Edge-to-edge transit geometry (C).

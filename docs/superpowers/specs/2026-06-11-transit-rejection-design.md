# Walk-through (transit) rejection

**Date:** 2026-06-11
**Status:** Approved design, pre-implementation
**Branch:** feat/realtime-pose

## Goal

Stop people **walking through** a zone (e.g. the non-Mashgin checkout sitting in a
walkway) from being counted as present. A person counts in a zone only once they
have **stopped** there; someone in sustained directional motion is in no zone. This
refines the project's earlier "no motion gating" stance — *directional transit* is
gated, but in-place movement (a line shuffler fetching items) still counts.

## Mechanism (analyzer only)

The discriminator is the **EMA of the velocity vector**, not instantaneous speed:
a shuffler's back-and-forth cancels to ~0, while a walk-through's stays large and
directional.

Per person, in `QueueAnalyzer._VisitState`: add `prev_center: tuple | None` and
`ema_v: tuple[float, float] = (0.0, 0.0)`. Each `update` frame, for each present
person:
- `cx, cy = box center`; `box_h = max(1, y2 - y1)`.
- If `prev_center` is set, `dt > 0`, and the person did **not** just return from a
  re-id gap (`absent_seconds` was 0): `inst = ((cx-px)/dt, (cy-py)/dt)` and
  `ema_v = (1-α)·ema_v + α·inst` with `α = 0.3`. (On a re-id return, skip the
  velocity update for that frame to avoid an inflated gap-displacement spike.)
- `prev_center = (cx, cy)`.
- `speed_norm = |ema_v| / box_h`  (body-heights per second).
- `transiting = transit_speed > 0 and speed_norm > transit_speed`.

Then gate **all** zone membership: `in_line = in_zone(line) and not transiting`,
and likewise `in_pos` / `in_alt`. A transiting person is in no zone — they never
accumulate `enter_frames`, never become waiting/serving; a counted person who
starts walking off leaves via the existing `exit_seconds` debounce.

## Config (`config.py`)
- `transit_speed: float = 0.4` + `--transit-speed` (body-heights/sec; `0` disables
  → count regardless of motion, the prior behavior). Validate `>= 0`.

## Runner
Pass `transit_speed=config.transit_speed` to `QueueAnalyzer`.

## Testing (`tests/queue/test_analyzer.py`)
- A person whose box moves steadily across the zone (e.g. +30 px/frame) never
  becomes waiting (`count == 0`) — transit rejected.
- A person who moves in, then **stops** and dwells, becomes waiting once stopped
  (the EMA decays below threshold).
- A **shuffler** (small oscillating moves, ±8 px) still becomes waiting — the
  vector EMA cancels, so they read as present.
- `transit_speed=0` restores count-everyone (a moving person becomes waiting).
- The existing analyzer suite stays green (default 0.4 doesn't disturb the
  stationary-person tests, whose boxes don't move).

## Docs
Update `docs/problem-definition.md`'s "no motion gating" note: directional transit
is now gated via `--transit-speed`; in-place movement still counts.

## Out of scope
Acceleration gating (velocity magnitude suffices); trajectory/path-curvature
modeling; per-zone thresholds (one global `--transit-speed`).

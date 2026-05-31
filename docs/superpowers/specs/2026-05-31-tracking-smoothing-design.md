# storePose — Per-Person Tracking, Permanence & Smoothing

**Date:** 2026-05-31
**Status:** Approved

## Goal

Give each detected person a **stable identity** whose bounding box is
**permanent** through brief occlusions/missed detections, and **reduce jitter**
in both boxes and pose skeletons. Builds on the existing detector→pose pipeline.

## Why not rtmlib's PoseTracker

rtmlib ships a `PoseTracker`, but it is insufficient:

- IoU-only association on pose-derived boxes, with **no smoothing** (no jitter
  reduction).
- **No coasting** — a track that misses a single detection is dropped, so boxes
  are not permanent through occlusion.
- Does not expose track IDs in its return value cleanly.

We therefore build a dedicated tracking + smoothing layer.

## Decisions (locked)

| Decision            | Choice                                                       |
|---------------------|--------------------------------------------------------------|
| ID permanence       | Coasting only (motion prediction through brief gaps). A person who fully leaves and returns gets a NEW id. No appearance/ReID model. |
| While occluded      | **Predicted box only** — skeleton hidden until re-detected (pose is not faked during occlusion). |
| Coast hold time     | Medium, **~1–2s** (default 1.5s), converted to `max_age` frames via fps. |
| Tracker             | **SORT-style**: per-track Kalman (constant velocity) + IoU/Hungarian association. |
| Keypoint smoothing  | **One-Euro filter** per keypoint, keyed by track id.         |
| Box smoothing       | Kalman filtered state (the drawn box is the filter output, not the raw detection). |
| New dependency      | `scipy` (`linear_sum_assignment`).                           |

## Data flow

```
frame
  → PosePipeline.process(frame) → FrameResult(boxes, keypoints, scores)
  → MultiObjectTracker.update(result, dt) → list[TrackedPerson]
  → drawing.annotate_tracked(frame, people, config, fps)
  → display / VideoSink
```

Detection→pose still runs every frame. Pose is only ever run on detector boxes
(never for coasting tracks, whose skeleton is hidden), so tracking adds no extra
pose inference.

## How each requirement is met

- **Permanent boxes:** each `Track` owns a Kalman filter with SORT's 7-dim state
  `[cx, cy, s(area), r(aspect), ẋ, ẏ, ṡ]` (measurement `[cx, cy, s, r]`, aspect
  velocity fixed at 0). Per frame: predict all tracks → IoU cost matrix →
  Hungarian match (gated by `iou_thr`) → update matched, **coast unmatched**
  (prediction only) up to `max_age`, then delete. The track keeps its integer id
  and stable color throughout.
- **Box jitter:** the drawn box is the Kalman *filtered* state, damping
  frame-to-frame wobble from the raw detector.
- **Keypoint jitter:** a One-Euro filter per keypoint (independent x and y),
  stored on the `Track` so state persists across frames. Adaptive cutoff:
  `cutoff = min_cutoff + beta·|derivative|` — low jitter when still, low lag when
  moving.
- **Flicker control:** new tracks are *tentative* and not drawn until `min_hits`
  (default 3) consecutive matches; coasting tracks draw **box only** (no
  skeleton).

## Track lifecycle (states)

- **Tentative:** newly created from an unmatched detection; not drawn. Promotes
  to *confirmed* after `min_hits` consecutive matches.
- **Confirmed (matched):** drawn with box + smoothed skeleton + `ID n` label.
- **Coasting (confirmed but unmatched this frame):** Kalman prediction only;
  drawn box only; `time_since_update` increments. Returns to matched on
  re-association.
- **Deleted:** `time_since_update > max_age`, or a tentative track missed before
  confirmation.

## Module structure (new)

```
src/storepose/tracking/
  __init__.py
  types.py        TrackedPerson(id, box, keypoints|None, scores, coasting, color)
  kalman.py       KalmanBoxTracker — SORT constant-velocity box model
  assignment.py   iou(), iou_cost_matrix(), match(detections, tracks, iou_thr)
  smoothing.py    OneEuroFilter (scalar) + KeypointSmoother (per-keypoint)
  track.py        Track — id, kalman, hit/age counters, state, color, smoothers
  tracker.py      MultiObjectTracker — predict → associate → lifecycle → tracks
```

Changes to existing modules:

- `drawing.py`: add `annotate_tracked(frame, people, config, fps)` — stable
  per-id color (palette keyed by id), `ID n` label, box always, skeleton only
  when not coasting. Existing `annotate` retained for `--no-track` mode.
- `runner.py`: build a `MultiObjectTracker` (unless `--no-track`); compute `dt`
  per frame (real time, or `1/fps` for files); call tracker then
  `annotate_tracked`.
- `config.py`: new flags (below).

## Config additions

| Flag              | Default | Meaning                                              |
|-------------------|---------|------------------------------------------------------|
| `--no-track`      | —       | Disable tracking; use raw per-frame `annotate`.      |
| `--hold-seconds`  | `1.5`   | Coast duration; → `max_age` frames via fps.          |
| `--min-hits`      | `3`     | Matches before a track is confirmed/drawn.           |
| `--iou-thr`       | `0.3`   | Min IoU to associate a detection to a track.         |
| `--no-smooth`     | —       | Disable One-Euro keypoint smoothing.                 |
| `--smooth-cutoff` | `1.0`   | One-Euro `min_cutoff` (lower = smoother/laggier).    |
| `--smooth-beta`   | `0.007` | One-Euro `beta` (higher = more responsive to speed). |

## Testing strategy

Pure logic, TDD:

- `assignment.iou` — known boxes → known IoU; disjoint → 0; identical → 1.
- `OneEuroFilter` — constant input converges to constant; reduces variance on a
  noisy constant signal; tracks a ramp with bounded lag.
- `KalmanBoxTracker` — `predict()` advances center by velocity; `update()` pulls
  state toward the measurement.
- `assignment.match` — correct detection↔track pairing; IoU gating leaves
  low-overlap pairs unmatched; returns unmatched dets/tracks.
- `MultiObjectTracker` lifecycle — tentative→confirmed after `min_hits`; coasts
  through misses up to `max_age` then deletes; stable ids across frames; two
  separated detections keep distinct ids; re-entry after deletion → new id.
- `KeypointSmoother` — output shape preserved; reduces jitter; coasting (no
  update) leaves last value available.
- Integration — `MultiObjectTracker.update(synthetic FrameResult)` returns the
  expected `TrackedPerson`s (ids, coasting flags) with no models loaded.

Manual verification: run on a store clip with `--save`, confirm stable ids,
boxes persisting through occlusion, and visibly calmer boxes/skeletons; A/B with
`--no-track` and `--no-smooth`.

## Trade-offs

Smoothing trades a little latency for stability; defaults are conservative.
`--no-smooth` and `--no-track` allow A/B comparison. Coasting-only (no ReID)
means re-entry produces a new id — acceptable for a fixed store camera and far
cheaper than an appearance model.

## Dependencies

Add `scipy` (Hungarian assignment) to the existing `rtmlib`, `onnxruntime`,
`opencv-python`, `numpy`.

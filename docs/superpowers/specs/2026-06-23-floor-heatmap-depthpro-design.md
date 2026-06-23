# Floor-Plane Foot-Traffic Heatmap with Depth Pro Calibration

- Date: 2026-06-23
- Branch: `feat/floor-heatmap-depthpro`
- Status: Design — pending review

## Problem

We want a heatmap of where people stand, walk, and dwell on a store floor,
derived from the existing storePose tracking output. Cameras are fixed,
monocular, one camera per store area. Different cameras cover different
(possibly non-overlapping) areas of the same store, and we want the *option,
available in code*, to later stitch multiple camera heatmaps into one store
frame.

The analytical product is the floor density map. "3D" is a viewing choice
layered on top, not a prerequisite. A photoreal 3D store render (NeRF/Instant-NGP
or photogrammetry) is explicitly parked as a later, decoupled visualization layer.

## Goal (v1)

From a fixed camera's tracked people, project each person's floor position into a
per-camera **metric floor frame**, accumulate into a density grid weighted by
dwell time, and render a viewable rectified top-down heatmap. Calibration of the
floor frame is automatic via Apple **Depth Pro** (run once per camera), with a
manual 4-point homography as a zero-dependency fallback and sanity check.

## Non-Goals (v1)

- Photoreal 3D store reconstruction / NeRF / Instant-NGP rendering.
- Per-frame depth estimation. Depth runs **once** at calibration; the live path
  is a matrix multiply.
- Survey-grade metric accuracy. Depth Pro meters are "good approximate"
  (~10-20% at retail range); we sanity-check against one real measurement when
  available.
- Multi-camera stitching *execution*. v1 ships the abstraction and the
  per-camera `T_cam_to_store` transform (identity for one camera); actual
  cross-camera alignment is a later milestone (see Open Questions).

## Key Decisions and Rationale

1. **Floor-plane homography is the load-bearing primitive, not NeRF.**
   A foot-traffic heatmap needs exactly one depth per person: where the ankle
   ray pierces the floor plane. That is a closed-form ray-plane intersection —
   a 3x3 homography. NeRF computes a whole volumetric field to recover depth
   everywhere, then we would discard all but the floor. NeRF also requires
   multi-view overlap (via COLMAP) that non-overlapping store cameras do not
   have, so it structurally cannot merge our cameras anyway.

2. **Shared floor frame is what makes stitching cheap.** Each camera registers
   into a floor coordinate system via its own homography. Heatmaps from
   different cameras compose because they share the floor frame, *not* because
   their images overlap. Adding a camera = adding one homography + one
   `T_cam_to_store`.

3. **Depth Pro for auto metric calibration.** Both NeRF and a manual homography
   need an externally injected real-world distance to get meters. Depth Pro is
   natively metric and self-estimates focal length, so one frame yields the
   floor plane *in meters* with no tile-measuring. That is the entire reason it
   earns a dependency. Depth Anything v2 (base) is affine-ambiguous (unknown
   scale and shift) and would still need a manual measurement, at which point
   the manual homography is simpler — so Depth Anything is not used (its
   metric-fine-tuned variant is a documented lighter fallback only).

4. **Depth Pro is offline and optional; the live pipeline stays ONNX-only.**
   The realtime stack is rtmlib + onnxruntime with no torch. Depth Pro is
   PyTorch. It runs once per camera in a separate calibration tool under an
   optional `depth` dependency group. The live heatmap accumulation imports
   only numpy. The realtime path never gains a torch/GPU dependency.

## Architecture

```
                         live pipeline (numpy, ONNX, per frame)
TrackedPerson ──anchor──▶ floor_pixel (u,v) ──H_cam──▶ floor_xy (m, cam frame)
                                                          │
                                              T_cam_to_store (identity for 1 cam)
                                                          ▼
                                                  store_xy ──▶ DensityGrid (dwell-weighted)
                                                                     │
                                                                     ▼
                                                          render: rectified top-down PNG/npy
                                                          (later: 3D drape)

                         offline, run once per camera (torch, optional dep)
one frame ──Depth Pro──▶ metric point cloud ──fit floor plane──▶ H_cam + plane  ──▶ calib JSON
            (or)  manual 4-point click ────────────────────────▶ H_cam (relative or metric)
```

### Components

- **`FloorFrame` / `CameraView`** (new `heatmap/floor.py`): a `CameraView` holds
  `camera_id`, homography `H` (image pixel -> floor meters, camera frame), and
  `T_cam_to_store` (2D similarity transform into the shared store frame;
  identity by default). `apply(u, v) -> (X, Y)` does the projection. A
  `StoreFloor` is a list of `CameraView`s sharing one coordinate system — the
  stitching hook.

- **Anchor selection** (`heatmap/anchor.py`): per `TrackedPerson`, estimate the
  floor-contact pixel using the *best available* body evidence. Critically,
  anchors use the **Kalman-smoothed track** (`TrackedPerson.box` and smoothed
  keypoints), never raw detector output — so detector jitter is already removed
  before projection. Hierarchy, each step carrying a quality score:

  1. Both ankles (COCO kpts 15/16) confident -> ankle midpoint (true floor
     contact, highest quality).
  2. One ankle confident -> that ankle.
  3. No ankles but hips (11/12) / knees (13/14) visible -> take the hip midpoint
     and **drop to the floor along the scene vertical** by the modeled remaining
     standing height. The vertical direction is the floor-plane **normal**
     recovered by Depth Pro calibration (its projection is the vertical
     vanishing direction), so "assume upright" is exact geometry, not image-+y.
     This localizes people whose feet are occluded.
  4. Nothing reliable -> smoothed box bottom-center, lowest quality, flagged.

  Returns floor pixel + quality. Skips coasting tracks (no keypoints). Box
  bottom-center is only the last resort precisely because it sits at the
  occluder, not the floor, when feet are hidden.

- **Calibration backends** (one interface, `heatmap/calibrate/`):
  - `depthpro.py` — load one frame, run Depth Pro (MPS/CPU), back-project to a
    metric point cloud, RANSAC-fit the dominant floor plane, derive `H` mapping
    image pixels to in-plane metric coordinates **and persist the plane normal**
    (the scene vertical) so the anchor module's hip-drop has a principled "down".
    Writes a floor-calib JSON. Imports torch + depth_pro lazily; only this module
    needs the optional dep.
  - `manual.py` — click/define 4+ floor points; with optional known distances
    -> metric `H`, else relative top-down `H`. Zero extra dependencies. Also
    the cross-check tool for Depth Pro output. Note: manual calibration yields
    `H` but no floor normal, so the hip-drop anchor (step 3) is unavailable under
    manual-only calibration — those frames fall back to box-bottom unless the
    user also marks one vertical reference line (optional, gives the normal).
    Depth Pro calibration is the path that unlocks hip-drop for free.

- **Accumulator** (`heatmap/accumulate.py`): a `DensityGrid` over the floor
  frame at a configurable cell size (e.g. 0.1 m). Two robustness layers make the
  aggregate tolerant of noisy per-frame anchors:
  - **Per-track floor-XY Kalman smoothing**: a track's floor point cannot
    teleport; smoothing each track's projected position rejects single-frame
    outliers (e.g. a brief box-bottom fallback).
  - **Quality-weighted dwell**: each frame contributes `anchor_quality / fps`
    (dwell seconds scaled by anchor confidence), so ankle frames count fully and
    occluded box-bottom frames count little. `1/fps` keeps weight
    fps-independent. Track-id awareness caps a stationary person so they do not
    infinitely dominate one cell.

  A heatmap is a distribution over time, so many good anchors carry the map and
  down-weighted occluded frames wash out — the accumulator is where per-frame
  anchor noise is absorbed.

- **Renderer** (`heatmap/render.py`): rasterize the grid to a colormapped image,
  default rectified top-down. The 3D drape is a later render target consuming
  the same grid — out of v1 scope but the grid is render-agnostic.

### Data / file layout

```
src/storepose/heatmap/
  __init__.py
  floor.py          # FloorFrame, CameraView, StoreFloor, projection
  anchor.py         # TrackedPerson -> floor pixel
  accumulate.py     # DensityGrid, dwell weighting
  render.py         # top-down colormap render
  calibrate/
    __init__.py     # backend interface
    manual.py       # 4-point homography (no deps)
    depthpro.py     # Depth Pro one-shot (optional torch dep)
calib/floor/<camera_id>.json   # persisted H, plane, T_cam_to_store, units, source
outputs/heatmap/<camera_id>.{png,npy}
```

Calibration is a sibling concern to the existing `calib/` (busy bands) and
`zones/` (pixel polygons); floor calibration gets its own `calib/floor/`
namespace, JSON-persisted so the live path never re-runs Depth Pro.

### Dependencies

- Live pipeline: unchanged (numpy/scipy/onnxruntime/opencv/rtmlib).
- New optional group in `pyproject.toml`: `[dependency-groups] depth = ["torch",
  "depth-pro @ <apple ml-depth-pro source>"]`. Only `calibrate/depthpro.py`
  imports it, lazily, with a clear error if the group is not installed.
- Runs on MPS or CPU on Mac (one frame; CPU latency acceptable).

## Multi-Camera Stitching (designed-in, not executed in v1)

Each camera's Depth Pro reconstruction is in its *own* camera frame; for
non-overlapping cameras Depth Pro alone cannot establish a shared store origin.
The architecture captures this in `T_cam_to_store` per camera. v1 sets it to
identity (single camera). A later milestone adds cross-camera alignment via a
few manually corresponded floor points (or a shared fiducial / known layout) to
solve each camera's `T_cam_to_store`. No accumulator/renderer changes are needed
when that lands — they already operate in the shared store frame.

## Testing

- `floor.py`: synthetic homography round-trips (known pixel -> known meter),
  `T_cam_to_store` composition, identity default.
- `anchor.py`: ankle-visible, one-ankle, ankle-occluded-but-hips-visible ->
  hip-drop along a known floor normal hits the expected floor point, no-pose ->
  box-bottom fallback, coasting -> skipped. Score-threshold boundaries; quality
  score ordering (ankle > hip-drop > box-bottom).
- `accumulate.py`: dwell weighting is fps-independent (same seconds -> same
  weight at 15 vs 30 fps); quality weighting (low-quality anchor contributes
  less); per-track floor-XY smoothing rejects a single-frame outlier;
  stationary-cap behavior; grid bounds/cell size.
- `calibrate/manual.py`: 4-point -> homography matches a known projective map.
- `calibrate/depthpro.py`: plane-fit on a synthetic/saved depth array (no model
  download in CI); the model run itself is exercised behind a marker/manual test.

## Open Questions / Risks

- **Metric trust.** Depth Pro absolute scale is approximate at retail range.
  Mitigation: optional one-distance sanity check via the manual backend; report
  the discrepancy.
- **Ankle occlusion behind counters.** Handled by the hip-drop anchor (step 3),
  which localizes the floor point from the visible upper body via the Depth Pro
  floor normal without needing feet. Residual error comes from the assumed
  standing height and grows with camera obliqueness (near-overhead -> small);
  box-bottom is only the final fallback. All sub-ankle anchors are quality-
  flagged and down-weighted.
- **Floor non-planarity / clutter** can corrupt the RANSAC plane fit; the manual
  backend is the escape hatch.
- **Cross-camera origin** for stitching is unsolved by Depth Pro alone (above).

## Sequencing

1. v1 (this spec): `heatmap/` package, manual + Depth Pro backends, single-camera
   floor frame, dwell-weighted grid, top-down render, persisted floor calib.
2. Later: cross-camera `T_cam_to_store` solving (stitching execution).
3. Later/backburner: 3D drape render target; photoreal store model
   (photogrammetry/NeRF) as an optional viz asset hung off the floor frame.

# Floor-Plane Foot-Traffic Heatmap with Depth Pro Calibration

- Date: 2026-06-23
- Branch: `feat/floor-heatmap-depthpro`
- Status: Design approved; de-risk validated; building

## De-risk result (validated 2026-06-23)

Ran Depth Pro on a real Counter Overview frame (`videos/cumberland/chunks/
Counter Overview-converted/part11.mp4`, mid-frame). Confirmed the core
assumption holds:

- Clean, sharp metric depth (range 0.67–8.60 m, median 2.54 m); self-estimated
  focal 943 px (~90 deg HFOV, correct for the cam).
- RANSAC floor plane: normal ~(0, -0.70, -0.71), camera tilt 44.7 deg, camera
  height 2.69 m — all physically correct for a ceiling security cam.
- Plane inliers land on the actual floor (walkway tile + aisle floor), avoiding
  counters/shelves/people.

Key implementation constraint discovered: **Depth Pro OOM-kills on MPS; run on
CPU** (~60 s/frame, fine for a once-per-camera gate). The `scene/depthpro.py`
module defaults to CPU. Depth is cached so this cost is paid once. Minor tuning
left: floor-region mask + inlier threshold to also capture lighter far tile.

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
dwell time, and render a viewable rectified top-down heatmap. The floor frame is
derived from a **required** Apple **Depth Pro** scene-calibration step (run once
per camera, cached); a manual 4-point homography exists only as a fallback /
sanity check when the depth plane-fit fails.

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

4. **Depth is foundational and required — a mandatory calibration gate, run
   once and cached.** Depth Pro (PyTorch) is a *core* dependency, not an optional
   group: depth underpins not just the heatmap but planned future work (gaze
   detection, richer 3D analytics), so the project does not install without it.
   It is **not** in the per-frame loop: cameras are fixed and the floor is
   static, so depth only changes if a camera moves. Depth Pro therefore runs
   **once per camera as a required calibration step that must complete before the
   pipeline runs**, and its output is cached to disk. The live loop reads the
   cache (numpy only) and never invokes torch at runtime — but torch is installed
   because calibration is non-negotiable.

5. **Cache the full scene geometry, not just the floor homography.** Because the
   depth pass is foundational for future features, its cached artifact is the
   whole scene model — depth map, estimated focal/intrinsics, floor plane,
   floor normal, and point cloud — so gaze and other downstream work reuse the
   same calibration instead of re-deriving geometry. The floor homography is one
   derived view of this cache, not the whole of it.

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

         REQUIRED calibration gate — run once per camera, cached (torch, core dep)
one frame ──Depth Pro──▶ metric point cloud + focal ──┬─ fit floor plane + normal
                                                      ├─ depth map, intrinsics
                                                      └─▶ scene-geometry cache (calib/scene/<cam>)
                                                            │ (H_cam derived from plane)
            (manual override / cross-check) ───────────────┘
```

The cache is the prerequisite for everything downstream; the live pipeline
refuses to run a camera that has no scene-geometry cache.

### Components

The foundational geometry lives in a new top-level `scene/` package (not under
`heatmap/`), because depth-derived scene geometry is shared infrastructure that
future features — gaze, 3D analytics — consume alongside the heatmap.

- **Scene calibration gate** (`scene/depthpro.py`): runs Depth Pro once per
  camera (MPS/CPU) on one frame, back-projects to a metric point cloud using the
  self-estimated focal, RANSAC-fits the dominant floor plane, and writes the
  **scene-geometry cache**: depth map + intrinsics/focal + floor plane + floor
  normal + point cloud. This is a required step; nothing downstream runs without
  its cache. A `scene/manual.py` override (4-point click, optional vertical-line
  mark) exists only as a cross-check / fallback when the plane fit fails — it is
  no longer a co-equal backend now that depth is mandatory.

- **Scene cache I/O** (`scene/cache.py`): load/save the cache to
  `calib/scene/<camera_id>.{json,npz}` (JSON for plane/normal/intrinsics/units,
  `.npz` for the depth map and point cloud). Validity note: the cache holds while
  the camera is fixed; re-run only if a camera is repositioned. Callers fail
  loudly if a camera has no cache.

- **`FloorFrame` / `CameraView`** (`scene/geometry.py`): a `CameraView` holds
  `camera_id`, homography `H` (image pixel -> floor meters, camera frame) derived
  from the cached plane, the floor `normal`, and `T_cam_to_store` (2D similarity
  transform into the shared store frame; identity by default). `apply(u, v) ->
  (X, Y)` does the projection. A `StoreFloor` is a list of `CameraView`s sharing
  one coordinate system — the stitching hook. Built *from* the scene cache.

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
src/storepose/scene/            # foundational depth-derived geometry (required gate)
  __init__.py
  depthpro.py       # run Depth Pro once per camera -> scene-geometry cache
  manual.py         # 4-point override / cross-check (fallback only)
  cache.py          # load/save scene cache
  geometry.py       # FloorFrame, CameraView, StoreFloor, projection (from cache)
src/storepose/heatmap/
  __init__.py
  anchor.py         # TrackedPerson -> floor pixel (best-available body evidence)
  accumulate.py     # DensityGrid, per-track smoothing, quality-weighted dwell
  render.py         # top-down colormap render
calib/scene/<camera_id>.json   # plane, normal, intrinsics/focal, H, T_cam_to_store, units
calib/scene/<camera_id>.npz    # depth map + point cloud (reused by future features)
outputs/heatmap/<camera_id>.{png,npy}
```

Scene calibration is a sibling concern to the existing `calib/` (busy bands) and
`zones/` (pixel polygons); it gets its own `calib/scene/` namespace, persisted so
the live path never re-runs Depth Pro and future features (gaze) reuse the cache.

### Dependencies

- **torch + depth-pro become core `dependencies`** in `pyproject.toml` (not an
  optional group): depth is foundational, so the project does not install without
  it. `depth-pro @ <apple ml-depth-pro source>` (Apple's `ml-depth-pro`).
- **Runtime separation, not dependency separation.** Only `scene/depthpro.py`
  imports torch, and only at calibration time. The live loop
  (`scene/geometry.py`, `heatmap/*`) imports numpy and reads the cache — it never
  invokes torch. So adding a heavy dep does not slow or burden the per-frame path.
- Runs on MPS or CPU on Mac (one frame per camera; CPU latency acceptable).

## Multi-Camera Stitching (designed-in, not executed in v1)

Each camera's Depth Pro reconstruction is in its *own* camera frame; for
non-overlapping cameras Depth Pro alone cannot establish a shared store origin.
The architecture captures this in `T_cam_to_store` per camera. v1 sets it to
identity (single camera). A later milestone adds cross-camera alignment via a
few manually corresponded floor points (or a shared fiducial / known layout) to
solve each camera's `T_cam_to_store`. No accumulator/renderer changes are needed
when that lands — they already operate in the shared store frame.

## Testing

- `scene/geometry.py`: synthetic homography round-trips (known pixel -> known
  meter), `T_cam_to_store` composition, identity default; FloorFrame built from
  a synthetic cache.
- `scene/cache.py`: round-trip save/load; loud failure when a camera has no cache.
- `anchor.py`: ankle-visible, one-ankle, ankle-occluded-but-hips-visible ->
  hip-drop along a known floor normal hits the expected floor point, no-pose ->
  box-bottom fallback, coasting -> skipped. Score-threshold boundaries; quality
  score ordering (ankle > hip-drop > box-bottom).
- `accumulate.py`: dwell weighting is fps-independent (same seconds -> same
  weight at 15 vs 30 fps); quality weighting (low-quality anchor contributes
  less); per-track floor-XY smoothing rejects a single-frame outlier;
  stationary-cap behavior; grid bounds/cell size.
- `scene/manual.py`: 4-point -> homography matches a known projective map.
- `scene/depthpro.py`: plane-fit + cache-write on a synthetic/saved depth array
  (no model download in CI); the model run itself is exercised behind a
  marker/manual test.

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

1. v1 (this spec): required `scene/` Depth Pro calibration gate + cache, `heatmap/`
   package, single-camera floor frame, dwell-weighted grid, top-down render.
2. Later: cross-camera `T_cam_to_store` solving (stitching execution).
3. Later: gaze detection and other features reusing the scene-geometry cache.
4. Later/backburner: 3D drape render target; photoreal store model
   (photogrammetry/NeRF) as an optional viz asset hung off the floor frame.

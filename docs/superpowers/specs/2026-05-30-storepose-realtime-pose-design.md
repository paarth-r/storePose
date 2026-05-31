# storePose — Realtime Multi-Person Pose Pipeline

**Date:** 2026-05-30
**Status:** Approved

## Goal

Realtime, in-frame multi-person detector that draws a bounding box around each
person and overlays a 17-keypoint (COCO body) pose skeleton, from a live webcam,
displayed in a window. Clean and modular codebase.

## Decisions (locked)

| Decision        | Choice                                                        |
|-----------------|---------------------------------------------------------------|
| Input           | Live webcam (camera index, default 0)                         |
| Output          | Live OpenCV display window                                    |
| Backend         | `rtmlib` (ONNX Runtime) — clean install on Apple Silicon      |
| Pose detail     | Body, 17 COCO keypoints                                       |
| Speed/accuracy  | `balanced` mode (YOLOX-m detector + RTMPose-m)                |
| Environment     | `uv`-managed venv pinned to Python 3.12                       |
| Compute device  | `cpu` (default) or `mps` → CoreMLExecutionProvider            |

## Pipeline

RTMPose is top-down: it needs person boxes first. The stack is therefore a
person **detector → pose** estimator, which also yields the bounding boxes we
want to draw.

```
frame → YOLOX (person boxes, N×4 xyxy)
      → RTMPose (per box → 17 keypoints + scores)
      → drawing overlay (boxes + skeletons + FPS)
      → cv2.imshow ; 'q' quits
```

rtmlib's `Body` solution bundles these and exposes `det_model` / `pose_model`,
but its top-level call hides the boxes. We construct YOLOX + RTMPose directly
(URLs taken from `Body.MODE[mode]`) so boxes are a first-class output and each
stage is independently swappable/testable.

## Module structure

```
storePose/
├── main.py                  # CLI entrypoint: argparse → AppConfig → Runner.run()
├── pyproject.toml           # deps + Python 3.12 pin (uv)
├── src/storepose/
│   ├── config.py            # AppConfig dataclass + from_args() + validation
│   ├── model_zoo.py         # mode → (det_url, det_size, pose_url, pose_size) from rtmlib
│   ├── detector.py          # PersonDetector: wraps YOLOX → boxes (N,4)
│   ├── pose.py              # PoseEstimator: wraps RTMPose; empty-box short-circuit
│   ├── pipeline.py          # PosePipeline.process(frame) → FrameResult(boxes, kpts, scores)
│   ├── drawing.py           # annotate(frame, result, fps) → frame
│   ├── fps.py               # FpsMeter: rolling-average FPS
│   ├── video_source.py      # VideoSource: webcam context manager + frame iterator
│   └── runner.py            # capture → process → annotate → imshow loop
└── tests/                   # unit tests for pure logic
```

## Component contracts

- **`AppConfig`** (`config.py`): `source:int`, `mode:str`, `det_conf:float`,
  `kpt_thr:float`, `device:str`, `show_fps:bool`. `from_args(argv)` parses CLI
  and validates (mode ∈ {lightweight,balanced,performance}; device ∈ {cpu,mps};
  thresholds ∈ [0,1]).
- **`PersonDetector`** (`detector.py`): `detect(frame) -> np.ndarray (N,4)` xyxy.
  Wraps `YOLOX(score_thr=det_conf, mode='human')`.
- **`PoseEstimator`** (`pose.py`): `estimate(frame, boxes) -> (kpts (N,17,2),
  scores (N,17))`. **Returns empty arrays when `boxes` is empty** (rtmlib would
  otherwise pose-estimate the whole frame — undesired).
- **`PosePipeline`** (`pipeline.py`): composes detector + pose; `process(frame)
  -> FrameResult`. Built from `model_zoo` URLs for the chosen mode/device.
- **`FrameResult`**: dataclass `boxes`, `keypoints`, `scores` + `count` property.
- **`drawing.annotate`**: draws each box (cv2.rectangle + `person N` label),
  reuses rtmlib `draw_skeleton` for correct COCO-17 topology, gates keypoints by
  `kpt_thr`, optional FPS overlay. Returns the annotated frame.
- **`FpsMeter`** (`fps.py`): `tick() -> float` rolling average over a window.
- **`VideoSource`** (`video_source.py`): context manager opening the webcam;
  iterating yields BGR frames; guarantees `release()`. Raises a clear error if
  the camera cannot be opened.
- **`Runner`** (`runner.py`): owns the loop; builds pipeline + source + meter;
  `q`/Ctrl-C exit; guaranteed window + capture cleanup.

## Testing strategy

- **Unit-tested (pure logic, TDD):** `AppConfig.from_args` parsing/validation;
  `FpsMeter` rolling average; `PoseEstimator` empty-box short-circuit (injected
  fake model); `PosePipeline.process` composition + `FrameResult.count` (injected
  fake det/pose); `drawing.annotate` returns same-shape image and is a no-op-safe
  on empty results.
- **Manual verification (visual/hardware):** real model inference and the live
  window. Verified headlessly by running the full pipeline on a frame from the
  `videos/` footage and writing an annotated image to disk (boxes + skeletons
  present), since a webcam/display isn't available in the build environment.

## Error handling

- Camera open failure → clear message, non-zero exit.
- Empty/dropped frame → skip iteration.
- First-run model download failure → network hint.
- `q` / `KeyboardInterrupt` → guaranteed `cap.release()` + `destroyAllWindows()`
  via context manager.

## Dependencies

`rtmlib`, `onnxruntime`, `opencv-python`, `numpy` — installed via `uv` into a
Python 3.12 venv.

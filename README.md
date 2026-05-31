# storePose

Realtime, in-frame multi-person detector: a bounding box around every person plus
a 17-keypoint (COCO body) **RTMPose** skeleton, streamed live from a webcam.

Built on [`rtmlib`](https://github.com/Tau-J/rtmlib) (ONNX Runtime) — YOLOX for
person detection, RTMPose for pose. No `mmcv`/`mmpose` build pain.

## Setup

Requires [`uv`](https://docs.astral.sh/uv/) (`brew install uv`). Then:

```bash
uv sync
```

This creates a Python 3.12 virtualenv and installs all dependencies. Models
(~140 MB) download automatically to `~/.cache/rtmlib` on first run.

## Run

```bash
uv run python main.py                 # default webcam, balanced mode, CoreML
uv run python main.py --source 1      # a different camera
uv run python main.py --mode lightweight   # faster, slightly less accurate
uv run python main.py --device cpu    # CPU fallback if CoreML misbehaves
```

Press **`q`** or **Esc** in the window to quit.

### Flags

| Flag          | Default    | Description                                        |
|---------------|------------|----------------------------------------------------|
| `--source`    | `0`        | Webcam index.                                      |
| `--mode`      | `balanced` | `lightweight` \| `balanced` \| `performance`.      |
| `--det-conf`  | `0.5`      | Person-detection confidence threshold.             |
| `--kpt-thr`   | `0.5`      | Keypoint confidence threshold for drawing.         |
| `--device`    | `mps`      | `mps` (CoreML) or `cpu`.                           |
| `--no-fps`    | —          | Hide the FPS overlay.                              |

## Performance

Measured on an Apple M5 Max (`balanced` mode). Top-down pose runs once per
person, so framerate scales with people in frame:

| Device       | Detector | Pose/person | 1 person | 3 people |
|--------------|----------|-------------|----------|----------|
| `mps` CoreML | ~18 ms   | ~4 ms       | ~45 fps  | ~32 fps  |
| `cpu`        | ~156 ms  | ~14 ms      | ~6 fps   | ~5 fps   |

CoreML is ~8× faster here — keep the default `--device mps` for realtime. Use
`--mode lightweight` for higher framerates in crowded scenes.

## Architecture

```
main.py            CLI entrypoint
src/storepose/
  config.py        AppConfig + CLI parsing/validation
  model_zoo.py     mode -> model URLs/sizes (from rtmlib)
  detector.py      PersonDetector  (YOLOX)      -> boxes
  pose.py          PoseEstimator   (RTMPose)    -> keypoints + scores
  pipeline.py      PosePipeline.process(frame)  -> FrameResult
  drawing.py       annotate(frame, result, fps) -> annotated frame
  fps.py           FpsMeter (rolling average)
  video_source.py  VideoSource (webcam, context manager)
  runner.py        capture -> process -> annotate -> display loop
```

Each stage has one job and a clean interface; detector and pose are injectable,
so the pipeline and pose short-circuit logic are unit-tested without weights.

## Tests

```bash
uv run pytest
```

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
uv run python main.py                       # default webcam, balanced, CoreML
uv run python main.py --source 1            # a different camera
uv run python main.py --source videos/clip.mp4          # run on a video file
uv run python main.py --source videos/clip.mp4 --save out.mp4   # display + save
uv run python main.py --mode lightweight    # faster, slightly less accurate
uv run python main.py --device cpu          # CPU fallback if CoreML misbehaves
```

`--source` is a webcam index when numeric, otherwise a path to a video file.
`--save PATH` writes the annotated stream to an .mp4 (display stays live).

Press **`q`** or **Esc** in the window to quit.

### Flags

| Flag          | Default    | Description                                        |
|---------------|------------|----------------------------------------------------|
| `--source`    | `0`        | Webcam index, or path to a video file.             |
| `--save`      | —          | Write annotated output to this .mp4 path.          |
| `--mode`      | `balanced` | `lightweight` \| `balanced` \| `performance`.      |
| `--det-conf`  | `0.5`      | Person-detection confidence threshold.             |
| `--kpt-thr`   | `0.5`      | Keypoint confidence threshold for drawing.         |
| `--device`    | `mps`      | `mps` (CoreML) or `cpu`.                           |
| `--no-fps`    | —          | Hide the FPS overlay.                              |
| `--no-track`     | —       | Disable tracking; draw raw per-frame detections.  |
| `--hold-seconds` | `1.5`   | How long a lost person's box keeps coasting.      |
| `--min-hits`     | `3`     | Detections before a track is confirmed/drawn.     |
| `--iou-thr`      | `0.3`   | Min IoU to associate a detection to a track.      |
| `--no-smooth`    | —       | Disable One-Euro keypoint smoothing.              |

## Tracking & smoothing

By default each person gets a stable `ID n` and a SORT-style tracker (Kalman +
IoU) keeps that box alive through brief occlusions (coasting), while a One-Euro
filter smooths the skeleton. A box that loses its detection is predicted forward
for `--hold-seconds` (skeleton hidden while coasting), then dropped. Someone who
fully leaves and returns gets a new id (no appearance re-identification).

A/B the behavior with `--no-track` (raw per-frame boxes) and `--no-smooth`.

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
  video_source.py  VideoSource (webcam or file, context manager)
  video_sink.py    VideoSink (annotated .mp4 writer, context manager)
  runner.py        capture -> process -> (track) -> annotate -> display loop
  tracking/        SORT tracker: assignment, kalman, smoothing, track, tracker
```

Each stage has one job and a clean interface; detector and pose are injectable,
so the pipeline and pose short-circuit logic are unit-tested without weights.

## Tests

```bash
uv run pytest
```

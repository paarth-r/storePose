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
| `--det-conf`  | `0.95`     | Person-detection confidence threshold.             |
| `--kpt-thr`   | `0.5`      | Keypoint confidence threshold for drawing.         |
| `--device`    | `mps`      | `mps` (CoreML) or `cpu`.                           |
| `--no-fps`    | —          | Hide the FPS overlay.                              |
| `--no-track`     | —       | Disable tracking; draw raw per-frame detections.  |
| `--hold-seconds` | `1.5`   | How long a lost person's box keeps coasting.      |
| `--min-hits`     | `3`     | Detections before a track is confirmed/drawn.     |
| `--iou-thr`      | `0.3`   | Min IoU to associate a detection to a track.      |
| `--max-overlap`  | `0.5`   | Drop a coasting ghost overlapping another box by more than this. |
| `--no-smooth`    | —       | Disable One-Euro keypoint smoothing.              |
| `--zone`         | —       | Queue-zone JSON; enables waiting-in-line detection. |
| `--define-zone`  | —       | Launch the interactive zone editor and exit.      |
| `--wait-speed`   | `0.15`  | Max speed (body-heights/sec) counted as "slow".   |
| `--wait-enter-frames`  | `5`   | Consecutive in-zone+slow frames before WAITING. |
| `--wait-exit-seconds`  | `2.0` | Out-of-condition time before WAITING ends.     |
| `--zone-coverage`      | `0.5` | Box-fraction inside the zone when ankles are occluded. |
| `--wait-log`     | —       | Append completed waits to this CSV.               |

## Tracking & smoothing

By default each person gets a stable `ID n` and a SORT-style tracker (Kalman +
IoU) keeps that box alive through brief occlusions (coasting), while a One-Euro
filter smooths the skeleton. A box that loses its detection is predicted forward
for `--hold-seconds` (skeleton hidden while coasting), then dropped. Someone who
fully leaves and returns gets a new id (no appearance re-identification).

A/B the behavior with `--no-track` (raw per-frame boxes) and `--no-smooth`.

## Waiting in line

Define a queue area once per (fixed) camera, then storePose flags each person
**waiting** in it, shows a live **count**, and logs **per-person wait time**.

```bash
# 1. draw the queue polygon on a frame (click points, 's' to save, 'q' to quit)
uv run python main.py --define-zone --source videos/clip.mp4

# 2. run with the zone; optionally log completed waits to CSV
uv run python main.py --source videos/clip.mp4 --zone zones/clip.json --wait-log waits.csv
```

**In-zone test:** if an ankle keypoint is confident, the visible-ankle midpoint
is tested against the polygon (precise, ignores box padding such as carts). When
the feet are occluded, it falls back to *coverage* — the fraction of the box
inside the zone must be ≥ `--zone-coverage`.

A person is "waiting" once that in-zone test holds while they move slowly
(`--wait-speed`, in body-heights/sec) for `--wait-enter-frames` (default 5)
consecutive frames; they stop after `--wait-exit-seconds` out of that condition
or when their track is lost.

Visual states:
- **Joining** (candidate): an amber "sheer" fill rises over the box as a flood
  animation, with a join `%`, while the 5 frames accrue.
- **In line**: a translucent **green** overlay on the box plus a `WAIT n.n s`
  timer; the header shows `in line: N`.

Boxes are already Kalman-smoothed by the tracker (constant-velocity, low process
noise), so in-line boxes are stable. The CSV rows are
`id, entered_s, exited_s, wait_seconds`.

Requires tracking (on by default) — waiting state is keyed by stable id.

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
  queue/           zone, analyzer (waiting state machine), zone_editor
```

Each stage has one job and a clean interface; detector and pose are injectable,
so the pipeline and pose short-circuit logic are unit-tested without weights.

## Tests

```bash
uv run pytest
```

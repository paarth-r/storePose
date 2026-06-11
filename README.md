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

> Quickstart with copy-paste commands for running, zone setup, and the busy
> signal: [`docs/usage.md`](docs/usage.md).

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
| `--det-conf`  | `0.7`      | Person-detection confidence threshold.             |
| `--det-overlap` | `0.8`    | Drop a box more than this fraction contained within a larger one (duplicate-on-one-person suppression). |
| `--kpt-thr`   | `0.5`      | Keypoint confidence threshold for drawing.         |
| `--device`    | `mps`      | `mps` (CoreML) or `cpu`.                           |
| `--no-fps`    | —          | Hide the FPS overlay.                              |
| `--no-track`     | —       | Disable tracking; draw raw per-frame detections.  |
| `--hold-seconds` | `1.5`   | How long a lost person's box keeps coasting.      |
| `--min-hits`     | `3`     | Detections before a track is confirmed/drawn.     |
| `--iou-thr`      | `0.3`   | Min IoU to associate a detection to a track.      |
| `--max-overlap`  | `0.5`   | Drop a coasting ghost overlapping another box by more than this. |
| `--no-reid`      | —       | Disable appearance re-id (returning person keeps their id). |
| `--reid-seconds` | `5.0`   | How long a lost track stays re-attachable.        |
| `--reid-thr`     | `0.6`   | Appearance similarity floor for re-attach (HSV histogram correlation). |
| `--no-smooth`    | —       | Disable One-Euro keypoint smoothing.              |
| `--zone`         | —       | Queue-zone JSON; enables waiting-in-line detection. |
| `--define-zone`  | —       | Launch the interactive zone editor and exit.      |
| `--pos-zone`     | —       | Mashgin POS-zone JSON; splits line time into waiting vs serving (needs `--zone`). |
| `--define-pos-zone` | —    | Launch the editor for the POS zone and exit.       |
| `--alt-zone`     | —       | Non-Mashgin checkout zone; enables the Mashgin-vs-traditional comparison. |
| `--define-alt-zone` | —    | Launch the editor for the non-Mashgin checkout and exit. |
| `--wait-enter-frames`  | `5`   | Consecutive in-zone frames before WAITING.      |
| `--pos-enter-frames`   | `3`   | Consecutive in-POS frames before SERVING (debounces the POS edge). |
| `--wait-exit-seconds`  | `2.0` | Out-of-condition time before WAITING ends.     |
| `--zone-coverage`      | `0.5` | Foot-region fraction inside the zone when ankles are occluded. |
| `--zone-foot-band`     | `0.3` | Bottom fraction of the box used as the foot region. |
| `--wait-min-dwell`     | `0.0` | Min in-zone dwell (s) before counting as in line; rejects pass-through bystanders. |
| `--wait-log`     | —       | Append completed waits to this CSV.               |
| `--busy`         | —       | Show a live Low/Medium/High busy badge (needs `--zone`). |
| `--busy-log`     | —       | Write the per-window busy report to this CSV (implies `--busy`). |
| `--busy-window`  | `600`   | Busy aggregation window, seconds (600 = 10 min).  |
| `--busy-subwindow` | `0.0` | Two-level smoothing sub-window, seconds; 0 = off (e.g. 60 = per-minute robust estimate). |
| `--busy-metric`  | `occupancy_p90` | Window feature driving the label.         |
| `--busy-low-max` | `1.0`   | Upper bound of the LOW band (metric units). Calibrate. |
| `--busy-medium-max` | `3.0` | Upper bound of the MEDIUM band (metric units). Calibrate. |
| `--busy-hysteresis` | `0.0` | Cross-window deadband to suppress label flapping. |
| `--no-dashboard`    | —     | Disable the live web dashboard.                   |
| `--dashboard-port`  | `8000`| Port for the live dashboard server.               |

## Tracking & smoothing

By default each person gets a stable `ID n` and a SORT-style tracker (Kalman +
IoU) keeps that box alive through brief occlusions (coasting), while a One-Euro
filter smooths the skeleton. A box that loses its detection is predicted forward
for `--hold-seconds` (skeleton hidden while coasting), then dropped. By default,
appearance re-id (an HSV torso-color histogram) re-attaches a returning person to
their original id within `--reid-seconds`; disable it with `--no-reid`. A
genuinely new person still gets a new id. Each id keeps a persistent overlay
color across re-attach.

A/B the behavior with `--no-track` (raw per-frame boxes) and `--no-smooth`.

## Waiting in line

Define a queue area once per (fixed) camera, then storePose flags each person
**waiting** in it, shows a live **count**, and logs **per-person wait time**.

```bash
# 1. draw zones: '1' line, '2' Mashgin POS, '3' non-Mashgin; 'n' new contour, 's' save, 'q' quit
uv run python main.py --define-zone --source videos/clip.mp4

# 2. run with the zone; optionally log completed waits to CSV
uv run python main.py --source videos/clip.mp4 --zone zones/clip.json --wait-log waits.csv
```

The editor draws both zones in one session — **`1`** for line contours, **`2`**
for POS — and either zone may be **several disjoint contours** (`n` starts a new
one); a person counts as in-zone if inside **any** contour. `s` writes the line
contours to `zones/<name>.json` and the POS contours to `zones/<name>_pos.json`.

**In-zone test (OR of two signals):** a person counts as in-zone if **either**
the visible-ankle midpoint is inside the polygon (precise, ignores box padding
such as carts) **or** the **foot region** of the box — its bottom
`--zone-foot-band` (default 30%) — is ≥ `--zone-coverage` inside the zone.
Coverage uses only the foot region, not the whole box, because a standing
person's box is mostly torso/head that projects *above* a floor zone. The OR
means a held position isn't lost when feet leave frame or an ankle drifts
outside while the body is still in the zone — the wait timer keeps running.

A person is "waiting" once that in-zone test holds for `--wait-enter-frames`
(default 5) consecutive frames; they stop after `--wait-exit-seconds` out of the
zone or when their track is lost. There is **no motion gating** — people in line
move around (fetching items, pushing carts), so presence in the zone is what
counts, not how still they are.

Visual states (drawn in each person's **persistent id color**, not a shared hue):
- **Joining** (candidate): a "sheer" fill rises over the box as a flood
  animation, with a join `%`, while the 5 frames accrue.
- **In line**: a translucent overlay on the box plus a `WAIT n.n s` timer; the
  header shows `in line: N`. The zone polygon stays orange.

Boxes are already Kalman-smoothed by the tracker (constant-velocity, low process
noise), so in-line boxes are stable. The CSV rows are
`id, entered_s, exited_s, wait_seconds`.

Requires tracking (on by default) — waiting state is keyed by stable id.

### Waiting vs at-POS

Add a second **POS zone** (the register area) to split each visit into *waiting*
(in the line zone, not yet at POS) and *serving* (at the POS):

```bash
uv run python main.py --define-pos-zone --source videos/clip.mp4   # draw the POS polygon
uv run python main.py --source videos/clip.mp4 --zone zones/clip.json --pos-zone zones/clip_pos.json --wait-log waits.csv
```

A person flows OUT → WAITING → SERVING → done; each frame's time is attributed to
the state they are in, and time spent out of detection (re-identified within the
re-id window) is credited to the state they held when lost. With a POS zone the
wait-log columns become `id, entered_s, exited_s, wait_seconds, serving_seconds,
outcome`, where `outcome` is `served` (reached POS) or `abandoned` (left the line
first). The overlay draws the POS zone in azure and tags people `POS n.n s` while
being served; the header shows `in line: N   at POS: M`.

## How busy is the line? (busy signal)

The headline question: given store video, **how busy is the checkout line?**
storePose answers it as a stable **Low / Medium / High** label per **10-minute
window**, by aggregating the noisy per-frame waiting count into a robust
per-window statistic and mapping it to a band.

```bash
# live: badge on screen + per-window report written at exit
uv run python main.py --source videos/clip.mp4 --zone zones/clip.json \
    --busy-log busy.csv

# offline: turn an existing wait log into per-window labels
uv run python busy_report.py aggregate waits.csv --window 600 -o busy.csv

# hand-label a video's windows to build a ground-truth set (resumable)
uv run python busy_report.py label videos/clip.mp4 -o truth.csv --window 600

# score predicted labels against the ground-truth CSV
uv run python busy_report.py eval busy.csv truth.csv
```

The label is driven by a **time-weighted robust statistic** of occupancy (the
90th-percentile waiting count by default), which ignores 1–2-second flicker but
reacts to a sustained crowd. Two further stabilizers: `--busy-subwindow` adds
**two-level smoothing** (compute the metric per short sub-window, then take the
median across the window, so one busy minute can't dominate), and
`--busy-hysteresis` adds a cross-window deadband so the output doesn't oscillate
near a threshold. Bystanders who merely pass through the line zone are rejected
by `--wait-min-dwell` (a person must linger N seconds before counting).

The `label` command steps through a video window-by-window and records your
Low/Medium/High judgement to a `window_index,level` CSV — resumable, and exactly
the format `eval` consumes. The band thresholds
(`--busy-low-max`, `--busy-medium-max`) are **placeholders that must be
calibrated** on real footage against a labeled evaluation set.

See [`docs/problem-definition.md`](docs/problem-definition.md) for the precise
definition of Low/Medium/High, the alternatives considered (occupancy vs.
parties vs. wait-time), and the evaluation plan (ordinal metrics, ground-truth
labeling, cross-store protocol).

## Live dashboard

Every run auto-starts a localhost web dashboard (no extra dependencies) and opens
it in your browser, with three live charts: **occupancy** (in line / at POS with
moving averages), **wait & serve** moving averages, and **throughput**
(served/min). Disable with `--no-dashboard`; change the port with
`--dashboard-port`. On a **file** source the timeline is video time; on a **webcam**
it is real time.

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
  busy/            occupancy timeline, BusyAggregator (window -> Low/Med/High)
  eval/            ordinal metrics + ground-truth labeling helpers
busy_report.py     offline CLI: aggregate waits.csv, label video, eval vs. truth
```

Each stage has one job and a clean interface; detector and pose are injectable,
so the pipeline and pose short-circuit logic are unit-tested without weights.

## Tests

```bash
uv run pytest
```

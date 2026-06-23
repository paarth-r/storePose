# CVAT Ground-Truth Annotation — Deploy, Label Schema, and Rubric

This document defines the workflow for hand-annotating store footage in CVAT to produce ground-truth labels for evaluating the occupancy and membership stages of the storePose pipeline. See also the design spec: `docs/superpowers/specs/2026-06-22-cvat-ground-truth-annotation-design.md`.

## Why CVAT, self-hosted

CVAT is chosen for its native support of **point tracks with frame-by-frame interpolation** (enabling future identity-persistence scoring), **per-frame mutable attributes** (reserved for state and intent beyond the current membership field), and a mature keyboard-driven annotation UI. We deploy CVAT **locally via Docker Compose, not the hosted `app.cvat.ai`**, because the clips are real store footage treated as privacy-sensitive — footage must not be uploaded to any third-party cloud service. Local deployment prioritizes privacy while preserving all annotation capabilities needed for occupancy and membership evaluation.

## Deploy (local)

Clone the official CVAT repository and run a local server:

```bash
git clone https://github.com/cvat-ai/cvat
cd cvat
docker compose up -d
```

Create an admin user to log in (required once, on first startup):

```bash
docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'
```

Open a web browser and navigate to:

```
http://localhost:8080
```

Log in with the credentials you created. You are now ready to create a task and upload a store clip.

## Label schema

### Structured data

Create a task with the following label configuration:

- **Label:** `person`
- **Shape:** `rectangle` (a bounding box per person)
- **Tracking mode:** track (each person gets one persistent track ID across frames)

> The box's **bottom-center** is taken as the person's ground point, so occupancy
> and membership scoring are identical whether a person is annotated as a box or
> (historically) a single point. Boxes are used because they are far faster to
> *review*: the pipeline pre-annotates every person as a box track (below), and a
> human drags, deletes, or re-labels rather than placing points from scratch.

### Attributes (all mutable frame-by-frame)

Each track has three attributes, all of which can change on every frame:

1. **`membership`** (select)
   - Values: `in_line`, `bystander`
   - Description: Whether the person is genuinely waiting in the checkout queue (`in_line`) or passing through without joining the queue (`bystander`).
   - Status: In active use.

2. **`state`** (text, default empty)
   - Reserved for future per-frame state labels (e.g., "looking_at_goods", "paying").
   - Status: Reserved; not scored in this phase.

3. **`intent`** (text, default empty)
   - Reserved for future per-frame intent labels (e.g., "approaching", "leaving").
   - Status: Reserved; not scored in this phase.

### CVAT internals

`track_id` is CVAT's native per-track persistent identifier. It is automatically assigned and carries forward across all frames of a person's presence in the video. No configuration is needed — it is automatic.

## Annotation rubric

### In-line vs. bystander definition

A person is **in line** if they are genuinely waiting to be served at the checkout. A person is a **bystander** if they are passing through the checkout area without queuing — e.g., walking across the zone, browsing nearby merchandise, or waiting for someone else without themselves intending to purchase.

**Operationally:**
- **In line:** The person is stationary or making minimal, localized movement (shuffling forward in the queue, looking around while standing still). They have a clear intent to wait their turn.
- **Bystander:** The person is transiting the zone (walking through with sustained displacement), standing still but facing away from the queue, or lingering for a reason unrelated to queuing (waiting for a friend, inspecting merchandise).

When unsure, err toward `in_line` if the person shows sustained presence with minimal net displacement; mark `bystander` if the person is clearly in motion across the zone.

### Marking entry and exit

When a person **enters** the frame or checkout area, create a new track and mark the first keyframe with the appropriate `membership`. If the person is outside the area of interest initially, mark the opening keyframe with CVAT's `outside` flag (do not change this flag later in that track).

When a person **leaves** the frame or area, mark the final keyframe with CVAT's `outside` flag. This flag signals to the converter that the person is no longer present at frames after this keyframe, ensuring the occupancy count does not include them past their departure.

### Keyframe discipline

Add a keyframe when:
- The person's **position changes materially** (e.g., they shuffle forward in line, walk to a different area).
- The person's **`membership` attribute changes** (e.g., they transition from bystander to in_line, or vice versa).
- The person **enters or leaves** the frame/area (mark the frame with `outside` as described above).

Between keyframes, CVAT's linear interpolation will smoothly move the point. You do **not** need to keyframe every frame — interpolation handles the motion. This approach reduces annotation effort while preserving positional accuracy for future association work (phase 2).

## Pre-annotation (model-assisted labeling)

Rather than annotate from a blank task, let the storePose pipeline detect and
track every person first, then **review** its output in CVAT — drag a box that
drifted, delete a false positive (e.g. a merchandise stand mis-detected as a
person), and flip `membership` where the default is wrong. This is the
Roboflow-style "approve/deny" loop, but the detector is your own pipeline, so
the tracks carry stable IDs for free.

Generate the pre-annotation XML:

```bash
uv run python busy_report.py export-cvat videos/clip.mp4 -o clip_preanno.xml
# or just the first N frames while iterating:
uv run python busy_report.py export-cvat videos/clip.mp4 -o clip_preanno.xml --max-frames 750
```

Each confirmed track becomes one CVAT `box` track (a keyframe per frame, stable
ID, `membership` pre-filled to `in_line`). Then in the CVAT task: **Menu ->
Upload annotations**, format **"CVAT for video 1.1"**, select `clip_preanno.xml`.
Upload into a task whose `person` label is already configured as a `rectangle`
with the attributes above and whose uploaded video matches the clip exactly
(same frames), or the boxes will land on the wrong frames.

Review, then re-export (next section) and score — the round-trip is closed:
`export-cvat` and `import-cvat` read/write the same box-track format.

## Export and score

The end-to-end workflow is a three-command loop:

### Step 1: Export from CVAT

In the CVAT UI, open the task and navigate to **Menu → Export task dataset**. Select the format **"CVAT for video 1.1"** and download the resulting `export.xml`.

### Step 2: Import to occupancy ground truth

```bash
uv run python busy_report.py import-cvat export.xml --fps 30 --step 1 -o gt_occupancy.csv
```

**Arguments:**
- `export.xml` — the downloaded CVAT export file.
- `--fps 30` — the clip's actual frame rate. **CRITICAL: this value must match the real frame rate of the clip you annotated, or timestamps will misalign.** Check your source footage (e.g., `ffprobe videos/clip.mp4 | grep fps`). If you get it wrong, the GT and predicted timelines will not align.
- `--step 1` — occupancy sampling interval in seconds (default: 1.0). **CRITICAL: this must match the `--step` value you use in the eval command (step 3), or the timelines will not align.**
- `-o gt_occupancy.csv` — output file containing the ground-truth occupancy timeline (columns: `t_s`, `occupancy`).

### Step 3: Run the pipeline to generate predictions

Run the main detection and tracking pipeline on the same clip:

```bash
uv run python main.py --source videos/clip.mp4 --zone zones/clip.json --wait-log waits.csv
```

This produces `waits.csv`, a log of detected wait intervals in the checkout zone.

### Step 4: Score predictions against ground truth

```bash
uv run python busy_report.py eval-occupancy gt_occupancy.csv waits.csv --step 1
```

**Arguments:**
- `gt_occupancy.csv` — the ground-truth occupancy CSV you created in step 2.
- `waits.csv` — the predicted wait log from step 3.
- `--step 1` — **CRITICAL: must match the `--step` value from the import-cvat command, or the timelines will not align and the metrics will be meaningless.**

The output includes MAE (mean absolute error), bias (systematic over/under-counting), and Pearson correlation.

### CRITICAL ALIGNMENT REQUIREMENTS

**If `--fps` is wrong, or if `--step` does not match between `import-cvat` and `eval-occupancy`, the ground-truth and predicted timelines will not align and the eval metrics will be invalid.** Always:

1. Verify the clip's true frame rate and pass it to `import-cvat` via `--fps`.
2. Use the same `--step` value in both `import-cvat` and `eval-occupancy`.
3. Document your choices in the eval report so results can be reproduced.

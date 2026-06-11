# storePose — Usage

Run the realtime pipeline, draw overlays, define a queue zone, and read the
busy signal. For *what the pipeline does* internally see the
[Architecture](../README.md#architecture) section; for the *definition* of
Low/Medium/High see [`problem-definition.md`](problem-definition.md).

---

## 1. Install

Requires [`uv`](https://docs.astral.sh/uv/) (`brew install uv`).

```bash
uv sync          # Python 3.12 venv + deps
```

Models (~140 MB, YOLOX + RTMPose) download to `~/.cache/rtmlib` on first run.

---

## 2. Run with overlays

Overlays are **on by default** — every run draws a box and a 17-keypoint
skeleton on each person, plus a rolling FPS counter. There is no "enable
overlays" flag; `--zone` and `--busy` simply add more overlay layers.

```bash
# default webcam, balanced model, CoreML — boxes + skeletons + FPS
uv run python main.py

# a video file instead of the webcam
uv run python main.py --source videos/clip.mp4

# full overlays: + queue waiting states + live Low/Med/High busy badge
uv run python main.py --source videos/clip.mp4 --zone zones/clip.json --busy

# also bake the overlays into an .mp4 (window stays live)
uv run python main.py --source videos/clip.mp4 --zone zones/clip.json --busy \
    --save out.mp4
```

`--source` is a **webcam index** when numeric (`0`, `1`, …), otherwise a **path
to a video file**. Press **`q`** or **Esc** in the window to quit.

### Overlay layers

| Layer | Shown when | What you see |
|-------|-----------|--------------|
| Boxes + skeletons | always | Per-person box with a stable `ID n` and a colored COCO skeleton. Skeleton is hidden while a box is *coasting* (predicted through an occlusion). |
| FPS | always (hide with `--no-fps`) | Rolling average framerate, top-left. |
| Queue | `--zone PATH` | The zone polygon (orange); a rising fill + `%` for candidates and a full fill + `WAIT n.n s` for people in line, each in that person's **persistent id color**; an `in line: N` header count. |
| Busy badge | `--busy` (needs `--zone`) | A `LOW / MEDIUM / HIGH` badge, the current metric value, and a countdown to the end of the active window. |

### Useful run flags

| Flag | Default | Effect |
|------|---------|--------|
| `--mode` | `balanced` | `lightweight` (faster) \| `balanced` \| `performance` (most accurate). |
| `--device` | `mps` | `mps` (CoreML, ~8× faster) or `cpu` fallback. |
| `--det-conf` | `0.7` | Person-detection confidence threshold. |
| `--det-overlap` | `0.8` | Drop a box more than this fraction contained within a larger one (duplicate-on-one-person suppression). |
| `--kpt-thr` | `0.5` | Keypoint confidence threshold for drawing / ankle test. |
| `--no-fps` | — | Hide the FPS overlay. |
| `--no-track` | — | Raw per-frame boxes, no stable IDs (disables queue/busy). |
| `--no-reid` | — | Disable appearance re-id; a returning person gets a new id. |
| `--reid-seconds` | `5.0` | How long a lost track stays re-attachable. |
| `--reid-thr` | `0.6` | Appearance similarity floor for re-attach. |
| `--no-smooth` | — | Disable One-Euro keypoint smoothing. |
| `--save PATH` | — | Write the annotated stream to an `.mp4`. |

Full flag list: `uv run python main.py --help`, or the table in the
[README](../README.md#flags).

### Live dashboard

Auto-starts at `http://127.0.0.1:8000/` (override `--dashboard-port`; disable with
`--no-dashboard`) and opens in your browser. Three live charts — occupancy,
wait/serve moving averages, and throughput — fed by the running pipeline. Most
useful with `--zone` / `--pos-zone`. The timeline is video time for a file source,
real time for a webcam.

---

## 3. Define a queue zone

A zone is a **polygon in image coordinates**, drawn once per fixed camera and
saved as JSON (`{"points": [[x, y], ...]}`). The same camera angle reuses the
same zone file; move the camera and you re-draw it.

```bash
# open the first frame of the source and draw the polygon
uv run python main.py --define-zone --source videos/clip.mp4
```

**Editor controls** (window title `define zones`). One session draws both the
line and POS zones, and a zone may have several disjoint contours — a person
counts as in-zone if inside **any** contour:

| Action | Key / mouse |
|--------|-------------|
| Switch to line / POS contours | `1` / `2` |
| Add a point | left-click |
| Finish contour, start a new one | `n` (needs ≥ 3 points) |
| Undo last point | `u` |
| Clear everything | `c` |
| Save (writes line + POS files) | `s` |
| Quit without saving | `q` / Esc |

Saved path defaults to `zones/<name>.json` — `zones/cam0.json` for a webcam
index, or `zones/<filename-stem>.json` for a video (e.g. `videos/clip.mp4` →
`zones/clip.json`). Pass `--source` so the editor reads a frame from the exact
camera/clip you'll analyze.

**Where to draw it:** trace the floor area people stand on while waiting — the
*foot region*, not head height. The in-zone test is anchored to ankles / the
bottom of each box, so a zone drawn around the floor of the line is what counts.

Then run with it (Section 2), optionally logging completed waits:

```bash
uv run python main.py --source videos/clip.mp4 --zone zones/clip.json \
    --wait-log waits.csv
```

`waits.csv` rows are `id, entered_s, exited_s, wait_seconds`.

### How "waiting" is decided

A person counts as **in-zone** if *either* signal holds (OR, so a held position
survives one signal dropping out):

- their **visible-ankle midpoint** (COCO 15/16) is inside the polygon — precise,
  ignores box padding such as carts; **or**
- the **foot region** of their box — its bottom `--zone-foot-band` (default 30%)
  — is ≥ `--zone-coverage` (default 0.5) inside the polygon — robust when ankles
  are occluded or out of frame.

They become **waiting** once that holds for `--wait-enter-frames` (default 5)
consecutive frames **and** `--wait-min-dwell` seconds of accumulated in-zone time
(0 = off; raise it to reject people who merely pass through). They stop after
`--wait-exit-seconds` (default 2.0) out of the zone, or when their track is lost.
There is **no motion gating** — people in line shuffle and fetch items, so
presence over time is what counts, not stillness.

Tuning knobs:

| Flag | Default | Effect |
|------|---------|--------|
| `--zone-coverage` | `0.5` | Min foot-region fraction inside the zone (occlusion fallback). |
| `--zone-foot-band` | `0.3` | Bottom fraction of the box treated as the foot region. |
| `--wait-enter-frames` | `5` | Consecutive in-zone frames before WAITING. |
| `--wait-exit-seconds` | `2.0` | Out-of-zone time before a wait ends. |
| `--wait-min-dwell` | `0.0` | Min in-zone dwell (s) before counting — the bystander filter. |
| `--wait-log PATH` | — | Append completed waits as CSV. |
| `--pos-zone PATH` | — | POS zone; splits line time into waiting vs serving (adds `serving_seconds,outcome` CSV columns). |
| `--define-pos-zone` | — | Draw the POS polygon and exit. |

---

## 4. Busy signal (Low / Medium / High)

storePose collapses the noisy per-frame waiting count into a stable
**Low / Medium / High** label per **10-minute window**.

```bash
# live: badge on screen + per-window report written at exit
uv run python main.py --source videos/clip.mp4 --zone zones/clip.json \
    --busy-log busy.csv

# offline: turn an existing wait log into per-window labels
uv run python busy_report.py aggregate waits.csv --window 600 -o busy.csv

# hand-label a video to build ground truth (resumable)
uv run python busy_report.py label videos/clip.mp4 -o truth.csv --window 600

# score predicted labels against ground truth
uv run python busy_report.py eval busy.csv truth.csv
```

The label comes from a **time-weighted robust statistic** of occupancy
(`--busy-metric`, default `occupancy_p90` = 90th-percentile waiting count),
compared to two cut points:

```
value <= --busy-low-max     -> LOW       (default 1.0)
value <= --busy-medium-max  -> MEDIUM    (default 3.0)
value >  --busy-medium-max  -> HIGH
```

> ⚠️ The default thresholds are **placeholders** — calibrate them on real
> footage against a labeled set (the `label` → `eval` loop above). See
> [`problem-definition.md`](problem-definition.md).

Stabilizers:

| Flag | Default | Effect |
|------|---------|--------|
| `--busy` | — | Show the live badge (needs `--zone`). |
| `--busy-log PATH` | — | Write the per-window report at exit (implies `--busy`). |
| `--busy-window` | `600` | Window length, seconds (600 = 10 min). |
| `--busy-metric` | `occupancy_p90` | `occupancy_{p90,mean,median,max}` or `mean_wait`. |
| `--busy-subwindow` | `0` | Two-level smoothing: per-sub-window stat, median across window (e.g. 60). 0 = off. |
| `--busy-hysteresis` | `0` | Cross-window deadband so the label doesn't flap near a cut point. |
| `--busy-low-max` | `1.0` | Upper bound of the LOW band (metric units). |
| `--busy-medium-max` | `3.0` | Upper bound of the MEDIUM band (metric units). |

---

## 5. Tests

```bash
uv run pytest
```

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
| Detector confidence | `--conf` | Each person's YOLOX detection score (e.g. `ID 3  0.91`) next to their box/ID. Hidden while coasting (no fresh detection). |
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
| `--conf` | — | Overlay each person's detector confidence next to their box/ID. |
| `--no-track` | — | Raw per-frame boxes, no stable IDs (disables queue/busy). |
| `--no-reid` | — | Disable appearance re-id; a returning person gets a new id. |
| `--reid-seconds` | `5.0` | How long a lost track stays re-attachable. |
| `--reid-thr` | `0.6` | Appearance similarity floor for re-attach. |
| `--no-smooth` | — | Disable One-Euro keypoint smoothing. |
| `--save PATH` | — | Write the annotated stream to an `.mp4`. |
| `--debug` | — | Step through frames (scrub a rolling buffer); read each person's classification in the dashboard Debug tab. |

Full flag list: `uv run python main.py --help`, or the table in the
[README](../README.md#flags).

### Live dashboard

Auto-starts at `http://127.0.0.1:8000/` (override `--dashboard-port`; disable with
`--no-dashboard`) and opens in your browser. Live charts — occupancy, wait/serve
moving averages, throughput, and the Mashgin-vs-non-Mashgin checkout comparison —
fed by the running pipeline. A **Debug** tab lists each tracked person's
classification (state, wait/serve timers, speed, line/POS/REG membership) for the
current frame. Most useful with `--zone` / `--pos-zone`. The timeline is video time
for a file source, real time for a webcam.

### Frame-by-frame debug view (`--debug`)

To verify correctness by eye, step through the video one frame at a time while the
dashboard **Debug** tab shows why each person is classified the way they are:

```bash
uv run python main.py --source videos/clip.mp4 --zone zones/clip.json \
    --pos-zone zones/clip_pos.json --debug
```

The pipeline only ever moves **forward**; `←` is review-only over a rolling
300-frame buffer (older frames already processed). Controls in the video window:

| Action | Key |
|--------|-----|
| Step one frame / advance the video | `→` or `Space` |
| Review the previous (older) buffered frame | `←` |
| Play / pause | `c` / `p` |
| Quit | `q` / Esc |

The Debug tab follows the **viewed** frame, so scrubbing back also rewinds the
per-person table (the cumulative charts stay at the latest processed frame).

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
| Switch to line / Mashgin POS / non-Mashgin contours | `1` / `2` / `3` |
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
| `--transit-speed` | `0.4` | Reject walk-throughs: directional speed (body-heights/sec) above which a person counts in no zone; `0` disables. |
| `--wait-log PATH` | — | Append completed waits as CSV. |
| `--pos-zone PATH` | — | Mashgin POS zone; splits line time into waiting vs serving (adds `serving_seconds,outcome` CSV columns). |
| `--define-pos-zone` | — | Draw the POS polygon and exit. |
| `--alt-zone PATH` | — | Non-Mashgin checkout zone; the dashboard shows avg serve time at Mashgin (green) vs non-Mashgin (red). |
| `--define-alt-zone` | — | Draw the non-Mashgin checkout polygon and exit. |

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

> ⚠️ The manual thresholds above are **placeholders**. Prefer **calibration**
> (below), which infers per-view bands from the clip itself.

Stabilizers:

| Flag | Default | Effect |
|------|---------|--------|
| `--busy` | — | Show the live badge (needs `--zone`). |
| `--busy-log PATH` | — | Write the per-window report at exit (implies `--busy`). |
| `--busy-window` | `600` | Window length, seconds (600 = 10 min). |
| `--busy-metric` | `occupancy_p90` | `occupancy_{p90,mean,median,max}` or `mean_wait`. |
| `--busy-subwindow` | `0` | Two-level smoothing: per-sub-window stat, median across window (e.g. 60). 0 = off. |
| `--busy-hysteresis` | `0` | Cross-window deadband so the label doesn't flap near a cut point. |
| `--busy-live-window` | `30` | Trailing seconds the live badge summarises, so it tracks recent activity (not the whole 10-min window). |
| `--busy-low-max` | `1.0` | Upper bound of the LOW band (metric units). |
| `--busy-medium-max` | `3.0` | Upper bound of the MEDIUM band (metric units). |

### Calibration (per-view bands, inferred from a clip)

Instead of guessing thresholds, infer them from a representative clip. Calibration
is its own command — **headless by default**, `-v` shows a preview window:

```bash
# one full-clip pass; writes calib/<stem>.json
uv run python main.py --calibrate --source videos/clip.mp4 --zone zones/clip.json \
    --pos-zone zones/clip_pos.json
```

It collects occupancy per sub-window across the whole clip and derives three
candidate band sets, then **auto-selects a default** from the clip's shape:

| strategy | cuts by | meaning |
|----------|---------|---------|
| `skewed` | time percentiles `p60`/`p85` | busy is rare (top ~15% of the time) |
| `thirds` | time percentiles `p33`/`p66` | busiest third of the time |
| `peak`   | `0.30`/`0.70` × peak occupancy | line is >70% as full as it ever got |

Auto-default uses `fill_ratio = median/peak`: a line that **empties out** (low
ratio) → `skewed`; one that **sits full** (ratio ≥ 0.5, no quiet baseline) →
`peak`. The pick is stored in the calib file, so runs need no flag.

```bash
# run using the calibrated bands (auto-default strategy, no flag needed)
uv run python main.py --source videos/clip.mp4 --zone zones/clip.json --busy \
    --calib calib/clip.json
# override the strategy to compare by eye:
uv run python main.py ... --busy --calib calib/clip.json --busy-strategy peak
```

`--calib` overrides the manual `--busy-*-max` flags. `view-setup.sh`-generated
run scripts pick up `calib/<stem>.json` automatically once it exists. Calib files
are per-view and gitignored (like `zones/`).

| Flag | Default | Effect |
|------|---------|--------|
| `--calibrate` | — | Infer bands for `--source` and write `calib/<stem>.json` (needs `--zone`). |
| `-v` / `--verbose` | — | During `--calibrate`, show an annotated preview window (else headless). |
| `--calib PATH` | — | Load calibrated bands at run time (overrides `--busy-*-max`). |
| `--busy-strategy` | auto | Override the calib file's auto-selected strategy (`skewed`/`thirds`/`peak`). |

---

## 5. Tests

```bash
uv run pytest
```

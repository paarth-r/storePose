# AGENTS.md — how to work in storePose

Onboarding for AI agents and new contributors. **Assume nothing**: this repo has
several non-obvious traps that are documented here so you don't re-trip them. When
a fact here disagrees with your assumption, trust this doc but verify against the
named source file before acting.

Deep project history and rationale live in the owner's Obsidian knowledgebase
(`Claude Memory/project_storepose.md` for durable engineering state; `wiki/topics/mashgin.md`
for the full multi-session build log). This file distills the parts an agent needs
to read and edit the code; it is not the complete history.

---

## 1. What this is

storePose turns fixed store-camera video into a live answer to "how busy is the
checkout line right now?" plus a measured speed comparison between a Mashgin
self-checkout and a staffed lane. It is the deliverable for a Mashgin research
project ("How Busy Is the Line?").

Pipeline (one direction, per frame):

```
Camera/video → YOLOX detection → RTMPose 17-keypoint pose →
SORT tracker (Kalman + IoU + OSNet appearance + OC-SORT motion) →
Queue state machine (OUT → WAITING → SERVING) →
Busy aggregator (Low/Med/High per 10-min window) → live Next.js dashboard
```

It runs on Apple Silicon via ONNX Runtime + CoreML (no `mmcv`/`mmpose`, no CUDA).

---

## 2. Quickstart

```
uv sync                                              # Python 3.12 venv + deps
uv run python main.py                                # default webcam
uv run python main.py --source videos/test.mp4       # a clip
./view-setup.sh -v videos/clip.mp4                   # draw zones → calibrate → run + dashboard (recommended)
./video-run.sh                                        # arrow-key launcher over saved views
uv run pytest                                         # ~427 tests; the ONLY quality gate
```

- **Package manager:** `uv`. Python is pinned `>=3.12,<3.13` (newer Pythons lack ML
  wheels). Deps: `rtmlib`, `onnxruntime`, `opencv-python`, `numpy<2.3`, `scipy`; dev: `pytest>=8`.
- **Models** (~140 MB YOLOX + RTMPose) auto-download to `~/.cache/rtmlib` on first run;
  OSNet re-id weights to `~/.cache/storepose/reid/`.
- **Entry points:** dev is `uv run python main.py <args>`; installed CLI is `uv run storepose <args>`.
- **Canonical smoke run** (`run.sh`, kept single-line on purpose):
  `uv run python main.py --source videos/test.mp4 --zone zones/test.json --pos-zone zones/test_pos.json --busy --wait-log waits.csv --busy-log busy.csv`
- **Device:** default `--device mps` (CoreML, ~8× faster than CPU). Only `cpu`/`mps`
  are valid. There is **no linter/formatter/type-checker** — `uv run pytest` is the
  only gate. Run it from the repo root before claiming anything works.

---

## 3. Repo map

### Top-level directories (read this before touching files)

- **`zones/`** — polygon JSONs per video stem. `<stem>.json` = line/queue zone,
  `<stem>_pos.json` = Mashgin POS, `<stem>_alt.json` = staffed lane, `<stem>_blur.json`
  = censor region. Format: `{"polygons": [[[x,y], ...]]}`.
- **`calib/`** — **BUSY-NESS calibration, NOT camera/geometry calibration.** The name
  is misleading. Each `<stem>.json` holds an occupancy distribution and derived
  Low/Med/High band thresholds (strategies: `skewed`/`thirds`/`peak`). `--calibrate`
  writes one; `--calib PATH` loads one. **There is no camera-intrinsics/homography
  calibration anywhere in this project.**
- `runs/` — run outputs (saved mp4s, `*_waits.csv`, `*_busy.csv`); gitignored.
- `outputs/` — eval/ground-truth artifacts (`gt.csv`, exported XML).
- `videos/` — source clips (`test.mp4`, `maricopa-{0,1}.mp4`, `cumberland/` chunks).
- `demo/` — pre-rendered demo assets for docs.
- `scripts/` — dev utilities (`chunk_video.py` splits long clips into `partNN.mp4`).
- `viewscripts/` — **auto-generated** per-view run scripts (gitignored except
  `cvat-annotate.sh`). Regenerate via `view-setup.sh`; do not hand-edit.
- `web/` — the Next.js dashboard (see §4).
- `docs/` — `problem-definition.md`, `tracking-id-switches.md`, `usage.md`,
  `annotation-cvat.md`, and `superpowers/specs|plans/` (per-feature design specs that
  double as the spec the tests are written against).

### Source modules (`src/storepose/`)

Top-level: `pipeline` (compose detector+pose → `FrameResult`), `detector`
(YOLOX + `filter_confident` + duplicate suppression), `pose` (RTMPose, 17 kpts),
`config` (all CLI flags + `AppConfig` — single source of truth), `drawing`,
`faces` (pixelation), `video_source`/`video_sink`, `model_zoo`, `runner`
(the realtime loop + `build_tracker`/`build_analyzer`), `launcher`/`launcher_core` (TUI).

- `tracking/` — `kalman`, `track`, `tracker` (SORT `update()`), `assignment`
  (IoU + appearance + motion cost, Hungarian), `osnet`/`appearance` (re-id models
  behind an `AppearanceModel` protocol), `smoothing` (One-Euro), `reid_zoo`, `types` (`TrackedPerson`).
- `queue/` — `analyzer` (OUT/WAITING/SERVING state machine), `zone`, `zone_editor`,
  `outliers`, `types` (`PersonStatus`, `CompletedWait`, `QueueResult`).
- `busy/` — `aggregator` (windowed Low/Med/High), `occupancy`, `calibrate`, `report`, `types`.
- `dashboard/` — `server` (stdlib HTTP), `metrics` (JSON payload builders), `state`
  (thread-safe buffers), `stream`/`stream_payload` (SSE feed), `page` (legacy HTML), `panel` (cv2 panel for recordings).
- `eval/` — `cvat_import`/`cvat_export` (CVAT XML ↔ GT), `occupancy_eval`, `metrics`, `labeling`.

### Per-frame data flow (loop in `runner.py:367`)

`VideoSource` → `PosePipeline.process` (`detector.detect` → `pose.estimate` →
`FrameResult`) → `MultiObjectTracker.update` (appearance extract → Kalman predict →
`assignment.match` → update/reattach/gallery → `TrackedPerson[]`) →
`QueueAnalyzer.update` (ankle-in-zone test → state machine → `QueueResult`) →
`DashboardState.observe/add_visit` + `BusyAggregator` → `StreamHub.publish`
(lazy JPEG + overlay over SSE) → `cv2.imshow` / `VideoSink`.

**Keypoints** are COCO-17 (`NUM_KEYPOINTS=17`). Ankles **L=15, R=16** (`queue/analyzer.py:12`;
only consumer is `_in_zone`). Face 0–4 (`faces.py`). Torso shoulders/hips 5/6/11/12
(`tracking/appearance.py` crop). The in-zone test is: ankle midpoint inside the
polygon, OR (if ankles occluded) bottom-band box coverage ≥ `--zone-coverage`.

### Files that are large — read fully before editing

- `config.py` (~708 lines) — intentionally monolithic; the single source of truth for
  every flag, default, and validation. Scan it before assuming any default.
- `runner.py` (~497) — the integration hub; `run()` is long and wires every subsystem.
- `queue/analyzer.py` (~407) — dense state machine; transit filter, in-zone test, and
  the three states are interleaved.
- `dashboard/page.py` (~415) — almost entirely a raw HTML/CSS/JS string (the legacy page).

---

## 4. The web dashboard

Next.js 15 / React 19 / Tailwind v4, charts are hand-rolled SVG (no chart library).
It is a **static export** (`output: "export"` in `web/next.config.mjs`) to `web/out/`,
served by the Python process — no Node runtime at display time.

- **`web/out/` is gitignored and never committed.** It must exist on disk for the
  polished UI to serve. **After editing anything under `web/{app,components,lib}` or
  the config, rebuild:** `cd web && npm run build` (first time also `npm install`).
- `video-run.sh` auto-rebuilds when `web/out/index.html` is older than the `web/`
  sources. A manual `uv run python -m storepose.launcher ...` does **not** rebuild.
- **Serving** (`dashboard/server.py`, stdlib `ThreadingHTTPServer`): `static_dir = web/out`
  if it exists, else `None` (`runner._web_export_dir`). Routes: `GET /metrics` (JSON,
  polled ~1 Hz), `GET /stream` (SSE: face-blurred JPEG + overlay per event), else static.
  **If `web/out` is absent/stale the server silently falls back to the legacy
  `dashboard/page.py` PAGE_HTML** — you'll see an older UI with no error. Rebuild to fix.
- Dev hot-reload: `npm run dev` (port 3000) proxies `/metrics` and `/stream` to the
  pipeline on :8000.

---

## 5. CLI surface

`config.py` is the authoritative list. Flag groups and the defaults worth knowing:

- **Detection:** `--det-conf` (0.5, data-backed), `--det-overlap` (0.8, containment
  suppression), `--kpt-thr` (0.5).
- **Tracking:** `--hold-seconds` (1.5), `--coast` (OFF), `--predict-drift` (OFF),
  `--stationary-seconds` (20) / `--stationary-radius` (0.03), `--min-hits` (3), `--iou-thr` (0.3).
- **Re-id:** `--reid-backend` (`osnet-x025`), `--reid-thr` (0.8 osnet / 0.6 histogram),
  `--reid-seconds` (CLI 10.0), `--reid-assoc-weight` (0.4), `--reid-assoc-floor` (0.6),
  `--reid-assoc-motion` (0.3), `--no-reid`.
- **Zones:** `--zone`, `--pos-zone`, `--alt-zone`, `--blur-zone`, the `--define-*`
  editors, `--zone-coverage` (0.5), `--zone-foot-band` (0.3), `--wait-enter-frames` (5),
  `--transit-speed` (0.4), `--pos-reassign-seconds` (20).
- **Busy:** `--busy`, `--busy-window` (600 s), `--busy-metric` (`occupancy_p90`),
  `--busy-low-max`/`--busy-medium-max` (1.0/3.0), `--calibrate`, `--calib PATH`, `--busy-strategy`.
- **Dashboard:** `--no-dashboard`, `--dashboard-port` (8000), `--num-mashgins` (1),
  `--debug`, `--save-mp4`.

---

## 6. Conventions

- **Single-line shell commands only.** Backslash-newline continuations silently drop
  trailing flags on this Mac — never split a command across lines.
- **No emojis anywhere** — commits, PRs, docs, code, and any output.
- **Git:** branch off `main`, PR back into `main`. Conventional-commit prefixes:
  `feat(scope):`, `fix(scope):`, `tune(scope):` (measurement-backed tuning), `docs:`;
  scopes seen: `track`, `reid`, `detect`, `annotate`, `launcher`, `eval`. Commit in
  logical steps as you go, not one big dump at the end.
- **Video codec is `avc1` (H.264), never `mp4v`** — `mp4v` renders as green static on macOS.

---

## 7. Privacy (hard rule)

Real store footage must **never** reach public GitHub (it has nearly leaked three times).
Face-blur is **on by default** (`--no-blur-faces` to disable); `--blur-zone` censors
proprietary regions in live view, recordings, and the browser feed. CVAT for ground-truth
annotation runs **locally only** (Docker) — never upload clips to a hosted service.
When adding README/doc images, use UI-only screenshots, not annotated real frames.

---

## 8. Gotchas / hard-won lessons

1. **`calib/` is busy-ness calibration, not camera geometry.** (§3.)
2. **Verify every knob is actually applied.** `--det-conf` was a silent no-op for
   months: the balanced YOLOX ONNX embeds NMS and `postprocess` filtered at a hardcoded
   `0.3`. Fixed via `filter_confident()` in `detector.py`. Default is 0.5 (props don't
   separate from people by score at any threshold — use the stationary filter, not a
   higher conf).
3. **`Runner.run()` always opens a cv2 window** (`imshow`/`waitKey`). Only `--calibrate`
   is headless; there is no general no-display batch mode. Headless work drives the
   `detector`→`pose`→`pipeline` stages directly.
4. **`web/out` is a gitignored build artifact** — rebuild for UI changes or you get the
   silent legacy fallback. (§4.)
5. **mps zero-detection crash on sparse frames** is pre-existing — run smoke tests with
   `--device cpu`.
6. **Re-id is an association/gating problem, not an embedding-quality one.** `osnet-x1`
   benchmarks as a wash vs the default `osnet-x025`; the limiter is the domain gap
   (street-trained → overhead store cam) and look-alikes. Don't chase a bigger model;
   fix the gating. Appearance is fused into the primary match cost
   (`--reid-assoc-weight`/`--reid-assoc-floor`), not just a fallback.
7. **`--coast` and `--predict-drift` are both OFF by default.** An undetected track is
   invisible but still re-attachable; it is not drawn from a coasted/extrapolated box.
8. **`viewscripts/<stem>.sh` are auto-generated** by `view-setup.sh` — regenerate, don't edit.
9. **`--reid-seconds` mismatch:** the `AppConfig` dataclass default is 15.0 but the CLI
   default is 10.0; the CLI value governs actual runs.

---

## 9. Editing guidance

- Read the large files (§3) in full before changing them; `config.py` first for any flag.
- Reuse what exists — `build_tracker`/`build_analyzer` in `runner.py`, the
  `AppearanceModel` protocol in `tracking/appearance.py`, the `Zone` helpers, the
  `metrics.py` payload builders — rather than adding parallel code.
- After any change, run `uv run pytest` from the repo root and report the real result.
  Don't claim a fix works from logs/CSVs alone — for visual/pipeline changes, view an
  actual rendered frame.
- Design specs and plans for past features live in `docs/superpowers/specs/` and
  `docs/superpowers/plans/` — check there for the rationale behind a subsystem before
  reworking it.

# Re-id: learned OSNet embedding behind the appearance seam

**Date:** 2026-06-22
**Status:** Approved design, pre-implementation
**Branch:** TBD (feature branch off `main`)

## Goal

Make re-identification robust enough to fix four observed failures: false
merges (a returning detection revived under the wrong id), id swaps during
crossings, confusion between people in similar-colored clothing / under lighting
shifts, and re-id after a person fully exits and re-enters the frame. The root
cause shared by three of these is the weak appearance descriptor — an HSV
torso-color histogram. This sub-project replaces it with a learned ReID
embedding (OSNet ONNX) behind the existing `AppearanceModel` seam, and upgrades
per-track appearance memory from a single EMA vector to a feature gallery.

This is **sub-project 1 of 3** in a decomposition agreed during brainstorming:

1. **Stronger appearance embedding (this spec)** — foundation; fixes
   similar-clothing, false merges, re-entry.
2. **Approach B** — fold appearance into *active* frame-to-frame matching with a
   matching cascade (DeepSORT-style); fixes id swaps during crossings. Separate
   spec, consumes the `score()` seam introduced here.
3. **Multi-camera cross-view re-id** — separate spec; only viable with a learned
   embedding, builds on this one.

Id swaps during crossings (#2) and multi-camera (#3) are **out of scope here**.
This spec deliberately builds the seam those two will consume.

## Non-goals

- No appearance-augmented cost matrix for *active* matching (that is Approach B).
- No cross-camera gallery or coordination layer (sub-project 3).
- No change to detector, pose, or the QueueAnalyzer. Reviving an id is enough;
  the analyzer keys waiting state by id and continues the wait timer.

## Approach overview

Keep the lost-track gallery + gated re-attach flow from the current design. Three
changes:

1. **Refactor the `AppearanceModel` seam** so the model — not `Track` — owns
   per-track appearance memory (`new_memory` / `update_memory` / `score`), plus a
   batched `extract_batch`.
2. **Add `OsnetAppearance`**: full-body crop → OSNet ONNX → L2-normalized 512-d
   embedding; per-track memory is a capped feature gallery; matching scores by
   max cosine (min cosine distance) over the gallery.
3. **Split the spatial gate** so gallery (lost) candidates re-attach on
   appearance alone (enabling full cross-exit re-entry), at a slightly stricter
   threshold; in-frame candidates keep the spatial gate.

Precision-biased throughout: when uncertain, spawn a new id rather than risk a
false merge.

## Components

### 1. `AppearanceModel` seam refactor — `src/storepose/tracking/appearance.py`

The protocol moves descriptor memory out of `Track` and behind the model so the
histogram, OSNet, and future embeddings all plug in identically. Approach B and
multi-camera consume `score()` unchanged.

```python
class AppearanceModel(Protocol):
    def extract(self, frame, box, keypoints, scores) -> np.ndarray | None: ...   # single crop (kept)
    def extract_batch(self, frame, boxes, keypoints, scores) -> list[np.ndarray | None]: ...
    def new_memory(self, desc) -> object: ...          # seed a track's appearance state
    def update_memory(self, mem, desc) -> object: ...   # ingest a new observation (returns new/mutated state)
    def score(self, mem, desc) -> float: ...            # similarity of a detection to a track's memory (higher = closer)
```

- `extract(frame, box, keypoints, scores)` is kept (single crop → descriptor or
  `None`), so existing per-crop tests stay valid.
- `extract_batch(frame, boxes, keypoints, scores)` returns a list aligned to
  `boxes`, with `None` for any crop that is degenerate/empty. It replaces the
  per-detection `extract` call in the tracker loop (one batched forward pass per
  frame). `HsvHistogramAppearance.extract_batch` defaults to mapping `extract`
  over the boxes; `OsnetAppearance.extract_batch` is a true single batched
  forward pass.
- `new_memory(desc)` / `update_memory(mem, desc)` may receive `desc is None`
  (extraction failed); they must no-op gracefully (return `mem` unchanged, or a
  memory that simply has nothing to score against).
- `score(mem, desc)` returns `-1.0` (or any below-threshold sentinel) when `mem`
  is empty or `desc is None`.

**`HsvHistogramAppearance`** (kept — the histogram backend and the offline
fallback) implements the seam with no behavior change:
- `extract` = current torso-localized, background-masked, normalized H–S
  histogram (unchanged).
- `new_memory(desc)` = store the single histogram.
- `update_memory(mem, desc)` = EMA blend at the current `alpha = 0.3`.
- `score(mem, desc)` = `cv2.compareHist(mem, desc, HISTCMP_CORREL)`.

### 2. `OsnetAppearance` — `src/storepose/tracking/osnet.py` (new)

- **Crop:** the **whole person box** (OSNet is trained on full-body 256×128
  crops), clamped to frame; reject crops < a few px → `None`. Keypoints/scores
  are accepted for protocol compatibility but ignored.
- **Preprocess** (vectorized over the batch): resize to 256×128, BGR→RGB,
  scale to [0,1], ImageNet-normalize (mean `[0.485,0.456,0.406]`,
  std `[0.229,0.224,0.225]`), HWC→CHW, stack to `(N,3,256,128)` float32.
- **Inference:** one `onnxruntime.InferenceSession`, built at construction using
  the same `config.device` → provider selection the detector/pose use
  (CPU/CUDA/CoreML). One `run` per frame over the whole batch → `(N,512)`;
  L2-normalize each row.
- **Memory:** a capped deque of embeddings per track, `K = 10` (module constant).
  `new_memory` seeds it with one; `update_memory` appends and drops the oldest
  past K.
- **`score(mem, desc)`** = max dot-product of `desc` against the gallery (all
  unit vectors → cosine in [-1, 1]).

**Weights & caching** — a small resolver mirroring `model_zoo.py`, holding a
pinned ONNX URL + SHA256 for **both** sizes:

| backend value | model | embedding dim |
|---------------|-------|---------------|
| `osnet-x1` (default) | `osnet_x1_0` | 512 |
| `osnet-x025` | `osnet_x0_25` | 512 |

Weights auto-download once to `~/.cache/storepose/reid/`, checksum-verified.
`--reid-weights PATH` overrides with a local ONNX file (offline/custom). The
exact URL + hash for each size are pinned during implementation and verified by
a real download in the plan's verification step.

**Graceful fallback:** OSNet is default-on, so construction must never hard-crash
an offline/first-run user. If weights can't be fetched or the session fails to
build, log a clear one-line warning and **fall back to
`HsvHistogramAppearance`** for the run. Re-id stays on, just weaker.

### 3. `Track` changes — `src/storepose/tracking/track.py`

- Replace the `descriptor` field with one opaque `appearance_mem` field. `Track`
  never inspects its contents.
- On spawn: `appearance_mem = model.new_memory(desc)` — but `Track` stays
  decoupled from the model. To keep that decoupling, the **tracker** owns the
  `AppearanceModel` and performs the memory calls, passing the resulting memory
  object into `Track`. Concretely: `Track.__init__`/`update`/`reactivate` accept
  an `appearance_mem` object (already seeded/updated by the tracker) and just
  store it. The EMA logic currently inside `Track._update_descriptor` is
  removed; blending now lives in the model's `update_memory`.
- `color_for(id)` stays deterministic, so a revived id re-derives its color.

### 4. Tracker flow — `src/storepose/tracking/tracker.py`

- `update(result, dt, frame)`: replace the per-detection `extract` list
  comprehension with a single `extract_batch(frame, boxes, keypoints, scores)`.
- Matched/spawned tracks: tracker calls `update_memory`/`new_memory` and hands
  the memory object to `Track`.
- `_reattach`: score candidates with `model.score(cand.appearance_mem, det_desc)`
  instead of `similarity(a, b)`. Greedy lowest-cost one-to-one assignment
  unchanged.
- **Split spatial gate:**
  - *Unmatched active tracks* (briefly occluded, still in frame): keep the
    current spatial gate (`0.12·diag`, growing `0.05·diag`/gap, capped `0.5·diag`),
    threshold = `reid_thr`.
  - *Gallery (lost) entries*: **no spatial gate** — appearance only — at a
    stricter threshold `reid_thr + GALLERY_MARGIN` (`GALLERY_MARGIN ≈ 0.05`,
    module constant). This enables cross-exit re-entry from anywhere while
    staying precision-biased.

When `reid` is off, the gallery is never populated and re-attach is skipped —
exact current behavior.

### 5. Config — `src/storepose/config.py`

| Flag | Default | Effect |
|------|---------|--------|
| `--no-reid` | (re-id on) | Disable appearance re-attach (unchanged). |
| `--reid-backend {osnet-x1, osnet-x025, histogram}` | `osnet-x1` | Appearance backend / model size. |
| `--reid-weights PATH` | — | Local OSNet ONNX override (offline/custom). |
| `--reid-seconds` | `15.0` | Gallery TTL in seconds → frames (unchanged). |
| `--reid-thr` | per-backend | Appearance similarity floor. Default resolves by backend: OSNet (either size) → `0.5`, histogram → `0.6`. Explicit value overrides. |

`reid_thr` is stored as `None` when unset; `runner.build_tracker` resolves the
per-backend default. Validation in `AppConfig.__post_init__`:
`reid_backend ∈ {osnet-x1, osnet-x025, histogram}`; `reid_thr`, when given, in
[-1, 1]. Re-id requires tracking; `--no-track` disables it.

`runner.build_tracker` selects and constructs the backend (OSNet with histogram
fallback on load failure, or histogram directly), resolves the threshold, and
passes the model into `MultiObjectTracker`.

### 6. Launcher TUI — `src/storepose/launcher_core.py`, `launcher.py`

A new cycle-column, mirroring `Column.STRATEGY`:

- `Column.REID` (label `reid`), added to `COLUMNS` / `COLUMN_LABELS`.
- `ColumnState` gains `reid: str = "osnet-x1"`.
- `REID_CYCLE = ("osnet-x1", "osnet-x025", "histogram", "off")`; `toggle` cycles
  it (always available — not gated on calib).
- `build_run` emits `--reid-backend <v>`, or `--no-reid` when `reid == "off"`.
- `default_state` sets `reid="osnet-x1"`.

So each saved view can switch re-id model (or off) per-run from the TUI without
editing viewscripts: x1.0 for accuracy, x0.25 for speed.

### 7. Drawing

No change. Per-id overlay color already keyed off the persistent track id.

## Data flow

```
runner: for frame in source:
    result = pipeline.process(frame)
    people = tracker.update(result, dt, frame)
        -> descs = appearance.extract_batch(frame, boxes, kpts, scores)   # one batched forward pass
        -> IoU match active tracks; matched -> update_memory
        -> re-attach leftover dets:  score(cand.appearance_mem, det_desc)
             active candidates: spatial gate + reid_thr
             gallery candidates: appearance only + (reid_thr + margin)
        -> revive id (reactivate) or spawn new id (new_memory)
        -> cull aged confirmed tracks -> gallery
    qresult = analyzer.update(people, dt)        # unchanged; keyed by id
    canvas = annotate_* (per-id colors)          # unchanged
```

## Testing

**`tests/tracking/test_osnet.py`** (new): monkeypatch `InferenceSession.run` to
return fixed vectors →
- preprocess produces `(N,3,256,128)` float32, ImageNet-normalized;
- embeddings are L2-normalized;
- `score` is max-cosine over the gallery;
- degenerate/empty crop → `None`;
- gallery caps at `K` (oldest dropped);
- a real-weights integration test marked `skipif` (runs only when weights are
  cached) so CI stays offline.

**`tests/tracking/test_appearance.py`** (extend): histogram model implements the
4-method seam (`new_memory`/`update_memory` EMA / `score` correlation /
`extract_batch`); existing extract tests stay.

**`tests/tracking/test_tracker.py`** (extend): update the stub appearance model
to the 4-method protocol (memory = list of label vectors, `score` = max
exact-match). Existing re-attach/gallery/crossing tests stay green. Add:
- **cross-exit re-entry:** person exits one side, returns the far side beyond the
  old spatial gate, same appearance → **same id** (gallery has no spatial gate).
- **stricter gallery threshold:** a borderline appearance that passes the active
  threshold but fails `reid_thr + margin` from the gallery → **new id**.

**`tests/test_config.py`** (extend): `--reid-backend` parses/validates the three
values; per-backend `reid_thr` default resolves (osnet → 0.5, histogram → 0.6);
explicit `--reid-thr` overrides.

**`tests/test_launcher_core.py`** (extend): `Column.REID` cycles
`osnet-x1 → osnet-x025 → histogram → off → osnet-x1`; `build_run` emits
`--reid-backend <v>` and `--no-reid` for `off`.

Full suite green; live A/B smoke on a real clip: `--reid-backend osnet-x1` vs
`osnet-x025` vs `histogram`, confirming fewer fragmented ids and successful
cross-exit re-entry under the OSNet backends.

## Risks

1. **Weight sourcing** — pinning working `osnet_x1_0` and `osnet_x0_25` ONNX
   URLs + hashes is the one unverified piece; the plan verifies by real download.
   `--reid-weights` + histogram fallback de-risk it.
2. **CPU cost** — x1.0 batched is ~5–8 ms/crop, x0.25 ~1–2 ms/crop; on a busy
   frame x1.0 is tens of ms. Acceptable for recorded-video analysis; batching
   keeps it to one `run`/frame. The TUI selector lets the user trade accuracy for
   speed per-run.
3. **Threshold calibration** — the OSNet default (≈0.5) and gallery margin
   (≈0.05) are seeded then tuned on real crops during implementation.

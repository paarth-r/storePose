# Re-id: lost-track gallery + gated appearance re-attach

**Date:** 2026-06-11
**Status:** Approved design, pre-implementation
**Branch:** feat/realtime-pose

## Goal

Stop one person from fragmenting into several track IDs. Fragmentation is the
root cause of the busy/wait metrics being wrong: each new ID is counted as a
fresh arrival and resets that person's wait timer. Re-attaching a dropped track
to the same person when they reappear keeps one ID, so arrivals/throughput and
per-person wait time stay accurate.

Ranked drivers (from brainstorming): (1) cleaner busy/wait metrics, (2) bridge
longer occlusions, (3) general ID stability, (4) re-id across full exits.
Approach A below directly serves 1–3; full long-horizon cross-exit re-id (4) is
explicitly out of scope for now but the appearance seam leaves the door open.

## Non-goals

- No long-horizon "left the store and came back minutes later" gallery.
- No appearance-augmented cost matrix for *active* matching (that is Approach B,
  a future upgrade — see end).
- No changes to detector, pose, or the QueueAnalyzer state machine. Reviving an
  ID is sufficient; the analyzer keys waiting state by ID, so a revived ID
  continues its wait timer with no analyzer change.

## Approach A overview

When a confirmed track ages past `max_age` it is not deleted — it moves to a
**lost gallery** with a TTL. Each frame, after the normal IoU match to active
tracks, any leftover detection is tested for **appearance re-attach** against
both unmatched active tracks and gallery entries; a gated match revives the
original ID instead of spawning a new one. Precision-biased: when uncertain,
spawn a new ID rather than risk merging two different people (a false merge
corrupts metrics worse than a miss).

## Components

### 1. Appearance seam — `src/storepose/tracking/appearance.py` (new)

A small protocol, injectable exactly like detector/pose are today:

```python
class AppearanceModel(Protocol):
    def extract(self, frame, box, keypoints, scores) -> np.ndarray | None: ...
    def similarity(self, a: np.ndarray, b: np.ndarray) -> float: ...  # higher = closer
```

**`HsvHistogramAppearance`** (v1 implementation):
- Localize the **torso** region: from shoulder/hip keypoints (COCO 5/6/11/12)
  when all four are above `kpt_thr`; otherwise fall back to the upper-center of
  the box (a centered sub-rectangle, roughly the chest band).
- Mask out low-saturation and low-value pixels (background, shadow, blown
  highlights) before histogramming.
- Build a normalized 2-D H–S histogram (e.g. 32 H bins × 32 S bins).
- `similarity` = `cv2.compareHist(a, b, cv2.HISTCMP_CORREL)`, range ~[-1, 1].
- `extract` returns `None` when the crop is empty/degenerate (too small, fully
  masked); the tracker then skips appearance for that detection (falls back to
  spawn).

Tests inject a **stub** `AppearanceModel` returning fixed label vectors, so the
tracker is tested without real frames or weights — mirrors the existing
"detector and pose are injectable" pattern.

Later, an `OsnetAppearance` implementing the same protocol is the *only* new
piece needed for Approach B.

### 2. `Track` changes — `src/storepose/tracking/track.py`

- Carries an EMA appearance descriptor, updated on each **non-coasting**
  detection update: `desc <- (1-alpha)*desc + alpha*new`, renormalized
  (`alpha = 0.3`). First descriptor seeds it directly.
- The descriptor is **passed in** by the tracker (computed via the injected
  `AppearanceModel`), so `Track` stays decoupled from the appearance model.
  `Track.__init__` and `Track.update` gain a `descriptor` parameter.
- Palette: prune entries near the zone orange so a person's color never collides
  with the zone fill (see Drawing). `color_for(id)` stays deterministic, so a
  revived ID re-derives the same color — color persists for free.

### 3. Tracker flow — `src/storepose/tracking/tracker.py`

`MultiObjectTracker.__init__` gains: `appearance: AppearanceModel | None`,
`reid: bool`, `reid_max_age: int` (TTL in frames), `reid_thr: float`.
A new `self._lost: list[LostEntry]` gallery holds aged-out confirmed tracks:
`id`, `color`, last box, velocity, descriptor, `lost_age` (frames).

`update(result, dt, frame)` (frame is new):
1. predict active tracks; increment each gallery entry's `lost_age`, expire
   those past `reid_max_age`.
2. IoU-match detections → active tracks (unchanged).
3. update matched active tracks, now also with their freshly-extracted
   descriptor.
4. **appearance re-attach** (only when `reid` and `appearance` are set):
   candidates = unmatched active tracks ∪ gallery entries. For each leftover
   detection build cost `1 - similarity(det_desc, cand_desc)`, **masked** by:
   - gallery entry within TTL (active tracks always pass TTL),
   - **spatial gate**: detection center within a plausibility radius of the
     candidate's last/predicted center. Radius = `SPATIAL_GATE_FRAC * frame_diag
     * (1 + GATE_GROWTH * gap_frames)`, capped — grows with the gap because a
     person can move while gone. (`SPATIAL_GATE_FRAC`, `GATE_GROWTH`, cap are
     internal constants, not flags, unless they prove fiddly.)
     `similarity >= reid_thr`.
   One-to-one assignment (greedy by lowest cost is sufficient at these counts).
   A match **revives** the candidate: re-seat its Kalman at the detection box,
   `time_since_update = 0`, keep `id`/`color`/`confirmed`, update descriptor;
   pull it out of the gallery if it came from there. This single step handles
   both "occluded then reappeared far away" (unmatched active track) and "aged
   out then returned" (gallery).
5. spawn new IDs only for detections still unmatched.
6. cull: a confirmed track aged past `max_age` moves to the gallery (instead of
   being dropped); tentative + missed tracks are dropped (unchanged otherwise).
7. suppress coasting duplicates; emit confirmed tracks (unchanged).

When `reid` is off, the gallery is never populated and step 4 is skipped — exact
current behavior.

### 4. Config — `src/storepose/config.py`

| Flag | Default | Effect |
|------|---------|--------|
| `--no-reid` | (re-id on) | Disable appearance re-attach. |
| `--reid-seconds` | `5.0` | Gallery TTL; converted to frames via source fps, like `--hold-seconds`. |
| `--reid-thr` | `0.6` | Appearance similarity floor (HISTCMP_CORREL). Precision-biased. |

Re-id requires tracking; `--no-track` disables it (with a note, like `--zone`).
`runner.py` constructs the `HsvHistogramAppearance`, passes it + the reid config
into `MultiObjectTracker`, and passes `frame` into `update()`.

### 5. Drawing — per-ID overlay color — `src/storepose/drawing.py`

Today `annotate_queue` fills every waiting box green (`IN_LINE_COLOR`) and every
candidate box orange (`ZONE_COLOR`). Change: a person's fill uses **their own
persistent track color** (`p.color`). State is conveyed by **fill extent +
label**, not hue:

- **Present, not in zone:** box in `p.color`, no fill (already in
  `annotate_tracked`).
- **Joining (candidate):** partial fill rising from the box bottom by
  `progress`, in `p.color`, plus the `NN%` label.
- **In line (waiting):** full translucent fill over the box in `p.color`, plus
  the `WAIT n.n s` timer. Tag background uses `p.color` with black text.

The **zone polygon stays orange** (`ZONE_COLOR`) — it is the zone, distinct from
people. The track `_PALETTE` is pruned of near-orange entries so no person's
color collides with the zone fill. `in line: N` header may stay a neutral color.

`annotate_queue` needs each person's color; it already receives the
`TrackedPerson` list (which carries `color`), so no signature change.

## Data flow

```
runner: for frame in source:
    result = pipeline.process(frame)
    people = tracker.update(result, dt, frame)   # frame is new arg
        -> tracker extracts descriptors via appearance.extract(frame, box, kpts, scores)
        -> IoU match, appearance re-attach (revive ids), spawn, cull->gallery
    qresult = analyzer.update(people, dt)         # unchanged; keyed by id
    canvas = annotate_tracked(...); annotate_queue(..., per-id colors); annotate_busy(...)
```

## Testing

**`tests/tracking/test_appearance.py`** (new):
- torso crop localized from keypoints when present; box fallback when absent.
- background masking excludes low-sat/low-val pixels.
- same crop → high similarity; different dominant color → low similarity.
- degenerate crop → `extract` returns `None`.

**`tests/tracking/test_tracker.py`** (extend), all with a **stub** appearance
model returning fixed vectors keyed by a per-detection label (no real frames):
- (a) confirmed track → detection gap beyond `max_age` → reappears with the same
  appearance near last position ⇒ **same id revived**.
- (b) reappears with a *different* appearance ⇒ **new id** (no false merge).
- (c) reappears past TTL ⇒ **new id**.
- (d) two people crossing paths keep their own ids (no swap).
- (e) `reid=False` ⇒ exact current behavior (new id on return).

**`tests/test_drawing.py`** (extend):
- waiting fill uses the person's color, not a fixed green.
- candidate fill uses the person's color, not orange.
- zone polygon still drawn in orange.

**`tests/test_config.py`** (extend): `--no-reid`, `--reid-seconds`,
`--reid-thr` parse and validate; defaults are reid-on / 5.0 / 0.6.

## Future: Approach B (no resistance to add later)

Approach B = fold appearance into a combined cost matrix for *all* matching with
a matching cascade by track age (DeepSORT-style), improving frame-to-frame ID
stability too. The `AppearanceModel` protocol and the per-`Track` descriptor
introduced here are exactly what B consumes; B swaps the association stage and
optionally a stronger `OsnetAppearance`, without touching the appearance seam,
config surface, or drawing.

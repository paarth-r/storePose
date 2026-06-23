# CVAT Ground-Truth Annotation + Occupancy/Bystander Eval — Design

*storePose · ground-truth tooling for the "How Busy Is the Line?" deliverable · 2026-06-22*

## 1. Goal

Build a hand-annotation workflow on **CVAT** (self-hosted) that produces per-frame
ground truth for store clips, then score the pipeline's upstream stages against it.

Each person is annotated as a **single point** (not a box) in **track mode**
(keyframe + interpolate). One annotation pass yields:

- **Occupancy GT** — how many people are genuinely waiting in line at any instant.
- **Membership GT** — for each person, in-line vs. bystander.
- **Future fields, carried for free** — persistent identity (`track_id`), and
  reserved mutable attributes `state` and `intent`.

This fills the gap that the existing tooling leaves: `busy_report.py label`
only hand-labels the *final* Low/Med/High window output, and `eval/metrics.py`
only scores that final ordinal label. Nothing today validates the **occupancy**
and **bystander** stages that drive it.

## 2. Scope

**In scope (this project):**

1. Self-hosted CVAT deployment + label schema + written annotation rubric.
2. Converter from CVAT export → repo-native GT (occupancy timeline + membership).
3. New eval metrics: **occupancy** (phase 1) and **membership/bystander** (phase 2).
4. `busy_report.py` subcommands to run the converter and the scorer.
5. Docs.

**Reserved (carried in the schema, not scored yet):** `track_id` identity eval,
`state`, `intent`. These are the next roadmap items after membership.

**Out of scope (YAGNI):** multi-annotator agreement tooling, party grouping,
bounding-box annotation, any change to the live pipeline's outputs.

## 3. Why CVAT (decision record)

Chosen over a custom browser tool and over Label Studio. CVAT natively provides
point tracks with interpolation (→ future id persistence), per-frame mutable
attributes (→ state/intent), and a mature keyboard-driven UI. The cost is Docker
ops and an export→eval converter; the converter is work we'd write for any tool,
and the ops cost is accepted in exchange for not rebuilding track interpolation
and identity by hand. **Self-hosted (local Docker Compose)** is required rather
than hosted `app.cvat.ai`: the clips are real store footage the project already
treats as privacy-sensitive (face-blur is on by default), so footage must not be
uploaded to a third-party cloud.

## 4. Annotation model

### 4.1 Label schema (configured in CVAT)

- One label **`person`**, shape **points** (a single point per person), used in
  **track** mode.
- Attributes, all **mutable** (can change frame-to-frame within one track):
  - `membership` — select, ∈ {`in_line`, `bystander`}. *Used now.*
  - `state` — text/select, default empty. *Reserved, future.*
  - `intent` — text/select, default empty. *Reserved, future.*
- `track_id` is CVAT's native per-track id — persistent identity, no config.

### 4.2 Presence semantics

A person's **presence interval** is defined by their track's keyframes and CVAT
`outside` flags, *not* by positional interpolation. Interpolation only moves the
point between keyframes; it never invents or removes a person. This is what makes
occupancy counts robust under the keyframe+interpolate choice: enter/leave are
discrete `outside` transitions, and a count at time *t* is well-defined.

### 4.3 Annotation rubric (summary; full text in docs)

- Drop a point on each visible person's torso/center; create a new track per
  person, mark `outside` when they leave frame or the area.
- Set `membership` per the written in-line-vs-bystander definition (mirrors the
  dwell/transit intent in `problem-definition.md` §2), keyframing the attribute
  if a person transitions (e.g., walks up and joins the line).
- Keyframe discipline: add a keyframe whenever position changes materially or an
  attribute changes; let interpolation handle the rest.

## 5. Components

### 5.1 Converter — `src/storepose/eval/cvat_import.py`

Parses **CVAT-for-video XML** export into a pure, unit-testable GT model
(mirroring how `eval/labeling.py` keeps logic separate from any UI shell).

- Input: CVAT XML export + clip **fps** (frame number → seconds, to align with
  `waits.csv`, which is in seconds).
- Reconstructs, per track: presence intervals (keyframes + `outside`),
  interpolated point positions, and per-frame attribute values.
- Produces:
  - **Occupancy GT timeline:** `occ_gt(t) = |{tracks present at t with
    membership == in_line}|`, sampleable on any time grid.
  - **Membership GT:** per (track, frame), in_line vs. bystander, with the point
    position (needed for phase-2 association).

A small dataclass GT model (e.g. `GtTrack`, `GtPoint`) is the interchange type.
The occupancy timeline is serializable to a `t_s,occupancy` CSV so it composes
with existing CSV-based tooling.

### 5.2 Eval — `src/storepose/eval/occupancy_eval.py`

**Phase 1 — occupancy.** Sample `occ_gt(t)` and the predicted occupancy timeline
on a shared grid, then report:

- **MAE** of predicted vs. GT occupancy (headline number).
- **Bias** — signed mean error (systematic over/under-counting).
- **Pearson correlation** — does the predicted signal track the true one.
- An occupancy confusion table / scatter for inspection.

The predicted timeline needs **no new pipeline logging**: it is reconstructed
from the existing `waits.csv` via `busy/occupancy.py` (`sample_occupancy` /
`occupancy_at`), sampled on the same grid as GT.

**Phase 2 — membership / bystander.** Requires GT↔predicted **association**:
match each predicted track to the nearest GT point per frame (greedy or
Hungarian by distance, with a max-distance gate), then score in_line-vs-bystander
as **precision/recall** (+ confusion). Association is the added complexity that
justifies phasing this after occupancy.

### 5.3 Wiring — `busy_report.py`

New subcommands, consistent with the existing `aggregate` / `eval` / `label`:

- `import-cvat <export.xml> --fps F -o gt_occupancy.csv` — run the converter,
  write the occupancy GT CSV (and, phase 2, a membership GT artifact).
- `eval-occupancy <gt_occupancy.csv> <waits.csv> [--step S]` — score predicted
  occupancy (from `waits.csv`) against GT; print MAE/bias/correlation.
- (Phase 2) `eval-membership <gt_membership> <pred-tracks>` — matched membership
  scoring.

### 5.4 Docs — `docs/annotation-cvat.md`

Local CVAT deploy steps, the exact label-schema definition (§4.1), and the full
annotation rubric (§4.3). The rubric is the artifact that makes future
inter-annotator agreement (`problem-definition.md` §4.1) meaningful.

## 6. Data flow

```
store clip ──▶ CVAT (point tracks, membership attr) ──▶ CVAT-for-video XML
                                                              │
                                  cvat_import.py (+ fps)      ▼
                                  ┌─────────── GT model (GtTrack/GtPoint) ───────────┐
                                  │                                                   │
                          occupancy GT timeline                          membership GT (+positions)
                                  │                                                   │
        waits.csv ─▶ sample_occupancy ─▶ predicted occ          predicted tracks ─▶ association
                                  │                                                   │
                         occupancy_eval (P1)                            membership_eval (P2)
                          MAE / bias / corr                            precision / recall / confusion
```

## 7. Testing

- **Converter:** unit tests on synthetic CVAT XML — presence intervals from
  keyframes + `outside`, interpolation between keyframes, mutable-attribute
  resolution at a queried frame, fps→seconds. Follows the
  `tests/test_runner_debug.py` / `eval/labeling.py` pattern of testing pure logic.
- **Occupancy eval:** known GT/predicted timelines → assert MAE/bias/correlation.
- **Membership eval (P2):** synthetic GT points + predicted tracks → assert
  association and precision/recall.
- **CLI:** smoke tests for the new `busy_report.py` subcommands.

## 8. Open questions / risks

- **fps source:** taken as a CLI arg for now; could later be read from the clip
  via OpenCV. Mis-set fps misaligns GT and predicted time — documented loudly.
- **Association tuning (P2):** the max-distance gate for matching predicted
  tracks to GT points is store/camera-dependent; expose it as a flag.
- **CVAT export format:** CVAT-for-video XML is the target; if Datumaro/COCO is
  preferred later, the converter's parse layer is the only thing that changes.

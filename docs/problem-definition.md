# How Busy Is the Line? — Problem Definition & Evaluation Plan

*storePose · Week 1 deliverable · draft for review with Saurabh*

## 1. The task, precisely

**Input:** continuous store video from a fixed security camera overlooking a
checkout area.

**Output:** one categorical **busy label ∈ {Low, Medium, High}** for every
non-overlapping **10-minute window** of that video.

**Not in scope:** identifying *who* is in line (no face recognition, no identity
persistence across windows). Re-identification is short-window, anonymous, and
used only to keep counts stable within a window — see the privacy note in the
brief.

The system is a pipeline of decreasing reversibility:

```
video → person detection → tracking (stable short-window ids)
      → line membership (in line vs. bystander)
      → per-frame occupancy (people waiting)
      → robust per-window aggregation → Low / Medium / High
```

The first three stages already exist in storePose (YOLOX + RTMPose, a SORT
tracker, and a queue-zone membership test). This document defines the **last
stage** — turning the noisy per-frame occupancy into a stable label — and how we
measure whether it is right.

## 2. What do Low / Medium / High *mean*?

This is the central modeling decision, and it is genuinely a choice. Three
candidate definitions, each measurable from what the pipeline already produces:

| Definition | Signal | Pro | Con |
|---|---|---|---|
| **A. Occupancy** — how many people are waiting at once | per-frame count in the line zone | simplest, most robust, no party logic | treats a family of 4 as "4 busy" |
| **B. Parties** — how many *transactions* are queued | occupancy ÷ grouped into parties | matches "how long until I'm served" | needs grouping, which is hard & store-specific |
| **C. Wait time** — how long people currently wait | dwell time of completed waits | closest to customer experience | lags (only known once someone leaves) |

**Recommendation (default in code): Definition A, occupancy**, summarised by a
**robust statistic** (the time-weighted 90th percentile of the per-frame waiting
count over the window). Rationale:

- It needs nothing the pipeline doesn't already produce reliably.
- A percentile (not the max) ignores 1–2-second flicker from detector noise and
  brief pass-throughs, while still reacting to a genuinely sustained crowd.
- Parties (B) and wait time (C) are strictly *refinements* we can layer on once
  the occupancy baseline is honestly evaluated — and the brief explicitly values
  a documented baseline plus negative results over a fragile clever solution.

The band thresholds are **placeholders pending calibration** on real footage:

| Label | Default rule (occupancy p90) |
|---|---|
| **Low** | ≤ 1 person typically waiting |
| **Medium** | 2–3 |
| **High** | ≥ 4 |

> ⚠️ These numbers are a starting point, not a finding. The right cut points are
> store-specific and must be set from the labeled evaluation set (§4), not from
> intuition. The code exposes them as `--busy-low-max` / `--busy-medium-max`.

**Stability — three layers.** Per-frame estimates flicker; we damp them at three
scales so the 10-minute label is stable:

1. *Within a sub-window* — a time-weighted percentile (p90) ignores 1–2-second
   detector flicker.
2. *Across the window* — `--busy-subwindow` computes the metric per short
   sub-window (e.g. per minute) and takes the **median**, so a single busy
   minute can't drag the whole window up.
3. *Between windows* — a **hysteresis** deadband (`--busy-hysteresis`) requires
   the metric to clear a boundary by a margin before the label changes, so
   windows hovering near a cut point don't oscillate.

**Bystanders (hard-part #1).** "Person in the zone" ≠ "person in line." A
shopper walking through the checkout area should not count. The membership test
applies a **minimum-dwell gate** (`--wait-min-dwell`): a person must accumulate N
seconds of in-zone time before they count as waiting. This is framerate-
independent (unlike a raw frame count) and cleanly separates lingerers (line
members) from transients (pass-throughs). A second, complementary filter rejects
**directional transit**: each person carries an EMA of their velocity *vector*
(`--transit-speed`, body-heights/sec) — a shopper walking *through* the zone keeps
a large directional velocity and counts in no zone, while a queued person shuffling
in place (whose back-and-forth cancels in the vector EMA) still counts. So we gate
*sustained directional movement*, not stillness — people who legitimately move
around while queued are unaffected.

## 3. Architecture of the aggregation layer

Implemented in `src/storepose/busy/`:

- `occupancy.py` — reconstructs an occupancy timeline from completed-wait
  intervals (for offline analysis of an existing `waits.csv`).
- `aggregator.py` — `BusyAggregator`: buckets occupancy samples into windows,
  computes **time-weighted** mean/median/p90/max (so a variable frame rate
  doesn't bias the stats), attributes throughput / mean-wait / arrivals per
  window, and maps the chosen metric to a label with hysteresis.
- `types.py` — `BusyLevel`, `BusyThresholds`, `WindowFeatures`, `BusyWindow`.
- `report.py` — read a wait log, write/read a per-window busy CSV.

The bystander dwell gate lives upstream in `queue/analyzer.py`; the evaluation
metrics and the hand-labeling helpers live in `src/storepose/eval/`.

Two entry points:

- **Live:** `main.py --zone … --busy --busy-log busy.csv` shows a running busy
  badge and writes the per-window report at exit.
- **Offline:** `busy_report.py aggregate waits.csv -o busy.csv` turns an existing
  wait log into windows; `busy_report.py eval busy.csv truth.csv` scores it.

## 4. Evaluation plan

### 4.1 Ground truth

No labeled dataset exists, so we build one. For each evaluation clip:

1. Split into 10-minute windows.
2. A human watches each window and assigns Low/Medium/High **by a written
   rubric** (the §2 definition). The `busy_report.py label` command does this:
   it steps through the video window-by-window and records each judgement to a
   `window_index,level` CSV (resumable), the exact format the evaluator reads.
3. To check the rubric isn't arbitrary, label a subset twice (or by two people)
   and report **inter-annotator agreement**; if humans can't agree, the label
   definition is too vague and must be revised before trusting any model number.

Hold out **≥ 2 different store layouts** so we measure generalization, not
overfitting to one camera.

### 4.2 Metrics (implemented in `src/storepose/eval/metrics.py`)

Low/Medium/High is **ordinal**, so we don't rely on plain accuracy alone:

- **Accuracy** — exact-match fraction.
- **Within-1 accuracy** — fraction within one level (the "no gross errors"
  metric; for 3 classes this is everything except Low↔High confusions). This is
  the headline number: confusing Low with High is the failure that matters.
- **Ordinal MAE** — mean |true−pred| on the 0/1/2 scale.
- **Per-class precision/recall** + the full **confusion matrix**, to see *which*
  errors dominate (e.g. systematically calling Medium "High").

### 4.3 Protocol

- **Calibrate** thresholds on a *training* split of the labeled windows; **report**
  on held-out windows and held-out stores. Never tune on the test set.
- Always report a **majority-class baseline** (always predict the most common
  label) so we know the model beats "guess the usual."
- For each store, document **failure modes** and **where it won't scale**
  (camera angle, occlusion, line geometry) — per the brief, honest negative
  results are a deliverable.

## 5. Open questions for Saurabh (Week 1)

1. **Definition:** occupancy (A), parties (B), or wait-time (C)? Default is A.
2. **Thresholds:** what does a store consider "High"? Best set from labeled data,
   but a domain prior from Mashgin would anchor it.
3. **Window:** is 10 min the reporting cadence, or should we also emit a faster
   internal estimate (e.g. 1 min) smoothed to a 10-min label?
4. **Parties:** is grouping in scope for the prototype, or a documented "next
   step"? It is the largest source of A-vs-B disagreement.
5. **Evaluation labels:** can Mashgin provide any partial ground truth (POS
   transaction timestamps?) to anchor or cross-check the hand labels?

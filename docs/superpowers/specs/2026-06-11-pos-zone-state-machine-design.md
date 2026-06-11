# POS zone + per-state time decomposition

**Date:** 2026-06-11
**Status:** Approved design, pre-implementation
**Branch:** feat/realtime-pose
**Sub-project 1 of 2** (the dashboard that consumes this data is a separate later spec.)

## Goal

Split each person's line visit into **waiting** time (in the line, not yet at the
register) and **serving** time (at the POS), per person, by adding a second
**POS zone**. Rebuild `QueueAnalyzer` as an explicit per-person **state machine**
whose `dt` accrues to the current state, including time spent out of detection
(a re-id gap is attributed to the state the person held when they vanished).

## State machine

Per tracked person, one of three live states plus terminal outcomes:

```
OUT     — present but not yet counted as in line (or in neither zone)
WAITING — in the line zone but NOT in the POS zone; accrues waiting_seconds
SERVING — in the POS zone; accrues serving_seconds

OUT     --in line-or-POS area for enter_frames + min_dwell-->  WAITING or SERVING
WAITING --ankle/foot enters POS zone-->                        SERVING (reached_pos=True)
WAITING --out of line zone for >= exit_seconds-->              done: ABANDONED
SERVING --out of POS zone for >= exit_seconds-->               done: SERVED
absent past the re-id grace window-->                          done (SERVED if reached_pos else ABANDONED)
```

Each frame, `dt` is added to the current state's accumulator (`waiting_seconds`
for WAITING, `serving_seconds` for SERVING; OUT accrues nothing). `enter_frames`
+ `min_dwell_seconds` is the bystander gate for `OUT → WAITING/SERVING`;
`exit_seconds` debounces the leaving transitions (a brief flicker out of a zone
does not end the visit). A walk-up customer who appears directly at the POS goes
`OUT → SERVING` (skips WAITING).

**Zone membership.** `in_pos = in_zone(person, pos_zone)`,
`in_line = in_zone(person, line_zone)`, reusing the existing ankle-OR-foot-coverage
test against each polygon. "At POS" = `in_pos`. "Waiting region" =
`in_line and not in_pos` (so a POS zone nested inside the line zone works: a
person at the register is in both polygons but counts as SERVING).

## Re-id gap attribution

When a person's track vanishes (lost into the re-id gallery), the gap is **not
paused** — each absent-frame `dt` is **added to the state the person held when
they disappeared**, assuming continuity (lost-for-2s-while-SERVING adds 2s to
`serving_seconds`). On return within the re-id grace window they resume in
whatever state their current position dictates; absent past the window finalizes
the visit with whatever accrued.

**Ambiguous case** (vanished WAITING, returns SERVING — reached POS during the
gap): the whole gap is attributed to the **pre-loss state (WAITING)**, the last
state we actually observed. This replaces the "pause the timer" behavior added in
the earlier re-id work with "carry the gap forward into the held state." The
grace window is still `reid_grace_seconds` (wired to `config.reid_seconds`).

## Components

### `src/storepose/queue/analyzer.py` (rebuilt)
`QueueAnalyzer` keyed by id, holding `_VisitState` per person:
`state, waiting_seconds, serving_seconds, entered_s, reached_pos, in_frames,
in_seconds, out_streak, absent_seconds`. Takes both zones:
`QueueAnalyzer(line_zone, pos_zone=None, enter_frames, exit_seconds, kpt_thr,
coverage_thr, foot_band, min_dwell_seconds, reid_grace_seconds)`. With
`pos_zone=None` it behaves like today (WAITING only; SERVING never entered),
preserving single-zone use. The per-frame and absent-id loops implement the
transitions and gap attribution above; finalizing a visit emits a
`CompletedWait` (with its new `serving_seconds` / `outcome` fields populated).

### `src/storepose/queue/types.py`
Types are **extended with defaulted fields, not renamed**, so the whole `busy/`
package (aggregator/occupancy/report) and `busy_report.py` keep working unchanged
(`CompletedWait.wait_seconds` stays the **waiting** portion they already consume).
- `PersonStatus`: keep `id, waiting, candidate, progress, wait_seconds`; add
  `serving: bool = False`, `serving_seconds: float = 0.0`. The analyzer sets
  `waiting = state == "waiting"`, `serving = state == "serving"`,
  `wait_seconds = waiting_seconds`.
- `CompletedWait`: keep `id, entered_s, exited_s, wait_seconds`; add
  `serving_seconds: float = 0.0`, `outcome: str = "served"`. `wait_seconds`
  remains the waiting portion.
- `QueueResult`: keep `statuses, count (number WAITING), completed`; add
  `serving_count: int = 0` (number SERVING).

### `src/storepose/config.py`
- `pos_zone: str | None = None` + `--pos-zone PATH`.
- `define_pos_zone: bool = False` + `--define-pos-zone` (runs the editor, saving
  to the `--pos-zone` path / its default).
- `--pos-zone` requires `--zone` (waiting needs the line polygon); if given
  without it, the runner prints a note and ignores POS (degrades to today's
  single-zone behavior).

### `src/storepose/queue/zone_editor.py` / `main.py`
Reuse `define_zone(source, out_path)`. `main.py` handles `--define-pos-zone` the
same way it handles `--define-zone`, targeting `config.pos_zone`
(`default_zone_path` gets a `pos_` variant for the default filename).

### `src/storepose/runner.py`
Load `pos_zone` when set, pass it to `QueueAnalyzer` and to `annotate_queue`.
Wait-log CSV columns become `id, entered_s, exited_s, wait_seconds,
serving_seconds, outcome` (the legacy `wait_seconds` column kept, two appended).
The busy aggregator is fed **only SERVED visits** (`busy.add_wait(c)`, which reads
`c.wait_seconds` = the waiting portion) so its throughput stays "served", while
all visits (served + abandoned) are written to the CSV. `busy_report.read_waits`
is extended to parse the two new columns when present (defaults when absent, so
old CSVs still load).

### `src/storepose/drawing.py`
- New `POS_COLOR` (cyan, distinct from the orange `ZONE_COLOR`); draw the POS
  polygon when present.
- A person's box fill stays their **per-id color**; the tag reads `WAIT n.n s`
  while `state == "waiting"` and `POS n.n s` while `state == "serving"`.
- Header gains `at POS: M` beside `in line: N`.

## Data flow

```
runner: people = tracker.update(result, dt, frame)
        qresult = analyzer.update(people, dt)        # state machine + gap attribution
        for c in qresult.completed:
            write CSV row (id, entered, exited, wait_seconds, serving_seconds, outcome)
            if c.outcome == "served": busy.add_wait(c)
        canvas = annotate_queue(canvas, people, qresult, line_zone, pos_zone, config)
```

## Busy signal

Unchanged aggregator. It now receives only served visits and `count` is still the
waiting occupancy, so existing busy behavior/tests hold. Serving/throughput
analytics are the dashboard's job (sub-project 2), driven by the new CSV columns.

## Testing

`tests/queue/test_analyzer.py` (extend/rebuild):
- `OUT → WAITING → SERVING → SERVED`: accumulators land in the right buckets;
  completed visit has `outcome="served"`, both seconds > 0.
- `OUT → WAITING → ABANDONED` (leaves line pre-POS): `outcome="abandoned"`,
  `serving_seconds == 0`.
- Walk-up `OUT → SERVING` (never in waiting region): `waiting_seconds == 0`.
- Gap attribution: vanish while SERVING within grace → `serving_seconds` includes
  the gap; vanish while WAITING → `waiting_seconds` includes the gap.
- Pre-loss-state rule: vanish WAITING, return SERVING → gap counted as waiting.
- Finalize past grace: SERVED if reached_pos else ABANDONED.
- `pos_zone=None` preserves today's single-zone behavior (existing tests pass).

`tests/test_config.py`: `--pos-zone` / `--define-pos-zone` parse; `--pos-zone`
without `--zone` is handled.

`tests/test_drawing.py`: POS polygon drawn in cyan; serving person tagged
`POS n.n s` in their id color; header shows both counts.

## Out of scope (sub-project 2: dashboard)
Graphs, moving averages, throughput-over-time, abandonment rate — a separate
consumer of the CSV, its own spec.

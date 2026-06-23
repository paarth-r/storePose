# CVAT Occupancy Ground-Truth (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert CVAT point-track annotations into an occupancy ground-truth timeline and score the pipeline's predicted occupancy (from `waits.csv`) against it.

**Architecture:** A pure CVAT-for-video XML parser builds an in-memory GT track model; presence is derived from keyframes + `outside` flags and membership from a per-shape `membership` attribute; occupancy at a frame is the count of present in-line tracks. A new eval module compares the GT occupancy timeline against the predicted timeline that the existing `busy/occupancy.py` reconstructs from `waits.csv`. Two `busy_report.py` subcommands wire it together. No changes to the live pipeline.

**Tech Stack:** Python (stdlib only — `xml.etree.ElementTree`, `csv`, `dataclasses`), pytest, `uv` for running. CVAT (self-hosted) is the upstream annotation tool but is external to this code.

## Global Constraints

- Python with `from __future__ import annotations` at the top of every new module (matches the codebase).
- **Stdlib only** for the `eval/` code — no new dependencies (mirrors `eval/metrics.py` / `eval/labeling.py`, which are pure).
- Keep logic pure and unit-testable, separate from any UI/CLI shell (the `eval/labeling.py` pattern).
- Tests live under `tests/eval/`, import from `storepose.*`, run with `uv run pytest`.
- No emojis anywhere (code, commits, docs).
- Commit messages: `feat:` / `docs:` prefix, no emojis, end with:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
- This work is on branch `cvat-ground-truth` (already created off `main`).

## File Structure

- Create `src/storepose/eval/cvat_import.py` — CVAT XML parse, GT track model, presence/membership/occupancy derivation, timeline sampling, occupancy-CSV I/O.
- Create `src/storepose/eval/occupancy_eval.py` — occupancy comparison metrics (MAE, bias, Pearson correlation).
- Modify `busy_report.py` — add `import-cvat` and `eval-occupancy` subcommands.
- Create `tests/eval/test_cvat_import.py` — parser + derivation + sampling + I/O tests.
- Create `tests/eval/test_occupancy_eval.py` — metric tests.
- Create `docs/annotation-cvat.md` — CVAT deploy, label schema, annotation rubric.

---

### Task 1: CVAT-for-video XML parser and GT track model

**Files:**
- Create: `src/storepose/eval/cvat_import.py`
- Test: `tests/eval/test_cvat_import.py`

**Interfaces:**
- Consumes: nothing (entry point of the module).
- Produces:
  - `GtShape(frame: int, outside: bool, x: float, y: float, attrs: dict[str, str])` (frozen dataclass)
  - `GtTrack(id: int, label: str, shapes: list[GtShape])` (dataclass; `shapes` sorted ascending by `frame`)
  - `parse_cvat_xml(text: str) -> list[GtTrack]`

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_cvat_import.py
from storepose.eval.cvat_import import GtShape, GtTrack, parse_cvat_xml

SAMPLE_XML = """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <track id="0" label="person">
    <points frame="10" outside="0" occluded="0" keyframe="1" points="100.0,200.0">
      <attribute name="membership">in_line</attribute>
    </points>
    <points frame="20" outside="0" occluded="0" keyframe="1" points="110.0,205.0">
      <attribute name="membership">in_line</attribute>
    </points>
    <points frame="30" outside="1" occluded="0" keyframe="1" points="120.0,210.0">
      <attribute name="membership">in_line</attribute>
    </points>
  </track>
  <track id="1" label="person">
    <points frame="12" outside="0" occluded="0" keyframe="1" points="300.0,400.0">
      <attribute name="membership">bystander</attribute>
    </points>
  </track>
</annotations>
"""


def test_parse_builds_tracks_and_shapes():
    tracks = parse_cvat_xml(SAMPLE_XML)
    assert [t.id for t in tracks] == [0, 1]
    assert tracks[0].label == "person"
    assert tracks[0].shapes[0] == GtShape(10, False, 100.0, 200.0, {"membership": "in_line"})
    assert tracks[0].shapes[2].outside is True
    assert tracks[1].shapes[0].attrs["membership"] == "bystander"


def test_parse_shapes_sorted_by_frame():
    xml = SAMPLE_XML.replace('frame="10"', 'frame="99"')  # out-of-order keyframe
    tracks = parse_cvat_xml(xml)
    frames = [s.frame for s in tracks[0].shapes]
    assert frames == sorted(frames)


def test_parse_multipoint_takes_first_point():
    xml = SAMPLE_XML.replace('points="100.0,200.0"', 'points="100.0,200.0;150.0,250.0"')
    tracks = parse_cvat_xml(xml)
    assert (tracks[0].shapes[0].x, tracks[0].shapes[0].y) == (100.0, 200.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval/test_cvat_import.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'storepose.eval.cvat_import'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/storepose/eval/cvat_import.py
"""Convert CVAT-for-video point-track exports into occupancy ground truth.

CVAT annotates each person as a single *point* in *track* mode (keyframe +
interpolate). A track's presence is defined by its keyframes and ``outside``
flags, not by positional interpolation, so per-frame occupancy counts are
well-defined. The pure logic here is unit-tested; the CLI shell lives in
``busy_report.py``.
"""

from __future__ import annotations

import csv
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GtShape:
    """One keyframe of a track: position, visibility, and attributes."""

    frame: int
    outside: bool
    x: float
    y: float
    attrs: dict[str, str]


@dataclass
class GtTrack:
    """A single person's point track. ``shapes`` are sorted by ``frame``."""

    id: int
    label: str
    shapes: list[GtShape]


def parse_cvat_xml(text: str) -> list[GtTrack]:
    """Parse a CVAT-for-video 1.1 XML export into a list of tracks."""
    root = ET.fromstring(text)
    tracks: list[GtTrack] = []
    for tr in root.findall("track"):
        shapes: list[GtShape] = []
        for pt in tr.findall("points"):
            coords = (pt.get("points") or "").split(";")[0]
            x_str, _, y_str = coords.partition(",")
            attrs = {
                a.get("name", ""): (a.text or "") for a in pt.findall("attribute")
            }
            shapes.append(
                GtShape(
                    frame=int(pt.get("frame", "0")),
                    outside=pt.get("outside") == "1",
                    x=float(x_str) if x_str else 0.0,
                    y=float(y_str) if y_str else 0.0,
                    attrs=attrs,
                )
            )
        shapes.sort(key=lambda s: s.frame)
        tracks.append(
            GtTrack(id=int(tr.get("id", "0")), label=tr.get("label", ""), shapes=shapes)
        )
    return tracks
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/eval/test_cvat_import.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/storepose/eval/cvat_import.py tests/eval/test_cvat_import.py
git commit -m "feat: parse CVAT-for-video XML into GT track model

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Presence, membership, and per-frame occupancy

**Files:**
- Modify: `src/storepose/eval/cvat_import.py`
- Test: `tests/eval/test_cvat_import.py`

**Interfaces:**
- Consumes: `GtTrack`, `GtShape` (Task 1).
- Produces:
  - `present_at(track: GtTrack, frame: int) -> bool`
  - `membership_at(track: GtTrack, frame: int) -> str | None` (the `membership` attr value while present, else `None`)
  - `occupancy_gt_at(tracks: list[GtTrack], frame: int) -> int` (count of tracks with `membership == "in_line"` at `frame`)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/eval/test_cvat_import.py
from storepose.eval.cvat_import import (
    membership_at,
    occupancy_gt_at,
    present_at,
)


def test_present_only_within_track_before_outside():
    (track, _bystander) = parse_cvat_xml(SAMPLE_XML)
    assert present_at(track, 9) is False     # before first keyframe
    assert present_at(track, 10) is True      # at first keyframe
    assert present_at(track, 25) is True      # between non-outside keyframes
    assert present_at(track, 30) is False     # outside keyframe
    assert present_at(track, 99) is False     # after outside persists


def test_membership_is_none_when_absent():
    (track, _b) = parse_cvat_xml(SAMPLE_XML)
    assert membership_at(track, 15) == "in_line"
    assert membership_at(track, 30) is None   # outside -> not a member


def test_occupancy_counts_only_in_line_and_present():
    tracks = parse_cvat_xml(SAMPLE_XML)
    # frame 15: track 0 in_line present, track 1 bystander present -> occ 1
    assert occupancy_gt_at(tracks, 15) == 1
    # frame 9: neither present -> 0
    assert occupancy_gt_at(tracks, 9) == 0
    # frame 30: track 0 outside, track 1 absent -> 0
    assert occupancy_gt_at(tracks, 30) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval/test_cvat_import.py -k "present or membership or occupancy" -v`
Expected: FAIL — `ImportError: cannot import name 'present_at'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/storepose/eval/cvat_import.py

def _active_shape(track: GtTrack, frame: int) -> GtShape | None:
    """The last keyframe at or before ``frame`` (the one that governs state)."""
    active: GtShape | None = None
    for s in track.shapes:
        if s.frame <= frame:
            active = s
        else:
            break
    return active


def present_at(track: GtTrack, frame: int) -> bool:
    """True if the track is visible at ``frame`` (governing keyframe not outside)."""
    active = _active_shape(track, frame)
    return active is not None and not active.outside


def membership_at(track: GtTrack, frame: int) -> str | None:
    """The ``membership`` attribute while present, else ``None``."""
    active = _active_shape(track, frame)
    if active is None or active.outside:
        return None
    return active.attrs.get("membership")


def occupancy_gt_at(tracks: list[GtTrack], frame: int) -> int:
    """Number of present, in-line people at ``frame``."""
    return sum(1 for t in tracks if membership_at(t, frame) == "in_line")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/eval/test_cvat_import.py -v`
Expected: PASS (all tests in file)

- [ ] **Step 5: Commit**

```bash
git add src/storepose/eval/cvat_import.py tests/eval/test_cvat_import.py
git commit -m "feat: derive presence, membership, and occupancy from GT tracks

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Occupancy timeline sampling and CSV I/O

**Files:**
- Modify: `src/storepose/eval/cvat_import.py`
- Test: `tests/eval/test_cvat_import.py`

**Interfaces:**
- Consumes: `occupancy_gt_at`, `GtTrack` (Task 2).
- Produces:
  - `sample_occupancy_gt(tracks, fps, step=1.0, t_start=0.0, t_end=None) -> list[tuple[float, int]]` — `(t_seconds, occupancy)` pairs on `[t_start, t_end)`; `t_end` defaults to `max_frame / fps`. Mirrors `busy.occupancy.sample_occupancy` so GT and predicted timelines share a shape.
  - `write_occupancy_csv(path, samples) -> None` — header `t_s,occupancy`.
  - `read_occupancy_csv(path) -> list[tuple[float, int]]`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/eval/test_cvat_import.py
from storepose.eval.cvat_import import (
    read_occupancy_csv,
    sample_occupancy_gt,
    write_occupancy_csv,
)


def test_sample_occupancy_gt_grid_and_values():
    tracks = parse_cvat_xml(SAMPLE_XML)  # in_line track present frames [10,30)
    # fps=10 -> 1s == 10 frames; sample every 1s over [0, t_end)
    samples = sample_occupancy_gt(tracks, fps=10.0, step=1.0)
    by_t = dict(samples)
    assert by_t[0.0] == 0    # frame 0
    assert by_t[1.0] == 0    # frame 10 -> present, but membership in_line -> 1
    # frame 10 is exactly the first keyframe; round(1.0*10)=10 -> present in_line
    assert by_t[1.0] == 1 or by_t[2.0] == 1  # presence visible at/after entry


def test_sample_occupancy_gt_validates_inputs():
    tracks = parse_cvat_xml(SAMPLE_XML)
    import pytest
    with pytest.raises(ValueError):
        sample_occupancy_gt(tracks, fps=0.0)
    with pytest.raises(ValueError):
        sample_occupancy_gt(tracks, fps=10.0, step=0.0)


def test_occupancy_csv_roundtrip(tmp_path):
    samples = [(0.0, 0), (1.0, 2), (2.0, 1)]
    p = tmp_path / "gt.csv"
    write_occupancy_csv(p, samples)
    assert read_occupancy_csv(p) == samples
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval/test_cvat_import.py -k "sample or roundtrip" -v`
Expected: FAIL — `ImportError: cannot import name 'sample_occupancy_gt'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/storepose/eval/cvat_import.py

def sample_occupancy_gt(
    tracks: list[GtTrack],
    fps: float,
    step: float = 1.0,
    t_start: float = 0.0,
    t_end: float | None = None,
) -> list[tuple[float, int]]:
    """Sample GT occupancy on ``[t_start, t_end)`` every ``step`` seconds.

    Frame numbers map to seconds via ``fps`` so the timeline aligns with the
    predicted timeline reconstructed from ``waits.csv`` (which is in seconds).
    """
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")
    if step <= 0:
        raise ValueError(f"step must be > 0, got {step}")
    if not tracks:
        return []
    max_frame = max((s.frame for t in tracks for s in t.shapes), default=0)
    if t_end is None:
        t_end = max_frame / fps
    out: list[tuple[float, int]] = []
    t = t_start
    while t < t_end:
        frame = round(t * fps)
        out.append((t, occupancy_gt_at(tracks, frame)))
        t += step
    return out


def write_occupancy_csv(path: str | Path, samples: list[tuple[float, int]]) -> None:
    """Write ``t_s,occupancy`` rows."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "occupancy"])
        for t, occ in samples:
            w.writerow([f"{t:.3f}", occ])


def read_occupancy_csv(path: str | Path) -> list[tuple[float, int]]:
    """Read a ``t_s,occupancy`` CSV back into ``(t, occ)`` pairs."""
    out: list[tuple[float, int]] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out.append((float(row["t_s"]), int(row["occupancy"])))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/eval/test_cvat_import.py -v`
Expected: PASS (all tests in file)

- [ ] **Step 5: Commit**

```bash
git add src/storepose/eval/cvat_import.py tests/eval/test_cvat_import.py
git commit -m "feat: sample GT occupancy timeline and CSV I/O

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Occupancy comparison metrics

**Files:**
- Create: `src/storepose/eval/occupancy_eval.py`
- Test: `tests/eval/test_occupancy_eval.py`

**Interfaces:**
- Consumes: timelines as `list[tuple[float, int]]` (Task 3 / `busy.occupancy.sample_occupancy`).
- Produces:
  - `OccupancyReport(n, mae, bias, corr, gt_mean, pred_mean)` dataclass with `format() -> str`.
  - `occupancy_eval(gt, pred) -> OccupancyReport` — aligns on shared timestamps (rounded to 3 dp); `bias` is mean `pred - gt` (positive = over-count); `corr` is Pearson, `0.0` when either series is constant or no overlap.

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_occupancy_eval.py
import math

from storepose.eval.occupancy_eval import OccupancyReport, occupancy_eval


def test_perfect_match():
    gt = [(0.0, 1), (1.0, 2), (2.0, 3)]
    rep = occupancy_eval(gt, list(gt))
    assert rep.n == 3
    assert rep.mae == 0.0
    assert rep.bias == 0.0
    assert math.isclose(rep.corr, 1.0, rel_tol=1e-9)


def test_mae_and_bias_signs():
    gt = [(0.0, 0), (1.0, 0), (2.0, 0)]
    pred = [(0.0, 1), (1.0, 2), (2.0, 0)]  # over-counts by 1, 2, 0
    rep = occupancy_eval(gt, pred)
    assert rep.mae == 1.0          # (1+2+0)/3
    assert rep.bias == 1.0         # pred - gt, positive = over-count
    assert rep.pred_mean == 1.0


def test_only_overlapping_timestamps_scored():
    gt = [(0.0, 1), (1.0, 1)]
    pred = [(1.0, 1), (2.0, 5)]    # only t=1.0 overlaps
    rep = occupancy_eval(gt, pred)
    assert rep.n == 1
    assert rep.mae == 0.0


def test_no_overlap_is_empty_report():
    rep = occupancy_eval([(0.0, 1)], [(9.0, 1)])
    assert rep.n == 0
    assert rep.corr == 0.0


def test_constant_series_correlation_zero():
    gt = [(0.0, 2), (1.0, 2), (2.0, 2)]   # zero variance
    pred = [(0.0, 1), (1.0, 3), (2.0, 2)]
    rep = occupancy_eval(gt, pred)
    assert rep.corr == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval/test_occupancy_eval.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'storepose.eval.occupancy_eval'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/storepose/eval/occupancy_eval.py
"""Compare a predicted occupancy timeline against ground truth.

Occupancy is the upstream signal that drives the busy label, so before trusting
any Low/Med/High number we check that the per-frame waiting count itself is
right. Both timelines are ``(t_seconds, occupancy)`` pairs; we align them on
shared timestamps and report error and correlation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OccupancyReport:
    n: int
    mae: float
    bias: float          # mean(pred - gt); positive = systematic over-count
    corr: float          # Pearson; 0.0 if a series is constant or no overlap
    gt_mean: float
    pred_mean: float

    def format(self) -> str:
        return "\n".join(
            [
                f"samples scored : {self.n}",
                f"occupancy MAE  : {self.mae:.3f}",
                f"bias (pred-gt) : {self.bias:+.3f}",
                f"correlation    : {self.corr:.3f}",
                f"mean occ gt    : {self.gt_mean:.3f}",
                f"mean occ pred  : {self.pred_mean:.3f}",
            ]
        )


def _pearson(xs: list[int], ys: list[int]) -> float:
    n = len(xs)
    if n == 0:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return 0.0
    return cov / ((vx ** 0.5) * (vy ** 0.5))


def occupancy_eval(
    gt: list[tuple[float, int]], pred: list[tuple[float, int]]
) -> OccupancyReport:
    """Align GT and predicted occupancy on shared timestamps and score them."""
    gd = {round(t, 3): occ for t, occ in gt}
    pdct = {round(t, 3): occ for t, occ in pred}
    keys = sorted(set(gd) & set(pdct))
    n = len(keys)
    if n == 0:
        return OccupancyReport(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    gs = [gd[k] for k in keys]
    ps = [pdct[k] for k in keys]
    mae = sum(abs(g - p) for g, p in zip(gs, ps)) / n
    bias = sum(p - g for g, p in zip(gs, ps)) / n
    return OccupancyReport(
        n=n,
        mae=mae,
        bias=bias,
        corr=_pearson(gs, ps),
        gt_mean=sum(gs) / n,
        pred_mean=sum(ps) / n,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/eval/test_occupancy_eval.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/storepose/eval/occupancy_eval.py tests/eval/test_occupancy_eval.py
git commit -m "feat: occupancy comparison metrics (MAE, bias, correlation)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Wire `import-cvat` and `eval-occupancy` subcommands

**Files:**
- Modify: `busy_report.py`
- Test: `tests/eval/test_cvat_cli.py` (create)

**Interfaces:**
- Consumes: `parse_cvat_xml`, `sample_occupancy_gt`, `write_occupancy_csv`, `read_occupancy_csv` (cvat_import); `occupancy_eval` (occupancy_eval); `read_waits`, `sample_occupancy` (existing busy modules).
- Produces: CLI subcommands `import-cvat <export.xml> --fps F [--step S] -o gt.csv` and `eval-occupancy <gt.csv> <waits.csv> [--step S]`, dispatched through the existing `main(argv)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_cvat_cli.py
import csv

import busy_report
from tests.eval.test_cvat_import import SAMPLE_XML


def _write(path, text):
    path.write_text(text)
    return str(path)


def test_import_cvat_writes_gt_csv(tmp_path, capsys):
    xml = _write(tmp_path / "export.xml", SAMPLE_XML)
    out = str(tmp_path / "gt.csv")
    rc = busy_report.main(["import-cvat", xml, "--fps", "10", "--step", "1", "-o", out])
    assert rc == 0
    with open(out) as f:
        rows = list(csv.DictReader(f))
    assert rows[0].keys() >= {"t_s", "occupancy"}


def test_eval_occupancy_runs(tmp_path, capsys):
    # GT: occupancy 1 at t=1.0; waits.csv: one wait covering [1,2)
    gt = tmp_path / "gt.csv"
    gt.write_text("t_s,occupancy\n1.000,1\n")
    waits = tmp_path / "waits.csv"
    waits.write_text("id,entered_s,exited_s,wait_seconds\n0,0.5,2.0,1.5\n")
    rc = busy_report.main(["eval-occupancy", str(gt), str(waits), "--step", "1"])
    assert rc == 0
    assert "occupancy MAE" in capsys.readouterr().out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval/test_cvat_cli.py -v`
Expected: FAIL — `argument cmd: invalid choice: 'import-cvat'`

- [ ] **Step 3: Write minimal implementation**

Add imports near the top of `busy_report.py` (with the other `storepose` imports):

```python
from storepose.eval.cvat_import import (
    parse_cvat_xml,
    read_occupancy_csv,
    sample_occupancy_gt,
    write_occupancy_csv,
)
from storepose.eval.occupancy_eval import occupancy_eval
```

Add the two handler functions (next to `_eval`):

```python
def _import_cvat(args: argparse.Namespace) -> int:
    with open(args.export) as f:
        tracks = parse_cvat_xml(f.read())
    if not tracks:
        print(f"No tracks in {args.export}; nothing to import.", file=sys.stderr)
        return 1
    samples = sample_occupancy_gt(tracks, fps=args.fps, step=args.step)
    write_occupancy_csv(args.output, samples)
    print(
        f"{len(tracks)} track(s) -> {len(samples)} occupancy sample(s) "
        f"at {args.step}s; wrote {args.output}"
    )
    return 0


def _eval_occupancy(args: argparse.Namespace) -> int:
    gt = read_occupancy_csv(args.gt)
    waits = read_waits(args.waits)
    if not waits:
        print(f"No waits in {args.waits}; nothing to score.", file=sys.stderr)
        return 1
    pred = sample_occupancy(waits, step=args.step)
    rep = occupancy_eval(gt, pred)
    if rep.n == 0:
        print("No overlapping timestamps between GT and predicted occupancy. "
              "Did you use the same --step for import-cvat and the pipeline?",
              file=sys.stderr)
        return 1
    print(rep.format())
    return 0
```

Register both subparsers in `_build_parser` (after the `eval` subparser, before `label`):

```python
    ic = sub.add_parser("import-cvat",
                        help="CVAT point-track XML -> occupancy GT CSV")
    ic.add_argument("export", help="Path to a CVAT-for-video XML export.")
    ic.add_argument("--fps", type=float, required=True,
                    help="Clip frame rate; maps CVAT frame numbers to seconds.")
    ic.add_argument("--step", type=float, default=1.0,
                    help="Occupancy sampling step in seconds (default: 1.0).")
    ic.add_argument("-o", "--output", required=True, metavar="PATH",
                    help="Occupancy GT CSV to write (t_s,occupancy).")
    ic.set_defaults(func=_import_cvat)

    eo = sub.add_parser("eval-occupancy",
                        help="score predicted occupancy (from waits) vs. GT")
    eo.add_argument("gt", help="Occupancy GT CSV (from import-cvat).")
    eo.add_argument("waits", help="Wait-log CSV (from --wait-log).")
    eo.add_argument("--step", type=float, default=1.0,
                    help="Sampling step; must match the import-cvat --step "
                         "(default: 1.0).")
    eo.set_defaults(func=_eval_occupancy)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/eval/test_cvat_cli.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run the full eval suite as a regression check**

Run: `uv run pytest tests/eval -v`
Expected: PASS (all eval tests, including pre-existing `test_labeling.py` / `test_metrics.py`)

- [ ] **Step 6: Commit**

```bash
git add busy_report.py tests/eval/test_cvat_cli.py
git commit -m "feat: import-cvat and eval-occupancy subcommands

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: CVAT deploy, schema, and annotation rubric docs

**Files:**
- Create: `docs/annotation-cvat.md`

**Interfaces:** none (documentation). Verification is a manual read-through plus the end-to-end command sequence below.

- [ ] **Step 1: Write the doc**

Create `docs/annotation-cvat.md` with these sections (write real content, not placeholders):

1. **Why CVAT, self-hosted** — one paragraph: point tracks + interpolation give future id-persistence; per-frame attributes give future state/intent; self-hosted (local Docker) because clips are privacy-sensitive store footage that must not leave the machine. Link back to `docs/superpowers/specs/2026-06-22-cvat-ground-truth-annotation-design.md`.
2. **Deploy (local)** — the standard CVAT self-host steps:
   ```bash
   git clone https://github.com/cvat-ai/cvat
   cd cvat
   docker compose up -d
   # create an admin user:
   docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'
   # open http://localhost:8080
   ```
3. **Label schema** — one label `person`, shape **points**, used in **track** mode, with attributes:
   - `membership` — select, values `in_line`, `bystander` (mutable). Used now.
   - `state` — text, default empty (mutable). Reserved, future.
   - `intent` — text, default empty (mutable). Reserved, future.
   Note that `track_id` is CVAT's native track id (persistent identity, no config).
4. **Annotation rubric** — the in-line vs. bystander definition (mirror `docs/problem-definition.md` Section 2: a person counts as `in_line` once they are genuinely waiting, not a pass-through; tag clear walk-throughs `bystander`); how to mark enter/leave with `outside`; keyframe discipline (add a keyframe when position changes materially or an attribute changes; let interpolation fill the rest).
5. **Export and score** — the end-to-end loop:
   ```bash
   # In CVAT: Menu -> Export task dataset -> format "CVAT for video 1.1" -> download export.xml
   uv run python busy_report.py import-cvat export.xml --fps 30 --step 1 -o gt_occupancy.csv
   uv run python main.py --source videos/clip.mp4 --zone zones/clip.json --wait-log waits.csv
   uv run python busy_report.py eval-occupancy gt_occupancy.csv waits.csv --step 1
   ```
   Call out loudly: **`--fps` must be the clip's real frame rate, and `--step` must match between `import-cvat` and `eval-occupancy`**, or the timelines will not align.

- [ ] **Step 2: Verify the documented commands match the implemented CLI**

Run: `uv run python busy_report.py import-cvat --help && uv run python busy_report.py eval-occupancy --help`
Expected: help text shows `--fps`, `--step`, `-o` for `import-cvat` and `gt`, `waits`, `--step` for `eval-occupancy`, matching the doc.

- [ ] **Step 3: Commit**

```bash
git add docs/annotation-cvat.md
git commit -m "docs: CVAT deploy, label schema, and annotation rubric

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage (against `2026-06-22-cvat-ground-truth-annotation-design.md`):**
- §4.1 label schema → Task 6 (doc) + parsed by Task 1.
- §4.2 presence semantics (keyframe + `outside`, not interpolation) → Task 2 (`present_at`).
- §5.1 converter `cvat_import.py` → Tasks 1-3.
- §5.2 occupancy eval (MAE/bias/correlation) → Task 4. *(Phase-2 membership/association is intentionally out of this plan — see note below.)*
- §5.3 wiring `import-cvat` / `eval-occupancy` → Task 5.
- §5.4 docs `annotation-cvat.md` → Task 6.
- §6 predicted side from `waits.csv` via `sample_occupancy` → Task 5 (`_eval_occupancy`).
- §8 fps as CLI arg, step-must-match risk → Task 5 (`--fps`, `--step` + mismatch error) and Task 6 (loud callout).

**Phase 2 (membership/bystander) is deliberately deferred to its own plan.** It needs per-frame predicted track *positions* for GT↔predicted association, which `waits.csv` does not contain — that requires new logging in `runner.py` not yet designed. The schema and GT model built here already carry the data (membership attribute, point positions), so Phase 2 builds on this without rework.

**Placeholder scan:** no TBD/TODO; every code step shows complete code; every command shows expected output.

**Type consistency:** `GtShape`/`GtTrack` (Task 1) are consumed unchanged in Tasks 2-3; `occupancy_gt_at` (Task 2) used by `sample_occupancy_gt` (Task 3); timelines are `list[tuple[float, int]]` everywhere (Tasks 3-5); `OccupancyReport`/`occupancy_eval` names match between Task 4 and Task 5. `bias = pred - gt` defined once (Task 4) and described consistently in Task 5 help text and Task 6 doc.

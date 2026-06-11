# Multi-contour Zones + Combined Editor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `Zone` hold several disjoint polygons (union `contains`/`coverage`) and turn `--define-zone` into one editor that draws both line (`1`) and POS (`2`) contours, saving two type-pure files — with no analyzer/runner/config changes.

**Architecture:** `Zone` stores `polygons: list[list[(x,y)]]` with a back-compatible single-polygon constructor and a `from_polygons` classmethod; JSON gains a `polygons` key and still reads legacy `points`. Drawing iterates contours. The editor collects per-type contour lists and writes line/POS files via `Zone.from_polygons`.

**Tech Stack:** Python 3.12, numpy, OpenCV, pytest, `uv`.

**Spec:** `docs/superpowers/specs/2026-06-11-multi-contour-zones-design.md`

---

## File Structure
- Modify: `src/storepose/queue/zone.py` — N-polygon `Zone`.
- Modify: `src/storepose/drawing.py` — iterate `zone.polygons`.
- Modify: `src/storepose/queue/zone_editor.py` — `define_zones` combined editor.
- Modify: `main.py` — route `--define-zone` / `--define-pos-zone` through `define_zones`.
- Modify: `README.md`, `docs/usage.md`.
- Tests: `tests/queue/test_zone.py`, `tests/test_drawing.py`.

---

## Task 1: `Zone` holds N polygons (union membership)

**Files:**
- Modify: `src/storepose/queue/zone.py`
- Test: `tests/queue/test_zone.py`

- [ ] **Step 1: Write the failing tests** — append to `tests/queue/test_zone.py`:

```python
def test_multi_contour_contains_union():
    z = Zone.from_polygons([
        [(0, 0), (100, 0), (100, 100), (0, 100)],
        [(200, 200), (300, 200), (300, 300), (200, 300)],
    ])
    assert z.contains((50, 50)) is True       # in first contour
    assert z.contains((250, 250)) is True      # in second contour
    assert z.contains((150, 150)) is False     # in neither


def test_multi_contour_coverage_union():
    z = Zone.from_polygons([
        [(0, 0), (100, 0), (100, 100), (0, 100)],
        [(200, 200), (300, 200), (300, 300), (200, 300)],
    ])
    assert z.coverage([10, 10, 90, 90]) == 1.0      # fully inside first
    assert z.coverage([210, 210, 290, 290]) == 1.0  # fully inside second
    assert z.coverage([120, 120, 180, 180]) == 0.0  # between them


def test_multi_contour_json_round_trip(tmp_path):
    z = Zone.from_polygons([
        [(0, 0), (100, 0), (100, 100), (0, 100)],
        [(200, 200), (300, 200), (300, 300), (200, 300)],
    ])
    path = str(tmp_path / "z.json")
    z.save(path)
    loaded = Zone.load(path)
    assert len(loaded.polygons) == 2
    assert loaded.contains((250, 250)) is True


def test_legacy_points_json_still_loads(tmp_path):
    import json
    path = str(tmp_path / "legacy.json")
    with open(path, "w") as f:
        json.dump({"points": [[0, 0], [100, 0], [100, 100], [0, 100]]}, f)
    z = Zone.load(path)
    assert len(z.polygons) == 1
    assert z.contains((50, 50)) is True


def test_single_polygon_back_compat():
    z = Zone([(0, 0), (100, 0), (100, 100), (0, 100)])
    assert len(z.polygons) == 1
    assert z.points == [(0, 0), (100, 0), (100, 100), (0, 100)]
    assert z.contains((50, 50)) is True
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/queue/test_zone.py -k "multi_contour or legacy or back_compat" -v`
Expected: FAIL — no `Zone.from_polygons`; `Zone` has no `polygons`.

- [ ] **Step 3: Replace `src/storepose/queue/zone.py` with:**

```python
"""Polygon zone (one or more contours) with point/coverage tests + JSON."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def _is_nested(seq) -> bool:
    """True if ``seq`` is a list of contours (vs a flat list of points)."""
    return bool(seq) and isinstance(seq[0][0], (list, tuple, np.ndarray))


class Zone:
    """One or more polygon contours in image coordinates.

    ``contains`` and ``coverage`` test membership against the **union** of the
    contours. A contour with fewer than 3 points contributes nothing. Construct
    with a flat list of points for a single contour (back-compatible) or with
    :meth:`from_polygons` for several.
    """

    def __init__(self, points_or_polygons):
        polys = points_or_polygons or []
        if polys and not _is_nested(polys):
            polys = [polys]  # a flat list of points is one contour
        self.polygons: list[list[tuple[int, int]]] = [
            [(int(x), int(y)) for x, y in poly] for poly in polys
        ]
        self._polys = [
            np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
            for poly in self.polygons
            if len(poly) >= 3
        ]

    @classmethod
    def from_polygons(cls, polygons) -> "Zone":
        return cls(list(polygons))

    @property
    def points(self) -> list[tuple[int, int]]:
        """The first contour (back-compat; ``[]`` when empty)."""
        return self.polygons[0] if self.polygons else []

    def contains(self, point: tuple[float, float]) -> bool:
        """True if ``point`` is inside or on any contour."""
        pt = (float(point[0]), float(point[1]))
        return any(cv2.pointPolygonTest(poly, pt, False) >= 0 for poly in self._polys)

    def coverage(self, box, grid: int = 7) -> float:
        """Fraction of ``box`` (xyxy) inside any contour, via a grid sample."""
        if not self._polys:
            return 0.0
        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inside = 0
        for i in range(grid):
            px = x1 + (x2 - x1) * (i + 0.5) / grid
            for j in range(grid):
                py = y1 + (y2 - y1) * (j + 0.5) / grid
                if any(cv2.pointPolygonTest(poly, (px, py), False) >= 0
                       for poly in self._polys):
                    inside += 1
        return inside / (grid * grid)

    def to_dict(self) -> dict:
        return {"polygons": [[list(p) for p in poly] for poly in self.polygons]}

    @classmethod
    def from_dict(cls, data: dict) -> "Zone":
        if "polygons" in data:
            return cls.from_polygons(
                [[tuple(p) for p in poly] for poly in data["polygons"]]
            )
        return cls([tuple(p) for p in data["points"]])  # legacy single-contour

    def save(self, path: str) -> None:
        p = Path(path)
        if p.parent and not p.parent.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict()))

    @classmethod
    def load(cls, path: str) -> "Zone":
        return cls.from_dict(json.loads(Path(path).read_text()))
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/queue/test_zone.py -v` → existing zone tests (incl. `test_json_round_trip`, `test_degenerate_zone_contains_nothing`) + 5 new pass.
Run: `uv run pytest -q` → full suite green (analyzer uses `contains`/`coverage`, unchanged signatures).

- [ ] **Step 5: Commit**

```bash
git add src/storepose/queue/zone.py tests/queue/test_zone.py
git commit -m "feat: Zone holds multiple contours with union contains/coverage"
```

---

## Task 2: Drawing iterates contours

**Files:**
- Modify: `src/storepose/drawing.py`
- Test: `tests/test_drawing.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_drawing.py`:

```python
def test_annotate_queue_draws_all_contours():
    from storepose.drawing import annotate_queue
    from storepose.queue.types import QueueResult
    from storepose.queue.zone import Zone
    frame = _blank()  # 120x160
    zone = Zone.from_polygons([
        [(0, 0), (40, 0), (40, 40), (0, 40)],       # top-left contour
        [(110, 70), (150, 70), (150, 110), (110, 110)],  # bottom-right contour
    ])
    out = annotate_queue(frame.copy(), [], QueueResult(statuses=[], count=0),
                         zone, AppConfig())
    # both contours leave orange (red+green) pixels in their regions
    assert out[5, 5].sum() > 0
    assert out[90, 130].sum() > 0
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_drawing.py -k all_contours -v`
Expected: FAIL — only the first contour (via `zone.points`) is drawn, the second region stays black.

- [ ] **Step 3: Implement** in `src/storepose/drawing.py`.

Add a helper above `annotate_queue`:

```python
def _draw_zone(canvas: np.ndarray, zone: Zone, color: tuple[int, int, int]) -> None:
    """Fill (faint) and outline every contour of ``zone`` in ``color``."""
    for poly in zone.polygons:
        if len(poly) < 2:
            continue
        pts = np.array(poly, np.int32).reshape(-1, 1, 2)
        if len(poly) >= 3:
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.15, canvas, 0.85, 0, canvas)
        cv2.polylines(canvas, [pts], True, color, 2)
```

Replace the line-zone drawing block (the `if zone is not None and len(zone.points) >= 2:` block) and the POS-zone block with calls to the helper:

```python
    if zone is not None:
        _draw_zone(canvas, zone, ZONE_COLOR)
    if pos_zone is not None:
        _draw_zone(canvas, pos_zone, POS_COLOR)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_drawing.py -v` → all drawing tests pass (single-contour zones still render: the helper loops the one contour).

- [ ] **Step 5: Commit**

```bash
git add src/storepose/drawing.py tests/test_drawing.py
git commit -m "feat: draw every contour of a multi-contour zone"
```

---

## Task 3: Combined `define_zones` editor + main.py wiring

**Files:**
- Modify: `src/storepose/queue/zone_editor.py`
- Modify: `main.py`

This task's editor loop is an interactive GUI; it has no unit test (its save step is
covered by the `Zone` round-trip tests). Verify it manually in Step 4.

- [ ] **Step 1: Add `define_zones`** to `src/storepose/queue/zone_editor.py`.

Add a POS color constant near `_COLOR` and the new function (keep `define_zone`,
`default_zone_path`, `default_pos_zone_path` as-is):

```python
_POS_COLOR = (255, 200, 0)  # azure, matches drawing.POS_COLOR


def define_zones(
    source: int | str,
    line_path: str | None = None,
    pos_path: str | None = None,
    pos_only: bool = False,
) -> dict[str, str]:
    """Draw line (key ``1``) and POS (key ``2``) contours in one session.

    Keys: left-click adds a point; ``n`` finishes the current contour (>=3 pts)
    and starts a new one; ``u`` undoes a point; ``c`` clears all; ``s`` saves;
    ``q`` quits. Saves all line contours to ``line_path`` and all POS contours to
    ``pos_path`` (each only if it has >=1 valid contour). Returns the saved paths.
    """
    line_path = line_path or default_zone_path(source)
    pos_path = pos_path or default_pos_zone_path(source)
    cap = cv2.VideoCapture(source)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read a frame from source {source!r}")

    line_contours: list[list[tuple[int, int]]] = []
    pos_contours: list[list[tuple[int, int]]] = []
    current: list[tuple[int, int]] = []
    mode = "pos" if pos_only else "line"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            current.append((x, y))

    def color_for(m):
        return _POS_COLOR if m == "pos" else _COLOR

    def commit_current(target_mode):
        if len(current) >= 3:
            (pos_contours if target_mode == "pos" else line_contours).append(list(current))
        current.clear()

    win = "define zones"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    saved: dict[str, str] = {}
    while True:
        disp = frame.copy()
        for poly in line_contours:
            cv2.polylines(disp, [np.array(poly, np.int32).reshape(-1, 1, 2)], True, _COLOR, 2)
        for poly in pos_contours:
            cv2.polylines(disp, [np.array(poly, np.int32).reshape(-1, 1, 2)], True, _POS_COLOR, 2)
        c = color_for(mode)
        for p in current:
            cv2.circle(disp, p, 4, c, -1)
        if len(current) >= 2:
            cv2.polylines(disp, [np.array(current, np.int32).reshape(-1, 1, 2)],
                          len(current) >= 3, c, 2)
        hint = (f"mode:{mode.upper()}  1:line 2:pos  n:new-contour  "
                f"u:undo c:clear s:save q:quit   line:{len(line_contours)} pos:{len(pos_contours)}")
        cv2.putText(disp, hint, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2, cv2.LINE_AA)
        cv2.imshow(win, disp)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("1") and not pos_only:
            commit_current(mode); mode = "line"
        elif key == ord("2"):
            commit_current(mode); mode = "pos"
        elif key == ord("n"):
            commit_current(mode)
        elif key == ord("u") and current:
            current.pop()
        elif key == ord("c"):
            current.clear(); line_contours.clear(); pos_contours.clear()
        elif key == ord("s"):
            commit_current(mode)
            if pos_contours:
                Zone.from_polygons(pos_contours).save(pos_path)
                saved["pos"] = pos_path
                print(f"Saved POS zone ({len(pos_contours)} contour(s)) to {pos_path}")
            if line_contours and not pos_only:
                Zone.from_polygons(line_contours).save(line_path)
                saved["line"] = line_path
                print(f"Saved line zone ({len(line_contours)} contour(s)) to {line_path}")
            if saved:
                break
            print("Draw at least one contour (>=3 points) before saving.")
    cv2.destroyAllWindows()
    return saved
```

- [ ] **Step 2: Wire `main.py`** — replace the existing `--define-zone` and
`--define-pos-zone` branches with calls to `define_zones`:

```python
    if config.define_zone:
        from storepose.queue.zone_editor import define_zones
        saved = define_zones(config.source, config.zone, config.pos_zone)
        parts = []
        if "line" in saved:
            parts.append(f"--zone {saved['line']}")
        if "pos" in saved:
            parts.append(f"--pos-zone {saved['pos']}")
        print("Run with: " + " ".join(parts) if parts else "Nothing saved.")
        return 0
    if config.define_pos_zone:
        from storepose.queue.zone_editor import define_zones
        saved = define_zones(config.source, pos_path=config.pos_zone, pos_only=True)
        print(f"Run with: --pos-zone {saved['pos']}" if "pos" in saved else "Nothing saved.")
        return 0
```

- [ ] **Step 3: Import/syntax check**

Run: `uv run pytest -q` → full suite still green (editor not unit-tested, nothing else touched).
Run: `uv run python -c "import main; from storepose.queue.zone_editor import define_zones; print('ok')"` → `ok`.

- [ ] **Step 4: Manual GUI verification** (the part that needs eyes)

Run: `uv run python main.py --define-zone --source /tmp/sco_clip.mp4`
Verify: window opens in LINE mode (orange); click a few points, press `n` to start a
second line contour, press `2` to switch to POS (azure), draw a contour, press `s`.
Expected: prints `Saved line zone (...) to zones/sco_clip.json` and
`Saved POS zone (...) to zones/sco_clip_pos.json`, then the run command. Confirm
both files exist and load:

Run: `uv run python -c "from storepose.queue.zone import Zone; print(len(Zone.load('zones/sco_clip.json').polygons), len(Zone.load('zones/sco_clip_pos.json').polygons))"`
Expected: prints the two contour counts.

- [ ] **Step 5: Commit**

```bash
git add src/storepose/queue/zone_editor.py main.py
git commit -m "feat: combined 1=line / 2=pos multi-contour zone editor"
```

---

## Task 4: Documentation

**Files:**
- Modify: `README.md`, `docs/usage.md`

- [ ] **Step 1: README** — in the "Waiting in line" / zone section, replace the
single-contour editor description so the editor keys read:

```markdown
Draw zones with one editor: `--define-zone` opens a frame where **`1`** draws
**line** contours and **`2`** draws **POS** contours. Left-click adds a point,
**`n`** finishes the current contour (≥3 pts) and starts a new one, **`u`** undo,
**`c`** clear all, **`s`** saves (line contours → `zones/<name>.json`, POS contours
→ `zones/<name>_pos.json`), **`q`** quit. A zone may have several disjoint contours
— a person counts as in-zone if inside **any** of them.
```

- [ ] **Step 2: usage.md** — in the "Define a queue zone" section, update the editor
controls table/text to list `1`/`2` (line/POS), `n` (new contour), and that a zone
can be multiple contours saved to the line and POS files. Replace the controls
table body with:

```markdown
| Switch to line / POS contours | `1` / `2` |
| Add a point | left-click |
| Finish contour, start a new one | `n` (needs ≥ 3 points) |
| Undo last point | `u` |
| Clear everything | `c` |
| Save (writes line + POS files) | `s` |
| Quit without saving | `q` / Esc |
```

- [ ] **Step 3: Commit**

```bash
git add README.md docs/usage.md
git commit -m "docs: document multi-contour zones + 1/2 editor keys"
```

---

## Task 5: Full verification

- [ ] **Step 1:** `uv run pytest -q` → all green.
- [ ] **Step 2:** Run the pipeline with an existing single-contour zone to confirm
back-compat (no regression):

Run: `uv run python main.py --source /tmp/sco_clip.mp4 --zone "zones/2706969 - Rock Hill, SC _SCO 1 _2026-05-07 12_09_21_370.json" --busy --save /tmp/mc_out.mp4`
Expected: runs to completion; the (single-contour) zone still draws and waiting works.

---

## Self-Review Notes
- **Spec coverage:** N-polygon `Zone` with union contains/coverage + `from_polygons` + `points` back-compat (Task 1) ✓; JSON `polygons` + legacy `points` (Task 1) ✓; drawing iterates contours (Task 2) ✓; combined `1`/`2` editor with `n`/`u`/`c`/`s`/`q`, two type-pure files, `pos_only` (Task 3) ✓; main wiring (Task 3) ✓; docs (Task 4) ✓; analyzer/runner/config untouched — verified by full suite staying green (Tasks 1, 3) ✓.
- **Type consistency:** `Zone(points)` single / `Zone.from_polygons(polygons)` multi; `Zone.polygons`, `Zone.points`, `contains`, `coverage`, `to_dict`/`from_dict`; `define_zones(source, line_path, pos_path, pos_only) -> dict` used consistently in main.py.
- **Back-compat:** existing `Zone([(x,y),...])`, `zone.points`, legacy `{"points":...}` files, and single-contour drawing all preserved; existing zone/drawing/analyzer tests stay valid.
```

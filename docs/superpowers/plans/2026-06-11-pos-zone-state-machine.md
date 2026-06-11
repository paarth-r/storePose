# POS Zone + Per-State Time Decomposition — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split each person's line visit into *waiting* (in line, not at register) and *serving* (at the POS) time via a second POS zone, by rebuilding `QueueAnalyzer` as an explicit per-person state machine that attributes re-id-gap time to the state the person held when lost.

**Architecture:** A new `--pos-zone` polygon plus a state machine (OUT → WAITING → SERVING → SERVED, or WAITING → ABANDONED) in `QueueAnalyzer`; each frame's `dt` (including absent frames within the re-id grace window) accrues to the current/held state. Public types are extended with defaulted fields so the `busy/` package is undisturbed. Overlay draws the POS zone in cyan and tags serving people `POS n.n s`.

**Tech Stack:** Python 3.12, numpy, OpenCV, pytest, `uv`.

**Spec:** `docs/superpowers/specs/2026-06-11-pos-zone-state-machine-design.md`

---

## File Structure

- Modify: `src/storepose/queue/types.py` — extend `PersonStatus`, `CompletedWait`, `QueueResult`.
- Modify: `src/storepose/queue/analyzer.py` — rebuild as the waiting/serving state machine.
- Modify: `src/storepose/config.py` — `pos_zone`, `define_pos_zone` + CLI.
- Modify: `src/storepose/queue/zone_editor.py` — `default_pos_zone_path`.
- Modify: `main.py` — handle `--define-pos-zone`.
- Modify: `src/storepose/drawing.py` — draw POS zone + serving tag + header count.
- Modify: `src/storepose/busy/report.py` — `read_waits` parses new columns.
- Modify: `src/storepose/runner.py` — load/pass pos zone; new CSV columns; feed busy served-only.
- Modify: `README.md`, `docs/usage.md`.
- Tests: `tests/queue/test_analyzer.py`, `tests/test_config.py`, `tests/test_drawing.py`, `tests/busy/test_report.py`.

---

## Task 1: Extend public queue types

**Files:**
- Modify: `src/storepose/queue/types.py`
- Test: `tests/queue/test_types_pos.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/queue/test_types_pos.py
from storepose.queue.types import PersonStatus, CompletedWait, QueueResult


def test_person_status_has_serving_fields_with_defaults():
    s = PersonStatus(id=1, waiting=True, candidate=False, progress=1.0, wait_seconds=2.0)
    assert s.serving is False
    assert s.serving_seconds == 0.0
    s2 = PersonStatus(id=2, waiting=False, candidate=False, progress=1.0,
                      wait_seconds=0.0, serving=True, serving_seconds=3.5)
    assert s2.serving is True and s2.serving_seconds == 3.5


def test_completed_wait_has_serving_and_outcome_defaults():
    c = CompletedWait(id=1, entered_s=0.0, exited_s=10.0, wait_seconds=8.0)
    assert c.serving_seconds == 0.0
    assert c.outcome == "served"
    c2 = CompletedWait(1, 0.0, 12.0, 8.0, 4.0, "served")
    assert c2.serving_seconds == 4.0 and c2.outcome == "served"


def test_queue_result_has_serving_count_default():
    r = QueueResult(statuses=[], count=0)
    assert r.serving_count == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/queue/test_types_pos.py -v`
Expected: FAIL — `PersonStatus` has no `serving` / `CompletedWait` has no `serving_seconds`.

- [ ] **Step 3: Write minimal implementation**

In `src/storepose/queue/types.py`, add fields to the three dataclasses. `PersonStatus`:

```python
@dataclass
class PersonStatus:
    """Waiting/serving state for one tracked person this frame.

    Attributes:
        id: Track id.
        waiting: True if the person is in line (waiting region, not at POS).
        candidate: True if accumulating frames toward inclusion (not yet counted).
        progress: Inclusion progress in ``[0, 1]`` (1.0 once waiting/serving).
        wait_seconds: Accumulated waiting time so far.
        serving: True if the person is at the POS being served.
        serving_seconds: Accumulated serving (at-POS) time so far.
    """

    id: int
    waiting: bool
    candidate: bool
    progress: float
    wait_seconds: float
    serving: bool = False
    serving_seconds: float = 0.0
```

`CompletedWait`:

```python
@dataclass
class CompletedWait:
    """A finished line visit, emitted the frame a person's visit ends.

    ``wait_seconds`` is the waiting portion; ``serving_seconds`` the at-POS
    portion; ``outcome`` is ``"served"`` (reached POS) or ``"abandoned"``.
    """

    id: int
    entered_s: float
    exited_s: float
    wait_seconds: float
    serving_seconds: float = 0.0
    outcome: str = "served"
```

`QueueResult`:

```python
@dataclass
class QueueResult:
    """Per-frame queue analysis output."""

    statuses: list[PersonStatus]
    count: int
    serving_count: int = 0
    completed: list[CompletedWait] = field(default_factory=list)
```

(Keep the existing `field` import; `completed` stays after `serving_count`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/queue/test_types_pos.py -v` → PASS.
Run: `uv run pytest -q` → full suite still green (defaults keep existing constructions valid).

- [ ] **Step 5: Commit**

```bash
git add src/storepose/queue/types.py tests/queue/test_types_pos.py
git commit -m "feat: extend queue types with serving + outcome fields"
```

---

## Task 2: Rebuild QueueAnalyzer as a waiting/serving state machine

**Files:**
- Modify: `src/storepose/queue/analyzer.py`
- Test: `tests/queue/test_analyzer.py`

- [ ] **Step 1: Write the failing tests** — append to `tests/queue/test_analyzer.py` (it already defines `ZONE`, `person(pid, box)`; add a POS zone and tests):

```python
# A POS zone occupying the right half; the line ZONE is the full (0,0)-(200,200).
POS = Zone([(120, 0), (200, 0), (200, 200), (120, 200)])


def pos_person(pid, x):
    # foot center at (x, 80); x>=120 is "at POS", x<120 is "waiting region"
    return person(pid, [x - 10, 40, x + 10, 80])


def test_waiting_then_serving_then_served():
    an = QueueAnalyzer(ZONE, pos_zone=POS, enter_frames=2, exit_seconds=1.0)
    # two frames in the waiting region -> WAITING
    an.update([pos_person(1, 40)], 0.5)
    r = an.update([pos_person(1, 40)], 0.5)
    assert r.statuses[0].waiting is True and r.statuses[0].serving is False
    an.update([pos_person(1, 40)], 0.5)              # accruing waiting
    r2 = an.update([pos_person(1, 160)], 0.5)        # step into POS -> SERVING
    assert r2.statuses[0].serving is True
    assert r2.serving_count == 1 and r2.count == 0
    an.update([pos_person(1, 160)], 0.5)             # accruing serving
    # leave POS for >= exit_seconds -> SERVED
    an.update([pos_person(1, 400)], 0.5)
    r3 = an.update([pos_person(1, 400)], 0.5)
    assert len(r3.completed) == 1
    c = r3.completed[0]
    assert c.outcome == "served" and c.wait_seconds > 0 and c.serving_seconds > 0


def test_waiting_then_abandoned_before_pos():
    an = QueueAnalyzer(ZONE, pos_zone=POS, enter_frames=2, exit_seconds=1.0)
    an.update([pos_person(1, 40)], 0.5)
    an.update([pos_person(1, 40)], 0.5)              # WAITING
    an.update([pos_person(1, 400)], 0.5)             # out of line
    r = an.update([pos_person(1, 400)], 0.5)         # out >= exit_seconds -> ABANDONED
    assert len(r.completed) == 1
    assert r.completed[0].outcome == "abandoned"
    assert r.completed[0].serving_seconds == 0.0


def test_walkup_straight_to_pos_has_no_waiting():
    an = QueueAnalyzer(ZONE, pos_zone=POS, enter_frames=2, exit_seconds=1.0)
    an.update([pos_person(1, 160)], 0.5)             # directly at POS
    r = an.update([pos_person(1, 160)], 0.5)         # 2 frames -> SERVING
    assert r.statuses[0].serving is True
    an.update([pos_person(1, 400)], 0.5)
    r2 = an.update([pos_person(1, 400)], 0.5)        # leave -> SERVED
    assert r2.completed[0].outcome == "served"
    assert r2.completed[0].wait_seconds == 0.0


def test_gap_while_serving_adds_to_serving_seconds():
    an = QueueAnalyzer(ZONE, pos_zone=POS, enter_frames=2, exit_seconds=5.0,
                       reid_grace_seconds=3.0)
    an.update([pos_person(1, 160)], 0.5)
    r = an.update([pos_person(1, 160)], 0.5)         # SERVING
    before = r.statuses[0].serving_seconds
    an.update([], 0.5)                               # vanished (within grace)
    an.update([], 0.5)
    r2 = an.update([pos_person(1, 160)], 0.5)        # re-id -> resume serving
    assert r2.statuses[0].serving is True
    # the two absent frames were attributed to serving
    assert r2.statuses[0].serving_seconds >= before + 1.0


def test_gap_while_waiting_attributed_to_waiting_even_if_returns_at_pos():
    an = QueueAnalyzer(ZONE, pos_zone=POS, enter_frames=2, exit_seconds=5.0,
                       reid_grace_seconds=3.0)
    an.update([pos_person(1, 40)], 0.5)
    r = an.update([pos_person(1, 40)], 0.5)          # WAITING
    wait_before = r.statuses[0].wait_seconds
    an.update([], 0.5)                               # vanished while waiting
    an.update([], 0.5)
    r2 = an.update([pos_person(1, 160)], 0.5)        # returns at POS
    # the gap counted as WAITING (pre-loss state), not serving
    assert r2.statuses[0].wait_seconds >= wait_before + 1.0
    assert r2.statuses[0].serving is True


def test_gap_past_grace_finalizes_with_outcome():
    an = QueueAnalyzer(ZONE, pos_zone=POS, enter_frames=2, exit_seconds=5.0,
                       reid_grace_seconds=1.0)
    an.update([pos_person(1, 160)], 0.5)
    an.update([pos_person(1, 160)], 0.5)             # SERVING (reached POS)
    an.update([], 0.5)                               # absent 0.5 < 1.0
    r = an.update([], 0.5)                           # absent 1.0 >= grace -> finalize
    assert len(r.completed) == 1 and r.completed[0].outcome == "served"
```

The existing single-zone tests in this file must keep passing (they construct `QueueAnalyzer(ZONE, ...)` with no `pos_zone` → `serving` never entered).

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/queue/test_analyzer.py -k "serving or abandoned or walkup or gap" -v`
Expected: FAIL — `QueueAnalyzer.__init__()` has no `pos_zone`.

- [ ] **Step 3: Write minimal implementation** — replace the body of `src/storepose/queue/analyzer.py` from the `_WaitState` dataclass through the end of `update` with:

```python
@dataclass
class _VisitState:
    state: str = "out"            # "out" | "waiting" | "serving"
    waiting_seconds: float = 0.0
    serving_seconds: float = 0.0
    entered_s: float = 0.0        # clock when the visit began (first waiting/serving)
    reached_pos: bool = False
    in_frames: int = 0
    in_seconds: float = 0.0
    out_streak: float = 0.0
    absent_seconds: float = 0.0   # time the track has been gone (re-id grace)


class QueueAnalyzer:
    """Per-person line state machine splitting a visit into waiting vs serving.

    States: OUT (not yet counted) -> WAITING (in the line zone, not at POS) ->
    SERVING (in the POS zone) -> done. ``enter_frames`` + ``min_dwell_seconds``
    is the bystander gate for entering the line; ``exit_seconds`` debounces the
    leaving transitions. Each frame's ``dt`` accrues to the current state; while
    a track is vanished (within ``reid_grace_seconds``) ``dt`` is attributed to
    the state held when it disappeared, so a re-identified person resumes with
    the gap counted into that state. A visit finalizes as ``"served"`` (reached
    POS) or ``"abandoned"`` and is reported in :attr:`QueueResult.completed`.
    With ``pos_zone=None`` no one ever enters SERVING (single-zone behavior).
    """

    def __init__(
        self,
        zone: Zone,
        pos_zone: Zone | None = None,
        enter_frames: int = 5,
        exit_seconds: float = 2.0,
        kpt_thr: float = 0.5,
        coverage_thr: float = 0.5,
        foot_band: float = 0.3,
        min_dwell_seconds: float = 0.0,
        reid_grace_seconds: float = 0.0,
    ):
        self.zone = zone
        self.pos_zone = pos_zone
        self.enter_frames = max(1, enter_frames)
        self.exit_seconds = exit_seconds
        self.min_dwell_seconds = max(0.0, min_dwell_seconds)
        self.reid_grace_seconds = max(0.0, reid_grace_seconds)
        self.kpt_thr = kpt_thr
        self.coverage_thr = coverage_thr
        self.foot_band = foot_band
        self._states: dict[int, _VisitState] = {}
        self._clock = 0.0

    def _foot_box(self, box) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        band = max(1.0, (y2 - y1) * self.foot_band)
        return (x1, y2 - band, x2, y2)

    def _in_zone(self, person: TrackedPerson, zone: Zone) -> bool:
        """Ankle-midpoint-inside OR foot-region-coverage test against ``zone``."""
        kpts, scores = person.keypoints, person.scores
        if kpts is not None and scores is not None:
            ankles = [
                (float(kpts[idx][0]), float(kpts[idx][1]))
                for idx in (_L_ANKLE, _R_ANKLE)
                if float(scores[idx]) >= self.kpt_thr
            ]
            if ankles:
                mid = (
                    sum(p[0] for p in ankles) / len(ankles),
                    sum(p[1] for p in ankles) / len(ankles),
                )
                if zone.contains(mid):
                    return True
        return zone.coverage(self._foot_box(person.box)) >= self.coverage_thr

    def _finalize(self, pid: int, st: _VisitState, completed: list) -> None:
        completed.append(
            CompletedWait(
                pid, st.entered_s, self._clock, st.waiting_seconds,
                st.serving_seconds, "served" if st.reached_pos else "abandoned",
            )
        )

    def _status(self, pid: int, st: _VisitState) -> PersonStatus:
        waiting = st.state == "waiting"
        serving = st.state == "serving"
        candidate = st.state == "out" and st.in_frames > 0
        frame_progress = st.in_frames / self.enter_frames
        if self.min_dwell_seconds > 0:
            frame_progress = min(frame_progress, st.in_seconds / self.min_dwell_seconds)
        progress = 1.0 if (waiting or serving) else min(frame_progress, 1.0)
        return PersonStatus(
            pid, waiting, candidate, progress, st.waiting_seconds,
            serving, st.serving_seconds,
        )

    def update(self, people: list[TrackedPerson], dt: float) -> QueueResult:
        self._clock += dt
        present: set[int] = set()
        statuses: list[PersonStatus] = []
        completed: list[CompletedWait] = []

        for person in people:
            present.add(person.id)
            st = self._states.setdefault(person.id, _VisitState())
            st.absent_seconds = 0.0

            in_line = self._in_zone(person, self.zone)
            in_pos = self.pos_zone is not None and self._in_zone(person, self.pos_zone)

            if st.state == "serving":
                st.serving_seconds += dt
                if in_pos:
                    st.out_streak = 0.0
                else:
                    st.out_streak += dt
                    if st.out_streak >= self.exit_seconds:
                        self._finalize(person.id, st, completed)
                        self._states.pop(person.id)
                        statuses.append(PersonStatus(person.id, False, False, 0.0, 0.0))
                        continue
            elif st.state == "waiting":
                st.waiting_seconds += dt
                if in_pos:
                    st.state = "serving"
                    st.reached_pos = True
                    st.out_streak = 0.0
                elif in_line:
                    st.out_streak = 0.0
                else:
                    st.out_streak += dt
                    if st.out_streak >= self.exit_seconds:
                        self._finalize(person.id, st, completed)  # abandoned
                        self._states.pop(person.id)
                        statuses.append(PersonStatus(person.id, False, False, 0.0, 0.0))
                        continue
            else:  # out
                if in_line or in_pos:
                    st.out_streak = 0.0
                    st.in_frames += 1
                    st.in_seconds += dt
                    if (
                        st.in_frames >= self.enter_frames
                        and st.in_seconds >= self.min_dwell_seconds
                    ):
                        st.entered_s = self._clock
                        if in_pos:
                            st.state = "serving"
                            st.reached_pos = True
                        else:
                            st.state = "waiting"
                else:
                    st.out_streak += dt
                    if st.out_streak >= self.exit_seconds:
                        st.in_frames = 0
                        st.in_seconds = 0.0

            statuses.append(self._status(person.id, st))

        # vanished tracks: carry the gap forward into the held state while within
        # the re-id grace window; finalize once gone past it.
        for pid in list(self._states):
            if pid in present:
                continue
            st = self._states[pid]
            st.absent_seconds += dt
            if st.absent_seconds >= self.reid_grace_seconds:
                self._states.pop(pid)
                if st.state in ("waiting", "serving"):
                    self._finalize(pid, st, completed)
            elif st.state == "waiting":
                st.waiting_seconds += dt
            elif st.state == "serving":
                st.serving_seconds += dt

        count = sum(1 for s in statuses if s.waiting)
        serving_count = sum(1 for s in statuses if s.serving)
        return QueueResult(
            statuses=statuses, count=count, serving_count=serving_count,
            completed=completed,
        )
```

Keep the module imports and `_L_ANKLE, _R_ANKLE` line at the top unchanged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/queue/test_analyzer.py -v` → existing single-zone tests + the new POS tests pass.
Run: `uv run pytest -q` → full suite green.

- [ ] **Step 5: Commit**

```bash
git add src/storepose/queue/analyzer.py tests/queue/test_analyzer.py
git commit -m "feat: waiting/serving state machine with POS zone + gap attribution"
```

---

## Task 3: Config — `--pos-zone` and `--define-pos-zone`

**Files:**
- Modify: `src/storepose/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_config.py`:

```python
def test_pos_zone_flags():
    cfg = from_args(["--pos-zone", "zones/p.json"])
    assert cfg.pos_zone == "zones/p.json"
    assert cfg.define_pos_zone is False
    assert from_args(["--define-pos-zone"]).define_pos_zone is True


def test_pos_zone_defaults_none():
    cfg = from_args([])
    assert cfg.pos_zone is None
    assert cfg.define_pos_zone is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -k pos_zone -v`
Expected: FAIL — `AppConfig` has no `pos_zone`.

- [ ] **Step 3: Write minimal implementation** in `src/storepose/config.py`:

(a) Add dataclass fields after the existing `zone: str | None = None` / `define_zone: bool = False` lines:

```python
    pos_zone: str | None = None
    define_pos_zone: bool = False
```

(b) Add docstring lines in the `AppConfig` Attributes block, after the `zone` / `define_zone` entries:

```python
        pos_zone: Path to a POS-zone JSON; enables waiting-vs-serving split.
        define_pos_zone: Launch the editor for the POS zone and exit.
```

(c) Add CLI args in `_build_parser`, right after the `--define-zone` argument:

```python
    parser.add_argument(
        "--pos-zone", default=None, metavar="PATH",
        help="POS-zone JSON to load; splits line time into waiting vs serving.",
    )
    parser.add_argument(
        "--define-pos-zone", dest="define_pos_zone", action="store_true",
        help="Launch the interactive editor for the POS zone and exit.",
    )
```

(d) Wire into `from_args`'s `AppConfig(...)`, after `define_zone=args.define_zone,`:

```python
        pos_zone=args.pos_zone,
        define_pos_zone=args.define_pos_zone,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/storepose/config.py tests/test_config.py
git commit -m "feat: --pos-zone / --define-pos-zone config flags"
```

---

## Task 4: Zone editor default path + main.py wiring

**Files:**
- Modify: `src/storepose/queue/zone_editor.py`
- Modify: `main.py`
- Test: `tests/test_io.py` (append a small unit test for the path helper)

- [ ] **Step 1: Write the failing test** — append to `tests/test_io.py`:

```python
def test_default_pos_zone_path():
    from storepose.queue.zone_editor import default_pos_zone_path
    assert default_pos_zone_path(0) == "zones/cam0_pos.json"
    assert default_pos_zone_path("videos/clip.mp4") == "zones/clip_pos.json"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_io.py -k pos_zone_path -v`
Expected: FAIL — no `default_pos_zone_path`.

- [ ] **Step 3: Write minimal implementation**

In `src/storepose/queue/zone_editor.py`, add after `default_zone_path`:

```python
def default_pos_zone_path(source: int | str) -> str:
    """Default ``zones/<name>_pos.json`` path for a POS zone."""
    name = f"cam{source}" if isinstance(source, int) else Path(str(source)).stem
    return str(Path("zones") / f"{name}_pos.json")
```

In `main.py`, add a `--define-pos-zone` branch right after the existing
`if config.define_zone:` block (before the `try:`):

```python
    if config.define_pos_zone:
        from storepose.queue.zone_editor import define_zone, default_pos_zone_path
        path = define_zone(config.source, config.pos_zone or default_pos_zone_path(config.source))
        print(f"Run with: --pos-zone {path}")
        return 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_io.py -k pos_zone_path -v` → PASS.
Run: `uv run python -c "import main; print('ok')"` → `ok` (main.py imports cleanly).

- [ ] **Step 5: Commit**

```bash
git add src/storepose/queue/zone_editor.py main.py tests/test_io.py
git commit -m "feat: POS zone editor default path + --define-pos-zone entrypoint"
```

---

## Task 5: Drawing — POS zone, serving tag, header count

**Files:**
- Modify: `src/storepose/drawing.py`
- Test: `tests/test_drawing.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_drawing.py`:

```python
def test_annotate_queue_draws_pos_zone_and_serving_tag():
    from storepose.drawing import annotate_queue, POS_COLOR
    from storepose.queue.types import PersonStatus, QueueResult
    from storepose.queue.zone import Zone
    frame = _blank()
    teal = (200, 200, 0)
    person = TrackedPerson(id=1, box=np.array([20, 20, 100, 110], float),
                           keypoints=None, scores=None, coasting=False, color=teal)
    result = QueueResult(
        statuses=[PersonStatus(id=1, waiting=False, candidate=False, progress=1.0,
                               wait_seconds=0.0, serving=True, serving_seconds=4.2)],
        count=0, serving_count=1,
    )
    line_zone = Zone([(0, 0), (160, 0), (160, 120), (0, 120)])
    pos_zone = Zone([(80, 0), (160, 0), (160, 120), (80, 120)])
    out = annotate_queue(frame.copy(), [person], result, line_zone, AppConfig(),
                         pos_zone=pos_zone)
    assert out.shape == frame.shape
    # POS_COLOR (cyan/azure, BGR) has a blue channel; some blue pixels appear
    assert (out[:, :, 0] > 0).sum() > 0


def test_annotate_queue_pos_zone_optional():
    from storepose.drawing import annotate_queue
    from storepose.queue.types import QueueResult
    out = annotate_queue(_blank(), [], QueueResult(statuses=[], count=0), None, AppConfig())
    assert out.shape == (120, 160, 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_drawing.py -k "pos_zone or serving_tag" -v`
Expected: FAIL — `annotate_queue` takes no `pos_zone`; no `POS_COLOR`.

- [ ] **Step 3: Write minimal implementation** in `src/storepose/drawing.py`:

(a) Add a color constant near `ZONE_COLOR`:

```python
POS_COLOR = (255, 200, 0)  # BGR azure/cyan — the POS zone, distinct from orange
```

(b) Change `annotate_queue`'s signature to accept `pos_zone`:

```python
def annotate_queue(
    canvas: np.ndarray,
    people: list[TrackedPerson],
    result: QueueResult,
    zone: Zone | None,
    config: AppConfig,
    pos_zone: Zone | None = None,
) -> np.ndarray:
```

(c) After the existing line-zone polygon drawing block (the `if zone is not None ...`
block ending with `cv2.polylines(canvas, [pts], True, ZONE_COLOR, 2)`), draw the
POS zone:

```python
    if pos_zone is not None and len(pos_zone.points) >= 2:
        ppts = np.array(pos_zone.points, np.int32).reshape(-1, 1, 2)
        if len(pos_zone.points) >= 3:
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [ppts], POS_COLOR)
            cv2.addWeighted(overlay, 0.15, canvas, 0.85, 0, canvas)
        cv2.polylines(canvas, [ppts], True, POS_COLOR, 2)
```

(d) In the per-person loop, add a `serving` branch. Change the existing
`if s.waiting:` chain so it reads `if s.waiting: ... elif s.serving: ... elif s.candidate: ...`.
Insert this `elif s.serving:` block between the waiting block and the candidate block:

```python
        elif s.serving:
            # at POS: solid translucent fill in the person's color + POS timer
            overlay = canvas.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), p.color, -1)
            cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, canvas)
            tag = f"POS {s.serving_seconds:0.1f}s"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            ty = min(y2 + th + 6, canvas.shape[0] - 2)
            cv2.rectangle(canvas, (x1, ty - th - 6), (x1 + tw + 6, ty), p.color, -1)
            cv2.putText(canvas, tag, (x1 + 3, ty - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 0), 2, cv2.LINE_AA)
```

(e) Replace the header line so it shows both counts:

```python
    header = f"in line: {result.count}   at POS: {result.serving_count}"
    cv2.putText(canvas, header, (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, ZONE_COLOR, 2, cv2.LINE_AA)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_drawing.py -v` → existing drawing tests + the 2 new pass.

- [ ] **Step 5: Commit**

```bash
git add src/storepose/drawing.py tests/test_drawing.py
git commit -m "feat: draw POS zone + serving tag + at-POS header count"
```

---

## Task 6: Busy report — parse new CSV columns

**Files:**
- Modify: `src/storepose/busy/report.py`
- Test: `tests/busy/test_report.py`

- [ ] **Step 1: Write the failing test** — append to `tests/busy/test_report.py` (it already imports/uses tmp paths; this test writes a CSV and reads it back):

```python
def test_read_waits_parses_serving_and_outcome(tmp_path):
    from storepose.busy.report import read_waits
    p = tmp_path / "w.csv"
    p.write_text(
        "id,entered_s,exited_s,wait_seconds,serving_seconds,outcome\n"
        "1,0.00,10.00,7.00,3.00,served\n"
    )
    waits = read_waits(p)
    assert len(waits) == 1
    assert waits[0].wait_seconds == 7.0
    assert waits[0].serving_seconds == 3.0
    assert waits[0].outcome == "served"


def test_read_waits_back_compat_without_new_columns(tmp_path):
    from storepose.busy.report import read_waits
    p = tmp_path / "old.csv"
    p.write_text("id,entered_s,exited_s,wait_seconds\n1,0.00,10.00,8.00\n")
    waits = read_waits(p)
    assert waits[0].serving_seconds == 0.0 and waits[0].outcome == "served"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/busy/test_report.py -k "serving or back_compat" -v`
Expected: FAIL — `read_waits` ignores / can't populate the new fields.

- [ ] **Step 3: Write minimal implementation** in `src/storepose/busy/report.py`, in `read_waits`, expand the `CompletedWait(...)` construction:

```python
            waits.append(
                CompletedWait(
                    id=int(row["id"]),
                    entered_s=float(row["entered_s"]),
                    exited_s=float(row["exited_s"]),
                    wait_seconds=float(row["wait_seconds"]),
                    serving_seconds=float(row.get("serving_seconds") or 0.0),
                    outcome=(row.get("outcome") or "served"),
                )
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/busy/test_report.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/storepose/busy/report.py tests/busy/test_report.py
git commit -m "feat: read_waits parses serving_seconds + outcome (back-compatible)"
```

---

## Task 7: Runner wiring — load POS zone, new CSV columns, busy served-only

**Files:**
- Modify: `src/storepose/runner.py`

- [ ] **Step 1: Load the POS zone and pass it to the analyzer**

In `src/storepose/runner.py`, the zone block currently reads:

```python
                zone, analyzer = None, None
                if config.zone:
                    if tracker is None:
                        print("Note: --zone needs tracking; ignoring (you passed --no-track).")
                    else:
                        zone = Zone.load(config.zone)
                        analyzer = QueueAnalyzer(
                            zone,
                            enter_frames=config.wait_enter_frames,
                            exit_seconds=config.wait_exit_seconds,
                            kpt_thr=config.kpt_thr,
                            coverage_thr=config.zone_coverage,
                            foot_band=config.zone_foot_band,
                            min_dwell_seconds=config.wait_min_dwell,
                            reid_grace_seconds=config.reid_seconds if config.reid else 0.0,
                        )
```

Replace that whole block with (adds `pos_zone` to the defaults line, loads it, passes
it to the analyzer, and prints a note when `--pos-zone` is given without `--zone`):

```python
                zone, analyzer, pos_zone = None, None, None
                if config.zone:
                    if tracker is None:
                        print("Note: --zone needs tracking; ignoring (you passed --no-track).")
                    else:
                        zone = Zone.load(config.zone)
                        pos_zone = Zone.load(config.pos_zone) if config.pos_zone else None
                        analyzer = QueueAnalyzer(
                            zone,
                            pos_zone=pos_zone,
                            enter_frames=config.wait_enter_frames,
                            exit_seconds=config.wait_exit_seconds,
                            kpt_thr=config.kpt_thr,
                            coverage_thr=config.zone_coverage,
                            foot_band=config.zone_foot_band,
                            min_dwell_seconds=config.wait_min_dwell,
                            reid_grace_seconds=config.reid_seconds if config.reid else 0.0,
                        )
                elif config.pos_zone:
                    print("Note: --pos-zone needs --zone (the line zone); ignoring POS.")
```

- [ ] **Step 2: New wait-log columns**

Change the wait-log header writer:

```python
                    wait_writer.writerow(
                        ["id", "entered_s", "exited_s", "wait_seconds",
                         "serving_seconds", "outcome"]
                    )
```

And the per-completed row writer:

```python
                                for c in qresult.completed:
                                    wait_writer.writerow(
                                        [c.id, f"{c.entered_s:.2f}", f"{c.exited_s:.2f}",
                                         f"{c.wait_seconds:.2f}", f"{c.serving_seconds:.2f}",
                                         c.outcome]
                                    )
```

- [ ] **Step 3: Pass pos_zone to the overlay; feed busy served-only**

Change the annotate_queue call:

```python
                            canvas = annotate_queue(canvas, people, qresult, zone, config, pos_zone=pos_zone)
```

Change the busy feed loop to only add served visits:

```python
                                for c in qresult.completed:
                                    if c.outcome == "served":
                                        busy.add_wait(c)
```

- [ ] **Step 4: Verify**

Run: `uv run pytest -q` → full suite green (no runner unit tests, but nothing else breaks).
Run: `uv run python -c "from storepose.runner import Runner; print('ok')"` → `ok`.

Smoke the two-zone path end to end on a short clip. Note: one line, no backslash continuations. First make a POS zone for the clip (or reuse the line zone as a stand-in for a smoke test only):

Run: `uv run python -c "from storepose.queue.zone import Zone; import json; z=json.load(open('zones/2706969 - Rock Hill, SC _SCO 1 _2026-05-07 12_09_21_370.json')); Zone([(p[0]//2, p[1]) for p in z['points']]).save('/tmp/pos_smoke.json'); print('wrote /tmp/pos_smoke.json')"`

Run: `uv run python main.py --source /tmp/sco_clip.mp4 --zone "zones/2706969 - Rock Hill, SC _SCO 1 _2026-05-07 12_09_21_370.json" --pos-zone /tmp/pos_smoke.json --wait-log /tmp/pos_waits.csv --save /tmp/pos_out.mp4`
Expected: runs to completion, `/tmp/pos_waits.csv` has the 6-column header and rows whose `outcome` is `served`/`abandoned`.

- [ ] **Step 5: Commit**

```bash
git add src/storepose/runner.py
git commit -m "feat: wire POS zone into runner — analyzer, overlay, CSV, busy served-only"
```

---

## Task 8: Documentation

**Files:**
- Modify: `README.md`, `docs/usage.md`

- [ ] **Step 1: README flags table** — add rows after the `--define-zone` row:

```markdown
| `--pos-zone`     | —       | POS-zone JSON; splits line time into waiting vs serving (needs `--zone`). |
| `--define-pos-zone` | —    | Launch the editor for the POS zone and exit.       |
```

- [ ] **Step 2: README "Waiting in line" section** — add a short subsection after the existing waiting description:

```markdown
### Waiting vs at-POS

Add a second **POS zone** (the register area) to split each visit into *waiting*
(in the line zone, not yet at POS) and *serving* (at the POS):

```bash
uv run python main.py --define-pos-zone --source videos/clip.mp4   # draw the POS polygon
uv run python main.py --source videos/clip.mp4 --zone zones/clip.json --pos-zone zones/clip_pos.json --wait-log waits.csv
```

A person flows OUT → WAITING → SERVING → done; their time is attributed to the
state they are in each frame, and time spent out of detection (re-identified
within the re-id window) is credited to the state they held when lost. The wait
log columns become `id, entered_s, exited_s, wait_seconds, serving_seconds,
outcome` where `outcome` is `served` (reached POS) or `abandoned` (left the line
first). The overlay draws the POS zone in azure and tags people `POS n.n s` while
being served; the header shows `in line: N   at POS: M`.
```

- [ ] **Step 3: usage.md** — add the two flags to the queue-zone flag table and a note that the wait-log gains `serving_seconds, outcome` columns when `--pos-zone` is set. Add after the `--wait-log` row in the waiting tuning table:

```markdown
| `--pos-zone PATH` | — | POS zone; splits line time into waiting vs serving (adds `serving_seconds,outcome` CSV columns). |
| `--define-pos-zone` | — | Draw the POS polygon and exit. |
```

- [ ] **Step 4: Commit**

```bash
git add README.md docs/usage.md
git commit -m "docs: document POS zone + waiting/serving split"
```

---

## Task 9: Full verification

- [ ] **Step 1:** Run `uv run pytest -q` → all tests pass.
- [ ] **Step 2:** Two-zone live smoke on the clip (single line, no backslashes), reusing `/tmp/pos_smoke.json` from Task 7:

Run: `uv run python main.py --source /tmp/sco_clip.mp4 --zone "zones/2706969 - Rock Hill, SC _SCO 1 _2026-05-07 12_09_21_370.json" --pos-zone /tmp/pos_smoke.json --busy --wait-log /tmp/pos_waits.csv --save /tmp/pos_out.mp4`
Expected: window shows the azure POS zone + `POS n.n s` tags + `in line / at POS` header; `pos_waits.csv` has `served`/`abandoned` rows with both seconds columns.

- [ ] **Step 3:** Confirm a sampled frame from `/tmp/pos_out.mp4` shows the POS zone and a serving tag.

---

## Self-Review Notes

- **Spec coverage:** state machine + per-state accrual (Task 2) ✓; re-id gap attribution incl. pre-loss-state rule (Task 2 tests) ✓; POS zone membership = `in_line and not in_pos` for waiting / `in_pos` for serving (Task 2) ✓; completion served/abandoned + finalize-past-grace (Task 2) ✓; types extended not renamed (Task 1) ✓; config `--pos-zone`/`--define-pos-zone` (Task 3) ✓; editor + main entrypoint (Task 4) ✓; overlay POS zone + serving tag + header (Task 5) ✓; busy untouched, fed served-only, read_waits extended (Tasks 6, 7) ✓; CSV columns (Task 7) ✓; docs (Task 8) ✓.
- **Type consistency:** `CompletedWait(id, entered_s, exited_s, wait_seconds, serving_seconds="0.0 default", outcome="served default")`; `PersonStatus(id, waiting, candidate, progress, wait_seconds, serving=False, serving_seconds=0.0)`; `QueueResult(statuses, count, serving_count=0, completed=[])`; `QueueAnalyzer(zone, pos_zone=None, enter_frames, exit_seconds, kpt_thr, coverage_thr, foot_band, min_dwell_seconds, reid_grace_seconds)`; `annotate_queue(canvas, people, result, zone, config, pos_zone=None)`. Used consistently across tasks.
- **Backward compatibility:** all new type fields and the `pos_zone` analyzer/overlay params default such that single-zone behavior and the whole `busy/` package are unchanged; existing tests stay valid.
```

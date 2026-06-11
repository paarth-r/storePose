"""Per-person waiting-in-line state machine over tracked people."""

from __future__ import annotations

from dataclasses import dataclass

from ..tracking.types import TrackedPerson
from .types import CompletedWait, PersonStatus, QueueResult
from .zone import Zone

_L_ANKLE, _R_ANKLE = 15, 16  # COCO keypoint indices


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
        """The bottom ``foot_band`` strip of the box — where floor contact is.

        Coverage is measured here, not over the whole box, because a standing
        person's box is mostly torso/head that projects above a floor zone.
        """
        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        band = max(1.0, (y2 - y1) * self.foot_band)
        return (x1, y2 - band, x2, y2)

    def _in_zone(self, person: TrackedPerson, zone: Zone) -> bool:
        """Whether the person is in ``zone``, as an OR of two signals so a held
        position isn't lost when one drops out: the visible-ankle midpoint inside
        the polygon (precise, ignores box padding/carts), OR the coverage of the
        foot region inside the zone meeting ``coverage_thr`` (robust when feet
        leave frame / are occluded)."""
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
        # Without a POS zone there is no abandonment concept: a finished wait is
        # simply "served" (preserves single-zone busy/throughput behavior).
        outcome = "served" if (st.reached_pos or self.pos_zone is None) else "abandoned"
        completed.append(
            CompletedWait(
                pid, st.entered_s, self._clock, st.waiting_seconds,
                st.serving_seconds, outcome,
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

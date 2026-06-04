"""Per-person waiting-in-line state machine over tracked people."""

from __future__ import annotations

from dataclasses import dataclass

from ..tracking.types import TrackedPerson
from .types import CompletedWait, PersonStatus, QueueResult
from .zone import Zone

_L_ANKLE, _R_ANKLE = 15, 16  # COCO keypoint indices


@dataclass
class _WaitState:
    waiting: bool = False
    wait_seconds: float = 0.0
    entered_s: float = 0.0
    in_frames: int = 0
    out_streak: float = 0.0


class QueueAnalyzer:
    """Decides whether each tracked person is waiting in the zone.

    A person is *waiting* once they are in the zone for at least ``enter_frames``
    consecutive frames; before that they are a *candidate* with a 0..1 inclusion
    ``progress``. There is no motion gating — people in line move around (fetching
    items, carts), so presence in the zone is what counts. They stop waiting after
    ``exit_seconds`` of being out of the zone, or when their track disappears.
    ``count`` is the number currently waiting; finished waits are reported in
    :attr:`QueueResult.completed`.
    """

    def __init__(
        self,
        zone: Zone,
        enter_frames: int = 5,
        exit_seconds: float = 2.0,
        kpt_thr: float = 0.5,
        coverage_thr: float = 0.5,
        foot_band: float = 0.3,
    ):
        self.zone = zone
        self.enter_frames = max(1, enter_frames)
        self.exit_seconds = exit_seconds
        self.kpt_thr = kpt_thr
        self.coverage_thr = coverage_thr
        self.foot_band = foot_band
        self._states: dict[int, _WaitState] = {}
        self._clock = 0.0

    def _foot_box(self, box) -> tuple[float, float, float, float]:
        """The bottom ``foot_band`` strip of the box — where floor contact is.

        Coverage is measured here, not over the whole box, because a standing
        person's box is mostly torso/head that projects above a floor zone.
        """
        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        band = max(1.0, (y2 - y1) * self.foot_band)
        return (x1, y2 - band, x2, y2)

    def _in_zone(self, person: TrackedPerson) -> bool:
        """Whether the person is in the zone, as an OR of two signals so a held
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
                if self.zone.contains(mid):
                    return True
        return self.zone.coverage(self._foot_box(person.box)) >= self.coverage_thr

    def update(self, people: list[TrackedPerson], dt: float) -> QueueResult:
        self._clock += dt
        present: set[int] = set()
        statuses: list[PersonStatus] = []
        completed: list[CompletedWait] = []

        for person in people:
            present.add(person.id)
            st = self._states.setdefault(person.id, _WaitState())

            in_cond = self._in_zone(person)

            # A brief loss of in_cond does not reset progress; only a loss
            # sustained for >= exit_seconds resets a candidate or ends a wait.
            if st.waiting:
                st.wait_seconds += dt
                if in_cond:
                    st.out_streak = 0.0
                else:
                    st.out_streak += dt
                    if st.out_streak >= self.exit_seconds:
                        completed.append(
                            CompletedWait(person.id, st.entered_s, self._clock, st.wait_seconds)
                        )
                        st.waiting = False
                        st.wait_seconds = 0.0
                        st.in_frames = 0
                        st.out_streak = 0.0
            else:
                if in_cond:
                    st.out_streak = 0.0
                    st.in_frames += 1
                    if st.in_frames >= self.enter_frames:
                        st.waiting = True
                        st.entered_s = self._clock
                        st.wait_seconds = 0.0
                else:
                    st.out_streak += dt
                    if st.out_streak >= self.exit_seconds:
                        st.in_frames = 0  # candidate truly gone; reset progress

            candidate = not st.waiting and st.in_frames > 0
            progress = 1.0 if st.waiting else min(st.in_frames / self.enter_frames, 1.0)
            statuses.append(
                PersonStatus(person.id, st.waiting, candidate, progress, st.wait_seconds)
            )

        # finalize people whose track vanished this frame
        for pid in list(self._states):
            if pid not in present:
                st = self._states.pop(pid)
                if st.waiting:
                    completed.append(
                        CompletedWait(pid, st.entered_s, self._clock, st.wait_seconds)
                    )

        count = sum(1 for s in statuses if s.waiting)
        return QueueResult(statuses=statuses, count=count, completed=completed)

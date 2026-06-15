"""Per-person waiting-in-line state machine over tracked people."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

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
    checkout: str | None = None   # "mashgin" | "other" — which checkout was reached
    in_frames: int = 0
    in_seconds: float = 0.0
    out_streak: float = 0.0
    pos_frames: int = 0           # consecutive in-POS frames (waiting -> serving debounce)
    absent_seconds: float = 0.0   # time the track has been gone (re-id grace)
    # trailing (t, cx, cy) box centers for the windowed-displacement transit filter
    centers: deque = field(default_factory=deque)


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
        alt_zone: Zone | None = None,
        enter_frames: int = 5,
        exit_seconds: float = 2.0,
        kpt_thr: float = 0.5,
        coverage_thr: float = 0.5,
        foot_band: float = 0.3,
        min_dwell_seconds: float = 0.0,
        reid_grace_seconds: float = 0.0,
        pos_enter_frames: int = 3,
        transit_speed: float = 0.4,
        transit_window: float = 1.0,
    ):
        self.zone = zone
        self.pos_zone = pos_zone
        self.alt_zone = alt_zone
        self.transit_speed = max(0.0, transit_speed)
        self.transit_window = max(1e-6, transit_window)
        self.enter_frames = max(1, enter_frames)
        self.pos_enter_frames = max(1, pos_enter_frames)
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
        # With no checkout zone there is no abandonment concept: a finished wait
        # is simply "served" (preserves single-zone busy/throughput behavior).
        no_checkout = self.pos_zone is None and self.alt_zone is None
        if st.checkout == "mashgin" or no_checkout:
            outcome = "served"
        elif st.checkout == "other":
            outcome = "served_other"
        else:
            outcome = "abandoned"
        completed.append(
            CompletedWait(
                pid, st.entered_s, self._clock, st.waiting_seconds,
                st.serving_seconds, outcome,
            )
        )

    def _status(self, pid: int, st: _VisitState) -> PersonStatus:
        waiting = st.state == "waiting"
        serving_now = st.state == "serving"
        serving = serving_now and st.checkout != "other"        # Mashgin (default)
        serving_other = serving_now and st.checkout == "other"  # non-Mashgin
        candidate = st.state == "out" and st.in_frames > 0
        frame_progress = st.in_frames / self.enter_frames
        if self.min_dwell_seconds > 0:
            frame_progress = min(frame_progress, st.in_seconds / self.min_dwell_seconds)
        progress = 1.0 if (waiting or serving_now) else min(frame_progress, 1.0)
        return PersonStatus(
            pid, waiting, candidate, progress, st.waiting_seconds,
            serving, st.serving_seconds, serving_other,
        )

    def update(self, people: list[TrackedPerson], dt: float) -> QueueResult:
        self._clock += dt
        present: set[int] = set()
        statuses: list[PersonStatus] = []
        completed: list[CompletedWait] = []

        for person in people:
            present.add(person.id)
            st = self._states.setdefault(person.id, _VisitState())
            returned = st.absent_seconds > 0
            st.absent_seconds = 0.0

            # Windowed net displacement: a person walking *through* a zone
            # accumulates large net displacement across it; a shuffler/pacer's
            # back-and-forth nets to ~0. Using net displacement over a trailing
            # window (an integral) instead of a per-frame velocity EMA avoids the
            # ramp lag that let walk-throughs slip through at low frame rates.
            box = person.box
            cx = (float(box[0]) + float(box[2])) / 2.0
            cy = (float(box[1]) + float(box[3])) / 2.0
            box_h = max(1.0, float(box[3]) - float(box[1]))
            if returned:
                st.centers.clear()  # don't read the re-id gap as a displacement spike
            st.centers.append((self._clock, cx, cy))
            while st.centers and self._clock - st.centers[0][0] > self.transit_window:
                st.centers.popleft()
            t0, cx0, cy0 = st.centers[0]
            elapsed = self._clock - t0
            if len(st.centers) >= 2 and elapsed > 0:
                net_disp = ((cx - cx0) ** 2 + (cy - cy0) ** 2) ** 0.5
                speed_norm = (net_disp / box_h) / elapsed
            else:
                speed_norm = 0.0
            transiting = self.transit_speed > 0.0 and speed_norm > self.transit_speed

            in_line = self._in_zone(person, self.zone) and not transiting
            in_pos = (self.pos_zone is not None
                      and self._in_zone(person, self.pos_zone) and not transiting)
            in_alt = (self.alt_zone is not None
                      and self._in_zone(person, self.alt_zone) and not transiting)
            if in_pos or in_alt:
                in_line = False  # a checkout beats the line (mutually exclusive)
            which = "mashgin" if in_pos else ("other" if in_alt else None)
            in_check = which is not None
            dbg = {"speed": round(speed_norm, 3), "transit": transiting,
                   "line": in_line, "pos": in_pos, "reg": in_alt}

            if st.state == "serving":
                st.serving_seconds += dt
                if in_check:
                    st.out_streak = 0.0
                    if which == st.checkout:
                        st.pos_frames = 0
                    else:
                        # moved toward the *other* checkout: once it's sustained,
                        # close out this visit and start a fresh timer there (the
                        # serving timer must not persist across checkouts).
                        st.pos_frames += 1
                        if st.pos_frames >= self.pos_enter_frames:
                            self._finalize(person.id, st, completed)
                            st.serving_seconds = 0.0
                            st.waiting_seconds = 0.0
                            st.entered_s = self._clock
                            st.checkout = which
                            st.pos_frames = 0
                else:
                    st.out_streak += dt
                    if st.out_streak >= self.exit_seconds:
                        self._finalize(person.id, st, completed)
                        self._states.pop(person.id)
                        statuses.append(PersonStatus(person.id, False, False, 0.0, 0.0, debug=dbg))
                        continue
            elif st.state == "waiting":
                st.waiting_seconds += dt
                if in_check:
                    st.out_streak = 0.0
                    st.pos_frames += 1
                    if st.pos_frames >= self.pos_enter_frames:
                        st.state = "serving"
                        st.checkout = which
                        st.pos_frames = 0
                elif in_line:
                    st.out_streak = 0.0
                    st.pos_frames = 0
                else:
                    st.out_streak += dt
                    if st.out_streak >= self.exit_seconds:
                        self._finalize(person.id, st, completed)  # abandoned
                        self._states.pop(person.id)
                        statuses.append(PersonStatus(person.id, False, False, 0.0, 0.0, debug=dbg))
                        continue
            else:  # out
                if in_line or in_check:
                    st.out_streak = 0.0
                    st.in_frames += 1
                    st.in_seconds += dt
                    if (
                        st.in_frames >= self.enter_frames
                        and st.in_seconds >= self.min_dwell_seconds
                    ):
                        st.entered_s = self._clock
                        if in_check:
                            st.state = "serving"
                            st.checkout = which
                        else:
                            st.state = "waiting"
                else:
                    st.out_streak += dt
                    if st.out_streak >= self.exit_seconds:
                        st.in_frames = 0
                        st.in_seconds = 0.0

            status = self._status(person.id, st)
            if in_check and status.waiting:
                # box is in a checkout: leave the line count instantly (the
                # serving timer still rides out the pos_enter_frames debounce, so
                # they are neither "in line" nor yet "at a checkout" meanwhile).
                status = PersonStatus(person.id, False, False, status.progress,
                                      status.wait_seconds, False, st.serving_seconds, False)
            status.debug = dbg
            statuses.append(status)

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
        serving_other_count = sum(1 for s in statuses if s.serving_other)
        return QueueResult(
            statuses=statuses, count=count, serving_count=serving_count,
            serving_other_count=serving_other_count, completed=completed,
        )

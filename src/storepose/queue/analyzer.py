"""Per-person waiting-in-line state machine over tracked people."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..tracking.types import TrackedPerson
from .types import CompletedWait, PersonStatus, QueueResult
from .zone import Zone

_SPEED_EMA_ALPHA = 0.5


@dataclass
class _WaitState:
    waiting: bool = False
    wait_seconds: float = 0.0
    entered_s: float = 0.0
    in_streak: float = 0.0
    out_streak: float = 0.0
    speed_ema: float = 0.0
    prev_foot: tuple[float, float] | None = field(default=None)


def _foot(box) -> tuple[float, float]:
    x1, _, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    return ((x1 + x2) / 2.0, y2)


class QueueAnalyzer:
    """Decides whether each tracked person is waiting in the zone.

    A person is *waiting* once their foot point stays inside the zone while
    moving slowly (in body-heights/sec) for at least ``enter_seconds``; they
    stop waiting after ``exit_seconds`` of failing that condition, or when their
    track disappears. ``count`` is the number currently waiting; finished waits
    are reported in :attr:`QueueResult.completed`.
    """

    def __init__(
        self,
        zone: Zone,
        wait_speed: float = 0.15,
        enter_seconds: float = 1.5,
        exit_seconds: float = 2.0,
    ):
        self.zone = zone
        self.wait_speed = wait_speed
        self.enter_seconds = enter_seconds
        self.exit_seconds = exit_seconds
        self._states: dict[int, _WaitState] = {}
        self._clock = 0.0

    def update(self, people: list[TrackedPerson], dt: float) -> QueueResult:
        self._clock += dt
        present: set[int] = set()
        statuses: list[PersonStatus] = []
        completed: list[CompletedWait] = []

        for person in people:
            present.add(person.id)
            st = self._states.setdefault(person.id, _WaitState())

            foot = _foot(person.box)
            box_h = max(float(person.box[3]) - float(person.box[1]), 1.0)
            if st.prev_foot is None:
                inst_speed = 0.0
            else:
                dist = math.hypot(foot[0] - st.prev_foot[0], foot[1] - st.prev_foot[1])
                inst_speed = dist / max(dt, 1e-6) / box_h
            st.prev_foot = foot
            st.speed_ema = _SPEED_EMA_ALPHA * inst_speed + (1 - _SPEED_EMA_ALPHA) * st.speed_ema

            in_cond = self.zone.contains(foot) and st.speed_ema < self.wait_speed

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
                        st.in_streak = 0.0
            else:
                if in_cond:
                    st.in_streak += dt
                    if st.in_streak >= self.enter_seconds:
                        st.waiting = True
                        st.entered_s = self._clock - st.in_streak
                        st.wait_seconds = st.in_streak
                        st.out_streak = 0.0
                else:
                    st.in_streak = 0.0

            statuses.append(PersonStatus(person.id, st.waiting, st.wait_seconds))

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

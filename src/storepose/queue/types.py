"""Result types for queue / waiting analysis."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PersonStatus:
    """Waiting state for one tracked person this frame.

    Attributes:
        id: Track id.
        waiting: True if the person is confirmed in line.
        candidate: True if accumulating frames toward inclusion (not yet in line).
        progress: Inclusion progress in ``[0, 1]`` (1.0 once waiting).
        wait_seconds: Accumulated waiting time so far (0 if not waiting).
    """

    id: int
    waiting: bool
    candidate: bool
    progress: float
    wait_seconds: float


@dataclass
class CompletedWait:
    """A finished wait, emitted the frame a person stops waiting."""

    id: int
    entered_s: float
    exited_s: float
    wait_seconds: float


@dataclass
class QueueResult:
    """Per-frame queue analysis output."""

    statuses: list[PersonStatus]
    count: int
    completed: list[CompletedWait] = field(default_factory=list)

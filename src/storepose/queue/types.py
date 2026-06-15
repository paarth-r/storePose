"""Result types for queue / waiting analysis."""

from __future__ import annotations

from dataclasses import dataclass, field


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
    serving_other: bool = False  # serving at the non-Mashgin checkout
    debug: dict | None = None    # per-person reasoning for the debug view


@dataclass
class CompletedWait:
    """A finished line visit, emitted the frame a person's visit ends.

    ``wait_seconds`` is the waiting portion; ``serving_seconds`` the at-POS
    portion; ``outcome`` is ``"served"`` (reached POS) or ``"abandoned"``.
    ``rejected`` marks a too-short outlier (see ``OutlierFilter``): kept for the
    wait log but excluded from busy/dashboard aggregation.
    """

    id: int
    entered_s: float
    exited_s: float
    wait_seconds: float
    serving_seconds: float = 0.0
    outcome: str = "served"
    rejected: bool = False


@dataclass
class QueueResult:
    """Per-frame queue analysis output."""

    statuses: list[PersonStatus]
    count: int
    serving_count: int = 0
    serving_other_count: int = 0
    completed: list[CompletedWait] = field(default_factory=list)

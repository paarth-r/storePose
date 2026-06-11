"""Helpers for building a ground-truth label set by hand.

The pure logic here — window enumeration and resume — is unit-tested; the actual
video playback lives in the ``label`` CLI command, which is a thin cv2 shell over
these functions. Labels are stored in the same ``window_index,level`` CSV the
evaluator reads, so a hand-labeled file drops straight into ``busy_report eval``.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..busy.types import BusyLevel


@dataclass(frozen=True)
class Window:
    index: int
    start_s: float
    end_s: float


def enumerate_windows(duration_s: float, window_s: float) -> list[Window]:
    """Tile ``[0, duration_s)`` into windows of ``window_s``.

    The final window is included even if partial (a short clip still yields one
    window), so every second of footage is labelable.
    """
    if duration_s <= 0:
        return []
    if window_s <= 0:
        raise ValueError(f"window_s must be > 0, got {window_s}")
    out: list[Window] = []
    i = 0
    while i * window_s < duration_s:
        start = i * window_s
        out.append(Window(i, start, min(start + window_s, duration_s)))
        i += 1
    return out


def unlabeled(
    windows: list[Window], existing: dict[int, BusyLevel]
) -> list[Window]:
    """Windows not yet present in ``existing`` — drives resume."""
    return [w for w in windows if w.index not in existing]


# Keys the labeling UI accepts, mapped to levels. Digits and initials both work.
KEY_TO_LEVEL = {
    "1": BusyLevel.LOW,
    "2": BusyLevel.MEDIUM,
    "3": BusyLevel.HIGH,
    "l": BusyLevel.LOW,
    "m": BusyLevel.MEDIUM,
    "h": BusyLevel.HIGH,
}


def level_for_key(key: str) -> BusyLevel | None:
    """Map a pressed key (case-insensitive) to a level, or ``None`` if not a
    label key."""
    return KEY_TO_LEVEL.get(key.lower())

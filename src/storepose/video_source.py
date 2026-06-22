"""Frame source (webcam or video file) with guaranteed cleanup."""

from __future__ import annotations

from types import TracebackType
from typing import Iterator

import cv2
import numpy as np


class SourceOpenError(RuntimeError):
    """Raised when the requested camera or video file cannot be opened."""


# Backwards-compatible alias.
CameraOpenError = SourceOpenError


class VideoSource:
    """Context manager yielding BGR frames from a webcam or a video file.

    Use as::

        with VideoSource(0) as source:           # webcam index
            for frame in source:
                ...

        with VideoSource("clip.mp4") as source:  # video file
            ...

    The capture device is always released on exit. Dropped/empty frames end
    iteration (a disconnected camera or the end of a video file).
    """

    def __init__(self, source: int | str = 0):
        self._source = source
        self._cap: cv2.VideoCapture | None = None

    def __enter__(self) -> "VideoSource":
        cap = cv2.VideoCapture(self._source)
        if not cap.isOpened():
            cap.release()
            kind = "camera" if isinstance(self._source, int) else "video file"
            raise SourceOpenError(
                f"Could not open {kind} source {self._source!r}. "
                "Check the path/index and camera permissions."
            )
        self._cap = cap
        return self

    @property
    def fps(self) -> float | None:
        """Source frame rate if the backend reports a sane value, else None."""
        if self._cap is None:
            return None
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        return fps if fps and fps > 0 else None

    @property
    def frame_count(self) -> int | None:
        """Total frames if the backend reports a positive count (files), else None.

        Live cameras and some containers report 0/negative; callers treat None as
        "unknown" and skip progress percentages.
        """
        if self._cap is None:
            return None
        n = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return n if n > 0 else None

    def __iter__(self) -> Iterator[np.ndarray]:
        assert self._cap is not None, "VideoSource must be used as a context manager"
        while True:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                break
            yield frame

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

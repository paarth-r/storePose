"""Webcam frame source with guaranteed cleanup."""

from __future__ import annotations

from types import TracebackType
from typing import Iterator

import cv2
import numpy as np


class CameraOpenError(RuntimeError):
    """Raised when the requested camera cannot be opened."""


class VideoSource:
    """Context manager yielding BGR frames from a webcam.

    Use as::

        with VideoSource(0) as source:
            for frame in source:
                ...

    The capture device is always released on exit. Dropped/empty frames end
    iteration (typical for a disconnected camera or end of a file source).
    """

    def __init__(self, source: int = 0):
        self._source = source
        self._cap: cv2.VideoCapture | None = None

    def __enter__(self) -> "VideoSource":
        cap = cv2.VideoCapture(self._source)
        if not cap.isOpened():
            cap.release()
            raise CameraOpenError(
                f"Could not open camera source {self._source!r}. "
                "Check the index and camera permissions."
            )
        self._cap = cap
        return self

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

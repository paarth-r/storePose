"""Annotated-frame writer (mp4) with guaranteed cleanup."""

from __future__ import annotations

from pathlib import Path
from types import TracebackType

import cv2
import numpy as np

DEFAULT_FPS = 30.0


class SinkOpenError(RuntimeError):
    """Raised when the output video writer cannot be created."""


class VideoSink:
    """Context manager that writes annotated BGR frames to an .mp4 file.

    The writer is created lazily on the first :meth:`write`, taking its frame
    size from that frame (so it always matches the annotated output). ``fps``
    falls back to ``DEFAULT_FPS`` when the source rate is unknown (webcams).

    Use as::

        with VideoSink("out.mp4", fps=30) as sink:
            sink.write(annotated_frame)
    """

    def __init__(self, path: str, fps: float | None = None):
        self._path = path
        self._fps = fps if fps and fps > 0 else DEFAULT_FPS
        self._writer: cv2.VideoWriter | None = None

    def __enter__(self) -> "VideoSink":
        parent = Path(self._path).parent
        if parent and not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)
        return self

    def write(self, frame: np.ndarray) -> None:
        """Append one annotated frame to the output video."""
        if self._writer is None:
            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(self._path, fourcc, self._fps, (width, height))
            if not writer.isOpened():
                raise SinkOpenError(f"Could not open video writer for {self._path!r}.")
            self._writer = writer
        self._writer.write(frame)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

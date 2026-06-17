"""Annotated-frame writer (mp4) with guaranteed cleanup."""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from types import TracebackType

import cv2
import numpy as np

DEFAULT_FPS = 30.0

DEFAULT_RUNS_DIR = "runs"


def run_output_path(source: int | str, runs_dir: str | Path = DEFAULT_RUNS_DIR) -> str:
    """Build an auto-named ``runs/<source>_<timestamp>_<id>.mp4`` path.

    Used by ``--save-mp4`` so a run records itself without the caller picking a
    name. ``source`` is the run's input — a video path (its stem is used) or a
    webcam index (named ``webcamN``). A second-granularity timestamp plus a short
    uuid suffix keeps back-to-back runs from colliding.
    """
    if isinstance(source, int):
        stem = f"webcam{source}"
    else:
        stem = Path(source).stem or "video"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = uuid.uuid4().hex[:6]
    return str(Path(runs_dir) / f"{stem}_{stamp}_{suffix}.mp4")

# H.264 in an .mp4 container. Plays natively in QuickTime/Preview/browsers;
# the older 'mp4v' (MPEG-4 Part 2) decodes as green static in many players.
FOURCC = "avc1"


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
            fourcc = cv2.VideoWriter_fourcc(*FOURCC)
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

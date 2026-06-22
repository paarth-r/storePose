"""Lazy SSE hub streaming the annotated browser feed (frame + overlay together).

The runner calls :meth:`StreamHub.publish` each frame; the HTTP server iterates
:meth:`StreamHub.events` per connected browser. Encoding is **lazy**: a frame is
only JPEG-encoded while at least one browser is subscribed (``active``), so the
stream costs nothing when nobody is watching.

Each Server-Sent Event carries the base64 JPEG and its overlay data in one JSON
object, so the client-side SVG overlay is always aligned to the frame it drew on.
"""
from __future__ import annotations

import base64
import json
import threading
from contextlib import contextmanager

import cv2

_JPEG_PREFIX = "data:image/jpeg;base64,"


def format_sse(event: dict) -> bytes:
    """Encode one event dict as an SSE ``data:`` record."""
    return b"data: " + json.dumps(event, separators=(",", ":")).encode("utf-8") + b"\n\n"


class StreamHub:
    """Thread-safe holder of the latest encoded frame + overlay for the SSE feed."""

    def __init__(self, max_width: int = 960, quality: int = 72):
        self._max_width = max_width
        self._quality = int(quality)
        self._cond = threading.Condition()
        self._latest: dict | None = None
        self._seq = 0
        self._subscribers = 0
        self._closed = False

    @property
    def active(self) -> bool:
        """True while at least one browser is subscribed."""
        with self._cond:
            return self._subscribers > 0

    def latest(self) -> dict | None:
        with self._cond:
            return self._latest

    def _encode(self, frame_bgr) -> str:
        h, w = frame_bgr.shape[:2]
        if w > self._max_width:
            scale = self._max_width / w
            frame_bgr = cv2.resize(frame_bgr, (self._max_width, max(1, round(h * scale))),
                                   interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame_bgr,
                               [cv2.IMWRITE_JPEG_QUALITY, self._quality])
        if not ok:
            return ""
        return _JPEG_PREFIX + base64.b64encode(buf.tobytes()).decode("ascii")

    def publish(self, frame_bgr, overlay: dict) -> None:
        """Encode ``frame_bgr`` + merge ``overlay`` into the latest event.

        No-op when inactive (lazy): the runner gates on :attr:`active`, and this
        guards against encoding when a race drops the last subscriber.
        """
        if not self.active:
            return
        jpeg = self._encode(frame_bgr)
        with self._cond:
            self._seq += 1
            self._latest = {**overlay, "seq": self._seq, "jpeg": jpeg}
            self._cond.notify_all()

    def close(self) -> None:
        """Wake every waiting subscriber so their generators can exit."""
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    @contextmanager
    def subscribe(self):
        with self._cond:
            self._subscribers += 1
        try:
            yield
        finally:
            with self._cond:
                self._subscribers -= 1

    def events(self, poll: float = 1.0):
        """Yield SSE records for one browser; blocks until a newer frame exists.

        Sends the current frame immediately on connect (if any), then each new
        frame as it is published. Exits when :meth:`close` is called.
        """
        with self.subscribe():
            last_seq = 0
            while True:
                with self._cond:
                    while not self._closed and (
                        self._latest is None or self._latest["seq"] == last_seq
                    ):
                        self._cond.wait(timeout=poll)
                    if self._closed:
                        return
                    event = self._latest
                last_seq = event["seq"]
                yield format_sse(event)

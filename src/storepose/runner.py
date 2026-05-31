"""Realtime loop: capture → process → annotate → display."""

from __future__ import annotations

import cv2

from .config import AppConfig
from .drawing import annotate
from .fps import FpsMeter
from .pipeline import PosePipeline
from .video_source import VideoSource

WINDOW_NAME = "storePose"
_QUIT_KEYS = {ord("q"), 27}  # 'q' or Esc


class Runner:
    """Owns the realtime display loop and its resources."""

    def __init__(self, config: AppConfig):
        self._config = config

    def run(self) -> None:
        """Open the camera and stream annotated frames until the user quits."""
        config = self._config
        print(f"Loading models (mode={config.mode}, device={config.device})...")
        pipeline = PosePipeline(config)
        meter = FpsMeter()
        print("Models ready. Press 'q' or Esc to quit.")

        try:
            with VideoSource(config.source) as source:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                for frame in source:
                    result = pipeline.process(frame)
                    fps = meter.tick()
                    canvas = annotate(frame, result, config, fps)
                    cv2.imshow(WINDOW_NAME, canvas)
                    if cv2.waitKey(1) & 0xFF in _QUIT_KEYS:
                        break
        except KeyboardInterrupt:
            pass
        finally:
            cv2.destroyAllWindows()

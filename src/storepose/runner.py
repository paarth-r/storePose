"""Realtime loop: capture → process → annotate → display (+ optional save)."""

from __future__ import annotations

from contextlib import ExitStack

import cv2

from .config import AppConfig
from .drawing import annotate
from .fps import FpsMeter
from .pipeline import PosePipeline
from .video_sink import VideoSink
from .video_source import VideoSource

WINDOW_NAME = "storePose"
_QUIT_KEYS = {ord("q"), 27}  # 'q' or Esc


class Runner:
    """Owns the realtime display loop and its resources."""

    def __init__(self, config: AppConfig):
        self._config = config

    def run(self) -> None:
        """Stream annotated frames to a window until the source ends or quit."""
        config = self._config
        print(f"Loading models (mode={config.mode}, device={config.device})...")
        pipeline = PosePipeline(config)
        meter = FpsMeter()
        print(f"Models ready. Source: {config.source}. Press 'q' or Esc to quit.")

        try:
            with ExitStack() as stack:
                source = stack.enter_context(VideoSource(config.source))
                sink = None
                if config.save:
                    sink = stack.enter_context(VideoSink(config.save, fps=source.fps))
                    print(f"Saving annotated video to {config.save}")

                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                for frame in source:
                    result = pipeline.process(frame)
                    fps = meter.tick()
                    canvas = annotate(frame, result, config, fps)
                    if sink is not None:
                        sink.write(canvas)
                    cv2.imshow(WINDOW_NAME, canvas)
                    if cv2.waitKey(1) & 0xFF in _QUIT_KEYS:
                        break
        except KeyboardInterrupt:
            pass
        finally:
            cv2.destroyAllWindows()

        if config.save:
            print(f"Done. Wrote {config.save}")

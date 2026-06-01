"""Realtime loop: capture → process → (track) → annotate → display (+ save)."""

from __future__ import annotations

import time
from contextlib import ExitStack

import cv2

from .config import AppConfig
from .drawing import annotate, annotate_tracked
from .fps import FpsMeter
from .pipeline import PosePipeline
from .tracking.tracker import MultiObjectTracker
from .video_sink import VideoSink
from .video_source import VideoSource

WINDOW_NAME = "storePose"
_QUIT_KEYS = {ord("q"), 27}  # 'q' or Esc
_DEFAULT_FPS = 30.0


class Runner:
    """Owns the realtime display loop and its resources."""

    def __init__(self, config: AppConfig):
        self._config = config

    def run(self) -> None:
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

                tracker = None
                if config.track:
                    base_fps = source.fps or _DEFAULT_FPS
                    max_age = max(1, round(config.hold_seconds * base_fps))
                    tracker = MultiObjectTracker(
                        max_age=max_age, min_hits=config.min_hits,
                        iou_thr=config.iou_thr, smooth=config.smooth,
                        min_cutoff=config.smooth_cutoff, beta=config.smooth_beta,
                    )

                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                prev = None
                for frame in source:
                    now = time.perf_counter()
                    dt = (now - prev) if prev else (1.0 / (source.fps or _DEFAULT_FPS))
                    prev = now

                    result = pipeline.process(frame)
                    fps = meter.tick()
                    if tracker is not None:
                        people = tracker.update(result, dt)
                        canvas = annotate_tracked(frame, people, config, fps)
                    else:
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

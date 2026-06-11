"""Realtime loop: capture → process → track → annotate → display (+ save)."""

from __future__ import annotations

import csv
import time
import webbrowser
from contextlib import ExitStack

import cv2

from .busy.aggregator import BusyAggregator
from .busy.report import write_busy
from .busy.types import BusyThresholds
from .config import AppConfig
from .dashboard.server import DashboardServer
from .dashboard.state import DashboardState
from .drawing import annotate, annotate_busy, annotate_queue, annotate_tracked
from .fps import FpsMeter
from .pipeline import PosePipeline
from .queue.analyzer import QueueAnalyzer
from .queue.zone import Zone
from .tracking.appearance import HsvHistogramAppearance
from .tracking.tracker import MultiObjectTracker
from .video_sink import VideoSink
from .video_source import VideoSource

WINDOW_NAME = "storePose"
_QUIT_KEYS = {ord("q"), 27}  # 'q' or Esc
_DEFAULT_FPS = 30.0
_BUSY_REFRESH_SECONDS = 10.0  # how often the Low/Med/High busy label is recomputed


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
                    appearance = (
                        HsvHistogramAppearance(kpt_thr=config.kpt_thr)
                        if config.reid else None
                    )
                    reid_max_age = max(1, round(config.reid_seconds * base_fps))
                    tracker = MultiObjectTracker(
                        max_age=max_age, min_hits=config.min_hits,
                        iou_thr=config.iou_thr, max_overlap=config.max_overlap,
                        smooth=config.smooth,
                        min_cutoff=config.smooth_cutoff, beta=config.smooth_beta,
                        appearance=appearance, reid=config.reid,
                        reid_max_age=reid_max_age, reid_thr=config.reid_thr,
                    )

                zone, analyzer, pos_zone, alt_zone = None, None, None, None
                if config.zone:
                    if tracker is None:
                        print("Note: --zone needs tracking; ignoring (you passed --no-track).")
                    else:
                        zone = Zone.load(config.zone)
                        pos_zone = Zone.load(config.pos_zone) if config.pos_zone else None
                        alt_zone = Zone.load(config.alt_zone) if config.alt_zone else None
                        analyzer = QueueAnalyzer(
                            zone,
                            pos_zone=pos_zone,
                            alt_zone=alt_zone,
                            enter_frames=config.wait_enter_frames,
                            exit_seconds=config.wait_exit_seconds,
                            kpt_thr=config.kpt_thr,
                            coverage_thr=config.zone_coverage,
                            foot_band=config.zone_foot_band,
                            min_dwell_seconds=config.wait_min_dwell,
                            # carry a vanished person's gap into their held state
                            # for the re-id window; a re-identified id resumes it
                            reid_grace_seconds=config.reid_seconds if config.reid else 0.0,
                            pos_enter_frames=config.pos_enter_frames,
                            transit_speed=config.transit_speed,
                        )
                elif config.pos_zone or config.alt_zone:
                    print("Note: --pos-zone/--alt-zone need --zone (the line zone); ignoring.")

                busy = None
                if config.busy and analyzer is not None:
                    busy = BusyAggregator(
                        BusyThresholds(
                            metric=config.busy_metric,
                            low_max=config.busy_low_max,
                            medium_max=config.busy_medium_max,
                            hysteresis=config.busy_hysteresis,
                        ),
                        window_seconds=config.busy_window,
                        sub_window_seconds=config.busy_subwindow or None,
                    )
                elif config.busy:
                    print("Note: --busy needs an active --zone; ignoring.")
                clock = 0.0

                wait_writer = None
                if config.wait_log and analyzer is not None:
                    log_file = stack.enter_context(open(config.wait_log, "w", newline=""))
                    wait_writer = csv.writer(log_file)
                    wait_writer.writerow(
                        ["id", "entered_s", "exited_s", "wait_seconds",
                         "serving_seconds", "outcome"]
                    )
                    print(f"Logging completed waits to {config.wait_log}")

                dash_state = None
                if config.dashboard:
                    dash_state = DashboardState()
                    dash_server = DashboardServer(dash_state, port=config.dashboard_port)
                    dash_server.start()
                    stack.callback(dash_server.stop)
                    print(f"Dashboard: {dash_server.url}")
                    try:
                        webbrowser.open(dash_server.url)
                    except Exception:
                        pass

                is_file = isinstance(config.source, str)
                base_dt = 1.0 / (source.fps or _DEFAULT_FPS)
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                prev = None
                next_busy = 0.0          # next clock time to refresh the busy label
                busy_label, busy_value = "—", 0.0
                for frame in source:
                    if is_file:
                        dt = base_dt                       # file: video time
                    else:
                        now = time.perf_counter()          # webcam: real time
                        dt = (now - prev) if prev else base_dt
                        prev = now
                    clock += dt

                    result = pipeline.process(frame)
                    fps = meter.tick()
                    if tracker is not None:
                        people = tracker.update(result, dt, frame)
                        canvas = annotate_tracked(frame, people, config, fps)
                        if analyzer is not None:
                            qresult = analyzer.update(people, dt)
                            canvas = annotate_queue(canvas, people, qresult, zone, config,
                                                    pos_zone=pos_zone, alt_zone=alt_zone)
                            if wait_writer is not None:
                                for c in qresult.completed:
                                    wait_writer.writerow(
                                        [c.id, f"{c.entered_s:.2f}", f"{c.exited_s:.2f}",
                                         f"{c.wait_seconds:.2f}", f"{c.serving_seconds:.2f}",
                                         c.outcome]
                                    )
                            if dash_state is not None:
                                dash_state.observe(clock, qresult.count, qresult.serving_count)
                                for c in qresult.completed:
                                    dash_state.add_visit(clock, c.wait_seconds,
                                                         c.serving_seconds, c.outcome)
                            if busy is not None:
                                busy.observe(clock, qresult.count, dt)
                                for c in qresult.completed:
                                    if c.outcome in ("served", "served_other"):
                                        busy.add_wait(c)
                                if clock >= next_busy:  # refresh the label every 10 s
                                    level, busy_value = busy.estimate(clock)
                                    busy_label = level.label
                                    next_busy = clock + _BUSY_REFRESH_SECONDS
                                    if dash_state is not None:
                                        dash_state.set_busy(clock, busy_label, busy_value)
                                canvas = annotate_busy(canvas, busy_label, busy_value,
                                                       next_busy - clock)
                    else:
                        canvas = annotate(frame, result, config, fps)

                    if dash_state is not None and analyzer is None:
                        dash_state.observe(clock, 0, 0)  # keep the dashboard ticking

                    if sink is not None:
                        sink.write(canvas)
                    cv2.imshow(WINDOW_NAME, canvas)
                    if cv2.waitKey(1) & 0xFF in _QUIT_KEYS:
                        break
                if config.busy_log and busy is not None:
                    windows = busy.windows()
                    write_busy(config.busy_log, windows)
                    print(f"Wrote busy report ({len(windows)} window(s)) to {config.busy_log}")
        except KeyboardInterrupt:
            pass
        finally:
            cv2.destroyAllWindows()

        if config.save:
            print(f"Done. Wrote {config.save}")

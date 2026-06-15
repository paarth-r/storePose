"""Realtime loop: capture → process → track → annotate → display (+ save)."""

from __future__ import annotations

import csv
import time
import webbrowser
from collections import deque
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

_DEBUG_BUFFER = 300                              # rolling frames kept for scrub-back
# arrow keycodes vary by platform (mac / linux / windows); accept all three
_RIGHT_KEYS = {63235, 65363, 2555904}
_LEFT_KEYS = {63234, 65361, 2424832}


def build_tracker(config: AppConfig, base_fps: float) -> "MultiObjectTracker":
    """Construct the multi-object tracker from config (shared by run + calibrate)."""
    max_age = max(1, round(config.hold_seconds * base_fps))
    appearance = (
        HsvHistogramAppearance(kpt_thr=config.kpt_thr) if config.reid else None
    )
    reid_max_age = max(1, round(config.reid_seconds * base_fps))
    return MultiObjectTracker(
        max_age=max_age, min_hits=config.min_hits,
        iou_thr=config.iou_thr, max_overlap=config.max_overlap,
        smooth=config.smooth,
        min_cutoff=config.smooth_cutoff, beta=config.smooth_beta,
        appearance=appearance, reid=config.reid,
        reid_max_age=reid_max_age, reid_thr=config.reid_thr,
    )


def build_analyzer(config: AppConfig):
    """Load zones and construct the queue analyzer (shared by run + calibrate).

    Assumes ``config.zone`` is set. Returns ``(zone, analyzer, pos_zone, alt_zone)``.
    """
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
        # carry a vanished person's gap into their held state for the re-id
        # window; a re-identified id resumes it
        reid_grace_seconds=config.reid_seconds if config.reid else 0.0,
        pos_enter_frames=config.pos_enter_frames,
        transit_speed=config.transit_speed,
    )
    return zone, analyzer, pos_zone, alt_zone


def _person_state(s) -> str:
    """One-word classification of a PersonStatus for the debug table."""
    if s.serving:
        return "serving-Mashgin"
    if s.serving_other:
        return "serving-REG"
    if s.waiting:
        return "waiting"
    if s.candidate:
        return f"candidate {int(round(s.progress * 100))}%"
    return "out"


def _debug_rows(statuses) -> list[dict]:
    """Per-person reasoning rows for the dashboard Debug tab."""
    rows = []
    for s in statuses:
        d = s.debug or {}
        rows.append({
            "id": s.id,
            "state": _person_state(s),
            "wait": round(s.wait_seconds, 2),
            "serve": round(s.serving_seconds, 2),
            "speed": d.get("speed", 0.0),
            "line": bool(d.get("line", False)),
            "pos": bool(d.get("pos", False)),
            "reg": bool(d.get("reg", False)),
            "transit": bool(d.get("transit", False)),
        })
    return rows


def _scrub(view: int, delta: int, length: int) -> int:
    """Clamp a frames-back view index into ``[0, length-1]`` (0 = newest)."""
    if length <= 0:
        return 0
    return max(0, min(view + delta, length - 1))


def _draw_debug_banner(img, text: str) -> None:
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 30), (24, 14, 8), -1)
    cv2.putText(img, text, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)


class Runner:
    """Owns the realtime display loop and its resources."""

    def __init__(self, config: AppConfig):
        self._config = config

    def _scrub_loop(self, buffer, scrub, dash_state) -> str:
        """Display the viewed buffer frame and read keys until the user advances
        to a new source frame ("advance") or quits ("quit").

        ``scrub`` holds ``view`` (frames back from newest; 0 = latest) and
        ``playing``. ``←`` reviews older frames, ``→``/space steps toward the
        newest and then advances the source; ``c``/``p`` play/pause; while playing
        a key-timeout auto-advances. The Debug tab follows the viewed frame.
        """
        while True:
            scrub["view"] = _scrub(scrub["view"], 0, len(buffer))
            b_idx, jpeg, b_rows = buffer[-1 - scrub["view"]]
            img = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
            state = "PLAY" if scrub["playing"] else "PAUSED"
            _draw_debug_banner(
                img, f"DEBUG  frame {b_idx}  {state}  [{scrub['view']} back]  "
                     f"<-/-> step  c play  p pause  q quit")
            if dash_state is not None:
                dash_state.set_debug(b_idx, b_rows)
            cv2.imshow(WINDOW_NAME, img)

            key = cv2.waitKeyEx(30 if scrub["playing"] else 0)
            low = key & 0xFF
            if low in _QUIT_KEYS:
                return "quit"
            if low == ord("c"):
                scrub["playing"] = True
                continue
            if low == ord("p"):
                scrub["playing"] = False
                continue
            if key in _LEFT_KEYS or low == ord("a"):
                scrub["view"] = _scrub(scrub["view"], +1, len(buffer))
                continue
            advance = key in _RIGHT_KEYS or low in (ord(" "), ord("d"))
            timeout = scrub["playing"] and key == -1
            if advance or timeout:
                if scrub["view"] > 0:
                    scrub["view"] = _scrub(scrub["view"], -1, len(buffer))
                    continue
                return "advance"
            # any other key while paused: just redraw

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
                    tracker = build_tracker(config, base_fps)

                zone, analyzer, pos_zone, alt_zone = None, None, None, None
                if config.zone:
                    if tracker is None:
                        print("Note: --zone needs tracking; ignoring (you passed --no-track).")
                    else:
                        zone, analyzer, pos_zone, alt_zone = build_analyzer(config)
                elif config.pos_zone or config.alt_zone:
                    print("Note: --pos-zone/--alt-zone need --zone (the line zone); ignoring.")

                busy = None
                if config.busy and analyzer is not None:
                    if config.calib:
                        from .busy.calibrate import (
                            DEFAULT_BUSY_STRATEGY,
                            load_calib,
                            thresholds_from_calib,
                        )
                        calib = load_calib(config.calib)
                        # explicit flag wins; else the calib's auto-selected default
                        strategy = (config.busy_strategy
                                    or calib.get("default_strategy")
                                    or DEFAULT_BUSY_STRATEGY)
                        chosen = " (auto)" if config.busy_strategy is None else " (--busy-strategy)"
                        thresholds = thresholds_from_calib(
                            calib, strategy, hysteresis=config.busy_hysteresis,
                        )
                        sub = calib.get("subwindow_seconds") or config.busy_subwindow or None
                        print(f"Busy bands from {config.calib} [{strategy}{chosen}]: "
                              f"not-busy <= {thresholds.low_max:g}, medium <= "
                              f"{thresholds.medium_max:g} ({thresholds.metric}); "
                              f"manual --busy-*-max ignored.")
                    else:
                        thresholds = BusyThresholds(
                            metric=config.busy_metric,
                            low_max=config.busy_low_max,
                            medium_max=config.busy_medium_max,
                            hysteresis=config.busy_hysteresis,
                        )
                        sub = config.busy_subwindow or None
                    busy = BusyAggregator(
                        thresholds,
                        window_seconds=config.busy_window,
                        sub_window_seconds=sub,
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
                frame_idx = -1
                buffer: deque = deque(maxlen=_DEBUG_BUFFER)  # debug scrub-back ring
                scrub = {"view": 0, "playing": False}        # debug loop state
                for frame in source:
                    frame_idx += 1
                    rows: list[dict] = []
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
                            rows = _debug_rows(qresult.statuses)
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
                                    level, busy_value = busy.estimate_recent(
                                        clock, config.busy_live_window)
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

                    if config.debug:
                        ok, jpeg = cv2.imencode(".jpg", canvas)
                        if ok:
                            buffer.append((frame_idx, jpeg, rows))
                        scrub["view"] = 0   # snap to newest after a real advance
                        if self._scrub_loop(buffer, scrub, dash_state) == "quit":
                            break
                    else:
                        if dash_state is not None:
                            dash_state.set_debug(frame_idx, rows)
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

import base64
import json
import threading
import time

import cv2
import numpy as np

from storepose.dashboard.stream import StreamHub, format_sse


def _frame(w=320, h=240):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _overlay():
    return {"w": 320, "h": 240, "people": [], "zones": {}, "busy": None}


def test_inactive_by_default():
    assert StreamHub().active is False


def test_publish_while_inactive_is_skipped():
    hub = StreamHub()
    hub.publish(_frame(), _overlay())
    assert hub.latest() is None  # lazy: nothing encoded when nobody is watching


def test_publish_while_active_encodes_and_bumps_seq():
    hub = StreamHub()
    with hub.subscribe():
        assert hub.active is True
        hub.publish(_frame(), _overlay())
        hub.publish(_frame(), _overlay())
        ev = hub.latest()
        assert ev["seq"] == 2
        assert ev["jpeg"].startswith("data:image/jpeg;base64,")
        assert ev["w"] == 320 and ev["people"] == []
    assert hub.active is False


def test_frame_downscaled_to_max_width():
    hub = StreamHub(max_width=480)
    with hub.subscribe():
        hub.publish(_frame(1920, 1080), {"w": 1920, "h": 1080, "people": [],
                                         "zones": {}, "busy": None})
        ev = hub.latest()
    b64 = ev["jpeg"].split(",", 1)[1]
    img = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[1] <= 480


def test_format_sse_framing():
    chunk = format_sse({"seq": 7, "w": 10})
    assert chunk.startswith(b"data: ")
    assert chunk.endswith(b"\n\n")
    assert json.loads(chunk[len(b"data: "):-2]) == {"seq": 7, "w": 10}


def test_events_streams_published_frames():
    hub = StreamHub()
    got: list[bytes] = []

    def consume():
        for chunk in hub.events():
            got.append(chunk)
            break

    th = threading.Thread(target=consume)
    th.start()
    for _ in range(200):  # wait for the generator to register as a subscriber
        if hub.active:
            break
        time.sleep(0.005)
    assert hub.active is True
    hub.publish(_frame(), _overlay())
    th.join(timeout=3)
    assert got and got[0].startswith(b"data: ")
    payload = json.loads(got[0][len(b"data: "):-2])
    assert payload["seq"] == 1 and "jpeg" in payload

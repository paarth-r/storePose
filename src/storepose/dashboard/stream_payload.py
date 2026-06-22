"""Build the per-frame overlay payload streamed to the browser dashboard.

Pure (no cv2): turns the tracked people, queue statuses, zones and busy label the
runner already has into the JSON the client-side SVG overlay renders. Coordinates
are in original frame space; the client scales them via an SVG ``viewBox``.
"""
from __future__ import annotations


def _state(status) -> str:
    """Map a PersonStatus to a single overlay state token."""
    if status is None:
        return "tracked"
    if status.serving:
        return "serving"
    if status.serving_other:
        return "serving_other"
    if status.waiting:
        return "waiting"
    if status.candidate:
        return "candidate"
    return "out"


def _keypoints(person) -> list:
    """[[x, y, score], ...] in original coords; ``[]`` while coasting."""
    if person.keypoints is None:
        return []
    scores = person.scores
    out = []
    for i, (x, y) in enumerate(person.keypoints):
        s = float(scores[i]) if scores is not None else 1.0
        out.append([round(float(x), 1), round(float(y), 1), round(s, 3)])
    return out


def _person_dict(person, status) -> dict:
    x1, y1, x2, y2 = (round(float(v), 1) for v in person.box)
    return {
        "id": int(person.id),
        "box": [x1, y1, x2, y2],
        "kpts": _keypoints(person),
        "state": _state(status),
        "wait": round(status.wait_seconds, 2) if status else 0.0,
        "serve": round(status.serving_seconds, 2) if status else 0.0,
        "progress": round(status.progress, 3) if status else 1.0,
    }


def _zone_polygons(zone) -> list:
    return [[[int(x), int(y)] for x, y in poly] for poly in zone.polygons]


def build_overlay(people, statuses, zones, busy, width: int, height: int) -> dict:
    """Assemble the overlay payload.

    Args:
        people: list of ``TrackedPerson`` this frame.
        statuses: list of ``PersonStatus`` (``QueueResult.statuses``); matched to
            people by id. May be empty when no analyzer is active.
        zones: mapping like ``{"line": Zone, "pos": Zone|None, "alt": Zone|None}``;
            ``None`` values are omitted from the output.
        busy: ``(level, value)`` tuple, or ``None``.
        width, height: original frame dimensions.
    """
    by_id = {s.id: s for s in statuses}
    out_people = [_person_dict(p, by_id.get(p.id)) for p in people]
    out_zones = {name: _zone_polygons(z) for name, z in zones.items() if z is not None}
    out_busy = {"level": busy[0], "value": busy[1]} if busy is not None else None
    return {
        "w": int(width),
        "h": int(height),
        "people": out_people,
        "zones": out_zones,
        "busy": out_busy,
    }

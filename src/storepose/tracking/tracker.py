"""SORT-style multi-object tracker producing stable, coasted tracks."""

from __future__ import annotations

from .assignment import iou, match
from .track import Track
from .types import TrackedPerson


def suppress_coasting_duplicates(tracks: list, max_overlap: float) -> list:
    """Remove coasting (predicted) tracks that duplicate a kept track.

    A coasting track is dropped when its box overlaps a higher-priority kept
    track by more than ``max_overlap``. Non-coasting tracks are never dropped,
    so two genuinely-detected people are never merged — this only removes the
    "ghost" left behind when a track coasts while the same person re-spawns a
    new track. Priority: actively-matched, then confirmed, then more hits.
    """
    order = sorted(tracks, key=lambda t: (t.coasting, not t.confirmed, -t.hits))
    kept: list = []
    for t in order:
        if t.coasting and any(iou(t.box, k.box) > max_overlap for k in kept):
            continue
        kept.append(t)
    return kept


class MultiObjectTracker:
    """Associates per-frame detections to persistent, smoothed tracks.

    Each ``update`` predicts existing tracks, associates them to the frame's
    detections by IoU, updates matches, coasts the rest (up to ``max_age``), and
    emits confirmed tracks as :class:`TrackedPerson`.
    """

    def __init__(
        self,
        max_age: int = 45,
        min_hits: int = 3,
        iou_thr: float = 0.3,
        smooth: bool = True,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        max_overlap: float = 0.5,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_thr = iou_thr
        self.smooth = smooth
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.max_overlap = max_overlap
        self._tracks: list[Track] = []
        self._next_id = 0

    def update(self, result, dt: float) -> list[TrackedPerson]:
        boxes = result.boxes
        keypoints = result.keypoints
        scores = result.scores

        # 1. predict existing tracks forward
        for t in self._tracks:
            t.predict()

        # 2. associate detections to predicted tracks
        det_boxes = [boxes[i] for i in range(len(boxes))]
        track_boxes = [t.box for t in self._tracks]
        matches, unmatched_dets, _ = match(det_boxes, track_boxes, self.iou_thr)

        # 3. update matched tracks
        for d, tr in matches:
            self._tracks[tr].update(boxes[d], keypoints[d], scores[d], dt)

        # 4. spawn tracks for unmatched detections
        for d in unmatched_dets:
            self._tracks.append(
                Track(
                    self._next_id, boxes[d], keypoints[d], scores[d], dt,
                    min_hits=self.min_hits, smooth=self.smooth,
                    min_cutoff=self.min_cutoff, beta=self.beta,
                )
            )
            self._next_id += 1

        # 5. cull dead tracks (aged out, or tentative + missed)
        self._tracks = [
            t for t in self._tracks
            if t.time_since_update <= self.max_age
            and not (t.time_since_update >= 1 and not t.confirmed)
        ]

        # 5b. drop coasting ghosts that duplicate another track
        self._tracks = suppress_coasting_duplicates(self._tracks, self.max_overlap)

        # 6. emit confirmed tracks
        people: list[TrackedPerson] = []
        for t in self._tracks:
            if not t.confirmed:
                continue
            coasting = t.coasting
            people.append(
                TrackedPerson(
                    id=t.id,
                    box=t.box,
                    keypoints=None if coasting else t.keypoints,
                    scores=None if coasting else t.scores,
                    coasting=coasting,
                    color=t.color,
                )
            )
        return people

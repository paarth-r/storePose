"""SORT-style multi-object tracker producing stable, coasted tracks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .appearance import AppearanceModel
from .assignment import iou, match
from .track import Track
from .types import TrackedPerson

# Appearance re-attach spatial gate: a candidate must lie within this radius of
# the detection. Radius grows with the gap (a person moves while gone), capped.
_SPATIAL_GATE_FRAC = 0.12   # base radius as a fraction of the frame diagonal
_GATE_GROWTH = 0.05         # extra radius per gap frame
_GATE_CAP_FRAC = 0.5        # never exceed half the frame diagonal


def _center(box: np.ndarray) -> tuple[float, float]:
    return ((float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0)


@dataclass
class _LostEntry:
    """A confirmed track that aged out, awaiting appearance re-attach."""
    track: Track
    center: tuple[float, float]
    lost_age: int


@dataclass
class _Candidate:
    """A re-attach candidate: an unmatched active track or a gallery entry."""
    track: Track
    lost: _LostEntry | None
    center: tuple[float, float]
    gap: int
    descriptor: np.ndarray


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
    emits confirmed tracks as :class:`TrackedPerson`. When ``reid`` is enabled
    and a frame is supplied, leftover detections are re-attached to unmatched
    active tracks or aged-out gallery entries by gated appearance similarity,
    reviving the original id instead of spawning a new one.
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
        appearance: AppearanceModel | None = None,
        reid: bool = False,
        reid_max_age: int = 150,
        reid_thr: float = 0.6,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_thr = iou_thr
        self.smooth = smooth
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.max_overlap = max_overlap
        self._appearance = appearance
        self._reid = reid
        self._reid_max_age = reid_max_age
        self._reid_thr = reid_thr
        self._tracks: list[Track] = []
        self._lost: list[_LostEntry] = []
        self._next_id = 0

    def update(self, result, dt: float, frame=None) -> list[TrackedPerson]:
        boxes = result.boxes
        keypoints = result.keypoints
        scores = result.scores
        n = len(boxes)

        use_reid = self._reid and self._appearance is not None and frame is not None
        if use_reid:
            det_descs = [
                self._appearance.extract(frame, boxes[i], keypoints[i], scores[i])
                for i in range(n)
            ]
        else:
            det_descs = [None] * n

        # 1. predict existing tracks forward; age the lost gallery
        for t in self._tracks:
            t.predict()
        if use_reid:
            for e in self._lost:
                e.lost_age += 1
            self._lost = [e for e in self._lost if e.lost_age <= self._reid_max_age]

        # 2. associate detections to predicted tracks by IoU
        det_boxes = [boxes[i] for i in range(n)]
        track_boxes = [t.box for t in self._tracks]
        matches, unmatched_dets, unmatched_tracks = match(
            det_boxes, track_boxes, self.iou_thr
        )

        # 3. update matched tracks
        for d, tr in matches:
            self._tracks[tr].update(
                boxes[d], keypoints[d], scores[d], dt, descriptor=det_descs[d]
            )

        # 4. appearance re-attach leftover detections
        if use_reid and unmatched_dets:
            unmatched_dets = self._reattach(
                unmatched_dets, unmatched_tracks, boxes, keypoints, scores,
                det_descs, frame, dt,
            )

        # 5. spawn tracks for still-unmatched detections
        for d in unmatched_dets:
            self._tracks.append(
                Track(
                    self._next_id, boxes[d], keypoints[d], scores[d], dt,
                    min_hits=self.min_hits, smooth=self.smooth,
                    min_cutoff=self.min_cutoff, beta=self.beta,
                    descriptor=det_descs[d],
                )
            )
            self._next_id += 1

        # 6. cull: aged-out confirmed tracks go to the gallery; drop tentatives
        survivors: list[Track] = []
        for t in self._tracks:
            if t.time_since_update >= 1 and not t.confirmed:
                continue
            if t.time_since_update > self.max_age:
                if use_reid and t.confirmed and t.descriptor is not None:
                    self._lost.append(
                        _LostEntry(track=t, center=_center(t.box), lost_age=0)
                    )
                continue
            survivors.append(t)
        self._tracks = survivors

        # 6b. drop coasting ghosts that duplicate another track
        self._tracks = suppress_coasting_duplicates(self._tracks, self.max_overlap)

        # 7. emit confirmed tracks
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

    def _reattach(
        self, unmatched_dets, unmatched_tracks, boxes, keypoints, scores,
        det_descs, frame, dt,
    ) -> list[int]:
        """Revive ids for leftover detections via gated appearance match.

        Returns the detection indices that remain unmatched (to be spawned).
        """
        diag = float(np.hypot(frame.shape[1], frame.shape[0]))

        cands: list[_Candidate] = []
        for tr_idx in unmatched_tracks:
            t = self._tracks[tr_idx]
            if not t.confirmed or t.descriptor is None:
                continue
            cands.append(_Candidate(t, None, _center(t.box), t.time_since_update, t.descriptor))
        for e in self._lost:
            if e.track.descriptor is None:
                continue
            cands.append(_Candidate(e.track, e, e.center, e.lost_age, e.track.descriptor))
        if not cands:
            return unmatched_dets

        # (cost, det_index, cand_index) for every gated, above-threshold pair
        pairs: list[tuple[float, int, int]] = []
        for d in unmatched_dets:
            dd = det_descs[d]
            if dd is None:
                continue
            dcx, dcy = _center(boxes[d])
            for ci, cand in enumerate(cands):
                radius = min(
                    _SPATIAL_GATE_FRAC * diag * (1.0 + _GATE_GROWTH * cand.gap),
                    _GATE_CAP_FRAC * diag,
                )
                if np.hypot(dcx - cand.center[0], dcy - cand.center[1]) > radius:
                    continue
                sim = self._appearance.similarity(dd, cand.descriptor)
                if sim < self._reid_thr:
                    continue
                pairs.append((1.0 - sim, d, ci))

        pairs.sort(key=lambda p: p[0])
        used_d: set[int] = set()
        used_c: set[int] = set()
        for _cost, d, ci in pairs:
            if d in used_d or ci in used_c:
                continue
            used_d.add(d)
            used_c.add(ci)
            cand = cands[ci]
            cand.track.reactivate(boxes[d], keypoints[d], scores[d], dt, descriptor=det_descs[d])
            if cand.lost is not None:
                self._lost.remove(cand.lost)
                self._tracks.append(cand.track)
        return [d for d in unmatched_dets if d not in used_d]

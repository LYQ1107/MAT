from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from mat.core.errors import ValidationError
from mat.core.types import AnimalObservation, FramePacket, LocalAssociation


def _iou(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-9)


@dataclass
class _Track:
    uid: str
    box: np.ndarray
    last_frame: int
    missed: int = 0


class ByteTrackBackend:
    """Index-preserving tracker boundary.

    If the pinned official dependency is installed, callers may inject it; the compact
    fallback is deterministic and exists for contracts/fixtures, not as an official
    ByteTrack score. In both paths association returns the source detection index.
    """

    def __init__(self, track_thresh: float = 0.5, match_thresh: float = 0.8,
                 track_buffer: int = 30, frame_rate: float = 30.0):
        if frame_rate <= 0 or track_buffer < 0:
            raise ValidationError("invalid tracker rate/buffer")
        self.track_thresh = float(track_thresh)
        self.match_thresh = float(match_thresh)
        self.track_buffer = int(track_buffer)
        self.frame_rate = float(frame_rate)
        self.reset("uninitialized")

    def reset(self, session_key: str) -> None:
        self.session_key = session_key
        self._tracks: list[_Track] = []
        self._next = 0
        self._frame_index = -1

    def _new(self, box: np.ndarray, frame: int) -> _Track:
        uid = f"{self.session_key}:local:{self._next}"
        self._next += 1
        track = _Track(uid, box.astype(np.float32).copy(), frame)
        self._tracks.append(track)
        return track

    def update(self, frame: FramePacket, observations: list[AnimalObservation]) -> list[LocalAssociation]:
        if frame.session_uid != self.session_key:
            self.reset(frame.session_uid)
        self._frame_index = frame.frame_index
        dets = [(i, np.asarray(o.bbox_xyxy, dtype=np.float32), float(o.detection_score), o.detection_uid)
                for i, o in enumerate(observations)]
        unmatched_tracks = set(range(len(self._tracks)))
        unmatched_dets = set(range(len(dets)))
        matches: list[tuple[int, int]] = []
        # Greedy highest IoU preserves the original detection index; no re-IoU pose guess.
        candidates = sorted((( _iou(self._tracks[ti].box, det[1]), ti, di)
                             for ti in range(len(self._tracks)) for di, det in enumerate(dets)
                             if det[2] >= self.track_thresh), reverse=True)
        for score, ti, di in candidates:
            if ti in unmatched_tracks and di in unmatched_dets and score >= self.match_thresh:
                unmatched_tracks.remove(ti); unmatched_dets.remove(di); matches.append((ti, di))
        associations: list[LocalAssociation] = []
        for ti, di in matches:
            track = self._tracks[ti]
            det = dets[di]
            track.box, track.last_frame, track.missed = det[1].copy(), frame.frame_index, 0
            associations.append(LocalAssociation(track.uid, det[3], di, tuple(map(float, det[1])), "matched"))
        for di in sorted(unmatched_dets):
            det = dets[di]
            if det[2] >= self.track_thresh:
                track = self._new(det[1], frame.frame_index)
                associations.append(LocalAssociation(track.uid, det[3], di, tuple(map(float, det[1])), "new"))
        for ti in sorted(unmatched_tracks):
            track = self._tracks[ti]
            track.missed += 1
            if track.missed <= self.track_buffer:
                associations.append(LocalAssociation(track.uid, None, None, tuple(map(float, track.box)), "predicted"))
        self._tracks = [t for t in self._tracks if t.missed <= self.track_buffer]
        return associations


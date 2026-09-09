"""Public SLEAP-IO prediction adapter.

The adapter is intentionally lazy with respect to the optional SLEAP runtime:
the audit/control interpreter can import MAT without importing ``sleap_io``.
Only detector outputs are read; provider track/identity annotations are not
copied into the returned model-facing contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping
import hashlib
import math

import numpy as np

from mat.core.errors import DependencyUnavailableError, ValidationError
from mat.core.types import PredictedPoseInstance


def _prediction_uid(session_uid: str, frame_index: int, instance_index: int) -> str:
    payload = f"{session_uid}|{int(frame_index)}|{int(instance_index)}".encode("utf-8")
    return "pred:" + hashlib.sha256(payload).hexdigest()[:24]


def _bbox(points: np.ndarray, valid: np.ndarray) -> np.ndarray | None:
    usable = np.isfinite(points).all(axis=1) & np.asarray(valid, dtype=bool)
    if int(usable.sum()) < 2:
        return None
    selected = points[usable]
    low = selected.min(axis=0)
    high = selected.max(axis=0)
    if not np.isfinite(low).all() or not np.isfinite(high).all():
        return None
    if high[0] <= low[0] or high[1] <= low[1]:
        # A detector can emit two points on one pixel row/column.  Keep the
        # instance valid with a one-pixel extent rather than inventing a large
        # crop or dropping otherwise useful pose evidence.
        high = np.maximum(high, low + 1.0)
    return np.asarray([low[0], low[1], high[0], high[1]], dtype=np.float32)


class SleapPosePredictionAdapter:
    """Load predicted SLP instances and assign deterministic local UIDs."""

    def __init__(self, *, pose_model_fingerprint: str, default_fps: float = 25.0):
        if not pose_model_fingerprint:
            raise ValidationError("pose_model_fingerprint must be non-empty")
        if not math.isfinite(float(default_fps)) or float(default_fps) <= 0:
            raise ValidationError("default_fps must be positive and finite")
        self.pose_model_fingerprint = str(pose_model_fingerprint)
        self.default_fps = float(default_fps)

    @staticmethod
    def _resolve_session(filename: str, video_index: int, session_map: Mapping[str, str]) -> str:
        base = Path(filename).name if filename else ""
        candidates = [filename, base, f"{base}#video{video_index}", f"video{video_index}", str(video_index)]
        for candidate in candidates:
            if candidate in session_map:
                return str(session_map[candidate])
        # Prepared inventories often preserve the provider basename while a
        # prediction SLP stores the materialized test/train basename.  Match
        # only an unambiguous ``#videoN`` suffix; never guess across sessions.
        suffix = f"#video{video_index}"
        matches = [str(value) for key, value in session_map.items() if str(key).endswith(suffix)]
        if len(set(matches)) == 1:
            return matches[0]
        raise ValidationError(f"prediction video {filename!r} index {video_index} is absent/ambiguous in session_map")

    def load(self, prediction_slp: Path, session_map: Mapping[str, str]) -> list[PredictedPoseInstance]:
        path = Path(prediction_slp).expanduser().resolve()
        if not path.is_file():
            raise ValidationError(f"missing SLEAP prediction SLP: {path}")
        if not session_map:
            raise ValidationError("session_map must not be empty")
        try:
            from sleap_io import load_slp
        except Exception as exc:  # pragma: no cover - depends on isolated runtime
            raise DependencyUnavailableError("SLEAP-IO is required to parse prediction SLP files") from exc
        labels = load_slp(path, open_videos=False, lazy=True)
        output: list[PredictedPoseInstance] = []
        try:
            videos = list(labels.videos)
            video_indices = {id(video): index for index, video in enumerate(videos)}
            for labeled_frame in labels.labeled_frames:
                video = labeled_frame.video
                video_index = video_indices.get(id(video))
                if video_index is None:
                    # SLEAP may materialize an equivalent Video object; the
                    # public list/index operation is the documented fallback.
                    try:
                        video_index = videos.index(video)
                    except ValueError as exc:
                        raise ValidationError("prediction frame references an unknown SLEAP video") from exc
                filename = str(getattr(video, "filename", "") or "")
                session_uid = self._resolve_session(filename, int(video_index), session_map)
                frame_index = int(labeled_frame.frame_idx)
                fps = getattr(video, "fps", None)
                try:
                    fps_value = float(fps) if fps is not None and math.isfinite(float(fps)) and float(fps) > 0 else self.default_fps
                except (TypeError, ValueError):
                    fps_value = self.default_fps
                for instance_index, instance in enumerate(labeled_frame.instances):
                    points_obj = getattr(instance, "points", None)
                    if points_obj is not None and getattr(points_obj, "dtype", None) is not None and getattr(points_obj.dtype, "names", None):
                        points = np.asarray(points_obj["xy"], dtype=np.float32)
                        scores = np.asarray(points_obj["score"], dtype=np.float32)
                        valid = np.asarray(points_obj["visible"], dtype=bool)
                    else:
                        points = np.asarray(instance.numpy(), dtype=np.float32)
                        scores = np.isfinite(points).all(axis=1).astype(np.float32)
                        valid = np.isfinite(points).all(axis=1)
                    valid &= np.isfinite(points).all(axis=1)
                    box = _bbox(points, valid)
                    if box is None:
                        # The public instance cannot satisfy the required crop
                        # contract.  Record the omission in parser statistics
                        # at the caller rather than emitting an invalid object.
                        continue
                    score = getattr(instance, "score", None)
                    try:
                        score = float(score) if score is not None and math.isfinite(float(score)) else None
                    except (TypeError, ValueError):
                        score = None
                    track = getattr(instance, "track", None)
                    local_track_uid = None
                    if track is not None:
                        # String conversion is a local handle only.  No SLEAP
                        # identity/track name is interpreted as persistent ID.
                        local_track_uid = str(getattr(track, "name", track))
                    output.append(PredictedPoseInstance(
                        prediction_uid=_prediction_uid(session_uid, frame_index, instance_index),
                        session_uid=session_uid, frame_index=frame_index,
                        timestamp_s=frame_index / fps_value, bbox_xyxy=box,
                        keypoints_xy=points, keypoint_scores=scores, keypoint_valid=valid,
                        detection_score=score, local_track_uid=local_track_uid,
                        pose_model_fingerprint=self.pose_model_fingerprint,
                    ))
        finally:
            close = getattr(labels, "close", None)
            if callable(close):
                close()
        return output


__all__ = ["SleapPosePredictionAdapter"]

"""Versioned, neutral data contracts crossing MAT module boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import math

import numpy as np

from .errors import ValidationError


SCHEMA_VERSION = "mat.v1"


def _array(value: Any, shape: tuple[int, ...] | None = None, dtype=np.float32) -> np.ndarray:
    arr = np.asarray(value, dtype=dtype)
    if shape is not None and arr.shape != shape:
        raise ValidationError(f"expected shape {shape}, got {arr.shape}")
    return arr


@dataclass(frozen=True)
class FramePacket:
    dataset_uid: str
    cohort_uid: str
    session_uid: str
    camera_uid: str
    frame_index: int
    timestamp_s: float
    rgb: np.ndarray

    def __post_init__(self) -> None:
        if self.frame_index < 0 or not math.isfinite(float(self.timestamp_s)):
            raise ValidationError("frame_index/timestamp must be finite and frame_index >= 0")
        if not isinstance(self.rgb, np.ndarray) or self.rgb.ndim != 3 or self.rgb.shape[2] != 3:
            raise ValidationError("rgb must be an H,W,3 array")
        if self.rgb.dtype != np.uint8:
            raise ValidationError("rgb must be uint8")

    @property
    def frame_uid(self) -> str:
        return f"{self.session_uid}:{self.frame_index}"


@dataclass(frozen=True)
class SessionSpec:
    session_uid: str
    cohort_uid: str
    recorded_at: str | None
    camera_uid: str
    observations_manifest: Path
    timebase: dict[str, Any]
    calibration_ref: str | None = None


@dataclass(frozen=True)
class SpeciesSpec:
    name: str
    skeleton_version: str
    keypoint_names: tuple[str, ...]
    edges: tuple[tuple[int, int], ...]
    part_groups: dict[str, tuple[int, ...]]
    flip_index: tuple[int, ...] | None
    supported_views: tuple[str, ...]
    pose_asset_id: str

    def __post_init__(self) -> None:
        k = len(self.keypoint_names)
        if k == 0 or any(i < 0 or i >= k for e in self.edges for i in e):
            raise ValidationError("invalid species skeleton")
        for name, ids in self.part_groups.items():
            if not ids or any(i < 0 or i >= k for i in ids):
                raise ValidationError(f"invalid part group {name}")
        if self.flip_index is not None and len(self.flip_index) != k:
            raise ValidationError("flip_index must cover every keypoint")


@dataclass
class AnimalObservation:
    observation_uid: str
    frame_uid: str
    detection_uid: str
    bbox_xyxy: np.ndarray
    detection_score: float
    keypoints_xy: np.ndarray
    keypoint_scores: np.ndarray
    keypoint_status: np.ndarray
    mask_ref: str | None
    coordinate_transform: dict[str, Any]
    evidence_provenance: dict[str, Any]

    def __post_init__(self) -> None:
        self.bbox_xyxy = _array(self.bbox_xyxy, (4,))
        self.keypoints_xy = _array(self.keypoints_xy)
        self.keypoint_scores = _array(self.keypoint_scores)
        self.keypoint_status = np.asarray(self.keypoint_status)
        if self.keypoints_xy.ndim != 2 or self.keypoints_xy.shape[1] != 2:
            raise ValidationError("keypoints_xy must be Kx2")
        if self.keypoint_scores.shape != (self.keypoints_xy.shape[0],):
            raise ValidationError("keypoint_scores must have K entries")
        if self.keypoint_status.shape != (self.keypoints_xy.shape[0],):
            raise ValidationError("keypoint_status must have K entries")
        if not 0 <= float(self.detection_score) <= 1:
            raise ValidationError("detection_score must be in [0,1]")
        if np.any(np.isinf(self.keypoints_xy)):
            raise ValidationError("keypoints may be finite or NaN, not infinity")


@dataclass
class LocalTracklet:
    tracklet_uid: str
    session_uid: str
    camera_uid: str
    observations: list[str]
    time_support: list[tuple[float, float]]
    quality: dict[str, Any] = field(default_factory=dict)


@dataclass
class IdentityDescriptor:
    global_feature: np.ndarray
    part_features: np.ndarray
    part_valid: np.ndarray
    part_quality: np.ndarray
    encoder_fingerprint: str

    def __post_init__(self) -> None:
        self.global_feature = _array(self.global_feature)
        self.part_features = _array(self.part_features)
        self.part_valid = np.asarray(self.part_valid, dtype=bool)
        self.part_quality = _array(self.part_quality)
        if self.global_feature.ndim != 1:
            raise ValidationError("global feature must be one-dimensional")
        if self.part_features.ndim != 2 or self.part_features.shape[0] != self.part_valid.size:
            raise ValidationError("part feature/valid shapes disagree")
        if self.part_quality.shape != self.part_valid.shape:
            raise ValidationError("part quality/valid shapes disagree")


@dataclass
class DescriptorBatch:
    global_features: np.ndarray
    part_features: np.ndarray
    part_valid: np.ndarray
    part_quality: np.ndarray
    encoder_fingerprint: str


@dataclass(frozen=True)
class PersistentIdentity:
    identity_uid: str
    cohort_uid: str
    anchor_refs: tuple[str, ...]
    committed_refs: tuple[str, ...]
    provisional_refs: tuple[str, ...]
    created_from_session: str
    provenance: dict[str, Any]


@dataclass(frozen=True)
class Assignment:
    tracklet_uid: str
    persistent_uid: str | None
    candidate_uids: tuple[str, ...]
    candidate_scores: tuple[float, ...]
    status: str
    reasons: tuple[str, ...]
    gallery_version: str
    model_version: str


@dataclass
class ScoreMatrix:
    tracklet_uids: tuple[str, ...]
    identity_uids: tuple[str, ...]
    values: np.ndarray

    def __post_init__(self) -> None:
        self.values = _array(self.values)
        expected = (len(self.tracklet_uids), len(self.identity_uids))
        if self.values.shape != expected:
            raise ValidationError(f"score matrix shape {self.values.shape} != {expected}")


@dataclass(frozen=True)
class LocalAssociation:
    local_track_uid: str
    detection_uid: str | None
    source_detection_index: int | None
    box_xyxy: tuple[float, float, float, float]
    state: str


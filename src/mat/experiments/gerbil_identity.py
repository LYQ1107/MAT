"""File-backed gerbil B0 identity diagnostics.

The model-facing side of this module consumes only neutral observation rows and
pose/bounding-box predictions.  Provider identity/keypoint truth is accepted
through a separate evaluator argument and is never attached to a model row.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from mat.core.errors import ValidationError
from mat.core.types import DescriptorBatch, IdentityDescriptor, LocalTracklet, PredictedPoseInstance
from mat.enrollment.base import ReferenceVerification
from mat.enrollment.manual import ManualRegistrar
from mat.identity.gallery import GalleryStore
from mat.identity.conflicts import ConflictGraphBuilder
from mat.identity.matching import MatchingPolicy, PersistentMatcher
from mat.data.sanitize import FORBIDDEN_MODEL_FIELDS
from mat.config.identity import IdentityMatchingConfig
from mat.identity.calibration import ThresholdCalibrationResult, calibrate_accept_threshold
from mat.evaluation.fixed_identity import EnrollmentMapping, PersistentIDEvaluator
from mat.evaluation.instance_matching import PoseInstanceMatcher


@dataclass(frozen=True)
class LazyRGBImage:
    """Shape-aware on-disk RGB image used by the file-backed runner.

    Prepared frames are large (1280x1024) and several instances can reference
    the same frame.  Keeping only the path in each neutral sample avoids
    materializing gigabytes before identity crops are actually encoded.
    ``np.asarray`` is the sole materialization point and is used by ``_crop``.
    """

    path: Path
    _shape: tuple[int, int, int]

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._shape

    @property
    def ndim(self) -> int:
        return 3

    @property
    def dtype(self):
        return np.dtype(np.uint8)

    def __array__(self, dtype=None) -> np.ndarray:
        from PIL import Image
        with Image.open(self.path) as image:
            # PIL may expose a read-only view.  The downstream tensor runtimes
            # are allowed to reuse their input buffers, so materialize a
            # writable RGB array at this single lazy boundary.
            array = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
        return array.astype(dtype, copy=False) if dtype is not None else array


@dataclass(frozen=True)
class GerbilInstanceSample:
    """Neutral one-instance input to the identity backend."""

    observation_uid: str
    session_uid: str
    frame_index: int
    timestamp_s: float
    image: Any
    bbox_xyxy: np.ndarray | None
    keypoints_xy: np.ndarray
    keypoint_scores: np.ndarray
    keypoint_valid: np.ndarray
    tracklet_uid: str
    detection_uid: str | None = None
    source_detection_index: int | None = None

    def __post_init__(self) -> None:
        if (getattr(self.image, "ndim", None) != 3
                or getattr(self.image, "shape", (0, 0, 0))[-1] != 3
                or np.dtype(getattr(self.image, "dtype", object)) != np.dtype(np.uint8)):
            raise ValidationError("gerbil sample image must be uint8 HWC RGB")
        if self.frame_index < 0 or not np.isfinite(float(self.timestamp_s)):
            raise ValidationError("sample frame/timestamp is invalid")
        points = np.asarray(self.keypoints_xy, dtype=np.float32)
        scores = np.asarray(self.keypoint_scores, dtype=np.float32)
        valid = np.asarray(self.keypoint_valid, dtype=bool)
        if points.ndim != 2 or points.shape[1] != 2 or scores.shape != (points.shape[0],) or valid.shape != scores.shape:
            raise ValidationError("sample keypoint shapes disagree")
        if np.isinf(points).any() or np.isinf(scores).any():
            raise ValidationError("sample keypoints/scores cannot contain infinity")
        object.__setattr__(self, "keypoints_xy", points)
        object.__setattr__(self, "keypoint_scores", scores)
        object.__setattr__(self, "keypoint_valid", valid)
        if self.bbox_xyxy is not None:
            box = np.asarray(self.bbox_xyxy, dtype=np.float32)
            if box.shape != (4,) or not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
                raise ValidationError("sample bbox must be finite xyxy with positive area")
            object.__setattr__(self, "bbox_xyxy", box)


def load_prepared_gerbil_samples(work_root: Path, *, keypoint_count: int = 14,
                                 lazy_images: bool = True) -> tuple[list[GerbilInstanceSample], list[dict[str, Any]]]:
    """Load neutral images/observations from the prepared file-backed bundle.

    ``lazy_images`` defaults to true because the real prepared bundle contains
    1561 instance rows over 1280x1024 frames.  Tests and small callers may set
    it false to obtain ordinary uint8 arrays.
    """
    root = Path(work_root).expanduser().resolve() / "prepared" / "sleap_gerbils"
    manifest = root / "manifests" / "observations.jsonl"
    if not manifest.is_file():
        raise ValidationError(f"missing prepared neutral observations: {manifest}")
    samples: list[GerbilInstanceSample] = []
    rows: list[dict[str, Any]] = []
    from PIL import Image
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if FORBIDDEN_MODEL_FIELDS.intersection(row):
            raise ValidationError("neutral observation manifest contains evaluator-only truth")
        image_path = root / str(row["image_ref"])
        if lazy_images:
            with Image.open(image_path) as probe:
                shape = (int(probe.height), int(probe.width), 3)
            image: Any = LazyRGBImage(image_path, shape)
        else:
            with Image.open(image_path) as opened:
                image = np.asarray(opened.convert("RGB"), dtype=np.uint8)
        points = np.full((keypoint_count, 2), np.nan, dtype=np.float32)
        samples.append(GerbilInstanceSample(
            observation_uid=str(row["observation_uid"]), session_uid=str(row["session_uid"]),
            frame_index=int(row["frame_index"]), timestamp_s=float(row.get("timestamp_s", row["frame_index"] / 25.0)),
            image=image, bbox_xyxy=None, keypoints_xy=points,
            keypoint_scores=np.zeros(keypoint_count, dtype=np.float32),
            keypoint_valid=np.zeros(keypoint_count, dtype=bool),
            tracklet_uid=str(row.get("tracklet_uid", row["observation_uid"])),
            detection_uid=None, source_detection_index=None,
        ))
        rows.append(row)
    return samples, rows


def load_private_gerbil_truth(work_root: Path) -> dict[str, dict[str, Any]]:
    """Load evaluator-only truth explicitly outside model-facing inputs."""
    path = Path(work_root).expanduser().resolve() / "prepared" / "sleap_gerbils" / "manifests" / "private_pose_identity_truth.jsonl"
    if not path.is_file():
        raise ValidationError(f"missing private evaluator truth: {path}")
    return {str(row["observation_uid"]): row for row in
            (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())}


def load_session_inventory(work_root: Path) -> list[dict[str, Any]]:
    path = Path(work_root).expanduser().resolve() / "prepared" / "sleap_gerbils" / "manifests" / "session_inventory.json"
    if not path.is_file():
        raise ValidationError(f"missing session inventory: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return list(raw.get("sessions", []))


def bbox_from_keypoints(keypoints_xy: Any, keypoint_valid: Any | None = None, *,
                        image_shape: tuple[int, int] | None = None,
                        padding: float = 0.10, min_points: int = 2) -> np.ndarray | None:
    """Return a finite padded xyxy box, or ``None`` when evidence is insufficient."""
    points = np.asarray(keypoints_xy, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2 or padding < 0 or min_points <= 0:
        raise ValidationError("keypoints must be Kx2 and bbox parameters valid")
    valid = np.isfinite(points).all(axis=1)
    if keypoint_valid is not None:
        mask = np.asarray(keypoint_valid, dtype=bool)
        if mask.shape != (points.shape[0],):
            raise ValidationError("keypoint_valid shape disagrees")
        valid &= mask
    selected = points[valid]
    if len(selected) < min_points:
        return None
    min_xy = selected.min(axis=0)
    max_xy = selected.max(axis=0)
    width = max(float(max_xy[0] - min_xy[0]), 1.0)
    height = max(float(max_xy[1] - min_xy[1]), 1.0)
    box = np.asarray([min_xy[0] - padding * width, min_xy[1] - padding * height,
                      max_xy[0] + padding * width, max_xy[1] + padding * height], dtype=np.float32)
    if image_shape is not None:
        h, w = image_shape
        box[[0, 2]] = np.clip(box[[0, 2]], 0, float(w))
        box[[1, 3]] = np.clip(box[[1, 3]], 0, float(h))
    if not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def select_s0_sessions(session_rows: Sequence[Mapping[str, Any]], truth_rows: Sequence[Mapping[str, Any]], *,
                       expected_identity_count: int = 4) -> tuple[str, ...]:
    """Choose the earliest deterministic sessions covering all provider IDs."""
    obs_to_session: dict[str, str] = {}
    # The caller may provide ``session_uid`` in truth rows; otherwise it joins
    # through the neutral observation rows supplied in the same records.
    for row in truth_rows:
        if row.get("session_uid") is not None:
            obs_to_session[str(row["observation_uid"])] = str(row["session_uid"])
    ordered = sorted(session_rows, key=lambda row: (row.get("recording_datetime_if_parseable") is None,
                                                     row.get("recording_datetime_if_parseable") or "",
                                                     str(row.get("session_uid", ""))))
    by_session: dict[str, set[str]] = defaultdict(set)
    for row in truth_rows:
        identity = row.get("gt_identity")
        session = row.get("session_uid") or obs_to_session.get(str(row.get("observation_uid", "")))
        if identity is not None and session is not None:
            by_session[str(session)].add(str(identity))
    selected: list[str] = []
    covered: set[str] = set()
    for row in ordered:
        uid = str(row.get("session_uid", ""))
        if not uid:
            continue
        selected.append(uid)
        covered.update(by_session.get(uid, set()))
        if len(covered) >= expected_identity_count:
            break
    return tuple(selected)


def select_oracle_reference_anchors(samples: Sequence[GerbilInstanceSample],
                                    truth_by_observation: Mapping[str, Mapping[str, Any]], *,
                                    s0_sessions: Sequence[str], max_per_identity: int = 16
                                    ) -> dict[str, tuple[str, ...]]:
    """Select at most ``max_per_identity`` visible, time-spread S0 anchors."""
    if max_per_identity <= 0:
        raise ValidationError("max_per_identity must be positive")
    grouped: dict[str, list[GerbilInstanceSample]] = defaultdict(list)
    allowed = set(s0_sessions)
    for sample in samples:
        truth = truth_by_observation.get(sample.observation_uid, {})
        identity = truth.get("gt_identity")
        if sample.session_uid not in allowed or identity is None:
            continue
        # Anchors use the evaluator's visible pose only to choose a crop; the
        # identity descriptor still sees the neutral image crop.
        visible = truth.get("gt_visibility", sample.keypoint_valid)
        if bbox_from_keypoints(truth.get("gt_keypoints", sample.keypoints_xy), visible,
                               image_shape=sample.image.shape[:2]) is None:
            continue
        grouped[str(identity)].append(sample)
    selected: dict[str, tuple[str, ...]] = {}
    for identity, values in sorted(grouped.items()):
        values = sorted(values, key=lambda item: (item.timestamp_s, item.observation_uid))
        if len(values) <= max_per_identity:
            chosen = values
        else:
            indices = np.linspace(0, len(values) - 1, max_per_identity, dtype=int)
            chosen = [values[int(index)] for index in indices]
        selected[identity] = tuple(item.observation_uid for item in chosen)
    return selected


def _crop(image: np.ndarray, box: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(round(float(x))) for x in box]
    x1, y1 = max(0, min(w - 1, x1)), max(0, min(h - 1, y1))
    x2, y2 = max(x1 + 1, min(w, x2)), max(y1 + 1, min(h, y2))
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValidationError("bbox produced an empty crop")
    return np.asarray(crop, dtype=np.uint8)


def _descriptor_rows(samples: Sequence[GerbilInstanceSample], encoder: Any,
                     boxes: Mapping[str, np.ndarray], *, batch_size: int = 32):
    if batch_size <= 0:
        raise ValidationError("descriptor batch_size must be positive")
    ordered = [sample for sample in samples if sample.observation_uid in boxes]
    if not ordered:
        return {}, []
    descriptors: dict[str, IdentityDescriptor] = {}
    for start in range(0, len(ordered), batch_size):
        chunk = ordered[start:start + batch_size]
        crops = [_crop(sample.image, boxes[sample.observation_uid]) for sample in chunk]
        shapes = {crop.shape for crop in crops}
        if len(shapes) == 1:
            encoded = encoder.encode(np.stack(crops))
        elif hasattr(encoder, "encode_crops"):
            # The verified MegaDescriptor backend preprocesses each ROI before
            # concatenating tensors, so heterogeneous crop dimensions never
            # reach np.stack and are not padded with model-visible pixels.
            encoded = encoder.encode_crops(crops)
        else:
            # Keep the generic backend contract usable for test fixtures and
            # third-party wrappers that only expose ``encode(B,H,W,C)``.  This
            # is slower, but each call remains a neutral model row and avoids
            # silently changing the feature-space preprocessing.
            encoded_batches = [encoder.encode(crop[None, ...]) for crop in crops]
            encoded = DescriptorBatch(
                np.concatenate([batch.global_features for batch in encoded_batches], axis=0),
                np.concatenate([batch.part_features for batch in encoded_batches], axis=0),
                np.concatenate([batch.part_valid for batch in encoded_batches], axis=0),
                np.concatenate([batch.part_quality for batch in encoded_batches], axis=0),
                encoded_batches[0].encoder_fingerprint,
            )
        if encoded.global_features.shape[0] != len(chunk):
            raise ValidationError("identity encoder returned a row count different from the crop batch")
        descriptors.update({sample.observation_uid: IdentityDescriptor(encoded.global_features[i], encoded.part_features[i],
                                                                        encoded.part_valid[i], encoded.part_quality[i],
                                                                        encoded.encoder_fingerprint)
                            for i, sample in enumerate(chunk)})
    return descriptors, ordered


def _gallery_from_oracle(samples: Sequence[GerbilInstanceSample], truth_by_observation: Mapping[str, Mapping[str, Any]],
                         anchors: Mapping[str, Sequence[str]], encoder: Any, gallery_store: GalleryStore,
                         cohort_uid: str):
    sample_by_uid = {sample.observation_uid: sample for sample in samples}
    anchor_samples = [sample_by_uid[uid] for values in anchors.values() for uid in values if uid in sample_by_uid]
    boxes = {}
    model_rows = []
    for sample in anchor_samples:
        truth = truth_by_observation[sample.observation_uid]
        box = bbox_from_keypoints(truth["gt_keypoints"], truth.get("gt_visibility"), image_shape=sample.image.shape[:2])
        if box is not None:
            boxes[sample.observation_uid] = box
            model_rows.append(sample)
    descriptors, ordered = _descriptor_rows(model_rows, encoder, boxes)
    tracklets = [LocalTracklet(f"anchor:{sample.observation_uid}", sample.session_uid, "camera",
                               [sample.observation_uid], [(sample.timestamp_s, sample.timestamp_s)]) for sample in ordered]
    groups = {identity: tuple(f"anchor:{uid}" for uid in values if uid in descriptors) for identity, values in anchors.items()}
    groups = {identity: refs for identity, refs in groups.items() if refs}
    enrollment = ManualRegistrar().build(type("Reference", (), {
        "tracklets": tracklets, "descriptors": {f"anchor:{uid}": descriptors[uid] for uid in descriptors},
        "session_uid": "+".join(sorted(set(sample.session_uid for sample in ordered))),
        "cohort_uid": cohort_uid,
    })(), ReferenceVerification(groups=groups, provenance="oracle_reference"), cohort_uid=cohort_uid)
    return gallery_store.create(enrollment), sample_by_uid


@dataclass(frozen=True)
class GerbilB0Result:
    status: str
    input_mode: str
    enrollment_mode: str
    s0_sessions: tuple[str, ...]
    anchor_counts: dict[str, int]
    assignments: tuple[dict[str, Any], ...]
    metrics: dict[str, Any]
    gallery_version: str | None
    blocker: str | None = None
    reference_sessions: tuple[str, ...] = ()
    development_sessions: tuple[str, ...] = ()
    sealed_test_sessions: tuple[str, ...] = ()
    threshold: float | None = None
    calibration: dict[str, Any] | None = None
    development_assignments: tuple[dict[str, Any], ...] = ()
    protocol_id: str | None = None

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _identity_label_mapping(snapshot) -> EnrollmentMapping:
    """Map frozen gallery persistent IDs to the explicit reference labels."""
    mapping: dict[str, str] = {}
    for uid, identity in snapshot.identities.items():
        label = identity.provenance.get("group_label") if isinstance(identity.provenance, dict) else None
        if label is not None:
            mapping[str(uid)] = str(label)
    result = EnrollmentMapping(mapping)
    result.freeze()
    return result


def _query_boxes(samples: Sequence[GerbilInstanceSample],
                 truth_by_observation: Mapping[str, Mapping[str, Any]],
                 session_uids: set[str], *, input_mode: str,
                 predicted_pose: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, np.ndarray]:
    """Build model-visible query boxes without adding evaluator fields to rows."""
    boxes: dict[str, np.ndarray] = {}
    predicted_pose = predicted_pose or {}
    for sample in samples:
        if sample.session_uid not in session_uids:
            continue
        if input_mode == "oracle":
            truth = truth_by_observation.get(sample.observation_uid, {})
            box = bbox_from_keypoints(truth.get("gt_keypoints", sample.keypoints_xy),
                                      truth.get("gt_visibility"), image_shape=sample.image.shape[:2])
        elif input_mode == "predicted":
            prediction = predicted_pose.get(sample.observation_uid, {})
            box = prediction.get("bbox_xyxy")
            if box is None and prediction.get("keypoints_xy") is not None:
                box = bbox_from_keypoints(prediction["keypoints_xy"], prediction.get("keypoint_valid"),
                                          image_shape=sample.image.shape[:2])
            if box is not None:
                box = np.asarray(box, dtype=np.float32)
                if box.shape != (4,) or not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
                    box = None
        else:
            raise ValidationError(f"unsupported B0 input_mode: {input_mode}")
        if box is not None:
            boxes[sample.observation_uid] = np.asarray(box, dtype=np.float32)
    return boxes


def _score_rows(samples: Sequence[GerbilInstanceSample], boxes: Mapping[str, np.ndarray], encoder: Any,
                snapshot: Any, *, batch_size: int = 32):
    ordered_samples = [sample for sample in samples if sample.observation_uid in boxes]
    descriptors, ordered = _descriptor_rows(ordered_samples, encoder, boxes, batch_size=batch_size)
    tracklets = [LocalTracklet(sample.observation_uid, sample.session_uid, "camera", [sample.observation_uid],
                               [(sample.timestamp_s, sample.timestamp_s)]) for sample in ordered]
    if not tracklets:
        return tracklets, descriptors, None
    scores = PersistentMatcher().score(tracklets, descriptors, snapshot)
    return tracklets, descriptors, scores


def _raw_prediction_rows(scores: Any, snapshot: Any, *, threshold: float | None = None,
                         assigned: Sequence[Any] | None = None) -> list[dict[str, Any]]:
    """Serialize top-1 labels for calibration/audit, without querying truth."""
    labels = {uid: (identity.provenance.get("group_label")
                    if isinstance(identity.provenance, dict) else None)
              for uid, identity in snapshot.identities.items()}
    assignment_by_tid = {item.tracklet_uid: item for item in (assigned or ())}
    rows: list[dict[str, Any]] = []
    if scores is None:
        return rows
    for i, tid in enumerate(scores.tracklet_uids):
        order = np.argsort(-scores.values[i], kind="stable")
        finite = [j for j in order if np.isfinite(scores.values[i, j])]
        top = finite[0] if finite else None
        top_uid = scores.identity_uids[top] if top is not None else None
        row: dict[str, Any] = {"observation_uid": tid, "tracklet_uid": tid,
                               "persistent_uid": assignment_by_tid.get(tid).persistent_uid
                               if tid in assignment_by_tid else top_uid,
                               "top1_persistent_uid": top_uid,
                               "top1_label": labels.get(top_uid) if top_uid is not None else None,
                               "top1_score": float(scores.values[i, top]) if top is not None else None}
        if threshold is not None:
            row["threshold"] = float(threshold)
        if tid in assignment_by_tid:
            row.update({"status": assignment_by_tid[tid].status,
                        "reasons": list(assignment_by_tid[tid].reasons),
                        "candidate_uids": list(assignment_by_tid[tid].candidate_uids),
                        "candidate_scores": list(assignment_by_tid[tid].candidate_scores)})
        rows.append(row)
    return rows


def _truth_rows_for_samples(samples: Sequence[GerbilInstanceSample],
                            truth_by_observation: Mapping[str, Mapping[str, Any]],
                            session_uids: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        if sample.session_uid not in session_uids:
            continue
        truth = truth_by_observation.get(sample.observation_uid)
        if truth is None:
            continue
        row = dict(truth)
        row["observation_uid"] = sample.observation_uid
        row["session_uid"] = sample.session_uid
        rows.append(row)
    return rows


def _predicted_query_samples(
    source_samples: Sequence[GerbilInstanceSample],
    predictions: Sequence[PredictedPoseInstance],
    session_uids: set[str],
) -> list[GerbilInstanceSample]:
    """Attach detector instances to neutral frame images without GT matching."""
    frame_images: dict[tuple[str, int], Any] = {}
    for sample in source_samples:
        frame_images.setdefault((sample.session_uid, int(sample.frame_index)), sample.image)
    output: list[GerbilInstanceSample] = []
    for prediction in predictions:
        if prediction.session_uid not in session_uids:
            continue
        image = frame_images.get((prediction.session_uid, int(prediction.frame_index)))
        if image is None:
            # No image means the prediction cannot be passed to the identity
            # model; it is retained as an explicit parser/data blocker by the
            # caller rather than being silently paired to another frame.
            continue
        output.append(GerbilInstanceSample(
            observation_uid=prediction.prediction_uid,
            session_uid=prediction.session_uid,
            frame_index=prediction.frame_index,
            timestamp_s=prediction.timestamp_s,
            image=image,
            bbox_xyxy=prediction.bbox_xyxy,
            keypoints_xy=prediction.keypoints_xy,
            keypoint_scores=prediction.keypoint_scores,
            keypoint_valid=prediction.keypoint_valid,
            tracklet_uid=prediction.local_track_uid or prediction.prediction_uid,
        ))
    return output


def run_b0_strict(
    samples: Sequence[GerbilInstanceSample],
    truth_by_observation: Mapping[str, Mapping[str, Any]],
    encoder: Any,
    gallery_store: GalleryStore,
    *,
    reference_sessions: Sequence[str],
    development_sessions: Sequence[str],
    sealed_test_sessions: Sequence[str],
    input_mode: str = "oracle",
    predicted_pose: Mapping[str, Mapping[str, Any]] | None = None,
    predicted_instances: Sequence[PredictedPoseInstance] | None = None,
    matching_config: IdentityMatchingConfig | None = None,
    cohort_uid: str = "gerbils-b0-strict",
    protocol_id: str | None = None,
    batch_size: int = 32,
    pose_match_threshold: float = 0.25,
) -> GerbilB0Result:
    """Run B0 with a frozen protocol and development-only threshold selection.

    The reference gallery is created once.  Development rows can influence
    only the rejection threshold; sealed rows are touched only for the final
    fixed assignment/evaluation.  No truth is passed to the encoder or
    matcher, and no session role is inferred inside this strict entry point.
    """
    config = matching_config or IdentityMatchingConfig()
    if config.mode != "global_only":
        return GerbilB0Result("FAILED_MATCHING_CONFIG", input_mode, "H_oracle_reference", tuple(reference_sessions), {}, (), {}, None,
                              "B0 strict requires matching.mode=global_only", tuple(reference_sessions),
                              tuple(development_sessions), tuple(sealed_test_sessions), protocol_id=protocol_id)
    refs = tuple(dict.fromkeys(str(value) for value in reference_sessions))
    dev = tuple(dict.fromkeys(str(value) for value in development_sessions))
    sealed = tuple(dict.fromkeys(str(value) for value in sealed_test_sessions))
    role_sets = [set(refs), set(dev), set(sealed)]
    if any(not role for role in role_sets):
        return GerbilB0Result("BLOCKED_MISSING_PROTOCOL_ROLE", input_mode, "H_oracle_reference", refs, {}, (), {}, None,
                              "reference/development/sealed session roles must all be non-empty",
                              refs, dev, sealed, protocol_id=protocol_id)
    if set(refs) & set(dev) or set(refs) & set(sealed) or set(dev) & set(sealed):
        return GerbilB0Result("FAILED_PROTOCOL_OVERLAP", input_mode, "H_oracle_reference", refs, {}, (), {}, None,
                              "session roles overlap", refs, dev, sealed, protocol_id=protocol_id)
    anchors = select_oracle_reference_anchors(samples, truth_by_observation, s0_sessions=refs)
    anchor_counts = {key: len(value) for key, value in anchors.items()}
    if not anchors:
        return GerbilB0Result("BLOCKED_MISSING_S0_INPUT", input_mode, "H_oracle_reference", refs, anchor_counts, (), {}, None,
                              "reference sessions have no finite visible anchor crops", refs, dev, sealed,
                              protocol_id=protocol_id)
    missing_labels = set(("female", "male", "pup shaved", "pup unshaved")) - set(anchors)
    if missing_labels:
        return GerbilB0Result("BLOCKED_INCOMPLETE_REFERENCE_LABELS", input_mode, "H_oracle_reference", refs, anchor_counts,
                              (), {}, None, f"reference gallery missing labels: {sorted(missing_labels)}", refs, dev, sealed,
                              protocol_id=protocol_id)
    try:
        snapshot, _ = _gallery_from_oracle(samples, truth_by_observation, anchors, encoder, gallery_store, cohort_uid)
    except Exception as exc:
        return GerbilB0Result("FAILED", input_mode, "H_oracle_reference", refs, anchor_counts, (), {}, None,
                              f"{type(exc).__name__}: {exc}", refs, dev, sealed, protocol_id=protocol_id)

    all_query_sessions = set(dev) | set(sealed)
    predicted_instance_mode = input_mode == "predicted" and predicted_instances is not None
    if predicted_instance_mode:
        query_samples = _predicted_query_samples(samples, list(predicted_instances or ()), all_query_sessions)
    else:
        query_samples = [sample for sample in samples if sample.session_uid in all_query_sessions]
    if not query_samples:
        return GerbilB0Result("BLOCKED_MISSING_QUERY_SESSIONS", input_mode, "H_oracle_reference", refs, anchor_counts, (), {},
                              snapshot.version, "no samples belong to development/sealed protocol roles", refs, dev, sealed,
                              protocol_id=protocol_id)
    if predicted_instance_mode:
        query_boxes = {sample.observation_uid: np.asarray(sample.bbox_xyxy, dtype=np.float32)
                       for sample in query_samples if sample.bbox_xyxy is not None}
    else:
        query_boxes = _query_boxes(query_samples, truth_by_observation, all_query_sessions,
                                   input_mode=input_mode, predicted_pose=predicted_pose)
    tracklets, _, scores = _score_rows(query_samples, query_boxes, encoder, snapshot, batch_size=batch_size)
    track_by_uid = {track.tracklet_uid: track for track in tracklets}
    label_mapping = _identity_label_mapping(snapshot)
    known_labels = tuple(sorted(set(label_mapping.mapping.values())))
    if not known_labels:
        return GerbilB0Result("BLOCKED_INCOMPLETE_REFERENCE_LABELS", input_mode, "H_oracle_reference", refs, anchor_counts, (), {},
                              snapshot.version, "gallery identities have no reference group labels", refs, dev, sealed,
                              protocol_id=protocol_id)

    # Development calibration uses every labeled truth instance in the role.
    if predicted_instance_mode:
        # Geometry is evaluated only after prediction/descriptor inference;
        # these maps are never used to create crops or identity descriptors.
        source_truth = _truth_rows_for_samples(samples, truth_by_observation, all_query_sessions)
        truth_by_frame: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for row in source_truth:
            sample = next((item for item in samples if item.observation_uid == row["observation_uid"]), None)
            if sample is not None:
                truth_by_frame[(sample.session_uid, sample.frame_index)].append(row)
        geometry = PoseInstanceMatcher()
        pred_object_by_uid = {item.prediction_uid: item for item in (predicted_instances or ())}
        dev_geometry: dict[str, str] = {}
        sealed_geometry: dict[str, str] = {}
        for role, role_set, target in (("development", set(dev), dev_geometry), ("sealed_test", set(sealed), sealed_geometry)):
            frame_keys = sorted({key for key in truth_by_frame if key[0] in role_set})
            for key in frame_keys:
                preds = [item for item in (predicted_instances or ()) if (item.session_uid, item.frame_index) == key]
                matches, _, _ = geometry.match_frame(preds, truth_by_frame[key],
                                                      normalized_distance_threshold=pose_match_threshold)
                target.update({item.prediction_uid: item.truth_observation_uid for item in matches})
        dev_truth = []
        truth_by_uid = {row["observation_uid"]: row for row in source_truth}
        for prediction_uid, truth_uid in sorted(dev_geometry.items()):
            row = dict(truth_by_uid[truth_uid]); row["observation_uid"] = prediction_uid
            dev_truth.append(row)
        # Include unmatched development GT instances as explicit unknowns for
        # rejection calibration.  Their score is filled below with -1.
        matched_dev_truth_uids = set(dev_geometry.values())
        for row in source_truth:
            if row["session_uid"] in set(dev) and row["observation_uid"] not in matched_dev_truth_uids:
                row = dict(row); row["observation_uid"] = f"unmatched_gt:{row['observation_uid']}"
                dev_truth.append(row)
    else:
        dev_geometry = {}
        sealed_geometry = {}
        dev_truth = _truth_rows_for_samples(query_samples, truth_by_observation, set(dev))
    score_by_tid: dict[str, tuple[float, str | None]] = {}
    if scores is not None:
        labels_by_uid = {uid: (identity.provenance.get("group_label")
                               if isinstance(identity.provenance, dict) else None)
                         for uid, identity in snapshot.identities.items()}
        for i, tid in enumerate(scores.tracklet_uids):
            finite = [j for j in np.argsort(-scores.values[i], kind="stable") if np.isfinite(scores.values[i, j])]
            if finite:
                j = finite[0]
                score_by_tid[tid] = (float(scores.values[i, j]), labels_by_uid.get(scores.identity_uids[j]))
    calibration: ThresholdCalibrationResult | None = None
    if config.threshold_calibration == "development":
        dev_scores: list[float] = []
        dev_preds: list[str | None] = []
        dev_targets: list[str] = []
        for row in dev_truth:
            score, label = score_by_tid.get(row["observation_uid"], (-1.0, None))
            dev_scores.append(score)
            dev_preds.append(label)
            dev_targets.append(str(row.get("gt_identity", row.get("gt_id"))))
        if not dev_targets:
            return GerbilB0Result("BLOCKED_MISSING_DEVELOPMENT_TRUTH", input_mode, "H_oracle_reference", refs, anchor_counts,
                                  (), {}, snapshot.version, "development role has no evaluator truth", refs, dev, sealed,
                                  protocol_id=protocol_id)
        calibration = calibrate_accept_threshold(np.asarray(dev_scores), dev_preds, dev_targets,
                                                 identity_labels=known_labels, split_role="development")
        threshold = calibration.threshold
    elif config.threshold_calibration == "fixed":
        if config.accept_threshold is None:
            return GerbilB0Result("BLOCKED_MISSING_THRESHOLD", input_mode, "H_oracle_reference", refs, anchor_counts, (), {},
                                  snapshot.version, "fixed threshold calibration requires accept_threshold", refs, dev, sealed,
                                  protocol_id=protocol_id)
        threshold = float(config.accept_threshold)
    else:
        if config.accept_threshold is None:
            return GerbilB0Result("BLOCKED_MISSING_THRESHOLD", input_mode, "H_oracle_reference", refs, anchor_counts, (), {},
                                  snapshot.version, "threshold_calibration=none requires accept_threshold", refs, dev, sealed,
                                  protocol_id=protocol_id)
        threshold = float(config.accept_threshold)
    policy = MatchingPolicy(accept_threshold=threshold, top_k=config.top_k, solver=config.solver,
                            model_version=snapshot.fingerprint, gallery_version=snapshot.version)
    dev_tracks = [track for track in tracklets if track.session_uid in set(dev)]
    sealed_tracks = [track for track in tracklets if track.session_uid in set(sealed)]
    # A score matrix is sliced by role before assignment.  There is no second
    # gallery fit or re-enrollment; each role uses the same frozen snapshot.
    def _slice_scores(selected: Sequence[LocalTracklet]):
        if scores is None:
            return None
        indices = [scores.tracklet_uids.index(track.tracklet_uid) for track in selected]
        return type(scores)(tuple(track.tracklet_uid for track in selected), scores.identity_uids,
                            scores.values[np.asarray(indices, dtype=int)] if indices else np.zeros((0, len(scores.identity_uids)), dtype=np.float32))
    matcher = PersistentMatcher()
    dev_assigned = matcher.assign(_slice_scores(dev_tracks), ConflictGraphBuilder().build(dev_tracks), policy) if dev_tracks else []
    sealed_assigned = matcher.assign(_slice_scores(sealed_tracks), ConflictGraphBuilder().build(sealed_tracks), policy) if sealed_tracks else []
    def _rows(items: Sequence[Any]) -> tuple[dict[str, Any], ...]:
        rows: list[dict[str, Any]] = []
        for item in items:
            row = {"tracklet_uid": item.tracklet_uid,
                   "persistent_uid": item.persistent_uid, "status": item.status,
                   "candidate_uids": list(item.candidate_uids), "candidate_scores": list(item.candidate_scores),
                   "reasons": list(item.reasons), "gallery_version": item.gallery_version,
                   "model_version": item.model_version,
                   "session_uid": track_by_uid[item.tracklet_uid].session_uid}
            # Predicted rows remain keyed by detector-owned prediction_uid even
            # when geometry did not locate them.  In particular, never emit a
            # prediction UID in the observation_uid slot: the evaluator must
            # only resolve a truth observation through an explicit
            # geometry-only evaluator_match.
            if predicted_instance_mode:
                row["prediction_uid"] = item.tracklet_uid
            else:
                row["observation_uid"] = item.tracklet_uid
            if predicted_instance_mode and item.tracklet_uid in sealed_geometry:
                row["evaluator_match"] = {"truth_observation_uid": sealed_geometry[item.tracklet_uid]}
            rows.append(row)
        return tuple(rows)
    sealed_rows = _rows(sealed_assigned)
    sealed_truth = _truth_rows_for_samples(samples, truth_by_observation, set(sealed))
    metrics_bundle = PersistentIDEvaluator().evaluate_labels(sealed_rows, sealed_truth, label_mapping,
                                                              known_identity_labels=known_labels)
    metrics = {"status": metrics_bundle.status, "counts": metrics_bundle.counts,
               "metrics": metrics_bundle.metrics, "errors": metrics_bundle.errors}
    return GerbilB0Result("SUCCEEDED", input_mode, "H_oracle_reference", refs, anchor_counts, sealed_rows, metrics,
                          snapshot.version, None, refs, dev, sealed, threshold,
                          calibration.to_dict() if calibration else None, _rows(dev_assigned), protocol_id)


def run_b0_oracle_crop_diagnostic(samples: Sequence[GerbilInstanceSample],
                                  truth_by_observation: Mapping[str, Mapping[str, Any]], encoder: Any,
                                  gallery_store: GalleryStore, session_rows: Sequence[Mapping[str, Any]], *,
                                  cohort_uid: str = "gerbils-b0-oracle") -> GerbilB0Result:
    """Legacy compatibility wrapper; strict callers must provide protocol roles."""
    enriched_truth = [dict(truth_by_observation[uid], observation_uid=uid, session_uid=sample.session_uid)
                      for sample in samples if (uid := sample.observation_uid) in truth_by_observation]
    s0 = select_s0_sessions(session_rows, enriched_truth)
    remaining = [str(row.get("session_uid")) for row in session_rows if str(row.get("session_uid")) not in set(s0)]
    # This wrapper is retained solely for old receipts/tests and is explicitly
    # not the strict protocol runner.
    return run_b0_strict(samples, truth_by_observation, encoder, gallery_store,
                         reference_sessions=s0,
                         development_sessions=tuple(remaining[:max(1, len(remaining) // 2)]),
                         sealed_test_sessions=tuple(remaining[max(1, len(remaining) // 2):]),
                         input_mode="oracle", cohort_uid=cohort_uid)


def run_b0_predicted_pose(samples: Sequence[GerbilInstanceSample],
                          truth_by_observation: Mapping[str, Mapping[str, Any]],
                          predicted_pose: Mapping[str, Mapping[str, Any]], encoder: Any,
                          gallery_store: GalleryStore, session_rows: Sequence[Mapping[str, Any]], *,
                          cohort_uid: str = "gerbils-b0-predicted") -> GerbilB0Result:
    """Legacy compatibility wrapper around the strict predicted-pose runner."""
    enriched_truth = [dict(truth_by_observation[uid], observation_uid=uid, session_uid=sample.session_uid)
                      for sample in samples if (uid := sample.observation_uid) in truth_by_observation]
    s0 = select_s0_sessions(session_rows, enriched_truth)
    remaining = [str(row.get("session_uid")) for row in session_rows if str(row.get("session_uid")) not in set(s0)]
    return run_b0_strict(samples, truth_by_observation, encoder, gallery_store,
                         reference_sessions=s0,
                         development_sessions=tuple(remaining[:max(1, len(remaining) // 2)]),
                         sealed_test_sessions=tuple(remaining[max(1, len(remaining) // 2):]),
                         input_mode="predicted", predicted_pose=predicted_pose,
                         cohort_uid=cohort_uid)


__all__ = ["LazyRGBImage", "GerbilInstanceSample", "GerbilB0Result", "bbox_from_keypoints", "select_s0_sessions",
           "select_oracle_reference_anchors", "run_b0_strict", "run_b0_oracle_crop_diagnostic", "run_b0_predicted_pose"]

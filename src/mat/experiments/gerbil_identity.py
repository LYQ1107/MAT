"""File-backed gerbil B0 identity diagnostics.

The model-facing side of this module consumes only neutral observation rows and
pose/bounding-box predictions.  Provider identity/keypoint truth is accepted
through a separate evaluator argument and is never attached to a model row.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from mat.core.errors import ValidationError
from mat.core.types import DescriptorBatch, IdentityDescriptor, LocalTracklet
from mat.enrollment.base import ReferenceVerification
from mat.enrollment.manual import ManualRegistrar
from mat.identity.gallery import GalleryStore
from mat.identity.conflicts import ConflictGraphBuilder
from mat.identity.matching import MatchingPolicy, PersistentMatcher
from mat.data.sanitize import FORBIDDEN_MODEL_FIELDS


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
            array = np.asarray(image.convert("RGB"), dtype=np.uint8)
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


def _evaluate(assignments: Sequence[Any], truth_by_observation: Mapping[str, Mapping[str, Any]],
              identity_names: Mapping[str, str]) -> dict[str, Any]:
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    correct = known = unknown = 0
    for assignment in assignments:
        uid = assignment.tracklet_uid
        truth = truth_by_observation.get(uid, {})
        target = truth.get("gt_identity")
        if target is None:
            continue
        known += 1
        predicted = identity_names.get(assignment.persistent_uid, "unknown") if assignment.persistent_uid else "unknown"
        confusion[str(target)][str(predicted)] += 1
        if predicted == target:
            correct += 1
        if predicted == "unknown":
            unknown += 1
    accuracy = correct / known if known else None
    # A conservative micro-F1 over known identities; unknown predictions are
    # counted as false negatives and never inflate identity performance.
    f1 = (2 * correct / (known + correct - unknown)) if known and (known + correct - unknown) else None
    return {"known_instances": known, "correct": correct, "unknown": unknown,
            "accuracy": accuracy, "f1": f1,
            "confusion": {key: dict(value) for key, value in sorted(confusion.items())}}


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

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def run_b0_oracle_crop_diagnostic(samples: Sequence[GerbilInstanceSample],
                                  truth_by_observation: Mapping[str, Mapping[str, Any]], encoder: Any,
                                  gallery_store: GalleryStore, session_rows: Sequence[Mapping[str, Any]], *,
                                  cohort_uid: str = "gerbils-b0-oracle") -> GerbilB0Result:
    """Run the first strict H_oracle_reference diagnostic with evaluator crops."""
    enriched_truth = [dict(truth_by_observation[uid], observation_uid=uid, session_uid=sample.session_uid)
                      for uid, sample in ((sample.observation_uid, sample) for sample in samples)
                      if uid in truth_by_observation]
    s0 = select_s0_sessions(session_rows, enriched_truth)
    if not s0:
        return GerbilB0Result("BLOCKED_MISSING_S0_INPUT", "oracle_crop_diagnostic", "H_oracle_reference",
                              (), {}, (), {}, None, "no session rows with a provider-labeled S0")
    anchors = select_oracle_reference_anchors(samples, truth_by_observation, s0_sessions=s0)
    if not anchors:
        return GerbilB0Result("BLOCKED_MISSING_S0_INPUT", "oracle_crop_diagnostic", "H_oracle_reference",
                              s0, {}, (), {}, None, "S0 has no finite visible anchor crops")
    try:
        snapshot, sample_by_uid = _gallery_from_oracle(samples, truth_by_observation, anchors, encoder, gallery_store, cohort_uid)
    except Exception as exc:
        return GerbilB0Result("FAILED", "oracle_crop_diagnostic", "H_oracle_reference", s0,
                              {key: len(value) for key, value in anchors.items()}, (), {}, None,
                              f"{type(exc).__name__}: {exc}")
    query = [sample for sample in samples if sample.session_uid not in set(s0)]
    if not query:
        return GerbilB0Result("BLOCKED_MISSING_QUERY_SESSIONS", "oracle_crop_diagnostic", "H_oracle_reference",
                              s0, {key: len(value) for key, value in anchors.items()}, (), {}, snapshot.version,
                              "no labeled session remains outside the fixed S0 reference")
    boxes = {}
    for sample in query:
        truth = truth_by_observation.get(sample.observation_uid, {})
        box = bbox_from_keypoints(truth.get("gt_keypoints", sample.keypoints_xy), truth.get("gt_visibility"),
                                  image_shape=sample.image.shape[:2])
        if box is not None:
            boxes[sample.observation_uid] = box
    descriptors, ordered = _descriptor_rows(query, encoder, boxes)
    tracklets = [LocalTracklet(sample.observation_uid, sample.session_uid, "camera", [sample.observation_uid],
                               [(sample.timestamp_s, sample.timestamp_s)]) for sample in ordered]
    scores = PersistentMatcher().score(tracklets, descriptors, snapshot)
    assigned = PersistentMatcher().assign(scores, ConflictGraphBuilder().build(tracklets),
                                           MatchingPolicy(model_version=snapshot.fingerprint))
    names = {identity.identity_uid: str(identity.provenance.get("group_label", "unknown"))
             for identity in snapshot.identities.values()}
    metrics = _evaluate(assigned, truth_by_observation, names)
    return GerbilB0Result("SUCCEEDED", "oracle_crop_diagnostic", "H_oracle_reference", s0,
                          {key: len(value) for key, value in anchors.items()}, tuple(item.__dict__ for item in assigned),
                          metrics, snapshot.version)


def run_b0_predicted_pose(samples: Sequence[GerbilInstanceSample],
                          truth_by_observation: Mapping[str, Mapping[str, Any]],
                          predicted_pose: Mapping[str, Mapping[str, Any]], encoder: Any,
                          gallery_store: GalleryStore, session_rows: Sequence[Mapping[str, Any]], *,
                          cohort_uid: str = "gerbils-b0-predicted") -> GerbilB0Result:
    """Run B0 using predicted pose boxes; truth is evaluator-only."""
    enriched_truth = [dict(truth_by_observation[uid], observation_uid=uid, session_uid=sample.session_uid)
                      for uid, sample in ((sample.observation_uid, sample) for sample in samples)
                      if uid in truth_by_observation]
    s0 = select_s0_sessions(session_rows, enriched_truth)
    anchors = select_oracle_reference_anchors(samples, truth_by_observation, s0_sessions=s0)
    if not anchors:
        return GerbilB0Result("BLOCKED_MISSING_S0_INPUT", "predicted_pose", "H_oracle_reference", s0, {}, (), {}, None,
                              "S0 oracle reference anchors unavailable")
    try:
        snapshot, _ = _gallery_from_oracle(samples, truth_by_observation, anchors, encoder, gallery_store, cohort_uid)
    except Exception as exc:
        return GerbilB0Result("FAILED", "predicted_pose", "H_oracle_reference", s0,
                              {key: len(value) for key, value in anchors.items()}, (), {}, None,
                              f"{type(exc).__name__}: {exc}")
    query = [sample for sample in samples if sample.session_uid not in set(s0)]
    if not query:
        return GerbilB0Result("BLOCKED_MISSING_QUERY_SESSIONS", "predicted_pose", "H_oracle_reference",
                              s0, {key: len(value) for key, value in anchors.items()}, (), {}, snapshot.version,
                              "no query session remains outside the fixed S0 reference")
    boxes: dict[str, np.ndarray] = {}
    for sample in query:
        prediction = predicted_pose.get(sample.observation_uid, {})
        box = prediction.get("bbox_xyxy")
        if box is None and prediction.get("keypoints_xy") is not None:
            box = bbox_from_keypoints(prediction["keypoints_xy"], prediction.get("keypoint_valid"),
                                      image_shape=sample.image.shape[:2])
        if box is not None:
            box = np.asarray(box, dtype=np.float32)
            if box.shape == (4,) and np.isfinite(box).all() and box[2] > box[0] and box[3] > box[1]:
                boxes[sample.observation_uid] = box
    descriptors, ordered = _descriptor_rows(query, encoder, boxes)
    tracklets = [LocalTracklet(sample.observation_uid, sample.session_uid, "camera", [sample.observation_uid],
                               [(sample.timestamp_s, sample.timestamp_s)]) for sample in ordered]
    scores = PersistentMatcher().score(tracklets, descriptors, snapshot)
    assigned = PersistentMatcher().assign(scores, ConflictGraphBuilder().build(tracklets),
                                           MatchingPolicy(model_version=snapshot.fingerprint))
    names = {identity.identity_uid: str(identity.provenance.get("group_label", "unknown"))
             for identity in snapshot.identities.values()}
    metrics = _evaluate(assigned, truth_by_observation, names)
    return GerbilB0Result("SUCCEEDED", "predicted_pose", "H_oracle_reference", s0,
                          {key: len(value) for key, value in anchors.items()}, tuple(item.__dict__ for item in assigned),
                          metrics, snapshot.version)


__all__ = ["LazyRGBImage", "GerbilInstanceSample", "GerbilB0Result", "bbox_from_keypoints", "select_s0_sessions",
           "select_oracle_reference_anchors", "run_b0_oracle_crop_diagnostic", "run_b0_predicted_pose"]

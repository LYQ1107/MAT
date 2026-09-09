"""File-backed B1 global-plus-pose-part identity diagnostics."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence
import math

import numpy as np

from mat.config.identity import IdentityMatchingConfig
from mat.core.errors import ValidationError
from mat.core.types import DescriptorBatch, IdentityDescriptor, LocalTracklet, PredictedPoseInstance, ScoreMatrix
from mat.enrollment.base import ReferenceVerification
from mat.enrollment.manual import ManualRegistrar
from mat.evaluation.fixed_identity import EnrollmentMapping, PersistentIDEvaluator
from mat.evaluation.instance_matching import PoseInstanceMatcher
from mat.identity.calibration import ThresholdCalibrationResult, calibrate_accept_threshold
from mat.identity.gallery import GalleryStore
from mat.identity.conflicts import ConflictGraphBuilder
from mat.identity.matching import MatchingPolicy
from mat.identity.part_matching import PartAwareStaticMatcher
from mat.models.part_encoder import PartAwareIdentityEncoder

from .gerbil_identity import (
    GerbilB0Result,
    GerbilInstanceSample,
    _predicted_query_samples,
    _truth_rows_for_samples,
    bbox_from_keypoints,
    select_oracle_reference_anchors,
)


def _part_descriptor_rows(
    samples: Sequence[GerbilInstanceSample],
    truth_by_observation: Mapping[str, Mapping[str, Any]],
    encoder: PartAwareIdentityEncoder,
    *,
    input_mode: str,
) -> tuple[dict[str, IdentityDescriptor], list[GerbilInstanceSample]]:
    """Encode one neutral/predicted crop at a time through the frozen encoder."""
    try:
        import torch
    except Exception as exc:  # pragma: no cover - production runtime only
        raise ValidationError("PyTorch is required for B1 part-aware inference") from exc
    descriptors: dict[str, IdentityDescriptor] = {}
    ordered: list[GerbilInstanceSample] = []
    for sample in samples:
        truth = truth_by_observation.get(sample.observation_uid, {})
        if sample.bbox_xyxy is not None:
            box = np.asarray(sample.bbox_xyxy, dtype=np.float32)
        elif input_mode == "oracle":
            box = bbox_from_keypoints(truth.get("gt_keypoints"), truth.get("gt_visibility"),
                                      image_shape=sample.image.shape[:2])
        else:
            box = None
        if box is None:
            continue
        if input_mode == "oracle":
            points = np.asarray(truth.get("gt_keypoints"), dtype=np.float32)
            valid = np.asarray(truth.get("gt_visibility"), dtype=bool)
            scores = valid.astype(np.float32)
        else:
            points = np.asarray(sample.keypoints_xy, dtype=np.float32)
            valid = np.asarray(sample.keypoint_valid, dtype=bool)
            scores = np.asarray(sample.keypoint_scores, dtype=np.float32)
            scores = np.nan_to_num(scores, nan=0.0)
        image = np.array(sample.image, dtype=np.uint8, copy=True)
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        encoded = encoder.encode(
            tensor,
            torch.as_tensor(box, dtype=torch.float32).reshape(1, 4),
            torch.as_tensor(points, dtype=torch.float32).unsqueeze(0),
            torch.as_tensor(scores, dtype=torch.float32).unsqueeze(0),
            torch.as_tensor(valid, dtype=torch.bool).unsqueeze(0),
        )
        if encoded.global_features.shape[0] != 1:
            raise ValidationError("part-aware identity encoder returned an unexpected row count")
        descriptors[sample.observation_uid] = IdentityDescriptor(
            encoded.global_features[0], encoded.part_features[0], encoded.part_valid[0],
            encoded.part_quality[0], encoded.encoder_fingerprint,
        )
        ordered.append(sample)
    return descriptors, ordered


def _part_gallery_from_oracle(
    samples: Sequence[GerbilInstanceSample],
    truth_by_observation: Mapping[str, Mapping[str, Any]],
    anchors: Mapping[str, Sequence[str]],
    encoder: PartAwareIdentityEncoder,
    gallery_store: GalleryStore,
    cohort_uid: str,
):
    sample_by_uid = {sample.observation_uid: sample for sample in samples}
    anchor_samples = [sample_by_uid[uid] for values in anchors.values() for uid in values if uid in sample_by_uid]
    descriptors, ordered = _part_descriptor_rows(anchor_samples, truth_by_observation, encoder, input_mode="oracle")
    tracklets = [LocalTracklet(f"anchor:{sample.observation_uid}", sample.session_uid, "camera",
                               [sample.observation_uid], [(sample.timestamp_s, sample.timestamp_s)]) for sample in ordered]
    groups = {identity: tuple(f"anchor:{uid}" for uid in values if uid in descriptors)
              for identity, values in anchors.items()}
    groups = {identity: refs for identity, refs in groups.items() if refs}
    enrollment = ManualRegistrar().build(type("Reference", (), {
        "tracklets": tracklets,
        "descriptors": {f"anchor:{uid}": descriptors[uid] for uid in descriptors},
        "session_uid": "+".join(sorted({sample.session_uid for sample in ordered})),
        "cohort_uid": cohort_uid,
    })(), ReferenceVerification(groups=groups, provenance="oracle_reference"), cohort_uid=cohort_uid)
    return gallery_store.create(enrollment)


def _label_mapping(snapshot) -> EnrollmentMapping:
    mapping = {
        str(uid): str(identity.provenance.get("group_label"))
        for uid, identity in snapshot.identities.items()
        if isinstance(identity.provenance, dict) and identity.provenance.get("group_label") is not None
    }
    result = EnrollmentMapping(mapping)
    result.freeze()
    return result


def _geometry_links(samples: Sequence[GerbilInstanceSample], truth: Mapping[str, Mapping[str, Any]],
                    predictions: Sequence[PredictedPoseInstance], role_sessions: set[str],
                    threshold: float) -> dict[str, str]:
    truth_by_frame: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    sample_by_uid = {sample.observation_uid: sample for sample in samples}
    for sample in samples:
        if sample.session_uid not in role_sessions:
            continue
        row = truth.get(sample.observation_uid)
        if row is None:
            continue
        item = dict(row)
        item.update({"observation_uid": sample.observation_uid, "session_uid": sample.session_uid,
                     "frame_index": sample.frame_index})
        truth_by_frame[(sample.session_uid, sample.frame_index)].append(item)
    links: dict[str, str] = {}
    matcher = PoseInstanceMatcher()
    for key in sorted(truth_by_frame):
        frame_predictions = [item for item in predictions
                             if (item.session_uid, item.frame_index) == key]
        matches, _, _ = matcher.match_frame(frame_predictions, truth_by_frame[key],
                                             normalized_distance_threshold=threshold)
        links.update({item.prediction_uid: item.truth_observation_uid for item in matches})
    return links


def _top1(scores: ScoreMatrix, snapshot) -> dict[str, tuple[float, str | None]]:
    labels = {uid: identity.provenance.get("group_label")
              for uid, identity in snapshot.identities.items()
              if isinstance(identity.provenance, dict)}
    result: dict[str, tuple[float, str | None]] = {}
    for i, tid in enumerate(scores.tracklet_uids):
        finite = [j for j in np.argsort(-scores.values[i], kind="stable") if np.isfinite(scores.values[i, j])]
        if finite:
            j = finite[0]
            result[tid] = (float(scores.values[i, j]), labels.get(scores.identity_uids[j]))
    return result


def _slice(scores: ScoreMatrix, tracks: Sequence[LocalTracklet]) -> ScoreMatrix:
    indices = [scores.tracklet_uids.index(track.tracklet_uid) for track in tracks]
    values = scores.values[np.asarray(indices, dtype=int)] if indices else np.zeros((0, len(scores.identity_uids)), dtype=np.float32)
    return ScoreMatrix(tuple(track.tracklet_uid for track in tracks), scores.identity_uids, values)


def _assignment_rows(items: Sequence[Any], track_by_uid: Mapping[str, LocalTracklet], *,
                     geometry_links: Mapping[str, str] | None = None,
                     predicted_mode: bool = False) -> tuple[dict[str, Any], ...]:
    rows = []
    geometry_links = geometry_links or {}
    for item in items:
        row = {"tracklet_uid": item.tracklet_uid, "persistent_uid": item.persistent_uid,
               "status": item.status, "candidate_uids": list(item.candidate_uids),
               "candidate_scores": list(item.candidate_scores), "reasons": list(item.reasons),
               "gallery_version": item.gallery_version, "model_version": item.model_version,
               "session_uid": track_by_uid[item.tracklet_uid].session_uid}
        if predicted_mode:
            row["prediction_uid"] = item.tracklet_uid
            if item.tracklet_uid in geometry_links:
                row["evaluator_match"] = {"truth_observation_uid": geometry_links[item.tracklet_uid]}
        else:
            row["observation_uid"] = item.tracklet_uid
        rows.append(row)
    return tuple(rows)


def run_b1(
    *,
    samples: Sequence[GerbilInstanceSample],
    truth: Mapping[str, Mapping[str, Any]],
    encoder: PartAwareIdentityEncoder,
    gallery_store: GalleryStore,
    reference_sessions: Sequence[str],
    development_sessions: Sequence[str],
    sealed_test_sessions: Sequence[str],
    identity_labels: Sequence[str],
    input_mode: str = "oracle",
    predicted_instances: Sequence[PredictedPoseInstance] | None = None,
    matching_config: IdentityMatchingConfig | None = None,
    global_weight_candidates: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    cohort_uid: str = "gerbils-b1-strict",
    protocol_id: str | None = None,
    pose_match_threshold: float = 0.25,
) -> GerbilB0Result:
    config = matching_config or IdentityMatchingConfig(mode="part_aware")
    refs = tuple(dict.fromkeys(str(v) for v in reference_sessions))
    dev = tuple(dict.fromkeys(str(v) for v in development_sessions))
    sealed = tuple(dict.fromkeys(str(v) for v in sealed_test_sessions))
    if config.mode not in {"part_aware", "global_only"}:
        return GerbilB0Result("FAILED_MATCHING_CONFIG", input_mode, "H_oracle_reference", refs, {}, (), {}, None,
                              "B1 matching mode must be part_aware", refs, dev, sealed, protocol_id=protocol_id)
    if not refs or not dev or not sealed or (set(refs) & set(dev)) or (set(refs) & set(sealed)) or (set(dev) & set(sealed)):
        return GerbilB0Result("FAILED_PROTOCOL_OVERLAP", input_mode, "H_oracle_reference", refs, {}, (), {}, None,
                              "B1 protocol roles must be non-empty and disjoint", refs, dev, sealed, protocol_id=protocol_id)
    anchors = select_oracle_reference_anchors(samples, truth, s0_sessions=refs)
    anchor_counts = {key: len(value) for key, value in anchors.items()}
    missing = set(identity_labels) - set(anchors)
    if missing:
        return GerbilB0Result("BLOCKED_INCOMPLETE_REFERENCE_LABELS", input_mode, "H_oracle_reference", refs, anchor_counts,
                              (), {}, None, f"reference gallery missing labels: {sorted(missing)}", refs, dev, sealed,
                              protocol_id=protocol_id)
    try:
        snapshot = _part_gallery_from_oracle(samples, truth, anchors, encoder, gallery_store, cohort_uid)
    except Exception as exc:
        return GerbilB0Result("FAILED", input_mode, "H_oracle_reference", refs, anchor_counts, (), {}, None,
                              f"{type(exc).__name__}: {exc}", refs, dev, sealed, protocol_id=protocol_id)
    predicted_mode = input_mode == "predicted" and predicted_instances is not None
    query_sessions = set(dev) | set(sealed)
    query_samples = (_predicted_query_samples(samples, list(predicted_instances or ()), query_sessions)
                     if predicted_mode else [sample for sample in samples if sample.session_uid in query_sessions])
    if not query_samples:
        return GerbilB0Result("BLOCKED_MISSING_QUERY_SESSIONS", input_mode, "H_oracle_reference", refs, anchor_counts, (), {},
                              snapshot.version, "no query samples for B1 roles", refs, dev, sealed, protocol_id=protocol_id)
    descriptors, ordered = _part_descriptor_rows(query_samples, truth, encoder, input_mode=input_mode)
    tracks = [LocalTracklet(sample.observation_uid, sample.session_uid, "camera", [sample.observation_uid],
                            [(sample.timestamp_s, sample.timestamp_s)]) for sample in ordered]
    if not tracks:
        return GerbilB0Result("BLOCKED_MISSING_QUERY_POSE", input_mode, "H_oracle_reference", refs, anchor_counts, (), {},
                              snapshot.version, "no finite query pose crops", refs, dev, sealed, protocol_id=protocol_id)
    track_by_uid = {track.tracklet_uid: track for track in tracks}
    mapping = _label_mapping(snapshot)
    labels = tuple(str(value) for value in identity_labels)
    if predicted_mode:
        dev_links = _geometry_links(samples, truth, list(predicted_instances or ()), set(dev), pose_match_threshold)
        sealed_links = _geometry_links(samples, truth, list(predicted_instances or ()), set(sealed), pose_match_threshold)
    else:
        dev_links, sealed_links = {}, {}
    matcher = PartAwareStaticMatcher()
    weights = tuple(float(v) for v in global_weight_candidates)
    if not weights:
        raise ValidationError("B1 global_weight_candidates must not be empty")
    if any(not math.isfinite(v) or not 0.0 <= v <= 1.0 for v in weights):
        raise ValidationError("B1 global_weight_candidates must lie in [0,1]")
    best = None
    best_scores = None
    best_calibration: ThresholdCalibrationResult | None = None
    best_weight = None
    dev_truth_source = _truth_rows_for_samples(samples, truth, set(dev))
    truth_by_uid = {row["observation_uid"]: row for row in _truth_rows_for_samples(samples, truth, query_sessions)}
    for weight in sorted(set(weights)):
        scores = matcher.score(tracks, descriptors, snapshot, global_weight=weight)
        top1 = _top1(scores, snapshot)
        calibration_scores, calibration_preds, calibration_truth = [], [], []
        if predicted_mode:
            for prediction_uid, truth_uid in sorted(dev_links.items()):
                if prediction_uid in top1 and truth_uid in truth_by_uid:
                    score, pred = top1[prediction_uid]
                    calibration_scores.append(score); calibration_preds.append(pred)
                    calibration_truth.append(str(truth_by_uid[truth_uid].get("gt_identity", truth_by_uid[truth_uid].get("gt_id"))))
            matched_truth = set(dev_links.values())
            for row in dev_truth_source:
                if row["observation_uid"] not in matched_truth:
                    calibration_scores.append(-1.0); calibration_preds.append(None)
                    calibration_truth.append(str(row.get("gt_identity", row.get("gt_id"))))
        else:
            for row in dev_truth_source:
                score, pred = top1.get(row["observation_uid"], (-1.0, None))
                calibration_scores.append(score); calibration_preds.append(pred)
                calibration_truth.append(str(row.get("gt_identity", row.get("gt_id"))))
        if not calibration_truth:
            continue
        if config.threshold_calibration == "development":
            calibration = calibrate_accept_threshold(np.asarray(calibration_scores), calibration_preds, calibration_truth,
                                                     identity_labels=labels, split_role="development")
            threshold = calibration.threshold
        elif config.accept_threshold is not None:
            calibration = None
            threshold = float(config.accept_threshold)
        else:
            return GerbilB0Result("BLOCKED_MISSING_THRESHOLD", input_mode, "H_oracle_reference", refs, anchor_counts, (), {},
                                  snapshot.version, "B1 requires development calibration or fixed accept_threshold", refs, dev, sealed,
                                  protocol_id=protocol_id)
        summary = calibration
        key = (
            float(summary.macro_f1) if summary and summary.macro_f1 is not None else -math.inf,
            -(float(summary.wrong_identity_rate) if summary and summary.wrong_identity_rate is not None else math.inf),
            float(summary.accepted_accuracy) if summary and summary.accepted_accuracy is not None else -math.inf,
            float(summary.accuracy) if summary and summary.accuracy is not None else -math.inf,
            float(weight),
        )
        if best is None or key > best:
            best, best_scores, best_calibration, best_weight = key, scores, calibration, weight
    if best_scores is None or best_weight is None:
        return GerbilB0Result("BLOCKED_MISSING_DEVELOPMENT_TRUTH", input_mode, "H_oracle_reference", refs, anchor_counts, (), {},
                              snapshot.version, "development rows could not calibrate B1", refs, dev, sealed, protocol_id=protocol_id)
    threshold = best_calibration.threshold if best_calibration is not None else float(config.accept_threshold)
    policy = MatchingPolicy(accept_threshold=threshold, top_k=config.top_k, solver=config.solver,
                            model_version=snapshot.fingerprint, gallery_version=snapshot.version)
    dev_tracks = [track for track in tracks if track.session_uid in set(dev)]
    sealed_tracks = [track for track in tracks if track.session_uid in set(sealed)]
    dev_assigned = matcher.assign(_slice(best_scores, dev_tracks), ConflictGraphBuilder().build(dev_tracks), policy) if dev_tracks else []
    sealed_assigned = matcher.assign(_slice(best_scores, sealed_tracks), ConflictGraphBuilder().build(sealed_tracks), policy) if sealed_tracks else []
    sealed_rows = _assignment_rows(sealed_assigned, track_by_uid,
                                   geometry_links=sealed_links if predicted_mode else {},
                                   predicted_mode=predicted_mode)
    sealed_truth = _truth_rows_for_samples(samples, truth, set(sealed))
    metrics_bundle = PersistentIDEvaluator().evaluate_labels(sealed_rows, sealed_truth, mapping,
                                                              known_identity_labels=labels)
    metrics = {"status": metrics_bundle.status, "counts": metrics_bundle.counts,
               "metrics": metrics_bundle.metrics, "errors": metrics_bundle.errors,
               "matcher": "part_static", "global_weight": best_weight,
               "calibration_source": "development" if best_calibration else "fixed"}
    return GerbilB0Result("SUCCEEDED", input_mode, "H_oracle_reference", refs, anchor_counts, sealed_rows, metrics,
                          snapshot.version, None, refs, dev, sealed, threshold,
                          (dict(best_calibration.to_dict(), global_weight=best_weight) if best_calibration else
                           {"global_weight": best_weight, "split_role": "fixed"}),
                          _assignment_rows(dev_assigned, track_by_uid,
                                          geometry_links=dev_links if predicted_mode else {},
                                          predicted_mode=predicted_mode),
                          protocol_id)


def run_b1_oracle_pose(**kwargs) -> GerbilB0Result:
    kwargs["input_mode"] = "oracle"
    return run_b1(**kwargs)


def run_b1_predicted_pose(**kwargs) -> GerbilB0Result:
    kwargs["input_mode"] = "predicted"
    return run_b1(**kwargs)


__all__ = ["run_b1", "run_b1_oracle_pose", "run_b1_predicted_pose"]

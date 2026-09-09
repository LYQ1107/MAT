"""Geometry-only matching of predicted and labeled pose instances.

This module is an evaluator primitive.  It runs *after* model inference and
identity assignment; identity labels are neither read nor used to construct
the cost matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import math

import numpy as np

from mat.core.errors import ValidationError
from mat.core.types import PredictedPoseInstance


@dataclass(frozen=True)
class PoseInstanceMatch:
    prediction_uid: str
    truth_observation_uid: str
    cost: float


def _truth_points(row: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    raw_points = row.get("gt_keypoints", row.get("keypoints_xy"))
    if raw_points is None:
        raise ValidationError("truth pose row is missing gt_keypoints/keypoints_xy")
    points = np.asarray(raw_points, dtype=np.float32)
    default_valid = np.isfinite(points).all(axis=1) if points.ndim == 2 and points.shape[1] == 2 else None
    raw_visibility = row.get("gt_visibility", row.get("keypoint_valid", default_valid))
    if raw_visibility is None:
        raise ValidationError("truth pose row is missing gt_visibility/keypoint_valid")
    visibility = np.asarray(raw_visibility, dtype=bool)
    if points.ndim != 2 or points.shape[1] != 2 or visibility.shape != (points.shape[0],):
        raise ValidationError("truth pose keypoints/visibility must have compatible Kx2/K shapes")
    valid = visibility & np.isfinite(points).all(axis=1)
    return points, valid


def _truth_frame(row: Mapping[str, Any]) -> int | None:
    value = row.get("frame_index", row.get("frame_idx"))
    return None if value is None else int(value)


def _truth_session(row: Mapping[str, Any]) -> str | None:
    value = row.get("session_uid")
    return None if value is None else str(value)


class PoseInstanceMatcher:
    """Match instances in one frame using normalized pose geometry only."""

    def match_frame(
        self,
        predictions: Sequence[PredictedPoseInstance],
        truth_instances: Sequence[Mapping[str, Any]],
        *,
        normalized_distance_threshold: float,
    ) -> tuple[list[PoseInstanceMatch], list[str], list[str]]:
        try:
            threshold = float(normalized_distance_threshold)
        except (TypeError, ValueError) as exc:
            raise ValidationError("normalized_distance_threshold must be finite and non-negative") from exc
        if not math.isfinite(threshold) or threshold < 0:
            raise ValidationError("normalized_distance_threshold must be finite and non-negative")
        predictions = list(predictions)
        truth_instances = list(truth_instances)
        pred_keys = [(str(item.session_uid), int(item.frame_index)) for item in predictions]
        if len(set(pred_keys)) > 1:
            raise ValidationError("match_frame predictions must belong to one session/frame")
        truth_keys = [(_truth_session(item), _truth_frame(item)) for item in truth_instances]
        if len({key for key in truth_keys if key != (None, None)}) > 1:
            raise ValidationError("match_frame truth instances must belong to one session/frame")
        if len({item.prediction_uid for item in predictions}) != len(predictions):
            raise ValidationError("prediction_uid values must be unique within a frame")
        truth_uids = [str(item.get("observation_uid")) for item in truth_instances]
        if any(item.get("observation_uid") is None for item in truth_instances):
            raise ValidationError("truth instances require observation_uid")
        if len(set(truth_uids)) != len(truth_uids):
            raise ValidationError("truth observation_uid values must be unique within a frame")
        if not predictions or not truth_instances:
            return [], [item.prediction_uid for item in predictions], truth_uids

        costs = np.full((len(predictions), len(truth_instances)), np.inf, dtype=np.float64)
        for i, prediction in enumerate(predictions):
            p_points = np.asarray(prediction.keypoints_xy, dtype=np.float32)
            p_valid = np.asarray(prediction.keypoint_valid, dtype=bool) & np.isfinite(p_points).all(axis=1)
            for j, truth in enumerate(truth_instances):
                t_points, t_valid = _truth_points(truth)
                if p_points.shape[0] != t_points.shape[0]:
                    continue
                common = p_valid & t_valid
                if int(common.sum()) < 2:
                    continue
                gt_visible = t_points[t_valid]
                low, high = gt_visible.min(axis=0), gt_visible.max(axis=0)
                diagonal = float(np.linalg.norm(high - low))
                distance = float(np.linalg.norm(p_points[common] - t_points[common], axis=1).mean())
                costs[i, j] = distance / max(diagonal, np.finfo(np.float64).eps)

        # scipy's Hungarian implementation expects finite values in practice;
        # use a sentinel larger than any accepted edge and filter it back out.
        finite = np.isfinite(costs)
        sentinel = max(1.0, threshold + 1.0) * 1e6
        work = np.where(finite, costs, sentinel)
        try:
            from scipy.optimize import linear_sum_assignment
        except Exception as exc:  # pragma: no cover - scipy is a declared dependency
            raise ValidationError("scipy is required for geometry-only pose matching") from exc
        row_ind, col_ind = linear_sum_assignment(work)
        matches: list[PoseInstanceMatch] = []
        matched_pred: set[int] = set()
        matched_truth: set[int] = set()
        for i, j in zip(row_ind.tolist(), col_ind.tolist()):
            cost = float(costs[i, j])
            if math.isfinite(cost) and cost <= threshold:
                matches.append(PoseInstanceMatch(predictions[i].prediction_uid, truth_uids[j], cost))
                matched_pred.add(i)
                matched_truth.add(j)
        matches.sort(key=lambda item: (item.truth_observation_uid, item.prediction_uid))
        unmatched_pred = [item.prediction_uid for i, item in enumerate(predictions) if i not in matched_pred]
        unmatched_truth = [uid for j, uid in enumerate(truth_uids) if j not in matched_truth]
        return matches, unmatched_pred, unmatched_truth


__all__ = ["PoseInstanceMatch", "PoseInstanceMatcher"]

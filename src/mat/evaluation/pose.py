from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np

from .fixed_identity import MetricBundle


class PoseEvaluator:
    def evaluate(self, predicted_xy: np.ndarray, truth_xy: np.ndarray, visible: np.ndarray,
                 scale: np.ndarray | float, threshold: float = 0.05) -> MetricBundle:
        pred, truth, visible = np.asarray(predicted_xy), np.asarray(truth_xy), np.asarray(visible, bool)
        if pred.shape != truth.shape or pred.shape[:-1] != visible.shape:
            raise ValueError("pose prediction/truth/visibility shape mismatch")
        scale = np.asarray(scale, dtype=float)
        if scale.ndim == 1 and pred.ndim >= 2 and scale.shape[0] == pred.shape[0]:
            scale = scale[:, None]
        error = np.linalg.norm(pred - truth, axis=-1) / np.maximum(scale, 1e-9)
        valid = visible & np.isfinite(error)
        values = error[valid]
        return MetricBundle("SUCCEEDED" if valid.any() else "BLOCKED_NEEDS_POSE_GT",
                            {"normalized_mean_error": float(values.mean()) if len(values) else None,
                             "pck": float(np.mean(values <= threshold)) if len(values) else None},
                            {"visible_points": int(valid.sum()), "total_points": int(visible.sum())}, [])


class IdentityAwarePoseEvaluator:
    def evaluate(self, predicted_xy: np.ndarray, truth_xy: np.ndarray, visible: np.ndarray,
                 located: np.ndarray, identity_correct: np.ndarray, scale: np.ndarray | float,
                 threshold: float = 0.05) -> MetricBundle:
        pred, truth = np.asarray(predicted_xy), np.asarray(truth_xy)
        base = np.asarray(visible, bool) & np.asarray(located, bool)[..., None] & np.asarray(identity_correct, bool)[..., None]
        scale = np.asarray(scale, float)
        if scale.ndim == 1 and pred.ndim >= 2 and scale.shape[0] == pred.shape[0]:
            scale = scale[:, None]
        error = np.linalg.norm(pred - truth, axis=-1) / np.maximum(scale, 1e-9)
        valid = base & np.isfinite(error)
        return MetricBundle("SUCCEEDED" if valid.any() else "BLOCKED_NEEDS_POSE_GT",
                            {"identity_aware_pck": float(np.mean(error[valid] <= threshold)) if valid.any() else None,
                             "identity_aware_mean_error": float(error[valid].mean()) if valid.any() else None},
                            {"joint_valid_points": int(valid.sum()), "candidate_visible_points": int(np.asarray(visible, bool).sum())}, [])

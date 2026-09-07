from __future__ import annotations

import numpy as np


def fuse_identity_scores(global_score: float, part_scores: np.ndarray,
                         part_valid: np.ndarray, alpha: float = 0.5,
                         part_weights: np.ndarray | None = None) -> float:
    """Fuse global and common-part scores; falls back to global with no valid parts."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    scores = np.asarray(part_scores, dtype=float)
    valid = np.asarray(part_valid, dtype=bool)
    if scores.shape != valid.shape or not np.any(valid):
        return float(global_score)
    weights = np.ones(scores.shape, dtype=float) if part_weights is None else np.asarray(part_weights, dtype=float)
    weights = np.where(valid, np.maximum(weights, 0.0), 0.0)
    if weights.sum() <= 0:
        return float(global_score)
    return float(alpha * global_score + (1.0 - alpha) * np.sum(scores * weights) / weights.sum())


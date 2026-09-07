from __future__ import annotations

import numpy as np


def retrieval_metrics(scores: np.ndarray, truth_indices: np.ndarray, top_k: tuple[int, ...] = (1, 5)) -> dict[str, float | None]:
    scores = np.asarray(scores)
    truth = np.asarray(truth_indices)
    if scores.ndim != 2 or truth.shape != (scores.shape[0],):
        raise ValueError("scores/truth shape mismatch")
    order = np.argsort(-scores, axis=1, kind="stable")
    result = {}
    for k in top_k:
        result[f"top{k}"] = float(np.mean([truth[i] in order[i, :k] for i in range(len(truth))])) if len(truth) else None
    return result


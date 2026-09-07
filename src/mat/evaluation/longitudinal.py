from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Any
import numpy as np


class LongitudinalMeasurementEvaluator:
    """Downstream-only measurements; never feeds results back into identity learning."""

    def displacement(self, keypoints_xy: np.ndarray, point_a: int, point_b: int) -> np.ndarray:
        points = np.asarray(keypoints_xy, dtype=float)
        delta = points[..., point_a, :] - points[..., point_b, :]
        return np.linalg.norm(delta, axis=-1)

    def summarize(self, values: dict[str, dict[str, float]]) -> dict[str, Any]:
        sessions = sorted(values)
        out = {"sessions": sessions, "per_session": values, "differences": {}}
        if sessions:
            ref = sessions[0]
            for session in sessions[1:]:
                out["differences"][session] = {uid: values[session][uid] - values[ref][uid]
                                                 for uid in set(values[session]) & set(values[ref])}
        return out


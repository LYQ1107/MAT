from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np


@dataclass(frozen=True)
class ScoreCalibration:
    threshold: float
    margin: float
    source_manifest_hash: str
    development_manifest_hash: str
    status: str = "CALIBRATED_SOURCE_DEV"

    def write(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")


def calibrate(scores: np.ndarray, correct: np.ndarray, source_hash: str, dev_hash: str) -> ScoreCalibration:
    scores, correct = np.asarray(scores, float), np.asarray(correct, bool)
    if scores.shape != correct.shape or scores.size == 0:
        raise ValueError("calibration requires source/dev scores and labels")
    thresholds = np.unique(scores)
    best = max(thresholds, key=lambda t: float(np.mean(correct[scores >= t])) if np.any(scores >= t) else -1)
    return ScoreCalibration(float(best), 0.0, source_hash, dev_hash)


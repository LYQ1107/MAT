"""Development-only rejection-threshold calibration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence
import json
import math

import numpy as np

from mat.core.errors import ValidationError


@dataclass(frozen=True)
class ThresholdCalibrationResult:
    threshold: float
    macro_f1: float | None
    accuracy: float | None
    accepted_accuracy: float | None
    unknown_rate: float | None
    wrong_identity_rate: float | None
    candidate_count: int
    split_role: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _div(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _summary(predicted: Sequence[str | None], truth: Sequence[str], labels: Sequence[str]) -> dict[str, float | None]:
    correct = wrong = unknown = 0
    for pred, target in zip(predicted, truth):
        if pred is None or pred not in labels:
            unknown += 1
        elif pred == target:
            correct += 1
        else:
            wrong += 1
    total = len(truth)
    accepted = correct + wrong
    f1s: list[float | None] = []
    for label in labels:
        tp = sum(int(target == label and pred == label) for target, pred in zip(truth, predicted))
        fp = sum(int(target != label and pred == label) for target, pred in zip(truth, predicted))
        fn = sum(int(target == label and pred != label) for target, pred in zip(truth, predicted))
        precision = _div(tp, tp + fp)
        recall = _div(tp, tp + fn)
        f1s.append(2 * precision * recall / (precision + recall)
                   if precision is not None and recall is not None and precision + recall > 0 else
                   (0.0 if precision is not None and recall is not None else None))
    macro = float(sum(value for value in f1s if value is not None) / len(f1s)) \
        if f1s and all(value is not None for value in f1s) else None
    return {
        "macro_f1": macro,
        "accuracy": _div(correct, total),
        "accepted_accuracy": _div(correct, accepted),
        "unknown_rate": _div(unknown, total),
        "wrong_identity_rate": _div(wrong, total),
        "coverage": _div(accepted, total),
    }


def calibrate_accept_threshold(
    top1_scores: np.ndarray,
    predicted_identity: Sequence[str],
    truth_identity: Sequence[str],
    *,
    identity_labels: Sequence[str],
    split_role: str = "development",
) -> ThresholdCalibrationResult:
    """Select a deterministic rejection threshold using development labels only."""
    scores = np.asarray(top1_scores, dtype=np.float64)
    predicted = [str(value) if value is not None else None for value in predicted_identity]
    truth = [str(value) for value in truth_identity]
    labels = tuple(str(value) for value in identity_labels)
    if scores.ndim != 1 or len(scores) == 0 or len(predicted) != len(scores) or len(truth) != len(scores):
        raise ValidationError("threshold calibration arrays must be non-empty and have equal length")
    if not np.isfinite(scores).all():
        raise ValidationError("threshold calibration scores must be finite")
    if not labels or len(set(labels)) != len(labels):
        raise ValidationError("identity_labels must be non-empty and unique")
    if any(value not in labels for value in truth):
        raise ValidationError("truth_identity contains a label outside identity_labels")
    low, high = float(scores.min()), float(scores.max())
    epsilon = max(np.finfo(np.float64).eps * max(1.0, abs(low), abs(high)), 1e-12)
    candidates = np.unique(np.concatenate(([low - epsilon], scores, [high + epsilon])))
    best_key: tuple[float, float, float, float, float] | None = None
    best: ThresholdCalibrationResult | None = None
    for threshold in sorted(float(value) for value in candidates):
        accepted_pred = [value if score >= threshold and value in labels else None
                         for score, value in zip(scores, predicted)]
        summary = _summary(accepted_pred, truth, labels)
        macro = summary["macro_f1"]
        wrong_rate = summary["wrong_identity_rate"]
        accepted_accuracy = summary["accepted_accuracy"]
        coverage = summary["coverage"]
        # None is ranked below any valid metric, while threshold is the final
        # descending tie-break.  The tuple is deliberately explicit/auditable.
        key = (
            float(macro) if macro is not None else -math.inf,
            -(float(wrong_rate) if wrong_rate is not None else math.inf),
            float(accepted_accuracy) if accepted_accuracy is not None else -math.inf,
            float(coverage) if coverage is not None else -math.inf,
            threshold,
        )
        if best_key is None or key > best_key:
            best_key = key
            best = ThresholdCalibrationResult(
                threshold=threshold, macro_f1=macro, accuracy=summary["accuracy"],
                accepted_accuracy=accepted_accuracy, unknown_rate=summary["unknown_rate"],
                wrong_identity_rate=wrong_rate, candidate_count=len(candidates), split_role=split_role,
            )
    assert best is not None
    return best


__all__ = ["ThresholdCalibrationResult", "calibrate_accept_threshold"]

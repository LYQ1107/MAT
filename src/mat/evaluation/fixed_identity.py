from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any, Iterable
import json


@dataclass(frozen=True)
class MetricBundle:
    status: str
    metrics: dict[str, float | None]
    counts: dict[str, int]
    errors: list[str]

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


class EnrollmentMapping:
    def __init__(self, mapping: dict[str, str], frozen: bool = False):
        self.mapping = dict(mapping)
        self._frozen = frozen

    @classmethod
    def fit_reference_only(cls, reference_predictions: Iterable[dict[str, Any]], reference_truth: Iterable[dict[str, Any]]):
        truth_by_obs = {str(r["observation_uid"]): str(r["gt_id"]) for r in reference_truth}
        votes: dict[str, Counter] = defaultdict(Counter)
        for row in reference_predictions:
            uid = row.get("persistent_uid") or row.get("model_uid")
            obs = str(row.get("observation_uid"))
            if uid is not None and obs in truth_by_obs:
                votes[str(uid)][truth_by_obs[obs]] += 1
        mapping = {uid: counts.most_common(1)[0][0] for uid, counts in votes.items() if counts}
        return cls(mapping)

    def freeze(self) -> None:
        self._frozen = True

    def add(self, model_uid: str, truth_uid: str) -> None:
        if self._frozen:
            raise RuntimeError("reference mapping is frozen")
        self.mapping[model_uid] = truth_uid


class PersistentIDEvaluator:
    def evaluate(self, predictions: Iterable[dict[str, Any]], sealed_truth: Iterable[dict[str, Any]],
                 mapping: EnrollmentMapping) -> MetricBundle:
        predictions = list(predictions)
        truth = list(sealed_truth)
        if not mapping._frozen:
            raise ValueError("EnrollmentMapping.freeze() is required before sealed evaluation")
        truth_by_obs = {str(r["observation_uid"]): str(r["gt_id"]) for r in truth}
        located = correct = unknown = wrong = 0
        by_session = defaultdict(lambda: [0, 0])
        for row in predictions:
            obs = str(row.get("observation_uid"))
            if obs not in truth_by_obs:
                continue
            located += 1
            pred_uid = row.get("persistent_uid")
            mapped = mapping.mapping.get(str(pred_uid)) if pred_uid is not None else None
            gt = truth_by_obs[obs]
            if mapped is None:
                unknown += 1
            elif mapped == gt:
                correct += 1
            else:
                wrong += 1
            session = str(row.get("session_uid", "unknown"))
            by_session[session][0] += int(mapped == gt)
            by_session[session][1] += 1
        total_truth = len(truth_by_obs)
        status = "SUCCEEDED" if total_truth else "BLOCKED_MISSING_PERSISTENT_TRUTH"
        metrics = {
            "located_persistent_id_accuracy": correct / located if located else None,
            "identity_correct_coverage": correct / total_truth if total_truth else None,
            "unknown_rate_on_located": unknown / located if located else None,
            "misidentification_rate_on_located": wrong / located if located else None,
            "located_recall": located / total_truth if total_truth else None,
        }
        return MetricBundle(status, metrics,
                            {"truth_instances": total_truth, "located": located, "correct": correct,
                             "unknown": unknown, "wrong": wrong, "sessions": len(by_session)}, [])


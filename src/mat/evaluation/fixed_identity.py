from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any, Iterable, Sequence
import json


@dataclass(frozen=True)
class MetricBundle:
    status: str
    metrics: dict[str, Any]
    counts: dict[str, int]
    errors: list[str]

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class IdentityClassMetrics:
    """Standard one-vs-rest metrics for one sealed identity label."""

    identity_uid: str
    tp: int
    fp: int
    fn: int
    precision: float | None
    recall: float | None
    f1: float | None


class EnrollmentMapping:
    def __init__(self, mapping: dict[str, str], frozen: bool = False):
        self.mapping = dict(mapping)
        self._frozen = frozen

    @classmethod
    def fit_reference_only(cls, reference_predictions: Iterable[dict[str, Any]], reference_truth: Iterable[dict[str, Any]]):
        truth_by_obs = {str(r["observation_uid"]): str(r.get("gt_id", r.get("gt_identity")))
                        for r in reference_truth}
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
    @staticmethod
    def _truth_label(row: dict[str, Any]) -> str:
        value = row.get("gt_id", row.get("gt_identity"))
        if value is None:
            raise ValueError("sealed truth row is missing gt_id/gt_identity")
        return str(value)

    @staticmethod
    def _truth_observation_uid(row: dict[str, Any]) -> str:
        value = row.get("observation_uid")
        if value is None:
            raise ValueError("sealed truth row is missing observation_uid")
        return str(value)

    @staticmethod
    def _prediction_observation_uid(row: dict[str, Any]) -> str | None:
        """Resolve direct or evaluator-matched observation identity.

        The model may emit ``prediction_uid`` only.  In that case the
        geometry-only evaluator can attach ``evaluator_match`` containing a
        ``truth_observation_uid``; this method never infers a match from an
        identity label.
        """
        value = row.get("observation_uid")
        if value is not None:
            return str(value)
        value = row.get("truth_observation_uid")
        if value is not None:
            return str(value)
        match = row.get("evaluator_match")
        if isinstance(match, dict):
            value = match.get("truth_observation_uid") or match.get("observation_uid")
            if value is not None:
                return str(value)
        return None

    @staticmethod
    def _prediction_persistent_uid(row: dict[str, Any]) -> str | None:
        value = row.get("persistent_uid", row.get("model_uid"))
        return str(value) if value is not None else None

    @staticmethod
    def _div(numerator: int | float, denominator: int | float) -> float | None:
        return float(numerator / denominator) if denominator else None

    def evaluate_labels(
        self,
        predictions: Iterable[dict[str, Any]],
        sealed_truth: Iterable[dict[str, Any]],
        mapping: EnrollmentMapping,
        *,
        known_identity_labels: Sequence[str],
    ) -> MetricBundle:
        """Evaluate fixed persistent-ID predictions against sealed labels.

        Every sealed truth row remains in the denominator.  Unknown predictions
        are rejected/absent IDs (one FN for the true class); a known wrong ID is
        one FP and one FN.  ``mapping`` must be reference-only and frozen before
        this method is called.
        """
        if not mapping._frozen:
            raise ValueError("EnrollmentMapping.freeze() is required before sealed evaluation")
        labels = tuple(str(label) for label in known_identity_labels)
        if not labels or len(set(labels)) != len(labels):
            raise ValueError("known_identity_labels must be a non-empty unique sequence")
        truth_rows = list(sealed_truth)
        truth_by_obs: dict[str, dict[str, Any]] = {}
        for row in truth_rows:
            obs = self._truth_observation_uid(row)
            if obs in truth_by_obs:
                raise ValueError(f"duplicate sealed truth observation_uid: {obs}")
            truth_by_obs[obs] = row
        prediction_by_obs: dict[str, dict[str, Any]] = {}
        for row in predictions:
            obs = self._prediction_observation_uid(row)
            if obs is None or obs not in truth_by_obs:
                continue
            if obs in prediction_by_obs:
                raise ValueError(f"duplicate prediction for sealed observation_uid: {obs}")
            prediction_by_obs[obs] = row

        per_identity: dict[str, IdentityClassMetrics] = {}
        confusion: dict[str, dict[str, int]] = {
            label: {candidate: 0 for candidate in (*labels, "unknown")}
            for label in labels
        }
        per_session: dict[str, dict[str, int]] = defaultdict(
            lambda: {"truth_instances": 0, "located_instances": 0, "accepted": 0,
                     "correct": 0, "wrong_identity": 0, "unknown": 0}
        )
        correct = wrong = unknown = located = 0
        truth_labels: list[str] = []
        predicted_labels: list[str | None] = []
        for obs, row in truth_by_obs.items():
            truth_label = self._truth_label(row)
            if truth_label not in labels:
                raise ValueError(f"sealed truth identity {truth_label!r} is not in known_identity_labels")
            prediction = prediction_by_obs.get(obs)
            mapped: str | None = None
            if prediction is not None:
                located += 1
                persistent_uid = self._prediction_persistent_uid(prediction)
                candidate = mapping.mapping.get(persistent_uid) if persistent_uid is not None else None
                if candidate in labels:
                    mapped = candidate
            outcome = mapped if mapped is not None else "unknown"
            confusion[truth_label][outcome] += 1
            session = str(row.get("session_uid", prediction.get("session_uid", "unknown") if prediction else "unknown"))
            session_counts = per_session[session]
            session_counts["truth_instances"] += 1
            session_counts["located_instances"] += int(prediction is not None)
            if mapped is None:
                unknown += 1
                session_counts["unknown"] += 1
            elif mapped == truth_label:
                correct += 1
                session_counts["accepted"] += 1
                session_counts["correct"] += 1
            else:
                wrong += 1
                session_counts["accepted"] += 1
                session_counts["wrong_identity"] += 1
            truth_labels.append(truth_label)
            predicted_labels.append(mapped)

        for label in labels:
            tp = sum(int(gt == label and pred == label) for gt, pred in zip(truth_labels, predicted_labels))
            fp = sum(int(gt != label and pred == label) for gt, pred in zip(truth_labels, predicted_labels))
            fn = sum(int(gt == label and pred != label) for gt, pred in zip(truth_labels, predicted_labels))
            precision = self._div(tp, tp + fp)
            recall = self._div(tp, tp + fn)
            f1 = (2.0 * precision * recall / (precision + recall)
                  if precision is not None and recall is not None and precision + recall > 0 else
                  (0.0 if precision is not None and recall is not None else None))
            per_identity[label] = IdentityClassMetrics(label, tp, fp, fn, precision, recall, f1)
        macro_values = [item.f1 for item in per_identity.values()]
        macro_f1 = float(sum(value for value in macro_values if value is not None) / len(macro_values)) \
            if len(macro_values) == len(labels) and all(value is not None for value in macro_values) else None
        micro_denominator = 2 * correct + (wrong) + (wrong + unknown)
        micro_f1 = self._div(2 * correct, micro_denominator)
        total = correct + wrong + unknown
        accepted = correct + wrong
        metrics: dict[str, Any] = {
            "accuracy": self._div(correct, total),
            "macro_f1": macro_f1,
            "micro_f1": micro_f1,
            "accepted_accuracy": self._div(correct, accepted),
            "unknown_rate": self._div(unknown, total),
            "wrong_identity_rate": self._div(wrong, total),
            "false_identity_rate_on_accepted": self._div(wrong, accepted),
            "per_identity": {label: asdict(value) for label, value in per_identity.items()},
            "per_session": {},
            "confusion_matrix": confusion,
        }
        for session, values in sorted(per_session.items()):
            session_total = values["truth_instances"]
            session_accepted = values["accepted"]
            metrics["per_session"][session] = {
                **values,
                "accuracy": self._div(values["correct"], session_total),
                "accepted_accuracy": self._div(values["correct"], session_accepted),
                "unknown_rate": self._div(values["unknown"], session_total),
                "wrong_identity_rate": self._div(values["wrong_identity"], session_total),
            }
        status = "SUCCEEDED" if truth_by_obs else "BLOCKED_MISSING_PERSISTENT_TRUTH"
        counts = {
            "truth_instances": total,
            "located_instances": located,
            "accepted": accepted,
            "correct": correct,
            "wrong_identity": wrong,
            "unknown": unknown,
            "sessions": len(per_session),
        }
        return MetricBundle(status, metrics, counts, [])

    def evaluate(self, predictions: Iterable[dict[str, Any]], sealed_truth: Iterable[dict[str, Any]],
                 mapping: EnrollmentMapping) -> MetricBundle:
        truth = list(sealed_truth)
        labels = tuple(sorted({self._truth_label(row) for row in truth}))
        bundle = self.evaluate_labels(predictions, truth, mapping, known_identity_labels=labels)
        total_truth = bundle.counts["truth_instances"]
        located = bundle.counts["located_instances"]
        metrics = {
            "located_persistent_id_accuracy": (bundle.counts["correct"] / located if located else None),
            "identity_correct_coverage": (bundle.counts["correct"] / total_truth if total_truth else None),
            "unknown_rate_on_located": (bundle.counts["unknown"] / located if located else None),
            "misidentification_rate_on_located": (bundle.counts["wrong_identity"] / located if located else None),
            "located_recall": (located / total_truth if total_truth else None),
        }
        return MetricBundle(bundle.status, metrics,
                            {"truth_instances": total_truth, "located": located, "correct": bundle.counts["correct"],
                             "unknown": bundle.counts["unknown"], "wrong": bundle.counts["wrong_identity"],
                             "sessions": bundle.counts["sessions"]}, bundle.errors)

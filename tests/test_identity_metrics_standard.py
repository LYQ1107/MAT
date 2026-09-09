import pytest

from mat.evaluation.fixed_identity import EnrollmentMapping, PersistentIDEvaluator


def test_standard_identity_metrics_include_unknowns_and_one_vs_rest_f1():
    labels = ("A", "B", "C", "D")
    truth = [{"observation_uid": f"o{i}", "gt_id": label, "session_uid": "sealed"}
             for i, label in enumerate(("A", "A", "B", "B", "C", "D"))]
    predictions = [
        {"observation_uid": "o0", "persistent_uid": "pA"},
        {"observation_uid": "o1", "persistent_uid": None},
        {"observation_uid": "o2", "persistent_uid": "pA"},
        {"observation_uid": "o3", "persistent_uid": "pB"},
        {"observation_uid": "o4", "persistent_uid": "pC"},
        {"observation_uid": "o5", "persistent_uid": "pD"},
    ]
    mapping = EnrollmentMapping({"pA": "A", "pB": "B", "pC": "C", "pD": "D"})
    mapping.freeze()
    result = PersistentIDEvaluator().evaluate_labels(
        predictions, truth, mapping, known_identity_labels=labels)

    assert result.status == "SUCCEEDED"
    assert result.counts == {
        "truth_instances": 6, "located_instances": 6, "accepted": 5,
        "correct": 4, "wrong_identity": 1, "unknown": 1, "sessions": 1,
    }
    assert result.metrics["accuracy"] == pytest.approx(4 / 6)
    assert result.metrics["accepted_accuracy"] == pytest.approx(4 / 5)
    assert result.metrics["unknown_rate"] == pytest.approx(1 / 6)
    assert result.metrics["wrong_identity_rate"] == pytest.approx(1 / 6)
    assert result.metrics["false_identity_rate_on_accepted"] == pytest.approx(1 / 5)
    assert result.metrics["macro_f1"] == pytest.approx((0.5 + 2 / 3 + 1 + 1) / 4)
    assert result.metrics["micro_f1"] == pytest.approx(8 / 11)

    per_identity = result.metrics["per_identity"]
    assert per_identity["A"] == {
        "identity_uid": "A", "tp": 1, "fp": 1, "fn": 1,
        "precision": pytest.approx(0.5), "recall": pytest.approx(0.5),
        "f1": pytest.approx(0.5),
    }
    assert per_identity["B"]["tp"] == 1
    assert per_identity["B"]["fp"] == 0
    assert per_identity["B"]["fn"] == 1


def test_evaluate_labels_accepts_geometry_evaluator_match_without_identity_lookup():
    mapping = EnrollmentMapping({"pA": "A"})
    mapping.freeze()
    result = PersistentIDEvaluator().evaluate_labels(
        [{"prediction_uid": "pred-0", "persistent_uid": "pA",
          "evaluator_match": {"truth_observation_uid": "truth-0"}}],
        [{"observation_uid": "truth-0", "gt_identity": "A"}], mapping,
        known_identity_labels=("A",))
    assert result.counts["correct"] == 1
    assert result.metrics["macro_f1"] == pytest.approx(1.0)

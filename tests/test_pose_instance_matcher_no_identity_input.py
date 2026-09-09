import numpy as np

from mat.core.types import PredictedPoseInstance
from mat.evaluation.instance_matching import PoseInstanceMatcher


def _prediction(uid, points):
    points = np.asarray(points, np.float32)
    return PredictedPoseInstance(uid, "s", 4, 0.16, np.asarray([0, 0, 10, 10], np.float32),
                                 points, np.ones(len(points), np.float32), np.ones(len(points), bool),
                                 0.8, None, "pose")


def test_geometry_matching_ignores_identity_labels_and_reports_unmatched():
    prediction = _prediction("p0", [[1, 1], [9, 1], [9, 9], [1, 9]])
    truth = [{"observation_uid": "o0", "session_uid": "s", "frame_index": 4,
              "gt_identity": "secret-A", "gt_keypoints": [[1, 1], [9, 1], [9, 9], [1, 9]],
              "gt_visibility": [True, True, True, True]},
             {"observation_uid": "o1", "session_uid": "s", "frame_index": 4,
              "gt_identity": "secret-B", "gt_keypoints": [[30, 30], [39, 30], [39, 39], [30, 39]],
              "gt_visibility": [True, True, True, True]}]
    matches, unmatched_pred, unmatched_truth = PoseInstanceMatcher().match_frame(
        [prediction], truth, normalized_distance_threshold=0.1)
    assert [(item.prediction_uid, item.truth_observation_uid) for item in matches] == [("p0", "o0")]
    assert unmatched_pred == []
    assert unmatched_truth == ["o1"]

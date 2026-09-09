import numpy as np

from mat.core.types import Assignment, LocalTracklet, PredictedPoseInstance
from mat.experiments.gerbil_part_identity import _assignment_rows


def test_predicted_pose_contract_has_no_gt_or_provider_identity_fields():
    prediction = PredictedPoseInstance(
        prediction_uid="pred:x", session_uid="s", frame_index=3, timestamp_s=0.12,
        bbox_xyxy=np.asarray([0, 0, 10, 10], np.float32),
        keypoints_xy=np.asarray([[1, 1], [2, 2]], np.float32),
        keypoint_scores=np.ones(2, np.float32), keypoint_valid=np.ones(2, bool),
        detection_score=0.9, local_track_uid="local:1", pose_model_fingerprint="pose:sha",
    )
    fields = set(prediction.__dataclass_fields__)
    assert not {"gt_identity", "gt_observation_uid", "provider_track_name"}.intersection(fields)
    assert prediction.prediction_uid == "pred:x"


def test_predicted_assignment_never_uses_observation_uid_for_unmatched_rows():
    assignment = Assignment("pred:x", None, (), (), "unregistered", ("below_threshold",), "v0", "pose:sha")
    track = LocalTracklet("pred:x", "s", "camera", ["pred:x"], [(0.0, 0.0)])
    row = _assignment_rows([assignment], {"pred:x": track}, geometry_links={}, predicted_mode=True)[0]
    assert row["prediction_uid"] == "pred:x"
    assert "observation_uid" not in row
    assert "evaluator_match" not in row

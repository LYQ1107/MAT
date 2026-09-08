import numpy as np
import pytest

from mat.backends.wildlife import NumpyFixtureEncoder
from mat.experiments.gerbil_identity import (GerbilInstanceSample, _descriptor_rows,
                                              bbox_from_keypoints, select_s0_sessions)


def test_bbox_from_keypoints_rejects_nonfinite_or_insufficient_points():
    assert bbox_from_keypoints(np.full((4, 2), np.nan)) is None
    assert bbox_from_keypoints([[1, 2], [np.inf, 3], [np.nan, 4]]) is None
    box = bbox_from_keypoints([[2, 3], [12, 13], [8, 9]], image_shape=(20, 30))
    assert box is not None and np.isfinite(box).all() and box[2] > box[0] and box[3] > box[1]


def test_s0_selection_is_earliest_sessions_covering_four_ids():
    sessions = [
        {"session_uid": "late", "recording_datetime_if_parseable": "2024-01-02"},
        {"session_uid": "early", "recording_datetime_if_parseable": "2024-01-01"},
    ]
    truth = [{"observation_uid": f"o{i}", "session_uid": "early", "gt_identity": name}
             for i, name in enumerate(("female", "male", "pup shaved"))]
    truth.append({"observation_uid": "o3", "session_uid": "late", "gt_identity": "pup unshaved"})
    assert select_s0_sessions(sessions, truth) == ("early", "late")


def test_neutral_sample_contract_has_no_identity_fields():
    sample = GerbilInstanceSample("o", "s", 0, 0.0, np.zeros((4, 5, 3), np.uint8),
                                  None, np.full((14, 2), np.nan, np.float32),
                                  np.zeros(14, np.float32), np.zeros(14, bool), "t")
    assert not hasattr(sample, "gt_identity")


def test_descriptor_rows_handles_heterogeneous_roi_shapes_without_stack_error():
    samples = [
        GerbilInstanceSample("o0", "s0", 0, 0.0, np.zeros((10, 12, 3), np.uint8),
                             None, np.full((4, 2), np.nan, np.float32),
                             np.zeros(4, np.float32), np.zeros(4, bool), "t0"),
        GerbilInstanceSample("o1", "s1", 1, 0.1, np.zeros((15, 20, 3), np.uint8),
                             None, np.full((4, 2), np.nan, np.float32),
                             np.zeros(4, np.float32), np.zeros(4, bool), "t1"),
    ]
    boxes = {"o0": np.asarray([0, 0, 6, 7], np.float32),
             "o1": np.asarray([0, 0, 11, 13], np.float32)}
    descriptors, ordered = _descriptor_rows(samples, NumpyFixtureEncoder(), boxes, batch_size=2)
    assert [sample.observation_uid for sample in ordered] == ["o0", "o1"]
    assert set(descriptors) == {"o0", "o1"}
    assert all(item.global_feature.shape == (6,) for item in descriptors.values())

import sys
import types
import numpy as np

from mat.data.predictions.sleap_pose import SleapPosePredictionAdapter
from mat.core.errors import ValidationError


def test_sleap_parser_generates_local_prediction_uid_without_gt(monkeypatch, tmp_path):
    video = types.SimpleNamespace(filename="materialized_test.pkg.slp", fps=25.0)
    points = np.array([
        ([1.0, 1.0], 0.9, True), ([8.0, 1.0], 0.8, True),
        ([8.0, 8.0], 0.7, True), ([1.0, 8.0], 0.6, True),
    ], dtype=[("xy", "f4", (2,)), ("score", "f4"), ("visible", "?")])
    instance = types.SimpleNamespace(points=points, score=0.75, track=None)
    frame = types.SimpleNamespace(video=video, frame_idx=10, instances=[instance])
    labels = types.SimpleNamespace(videos=[video], labeled_frames=[frame], close=lambda: None)
    fake = types.ModuleType("sleap_io")
    fake.load_slp = lambda path, open_videos=False, lazy=True: labels
    monkeypatch.setitem(sys.modules, "sleap_io", fake)
    path = tmp_path / "predictions.slp"; path.write_bytes(b"fixture")
    adapter = SleapPosePredictionAdapter(pose_model_fingerprint="pose-sha")
    values = adapter.load(path, {"day001.pkg.slp#video0": "session-0"})
    assert len(values) == 1
    value = values[0]
    assert value.session_uid == "session-0"
    assert value.prediction_uid.startswith("pred:")
    assert value.frame_index == 10 and value.local_track_uid is None
    assert not hasattr(value, "gt_identity")


def test_sleap_parser_rejects_ambiguous_video_alias(monkeypatch, tmp_path):
    video = types.SimpleNamespace(filename="same.pkg.slp", fps=25.0)
    points = np.array([
        ([1.0, 1.0], 0.9, True), ([8.0, 1.0], 0.8, True),
        ([8.0, 8.0], 0.7, True), ([1.0, 8.0], 0.6, True),
    ], dtype=[("xy", "f4", (2,)), ("score", "f4"), ("visible", "?")])
    instance = types.SimpleNamespace(points=points, score=0.75, track=None)
    frame = types.SimpleNamespace(video=video, frame_idx=10, instances=[instance])
    labels = types.SimpleNamespace(videos=[video], labeled_frames=[frame], close=lambda: None)
    fake = types.ModuleType("sleap_io")
    fake.load_slp = lambda path, open_videos=False, lazy=True: labels
    monkeypatch.setitem(sys.modules, "sleap_io", fake)
    path = tmp_path / "predictions.slp"; path.write_bytes(b"fixture")
    adapter = SleapPosePredictionAdapter(pose_model_fingerprint="pose-sha")
    # No exact #video0 alias is supplied; the basename is deliberately mapped
    # to two sessions and must not be guessed.
    import pytest
    with pytest.raises(ValidationError, match="absent/ambiguous"):
        adapter.load(path, {"a.pkg.slp#video0": "session-0", "b.pkg.slp#video0": "session-1"})

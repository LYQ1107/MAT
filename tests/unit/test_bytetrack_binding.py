import numpy as np

from mat.backends.bytetrack import ByteTrackBackend
from mat.core.types import AnimalObservation, FramePacket


def obs(uid, box, score=0.95):
    return AnimalObservation(uid, "f", uid, np.asarray(box, np.float32), score,
                             np.full((2, 2), np.nan, np.float32), np.zeros(2, np.float32),
                             np.array(["unknown", "unknown"]), None, {}, {"source": "TEST_FIXTURE"})


def frame(index):
    return FramePacket("d", "c", "s", "cam", index, float(index), np.zeros((8, 8, 3), np.uint8))


def test_detection_index_survives_reordered_detections_and_empty_frame():
    tracker = ByteTrackBackend(track_thresh=0.2, match_thresh=0.2, track_buffer=2)
    first = tracker.update(frame(0), [obs("a", [0, 0, 2, 2]), obs("b", [5, 5, 7, 7])])
    assert {a.detection_uid: a.source_detection_index for a in first if a.detection_uid} == {"a": 0, "b": 1}
    second = tracker.update(frame(1), [obs("b2", [5, 5, 7, 7]), obs("a2", [0, 0, 2, 2])])
    assert {a.detection_uid: a.source_detection_index for a in second if a.detection_uid} == {"b2": 0, "a2": 1}
    predicted = tracker.update(frame(2), [])
    assert predicted and all(a.detection_uid is None and a.source_detection_index is None for a in predicted)


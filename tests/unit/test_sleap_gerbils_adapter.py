import json
from pathlib import Path

import numpy as np

from mat.data.adapters.sleap_gerbils import SleapGerbilsAdapter
from mat.data.manifests import iter_manifest_records


class _Backend:
    source_filename = "day001.pkg.slp"
    dataset = "video0/video"

    def has_embedded_images(self):
        return True


class _Video:
    backend = _Backend()
    filename = "package.slp"

    def __getitem__(self, _frame_idx):
        return np.zeros((8, 8, 3), dtype=np.uint8)

    def __eq__(self, other):
        return self is other


class _Node:
    def __init__(self, name):
        self.name = name


class _Skeleton:
    nodes = [_Node(name) for name in (
        "nose", "lefteye", "righteye", "leftear", "rightear",
        "spine1", "spine2", "spine3", "spine4", "spine5",
        "tail1", "tail2", "tail3", "tail4",
    )]


class _Track:
    def __init__(self, name):
        self.name = name


class _Instance:
    skeleton = _Skeleton()

    def __init__(self, name):
        self.track = _Track(name)
        dtype = [("xy", "f8", (2,)), ("visible", "?"), ("complete", "?")]
        self.points = np.zeros(14, dtype=dtype)
        self.points["xy"] = 2.0
        self.points["visible"] = True

    def numpy(self):
        return self.points["xy"]


class _Frame:
    frame_idx = 7
    video = _Video()
    instances = [_Instance("female"), _Instance("male")]


class _Labels:
    videos = [_Video()]
    tracks = [_Track("female"), _Track("male"), _Track("pup shaved"), _Track("pup unshaved")]
    skeletons = [_Skeleton()]
    labeled_frames = [_Frame()]

    def __len__(self):
        return len(self.labeled_frames)

    def close(self):
        return None


def test_gt_identity_stays_out_of_observation_manifest(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for name in ("train.pkg.slp", "val.pkg.slp", "test.pkg.slp", "example_5min.mp4", "example_tracking.slp"):
        (raw / name).write_bytes(b"fixture")
    adapter = SleapGerbilsAdapter()
    adapter._load = lambda _path, open_videos=True: _Labels()
    out = tmp_path / "prepared"
    adapter.build_manifests(raw, out)
    observations = list(iter_manifest_records(out / "manifests" / "observations.jsonl"))
    truth = list(iter_manifest_records(out / "manifests" / "private_pose_identity_truth.jsonl"))
    assert observations and truth
    forbidden = {"gt_identity", "gt_keypoints", "gt_visibility", "female", "male", "pup shaved", "pup unshaved"}
    for row in observations:
        assert not forbidden.intersection(row)
        assert not any(value in forbidden for value in row.values() if isinstance(value, str))
    assert {row["gt_identity"] for row in truth} == {"female", "male"}

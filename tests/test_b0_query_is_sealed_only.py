import numpy as np

from mat.backends.wildlife import NumpyFixtureEncoder
from mat.config.identity import IdentityMatchingConfig
from mat.core.types import LocalTracklet
from mat.experiments.gerbil_identity import GerbilInstanceSample, run_b0_strict
from mat.identity.gallery import GalleryStore


def _sample(uid, session, label, value):
    image = np.full((24, 24, 3), value, dtype=np.uint8)
    points = np.asarray([[2, 2], [18, 2], [18, 18], [2, 18]], dtype=np.float32)
    return GerbilInstanceSample(uid, session, 0, 0.0, image, None,
                                np.full((4, 2), np.nan, np.float32), np.zeros(4), np.zeros(4, bool), uid), {
        "gt_identity": label, "gt_keypoints": points.tolist(), "gt_visibility": [True] * 4,
    }


def test_strict_b0_assignments_are_sealed_only(tmp_path):
    labels = ("female", "male", "pup shaved", "pup unshaved")
    samples, truth = [], {}
    for role, session in (("reference", "r"), ("development", "d"), ("sealed", "s")):
        for i, label in enumerate(labels):
            sample, row = _sample(f"{role}:{i}", session, label, 40 + i * 40)
            samples.append(sample); truth[sample.observation_uid] = row
    store = GalleryStore(tmp_path / "gallery.sqlite")
    result = run_b0_strict(samples, truth, NumpyFixtureEncoder(), store,
                           reference_sessions=("r",), development_sessions=("d",), sealed_test_sessions=("s",),
                           matching_config=IdentityMatchingConfig())
    assert result.status == "SUCCEEDED"
    assert result.reference_sessions == ("r",)
    assert result.development_sessions == ("d",)
    assert result.sealed_test_sessions == ("s",)
    assert result.assignments
    assert all(row["session_uid"] == "s" for row in result.assignments)
    assert all(row["observation_uid"].startswith("sealed:") for row in result.assignments)
    assert result.metrics["counts"]["sessions"] == 1
    store.close()

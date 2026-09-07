"""TEST_FIXTURE only: exercises the B0 chain without claiming a dataset result."""

import numpy as np

from mat.core.types import IdentityDescriptor, LocalTracklet
from mat.enrollment.automatic import AutoRegistrar
from mat.identity.gallery import GalleryStore
from mat.identity.matching import PersistentMatcher, MatchingPolicy
from mat.identity.conflicts import ConflictGraphBuilder
from mat.backends.wildlife import NumpyFixtureEncoder
from mat.experiments.b0 import run_prelocalized_b0


class Reference:
    session_uid = "s0"
    cohort_uid = "fixture-cohort"

    def __init__(self):
        self.tracklets = [LocalTracklet("s0:t0", "s0", "cam", ["o0"], [(0.0, 1.0)]),
                          LocalTracklet("s0:t1", "s0", "cam", ["o1"], [(2.0, 3.0)])]
        self.descriptors = {"s0:t0": self.d([1, 0]), "s0:t1": self.d([0, 1])}

    @staticmethod
    def d(x):
        return IdentityDescriptor(np.asarray(x, np.float32), np.zeros((0, 2), np.float32), np.zeros(0, bool), np.zeros(0), "TEST_FIXTURE")


def test_b0_reference_gallery_and_query_chain(tmp_path):
    gallery = GalleryStore(tmp_path / "registry.sqlite")
    snapshot = gallery.create(AutoRegistrar(merge_threshold=0.99).build(Reference()))
    query_tracks = [LocalTracklet("s1:t0", "s1", "cam", ["q0"], [(0.0, 1.0)])]
    query_desc = {"s1:t0": Reference.d([1, 0])}
    scores = PersistentMatcher().score(query_tracks, query_desc, snapshot)
    assignments = PersistentMatcher().assign(scores, ConflictGraphBuilder().build(query_tracks), MatchingPolicy(accept_threshold=0.5))
    assert assignments[0].status == "accepted"
    assert assignments[0].persistent_uid in snapshot.identities
    gallery.close()


def test_b0_runner_rejects_gt_in_model_rows(tmp_path):
    store = GalleryStore(tmp_path / "registry.sqlite")
    result = run_prelocalized_b0([{"observation_uid": "o", "session_uid": "s0", "image": np.zeros((4, 4, 3), np.uint8), "gt_id": "secret"}],
                                 NumpyFixtureEncoder(), store, "s0", ["s1"])
    assert result.status == "FAILED_LEAKAGE_CONTRACT"
    store.close()

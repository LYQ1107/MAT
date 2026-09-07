import numpy as np

from mat.core.types import IdentityDescriptor, LocalTracklet, PersistentIdentity
from mat.identity.conflicts import ConflictGraphBuilder
from mat.identity.matching import PersistentMatcher, MatchingPolicy
from mat.evaluation.fixed_identity import EnrollmentMapping, PersistentIDEvaluator
from mat.enrollment.base import EnrollmentResult


def desc(value, fp="fixture"):
    return IdentityDescriptor(np.asarray(value, np.float32), np.zeros((0, len(value)), np.float32), np.zeros(0, bool), np.zeros(0, np.float32), fp)


class G: pass


def test_nonoverlap_reuse_and_overlap_cannot_link():
    gallery = G(); gallery.descriptors = {"id:a": desc([1, 0]), "id:b": desc([0, 1])}
    tracks = [LocalTracklet("t0", "s", "cam", [], [(0, 1)]), LocalTracklet("t1", "s", "cam", [], [(2, 3)]), LocalTracklet("t2", "s", "cam", [], [(0.5, 1.5)])]
    queries = {"t0": desc([1, 0]), "t1": desc([1, 0]), "t2": desc([1, 0])}
    scores = PersistentMatcher().score(tracks, queries, gallery)
    assigned = PersistentMatcher().assign(scores, ConflictGraphBuilder().build(tracks), MatchingPolicy(accept_threshold=0.5))
    by = {a.tracklet_uid: a.persistent_uid for a in assigned}
    assert by["t0"] == "id:a" and by["t1"] == "id:a"
    assert by["t2"] != "id:a"  # conflict leaves unknown rather than violating identity exclusivity


def test_reference_mapping_is_frozen_and_not_rehungarianed():
    mapping = EnrollmentMapping.fit_reference_only(
        [{"observation_uid": "o0", "persistent_uid": "id:a"}, {"observation_uid": "o1", "persistent_uid": "id:b"}],
        [{"observation_uid": "o0", "gt_id": "A"}, {"observation_uid": "o1", "gt_id": "B"}],
    )
    mapping.freeze()
    metrics = PersistentIDEvaluator().evaluate(
        [{"observation_uid": "q0", "persistent_uid": "id:b"}], [{"observation_uid": "q0", "gt_id": "A"}], mapping)
    assert metrics.metrics["located_persistent_id_accuracy"] == 0.0


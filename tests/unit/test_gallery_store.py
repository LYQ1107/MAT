import numpy as np
import pytest

from mat.core.types import IdentityDescriptor, PersistentIdentity, Assignment
from mat.enrollment.base import EnrollmentResult
from mat.identity.gallery import GalleryStore


def test_gallery_anchor_immutable_and_versioned(tmp_path):
    d = IdentityDescriptor(np.array([1., 0.]), np.zeros((0, 2)), np.zeros(0, bool), np.zeros(0), "fixture")
    ident = PersistentIdentity("id:a", "cohort", ("anchor-0",), (), (), "s0", {"protocol": "A_auto"})
    store = GalleryStore(tmp_path / "registry.sqlite")
    snap = store.create(EnrollmentResult("cohort", "s0", (ident,), {"id:a": d}, (), 1.0, "A_auto"))
    assignment = Assignment("t", "id:a", ("id:a",), (0.9,), "accepted", (), snap.version, "fixture")
    pid = store.propose(assignment, {"cohort_uid": "cohort", "descriptor": {"global_feature": [1., 0.], "part_features": [], "part_valid": [], "part_quality": [], "encoder_fingerprint": "fixture"}, "source_observation_uids": ["q"]})
    new = store.commit([pid], snap.version)
    assert new.version != snap.version
    assert new.identities["id:a"].anchor_refs == ("anchor-0",)
    with pytest.raises(Exception): store.commit([pid], new.version)
    store.close()


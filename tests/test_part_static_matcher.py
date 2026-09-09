import numpy as np

from mat.core.types import IdentityDescriptor, LocalTracklet, PersistentIdentity
from mat.identity.gallery import GallerySnapshot
from mat.identity.part_matching import PartAwareStaticMatcher


def _desc(global_feature, parts, valid=None, quality=None):
    parts = np.asarray(parts, np.float32)
    valid = np.ones(parts.shape[0], bool) if valid is None else np.asarray(valid, bool)
    quality = np.ones(parts.shape[0], np.float32) if quality is None else np.asarray(quality, np.float32)
    return IdentityDescriptor(np.asarray(global_feature, np.float32), parts, valid, quality, "fixture")


def test_part_static_score_uses_quality_weighted_common_parts_and_global_fallback():
    ids = {
        "id:a": PersistentIdentity("id:a", "c", (), (), (), "r", {"group_label": "A"}),
        "id:b": PersistentIdentity("id:b", "c", (), (), (), "r", {"group_label": "B"}),
    }
    gallery = GallerySnapshot("c", "v0", ids, {
        "id:a": _desc([1, 0], [[1, 0], [0, 1]], quality=[1, 1]),
        "id:b": _desc([0, 1], [[0, 1], [1, 0]], quality=[1, 1]),
    }, "fixture")
    query = {"q": _desc([0, 1], [[1, 0], [0, 1]], quality=[1, 0])}
    tracklets = [LocalTracklet("q", "s", "cam", ["o"], [(0, 0)])]
    scores = PartAwareStaticMatcher().score(tracklets, query, gallery, global_weight=0.25)
    # Only the first part has quality, so id:a receives the higher fused score
    # despite its lower global cosine.
    assert scores.values[0, scores.identity_uids.index("id:a")] > scores.values[0, scores.identity_uids.index("id:b")]


def test_part_static_no_common_part_is_global_only():
    ids = {"id:a": PersistentIdentity("id:a", "c", (), (), (), "r", {})}
    gallery = GallerySnapshot("c", "v0", ids, {"id:a": _desc([1, 0], [[1, 0]], [True], [1])}, "fixture")
    query = {"q": _desc([0, 1], [[1, 0]], [False], [0])}
    score = PartAwareStaticMatcher().score([LocalTracklet("q", "s", "c", ["o"], [(0, 0)])], query, gallery,
                                            global_weight=0.25).values[0, 0]
    assert np.isclose(score, 0.0)

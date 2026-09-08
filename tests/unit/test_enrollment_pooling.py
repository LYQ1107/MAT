import numpy as np

from mat.core.types import IdentityDescriptor
from mat.enrollment.pooling import pool_identity_descriptors


def _d(global_feature, parts, valid, quality):
    return IdentityDescriptor(np.asarray(global_feature, np.float32), np.asarray(parts, np.float32),
                              np.asarray(valid, bool), np.asarray(quality, np.float32), "fixture")


def test_pooling_uses_all_refs_and_quality_weighted_parts():
    first = _d([1, 0], [[1, 0], [0, 1]], [True, True], [1.0, 0.1])
    second = _d([0, 1], [[0, 1], [1, 0]], [True, False], [0.1, 0.0])
    pooled = pool_identity_descriptors([first, second])
    # Global vector is the normalized mean, proving the second reference is
    # included rather than silently selecting refs[0].
    assert np.allclose(pooled.global_feature, [2 ** -0.5, 2 ** -0.5], atol=1e-6)
    # Head quality gives both, while the unavailable second part stays masked.
    assert pooled.part_valid.tolist() == [True, True]
    assert pooled.part_features[0, 0] > pooled.part_features[0, 1] > 0
    assert np.isclose(np.linalg.norm(pooled.part_features[0]), 1.0)
    assert np.allclose(pooled.part_features[1], [0, 1], atol=1e-6)

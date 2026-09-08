import numpy as np
import pytest

from mat.core.types import IdentityDescriptor
from mat.models.evidence_matcher import fixed_fusion_score


def _descriptor(global_value, part_value=None, fp="fixture"):
    global_value = np.asarray(global_value, dtype=np.float32)
    if part_value is None:
        parts = np.zeros((0, global_value.size), dtype=np.float32)
        valid = np.zeros(0, dtype=bool)
        quality = np.zeros(0, dtype=np.float32)
    else:
        parts = np.asarray(part_value, dtype=np.float32)
        valid = np.ones(parts.shape[0], dtype=bool)
        quality = np.ones(parts.shape[0], dtype=np.float32)
    return IdentityDescriptor(global_value, parts, valid, quality, fp)


def test_fixed_fusion_falls_back_to_global_without_common_parts():
    left = _descriptor([1, 0], [[1, 0]])
    right = _descriptor([1, 0], [[0, 1]])
    assert fixed_fusion_score(left, right) == pytest.approx(0.5)
    no_part = _descriptor([1, 0])
    assert fixed_fusion_score(no_part, _descriptor([1, 0], [[0, 1]])) == pytest.approx(1.0)


def test_trainable_matcher_masks_invalid_parts():
    torch = pytest.importorskip("torch")
    from mat.models.evidence_matcher import EvidenceMatcher
    matcher = EvidenceMatcher(feature_dim=4, num_parts=2)
    qg = torch.ones(1, 4)
    rg = torch.ones(1, 4)
    qp = torch.ones(1, 2, 4)
    rp = torch.ones(1, 2, 4)
    valid = torch.tensor([[True, False]])
    quality = torch.ones(1, 2)
    logits = matcher(qg, rg, qp, rp, valid, valid, quality, quality)
    assert logits.shape == (1,) and torch.isfinite(logits).all()

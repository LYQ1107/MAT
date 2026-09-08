"""Global/pose-part evidence matching.

``fixed_fusion_score`` is the deterministic B1a diagnostic.  The
``EvidenceMatcher`` module is a trainable, identity-label-free-at-inference
pair scorer; callers must train it only on S0/source identities.
"""

from __future__ import annotations

from typing import Any
import numpy as np

from mat.core.errors import DependencyUnavailableError, ValidationError
from mat.core.types import IdentityDescriptor

try:  # Optional on the lightweight interpreter.
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = object


def _cosine_numpy(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 0 else 0.0


def fixed_fusion_score(query: IdentityDescriptor, reference: IdentityDescriptor) -> float:
    """B1a score: global cosine plus quality-weighted common-part cosine."""
    if query.encoder_fingerprint != reference.encoder_fingerprint:
        raise ValidationError("feature-space fingerprint mismatch")
    global_score = _cosine_numpy(query.global_feature, reference.global_feature)
    common = np.asarray(query.part_valid, bool) & np.asarray(reference.part_valid, bool)
    if not np.any(common):
        return global_score
    part_scores = np.asarray([
        _cosine_numpy(query.part_features[i], reference.part_features[i])
        for i in np.flatnonzero(common)
    ], dtype=np.float32)
    weights = np.asarray(query.part_quality)[common] * np.asarray(reference.part_quality)[common]
    part_score = float(np.average(part_scores, weights=np.maximum(weights, 1e-6)))
    return float(0.5 * global_score + 0.5 * part_score)


class EvidenceMatcher(nn.Module if torch is not None else object):
    """Shared part-pair MLP and global-pair MLP match scorer.

    The forward method accepts either two :class:`IdentityDescriptor`
    instances or explicit batched tensors.  It returns one match logit per
    query/reference pair.  Invalid parts are masked before aggregation.
    """

    def __init__(self, feature_dim: int, num_parts: int, hidden_dim: int = 64):
        if torch is None:
            raise DependencyUnavailableError("PyTorch is required for EvidenceMatcher")
        super().__init__()
        if feature_dim <= 0 or num_parts < 0 or hidden_dim <= 0:
            raise ValidationError("invalid EvidenceMatcher dimensions")
        self.feature_dim = int(feature_dim)
        self.num_parts = int(num_parts)
        self.hidden_dim = int(hidden_dim)
        self.part_pair_mlp = nn.Sequential(
            nn.Linear(2 * feature_dim + 2, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.global_pair_mlp = nn.Sequential(
            nn.Linear(2 * feature_dim, 128), nn.GELU(),
            nn.Linear(128, hidden_dim),
        )
        self.match_head = nn.Sequential(
            nn.Linear(2 * hidden_dim + 2, hidden_dim), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _tensor(value: Any, *, dtype=None, device=None):
        if isinstance(value, torch.Tensor):
            result = value.to(device=device) if device is not None else value
            return result.to(dtype=dtype) if dtype is not None else result
        return torch.as_tensor(value, dtype=dtype, device=device)

    def _descriptor_args(self, query, reference, rest):
        if isinstance(query, IdentityDescriptor) and isinstance(reference, IdentityDescriptor):
            return (query.global_feature, reference.global_feature,
                    query.part_features, reference.part_features,
                    query.part_valid, reference.part_valid,
                    query.part_quality, reference.part_quality)
        if rest:
            raise ValidationError("explicit EvidenceMatcher tensors require eight arguments")
        raise ValidationError("query/reference must both be IdentityDescriptor instances")

    def forward(self, query, reference=None, *rest):
        if torch is None:  # pragma: no cover
            raise DependencyUnavailableError("PyTorch unavailable")
        if isinstance(query, IdentityDescriptor):
            if not isinstance(reference, IdentityDescriptor):
                raise ValidationError("query/reference descriptor types disagree")
            values = self._descriptor_args(query, reference, rest)
        else:
            if reference is None or len(rest) != 6:
                raise ValidationError("explicit EvidenceMatcher tensors require query and reference fields")
            values = (query, reference, *rest)
        qg, rg, qp, rp, qv, rv, qq, rq = values
        device = next(self.parameters()).device
        qg = self._tensor(qg, dtype=torch.float32, device=device)
        rg = self._tensor(rg, dtype=torch.float32, device=device)
        qp = self._tensor(qp, dtype=torch.float32, device=device)
        rp = self._tensor(rp, dtype=torch.float32, device=device)
        qv = self._tensor(qv, dtype=torch.bool, device=device)
        rv = self._tensor(rv, dtype=torch.bool, device=device)
        qq = self._tensor(qq, dtype=torch.float32, device=device)
        rq = self._tensor(rq, dtype=torch.float32, device=device)
        if qg.ndim == 1:
            qg, rg = qg.unsqueeze(0), rg.unsqueeze(0)
        if qp.ndim == 2:
            qp, rp = qp.unsqueeze(0), rp.unsqueeze(0)
        if qv.ndim == 1:
            qv, rv = qv.unsqueeze(0), rv.unsqueeze(0)
        if qq.ndim == 1:
            qq, rq = qq.unsqueeze(0), rq.unsqueeze(0)
        expected = (qg.shape[0], self.num_parts, self.feature_dim)
        if qp.shape != expected or rp.shape != expected or qv.shape != expected[:2] or rv.shape != expected[:2]:
            raise ValidationError("EvidenceMatcher part tensor shapes disagree")
        if qq.shape != expected[:2] or rq.shape != expected[:2] or rg.shape != qg.shape:
            raise ValidationError("EvidenceMatcher quality/global tensor shapes disagree")
        global_pair = torch.cat([qg * rg, torch.abs(qg - rg)], dim=-1)
        global_evidence = self.global_pair_mlp(global_pair)
        common = qv & rv
        part_pair = torch.cat([qp * rp, torch.abs(qp - rp), qq.unsqueeze(-1), rq.unsqueeze(-1)], dim=-1)
        part_evidence = self.part_pair_mlp(part_pair)
        mask = common.unsqueeze(-1).to(part_evidence.dtype)
        part_evidence = (part_evidence * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        common_fraction = common.float().mean(dim=1, keepdim=True) if self.num_parts else qg.new_zeros((qg.shape[0], 1))
        if self.num_parts:
            quality_mask = common.float()
            mean_quality = ((qq + rq) * 0.5 * quality_mask).sum(dim=1, keepdim=True) / quality_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        else:
            mean_quality = qg.new_zeros((qg.shape[0], 1))
        return self.match_head(torch.cat([global_evidence, part_evidence, common_fraction, mean_quality], dim=-1)).squeeze(-1)


__all__ = ["EvidenceMatcher", "fixed_fusion_score"]

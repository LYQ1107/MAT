from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from mat.core.errors import ValidationError
from mat.core.types import DescriptorBatch, IdentityDescriptor


@dataclass(frozen=True)
class TrackletDescriptorSummary:
    descriptor: IdentityDescriptor
    selected_sample_indices: tuple[int, ...]
    temporal_coverage: float
    per_part_support_count: tuple[int, ...]
    descriptor_consistency: float


class TrackletPooler:
    """Quality-weighted pooling with deterministic temporal coverage."""

    def __init__(self, max_samples: int = 32, trim_fraction: float = 0.1):
        if max_samples < 1 or not 0 <= trim_fraction < 0.5:
            raise ValidationError("invalid tracklet pooling parameters")
        self.max_samples = int(max_samples)
        self.trim_fraction = float(trim_fraction)

    def _select_indices(self, timestamps: np.ndarray, quality: np.ndarray) -> np.ndarray:
        order = np.argsort(timestamps, kind="mergesort")
        if len(order) <= self.max_samples:
            return order
        # Divide the complete time interval into bins and take the best quality
        # sample in each bin.  This prevents a 32-frame budget collapsing onto
        # one moment while still preferring clean observations.
        selected: list[int] = []
        edges = np.linspace(0, len(order), self.max_samples + 1)
        for start, stop in zip(edges[:-1], edges[1:]):
            members = order[int(np.floor(start)):max(int(np.ceil(stop)), int(np.floor(start)) + 1)]
            best = sorted(members.tolist(), key=lambda idx: (-float(quality[idx]), int(idx)))[0]
            selected.append(int(best))
        return np.asarray(selected, dtype=int)

    def aggregate_summary(self, descriptors: DescriptorBatch, timestamps: np.ndarray,
                          quality: np.ndarray) -> TrackletDescriptorSummary:
        g = np.asarray(descriptors.global_features, dtype=np.float32)
        p = np.asarray(descriptors.part_features, dtype=np.float32)
        valid = np.asarray(descriptors.part_valid, dtype=bool)
        part_quality = np.asarray(descriptors.part_quality, dtype=np.float32)
        q = np.asarray(quality, dtype=np.float32)
        t = np.asarray(timestamps, dtype=np.float64)
        if g.ndim != 2 or g.shape[0] == 0:
            raise ValidationError("global features must be a non-empty [N,D] array")
        n = g.shape[0]
        if q.shape != (n,) or t.shape != (n,):
            raise ValidationError("descriptor/timestamp/quality batch shape mismatch")
        if p.ndim != 3 or p.shape[0] != n or valid.shape != p.shape[:2] or part_quality.shape != p.shape[:2]:
            raise ValidationError("part descriptor batch shape mismatch")
        selected = self._select_indices(t, np.nan_to_num(q, nan=0.0))
        selected_t = t[selected]
        g, p, valid, part_quality, q = g[selected], p[selected], valid[selected], part_quality[selected], q[selected]
        q = np.clip(np.nan_to_num(q, nan=0.0), 0.0, None)
        if not np.any(q > 0):
            q = np.ones_like(q)
        weights = q / q.sum()
        pooled_g = np.sum(g * weights[:, None], axis=0)
        norm = np.linalg.norm(pooled_g)
        pooled_g = pooled_g / norm if norm > 0 else pooled_g
        parts = np.zeros((p.shape[1], p.shape[2]), dtype=np.float32)
        pooled_part_quality = np.zeros((p.shape[1],), dtype=np.float32)
        pooled_valid = np.any(valid, axis=0)
        support: list[int] = []
        for j in range(p.shape[1]):
            mask = valid[:, j] & (q > 0)
            support.append(int(mask.sum()))
            if np.any(mask):
                w = q[mask]
                parts[j] = np.sum(p[mask, j] * (w / w.sum())[:, None], axis=0)
                nrm = np.linalg.norm(parts[j])
                if nrm > 0:
                    parts[j] /= nrm
                pooled_part_quality[j] = float(np.average(np.clip(part_quality[mask, j], 0.0, 1.0), weights=w))
            else:
                pooled_valid[j] = False
        descriptor = IdentityDescriptor(pooled_g.astype(np.float32), parts, pooled_valid,
                                         pooled_part_quality, descriptors.encoder_fingerprint)
        # Mean cosine to the pooled vector is a compact consistency diagnostic;
        # it is never treated as a calibrated probability.
        norms = np.linalg.norm(g, axis=1) * max(float(np.linalg.norm(pooled_g)), 1e-12)
        cosines = np.sum(g * pooled_g[None, :], axis=1) / np.maximum(norms, 1e-12)
        consistency = float(np.clip(np.average(cosines, weights=weights), -1.0, 1.0))
        all_span = float(t.max() - t.min())
        temporal_coverage = 1.0 if all_span <= 0 else float(np.clip((selected_t.max() - selected_t.min()) / all_span, 0.0, 1.0))
        return TrackletDescriptorSummary(descriptor, tuple(int(i) for i in selected), temporal_coverage,
                                         tuple(support), consistency)

    def aggregate(self, descriptors: DescriptorBatch, timestamps: np.ndarray,
                  quality: np.ndarray, *, return_summary: bool = False):
        summary = self.aggregate_summary(descriptors, timestamps, quality)
        return summary if return_summary else summary.descriptor


__all__ = ["TrackletPooler", "TrackletDescriptorSummary"]

from __future__ import annotations

import numpy as np

from mat.core.errors import ValidationError
from mat.core.types import DescriptorBatch, IdentityDescriptor


class TrackletPooler:
    """Quality-weighted, temporally spread pooling for a bounded tracklet."""

    def __init__(self, max_samples: int = 32, trim_fraction: float = 0.1):
        if max_samples < 1 or not 0 <= trim_fraction < 0.5:
            raise ValidationError("invalid tracklet pooling parameters")
        self.max_samples = max_samples
        self.trim_fraction = trim_fraction

    def aggregate(self, descriptors: DescriptorBatch, timestamps: np.ndarray,
                  quality: np.ndarray) -> IdentityDescriptor:
        g = np.asarray(descriptors.global_features, dtype=np.float32)
        p = np.asarray(descriptors.part_features, dtype=np.float32)
        valid = np.asarray(descriptors.part_valid, dtype=bool)
        q = np.asarray(quality, dtype=np.float32)
        t = np.asarray(timestamps, dtype=np.float64)
        if g.ndim != 2 or g.shape[0] == 0 or q.shape != (g.shape[0],) or t.shape != q.shape:
            raise ValidationError("descriptor/timestamp/quality batch shape mismatch")
        order = np.argsort(t, kind="mergesort")
        if len(order) > self.max_samples:
            # Evenly cover the full time support; nearest ties are deterministic.
            positions = np.linspace(0, len(order) - 1, self.max_samples).round().astype(int)
            order = order[positions]
        g, p, valid, q = g[order], p[order], valid[order], q[order]
        q = np.clip(np.nan_to_num(q, nan=0.0), 0.0, None)
        if not np.any(q > 0):
            q = np.ones_like(q)
        weights = q / q.sum()
        pooled_g = np.sum(g * weights[:, None], axis=0)
        norm = np.linalg.norm(pooled_g)
        pooled_g = pooled_g / norm if norm > 0 else pooled_g
        parts = np.zeros((p.shape[1], p.shape[2]), dtype=np.float32)
        part_q = np.zeros((p.shape[1],), dtype=np.float32)
        pooled_valid = np.any(valid, axis=0)
        for j in range(p.shape[1]):
            mask = valid[:, j] & (q > 0)
            if np.any(mask):
                w = q[mask]
                parts[j] = np.sum(p[mask, j] * (w / w.sum())[:, None], axis=0)
                n = np.linalg.norm(parts[j])
                if n > 0:
                    parts[j] /= n
                part_q[j] = float(np.average(q[mask], weights=w))
        return IdentityDescriptor(pooled_g.astype(np.float32), parts, pooled_valid, part_q,
                                  descriptors.encoder_fingerprint)


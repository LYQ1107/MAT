"""Quality-weighted descriptor pooling shared by H and A enrollment."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any
import numpy as np

from mat.core.types import IdentityDescriptor
from mat.core.errors import ValidationError


def pool_identity_descriptors(descriptors: Iterable[IdentityDescriptor] | Mapping[str, IdentityDescriptor],
                              sample_quality: Sequence[float] | Mapping[str, float] | None = None
                              ) -> IdentityDescriptor:
    """Pool every reference descriptor with quality-weighted valid parts.

    Global descriptors use one non-negative sample-quality weight.  Each part
    additionally multiplies that weight by its own quality and is omitted when
    ``part_valid`` is false or the feature is non-finite.  The result is L2
    normalized in the same feature space and carries the common encoder
    fingerprint; no part gate or learned projection is introduced.
    """
    if isinstance(descriptors, Mapping):
        keys = list(descriptors)
        members = [descriptors[key] for key in keys]
    else:
        keys = list(range(len(descriptors))) if isinstance(descriptors, Sequence) else []
        members = list(descriptors)
    if not members:
        raise ValidationError("cannot pool an empty descriptor collection")
    if any(not isinstance(item, IdentityDescriptor) for item in members):
        raise ValidationError("pooling expects IdentityDescriptor values")
    fingerprints = {item.encoder_fingerprint for item in members}
    if len(fingerprints) != 1:
        raise ValidationError("reference descriptors use multiple encoder fingerprints")
    if sample_quality is None:
        weights = np.ones(len(members), dtype=np.float32)
    elif isinstance(sample_quality, Mapping):
        if not isinstance(descriptors, Mapping):
            raise ValidationError("mapping sample_quality requires mapping descriptors")
        weights = np.asarray([float(sample_quality.get(key, 1.0)) for key in keys], dtype=np.float32)
    else:
        weights = np.asarray(list(sample_quality), dtype=np.float32)
    if weights.shape != (len(members),) or np.any(~np.isfinite(weights)) or np.any(weights < 0):
        raise ValidationError("sample_quality must contain finite non-negative values for every descriptor")
    if not np.any(weights > 0):
        weights = np.ones(len(members), dtype=np.float32)

    global_values = np.stack([np.asarray(item.global_feature, dtype=np.float32) for item in members])
    if np.any(~np.isfinite(global_values)):
        raise ValidationError("global descriptor contains non-finite values")
    pooled_global = np.average(global_values, axis=0, weights=weights).astype(np.float32)
    norm = float(np.linalg.norm(pooled_global))
    if norm > 0:
        pooled_global /= norm

    part_count = members[0].part_features.shape[0]
    part_dim = members[0].part_features.shape[1]
    if any(item.part_features.shape != (part_count, part_dim) for item in members):
        raise ValidationError("part descriptor shapes disagree")
    pooled_parts = np.zeros((part_count, part_dim), dtype=np.float32)
    pooled_valid = np.zeros(part_count, dtype=bool)
    pooled_quality = np.zeros(part_count, dtype=np.float32)
    for index in range(part_count):
        valid_members = []
        valid_weights = []
        for member, base_weight in zip(members, weights):
            feature = member.part_features[index]
            quality = float(member.part_quality[index])
            if not bool(member.part_valid[index]) or not np.isfinite(feature).all() or not np.isfinite(quality):
                continue
            effective = float(base_weight) * max(quality, 0.0)
            if effective > 0:
                valid_members.append(member)
                valid_weights.append(effective)
        if not valid_members:
            continue
        values = np.stack([member.part_features[index] for member in valid_members]).astype(np.float32)
        part_weights = np.asarray(valid_weights, dtype=np.float32)
        pooled = np.average(values, axis=0, weights=part_weights).astype(np.float32)
        part_norm = float(np.linalg.norm(pooled))
        if part_norm > 0:
            pooled /= part_norm
        pooled_parts[index] = pooled
        pooled_valid[index] = True
        pooled_quality[index] = float(np.average(
            [float(member.part_quality[index]) for member in valid_members], weights=part_weights))
    return IdentityDescriptor(pooled_global, pooled_parts, pooled_valid, pooled_quality,
                              next(iter(fingerprints)))


__all__ = ["pool_identity_descriptors"]

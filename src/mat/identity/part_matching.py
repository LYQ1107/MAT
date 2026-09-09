"""Static-gallery matcher for the B1 global-plus-pose-part ablation."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from mat.core.errors import ProtocolError, ValidationError
from mat.core.types import IdentityDescriptor, LocalTracklet, ScoreMatrix
from mat.identity.matching import PersistentMatcher, _cosine


class PartAwareStaticMatcher(PersistentMatcher):
    """Score a frozen gallery with global and quality-weighted common parts.

    Assignment remains the deterministic B0 solver inherited from
    :class:`PersistentMatcher`; only the score matrix changes.  The gallery is
    never mutated and no identity truth is accepted by this class.
    """

    def score(
        self,
        tracklets: list[LocalTracklet],
        descriptors: dict[str, IdentityDescriptor],
        gallery: Any,
        *,
        global_weight: float = 0.5,
    ) -> ScoreMatrix:
        try:
            weight = float(global_weight)
        except (TypeError, ValueError) as exc:
            raise ValidationError("global_weight must be finite in [0,1]") from exc
        if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
            raise ValidationError("global_weight must be finite in [0,1]")
        identity_uids = tuple(sorted(gallery.descriptors))
        values = np.full((len(tracklets), len(identity_uids)), -np.inf, dtype=np.float32)
        for i, track in enumerate(tracklets):
            if track.tracklet_uid not in descriptors:
                raise ProtocolError(f"missing descriptor for tracklet {track.tracklet_uid}")
            query = descriptors[track.tracklet_uid]
            for j, uid in enumerate(identity_uids):
                reference = gallery.descriptors[uid]
                if query.encoder_fingerprint != reference.encoder_fingerprint:
                    raise ProtocolError(
                        f"feature-space mismatch for {track.tracklet_uid}/{uid}: "
                        f"{query.encoder_fingerprint} != {reference.encoder_fingerprint}"
                    )
                global_score = _cosine(query.global_feature, reference.global_feature)
                q_parts = np.asarray(query.part_features)
                r_parts = np.asarray(reference.part_features)
                q_valid = np.asarray(query.part_valid, dtype=bool)
                r_valid = np.asarray(reference.part_valid, dtype=bool)
                common_count = min(len(q_valid), len(r_valid), q_parts.shape[0], r_parts.shape[0])
                if common_count <= 0:
                    values[i, j] = global_score
                    continue
                common = q_valid[:common_count] & r_valid[:common_count]
                if not np.any(common):
                    values[i, j] = global_score
                    continue
                part_scores = np.asarray([
                    _cosine(q_parts[k], r_parts[k]) for k in range(common_count) if common[k]
                ], dtype=np.float64)
                q_quality = np.asarray(query.part_quality, dtype=np.float64)[:common_count][common]
                r_quality = np.asarray(reference.part_quality, dtype=np.float64)[:common_count][common]
                quality = np.clip(q_quality, 0.0, None) * np.clip(r_quality, 0.0, None)
                if not np.any(quality > 0):
                    part_score = float(part_scores.mean())
                else:
                    part_score = float(np.average(part_scores, weights=quality))
                values[i, j] = float(weight * global_score + (1.0 - weight) * part_score)
        return ScoreMatrix(tuple(track.tracklet_uid for track in tracklets), identity_uids, values)


__all__ = ["PartAwareStaticMatcher"]

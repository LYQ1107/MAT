from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np

from mat.core.types import Assignment, IdentityDescriptor, LocalTracklet, ScoreMatrix
from mat.core.errors import ProtocolError
from mat.identity.conflicts import ConflictGraph, ConflictGraphBuilder

__all__ = ["ConflictGraph", "ConflictGraphBuilder", "MatchingPolicy", "PersistentMatcher", "StaticGalleryMatcher"]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na <= 0 or nb <= 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


@dataclass(frozen=True)
class MatchingPolicy:
    accept_threshold: float = 0.35
    unknown_cost: float = 0.0
    top_k: int = 3
    solver: str = "deterministic_greedy"
    model_version: str = "unresolved"


class PersistentMatcher:
    def score(self, tracklets: list[LocalTracklet], descriptors: dict[str, IdentityDescriptor], gallery) -> ScoreMatrix:
        identity_uids = tuple(sorted(gallery.descriptors))
        values = np.full((len(tracklets), len(identity_uids)), -np.inf, dtype=np.float32)
        for i, track in enumerate(tracklets):
            query = descriptors[track.tracklet_uid]
            for j, uid in enumerate(identity_uids):
                ref = gallery.descriptors[uid]
                if query.encoder_fingerprint != ref.encoder_fingerprint:
                    raise ProtocolError(
                        f"feature-space mismatch for {track.tracklet_uid}/{uid}: "
                        f"{query.encoder_fingerprint} != {ref.encoder_fingerprint}"
                    )
                global_score = _cosine(query.global_feature, ref.global_feature)
                common = query.part_valid & ref.part_valid
                if np.any(common):
                    part_scores = np.array([_cosine(query.part_features[k], ref.part_features[k]) for k in range(len(common))])
                    weights = query.part_quality * ref.part_quality
                    part = float(np.average(part_scores[common], weights=np.maximum(weights[common], 1e-6)))
                    score = 0.5 * global_score + 0.5 * part
                else:
                    score = global_score
                values[i, j] = score
        return ScoreMatrix(tuple(t.tracklet_uid for t in tracklets), identity_uids, values)

    def assign(self, scores: ScoreMatrix, conflicts: ConflictGraph,
               policy: MatchingPolicy | None = None) -> list[Assignment]:
        policy = policy or MatchingPolicy()
        assignments: dict[str, str | None] = {}
        reasons: dict[str, list[str]] = {tid: [] for tid in scores.tracklet_uids}
        # Sort all candidate edges globally for deterministic maximum-score behavior;
        # identity capacity is only constrained by conflict pairs, not by session-wide 1:1.
        edges = []
        for i, tid in enumerate(scores.tracklet_uids):
            order = np.argsort(-scores.values[i], kind="stable")[:policy.top_k]
            for j in order:
                if np.isfinite(scores.values[i, j]):
                    edges.append((float(scores.values[i, j]), tid, scores.identity_uids[j]))
        edges.sort(key=lambda e: (-e[0], e[1], e[2]))
        assigned: dict[str, str] = {}
        for score, tid, identity in edges:
            if tid in assignments:
                continue
            if score < policy.accept_threshold:
                continue
            if any(other_tid != tid and other_id == identity and conflicts.conflicts(tid, other_tid)
                   for other_tid, other_id in assigned.items()):
                reasons[tid].append(f"cannot_link:{identity}")
                continue
            assignments[tid] = identity
            assigned[tid] = identity
        result = []
        for i, tid in enumerate(scores.tracklet_uids):
            row = scores.values[i]
            order = np.argsort(-row, kind="stable")[:policy.top_k]
            candidates = tuple(scores.identity_uids[j] for j in order if np.isfinite(row[j]))
            candidate_scores = tuple(float(row[j]) for j in order if np.isfinite(row[j]))
            if tid in assignments:
                result.append(Assignment(tid, assignments[tid], candidates, candidate_scores, "accepted",
                                         tuple(reasons[tid]), "unresolved", policy.model_version))
            else:
                why = reasons[tid] or (["below_threshold"] if len(candidate_scores) else ["no_gallery_evidence"])
                result.append(Assignment(tid, None, candidates, candidate_scores, "unregistered",
                                         tuple(why), "unresolved", policy.model_version))
        return result


class StaticGalleryMatcher(PersistentMatcher):
    """B0 alias: a matcher that never mutates the gallery."""

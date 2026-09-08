from __future__ import annotations

from dataclasses import replace
from typing import Iterable
import hashlib
import numpy as np

from mat.core.types import IdentityDescriptor, LocalTracklet, PersistentIdentity
from mat.identity.conflicts import ConflictGraphBuilder
from .base import EnrollmentResult


def _similar(a: IdentityDescriptor, b: IdentityDescriptor) -> float:
    na, nb = np.linalg.norm(a.global_feature), np.linalg.norm(b.global_feature)
    return float(np.dot(a.global_feature, b.global_feature) / max(na * nb, 1e-9))


def _prototype(tracklet_uids: list[str], descriptors: dict[str, IdentityDescriptor]) -> IdentityDescriptor:
    """Pool every member of a cluster instead of anchoring on cluster[0]."""
    members = [descriptors[uid] for uid in tracklet_uids]
    fingerprints = {item.encoder_fingerprint for item in members}
    if len(fingerprints) != 1:
        raise ValueError("reference descriptors use multiple encoder fingerprints")
    global_feature = np.mean(np.stack([item.global_feature for item in members]), axis=0)
    norm = np.linalg.norm(global_feature)
    if norm > 0:
        global_feature = global_feature / norm
    part_count = members[0].part_features.shape[0]
    part_dim = members[0].part_features.shape[1] if members[0].part_features.ndim == 2 else 0
    parts = np.zeros((part_count, part_dim), dtype=np.float32)
    valid = np.zeros(part_count, dtype=bool)
    quality = np.zeros(part_count, dtype=np.float32)
    for index in range(part_count):
        available = [item for item in members if item.part_valid[index]]
        if not available:
            continue
        weights = np.asarray([max(float(item.part_quality[index]), 1e-6) for item in available], dtype=np.float32)
        values = np.stack([item.part_features[index] for item in available])
        parts[index] = np.average(values, axis=0, weights=weights)
        part_norm = np.linalg.norm(parts[index])
        if part_norm > 0:
            parts[index] /= part_norm
        valid[index] = True
        quality[index] = float(np.average([item.part_quality[index] for item in available], weights=weights))
    return IdentityDescriptor(global_feature.astype(np.float32), parts, valid, quality, members[0].encoder_fingerprint)


class AutoRegistrar:
    def __init__(self, merge_threshold: float = 0.82):
        self.merge_threshold = merge_threshold

    def build(self, reference, *, cohort_uid: str | None = None) -> EnrollmentResult:
        tracklets: list[LocalTracklet] = list(reference.tracklets if hasattr(reference, "tracklets") else reference["tracklets"])
        descriptors: dict[str, IdentityDescriptor] = reference.descriptors if hasattr(reference, "descriptors") else reference["descriptors"]
        session_uid = reference.session_uid if hasattr(reference, "session_uid") else reference["session_uid"]
        cohort_uid = cohort_uid or (reference.cohort_uid if hasattr(reference, "cohort_uid") else reference.get("cohort_uid", "cohort:unresolved"))
        graph = ConflictGraphBuilder().build(tracklets)
        clusters: list[list[str]] = []
        for track in sorted(tracklets, key=lambda t: t.tracklet_uid):
            placed = False
            for cluster in clusters:
                if any(graph.conflicts(track.tracklet_uid, other) for other in cluster):
                    continue
                mean = _prototype(cluster, descriptors)
                if _similar(descriptors[track.tracklet_uid], mean) >= self.merge_threshold:
                    cluster.append(track.tracklet_uid)
                    placed = True
                    break
            if not placed:
                clusters.append([track.tracklet_uid])
        identities: list[PersistentIdentity] = []
        out_desc: dict[str, IdentityDescriptor] = {}
        for index, cluster in enumerate(clusters):
            uid = "id:" + hashlib.sha256(f"{cohort_uid}:{session_uid}:{index}:{','.join(cluster)}".encode()).hexdigest()[:20]
            identities.append(PersistentIdentity(uid, cohort_uid, tuple(cluster), (), (), session_uid,
                                                 {"protocol": "A_auto", "merge_threshold": self.merge_threshold}))
            out_desc[uid] = _prototype(cluster, descriptors)
        unresolved = tuple(t for cluster in clusters if len(cluster) == 1 for t in cluster)
        return EnrollmentResult(cohort_uid, session_uid, tuple(identities), out_desc, unresolved,
                                None, "A_auto", "SUCCEEDED")

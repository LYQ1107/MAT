from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict
from typing import Any
import json
import numpy as np

from mat.core.types import LocalTracklet, DescriptorBatch
from mat.models.tracklet_pooler import TrackletPooler
from mat.identity.gallery import GalleryStore
from mat.enrollment.automatic import AutoRegistrar
from mat.identity.conflicts import ConflictGraphBuilder
from mat.identity.matching import PersistentMatcher, MatchingPolicy
from mat.data.sanitize import FORBIDDEN_MODEL_FIELDS


@dataclass(frozen=True)
class B0Result:
    status: str
    input_mode: str
    enrollment_mode: str
    reference_session_uid: str
    query_sessions: tuple[str, ...]
    assignments: tuple[dict[str, Any], ...]
    gallery_version: str | None
    blocker: str | None = None

    def write(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _tracklets(rows: list[dict[str, Any]]) -> list[LocalTracklet]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get("tracklet_uid", row["observation_uid"])].append(row)
    out = []
    for uid, members in sorted(grouped.items()):
        members.sort(key=lambda r: (float(r.get("timestamp_s", 0)), str(r["observation_uid"])))
        out.append(LocalTracklet(uid, str(members[0]["session_uid"]), str(members[0].get("camera_uid", "unknown")),
                                  [str(m["observation_uid"]) for m in members],
                                  [(float(members[0].get("timestamp_s", 0)), float(members[-1].get("timestamp_s", 0)) + 1e-9)],
                                  {"sample_count": len(members), "input_mode": "prelocalized_crops"}))
    return out


def _pool(rows, encoder, max_samples=32):
    by_track = defaultdict(list)
    for i, row in enumerate(rows):
        by_track[row.get("tracklet_uid", row["observation_uid"])].append(i)
    images = np.asarray([row["image"] for row in rows])
    encoded = encoder.encode(images)
    result = {}
    pooler = TrackletPooler(max_samples=max_samples)
    for uid, indices in by_track.items():
        batch = DescriptorBatch(encoded.global_features[indices], encoded.part_features[indices],
                                encoded.part_valid[indices], encoded.part_quality[indices], encoded.encoder_fingerprint)
        timestamps = np.asarray([rows[i].get("timestamp_s", i) for i in indices], float)
        quality = np.asarray([rows[i].get("quality", 1.0) for i in indices], float)
        result[uid] = pooler.aggregate(batch, timestamps, quality)
    return result


def run_prelocalized_b0(observation_rows: list[dict[str, Any]], encoder, gallery_store: GalleryStore,
                        reference_session_uid: str, query_session_uids: list[str], *, cohort_uid: str = "pilot-cohort") -> B0Result:
    """Run B0 when neutral rows and a verified/frozen encoder are supplied.

    This path deliberately rejects identity labels in model rows and always marks the
    result ``prelocalized_crops``; it cannot be used to claim A end-to-end enrollment.
    """
    if any(FORBIDDEN_MODEL_FIELDS.intersection(row) for row in observation_rows):
        return B0Result("FAILED_LEAKAGE_CONTRACT", "prelocalized_crops", "A_auto", reference_session_uid, tuple(query_session_uids), (), None, "ground truth field in model rows")
    ref_rows = [r for r in observation_rows if r.get("session_uid") == reference_session_uid]
    if not ref_rows:
        return B0Result("BLOCKED_MISSING_REFERENCE", "prelocalized_crops", "A_auto", reference_session_uid, tuple(query_session_uids), (), None, "no S0 rows")
    ref_desc = _pool(ref_rows, encoder)
    reference = type("Reference", (), {"tracklets": _tracklets(ref_rows), "descriptors": ref_desc,
                                       "session_uid": reference_session_uid, "cohort_uid": cohort_uid})()
    enrollment = AutoRegistrar().build(reference, cohort_uid=cohort_uid)
    snapshot = gallery_store.create(enrollment)
    assignments = []
    for session_uid in query_session_uids:
        rows = [r for r in observation_rows if r.get("session_uid") == session_uid]
        tracks = _tracklets(rows)
        if not tracks:
            continue
        desc = _pool(rows, encoder)
        scores = PersistentMatcher().score(tracks, desc, snapshot)
        assigned = PersistentMatcher().assign(scores, ConflictGraphBuilder().build(tracks),
                                               MatchingPolicy(model_version=snapshot.fingerprint))
        assignments.extend(a.__dict__ for a in assigned)
    return B0Result("SUCCEEDED", "prelocalized_crops", "A_auto", reference_session_uid, tuple(query_session_uids),
                    tuple(assignments), snapshot.version)

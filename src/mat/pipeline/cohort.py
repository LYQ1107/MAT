from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
import hashlib
import json

import numpy as np

from mat.core.types import SessionSpec, LocalTracklet, IdentityDescriptor
from mat.enrollment.automatic import AutoRegistrar
from mat.enrollment.manual import ManualRegistrar
from mat.identity.gallery import GalleryStore, GallerySnapshot
from mat.identity.conflicts import ConflictGraphBuilder
from mat.identity.matching import PersistentMatcher, MatchingPolicy
from mat.identity.longitudinal_gallery import (
    IdentityExemplar,
    LongitudinalGallerySnapshot,
    LongitudinalGalleryStore,
    score_profile,
)
from mat.models.memory_commit_gate import MemoryCommitGate


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    status: str
    enrollment_mode: str
    registry_start_version: str
    registry_end_version: str
    sessions: tuple[str, ...]
    blockers: tuple[str, ...]
    metrics: dict[str, Any]

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.__dict__, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


class CohortRunner:
    """Chronological runner; detector/pose extraction is injected at the boundary."""

    def __init__(self, gallery_store: GalleryStore):
        self.gallery_store = gallery_store

    def enroll(self, reference: Any, protocol: str = "auto", verification: Any = None) -> GallerySnapshot:
        if protocol == "auto":
            result = AutoRegistrar().build(reference)
        elif protocol in {"human", "oracle_reference"}:
            if protocol == "oracle_reference" and verification is not None:
                if isinstance(verification, dict):
                    verification = dict(verification); verification["provenance"] = "oracle_reference"
            result = ManualRegistrar().build(reference, verification)
        else:
            raise ValueError(f"unknown enrollment protocol {protocol}")
        return self.gallery_store.create(result)

    def run(self, sessions: list[SessionSpec], gallery: GallerySnapshot,
            config: dict[str, Any], *, extract: Callable[[SessionSpec], tuple[list[LocalTracklet], dict[str, IdentityDescriptor]]] | None = None) -> RunManifest:
        blockers: list[str] = []
        current = gallery
        all_ids = []
        for session in sorted(sessions, key=lambda s: (s.recorded_at or "", s.session_uid)):
            # The caller must supply a session-start snapshot; no future session is inspected.
            current = self.gallery_store.snapshot(current.cohort_uid, current.version)
            if extract is None:
                blockers.append(f"{session.session_uid}:BLOCKED_MISSING_FRONTEND")
                continue
            tracklets, descriptors = extract(session)
            scores = PersistentMatcher().score(tracklets, descriptors, current)
            assignments = PersistentMatcher().assign(scores, ConflictGraphBuilder().build(tracklets),
                                                      MatchingPolicy(model_version=current.fingerprint))
            all_ids.extend(a.persistent_uid for a in assignments if a.persistent_uid)
        run_id = hashlib.sha256(f"{gallery.cohort_uid}:{gallery.version}:{','.join(s.session_uid for s in sessions)}".encode()).hexdigest()[:16]
        status = "SUCCEEDED" if not blockers else "BLOCKED_MISSING_FRONTEND"
        return RunManifest(run_id, status, str(config.get("protocol", {}).get("enrollment", "unknown")),
                           gallery.version, current.version, tuple(s.session_uid for s in sessions), tuple(blockers),
                           {"assigned_tracklets": len(all_ids)})


class LongitudinalCohortRunner:
    """End-of-session runner for the three-tier longitudinal gallery.

    ``extract`` is called once per session with the immutable snapshot at that
    session's start and must return ``(tracklets, descriptors[, evidence])``.
    The optional evidence mapping is keyed by tracklet UID and may contain an
    :class:`IdentityDescriptor`, quality and independent-window count.  All
    proposals are committed only after the session callback returns.
    """

    def __init__(self, gallery_store: LongitudinalGalleryStore,
                 gate: MemoryCommitGate | None = None):
        self.gallery_store = gallery_store
        self.gate = gate or MemoryCommitGate()

    @staticmethod
    def _score_matrix(tracklets, descriptors, snapshot: LongitudinalGallerySnapshot):
        identity_uids = tuple(sorted(snapshot.profiles))
        values = np.full((len(tracklets), len(identity_uids)), -np.inf, dtype=np.float32)
        for row, tracklet in enumerate(tracklets):
            query = descriptors[tracklet.tracklet_uid]
            for col, identity_uid in enumerate(identity_uids):
                values[row, col] = score_profile(query, snapshot.profiles[identity_uid])
        from mat.core.types import ScoreMatrix
        return ScoreMatrix(tuple(t.tracklet_uid for t in tracklets), identity_uids, values)

    def run(self, sessions: list[SessionSpec], gallery: LongitudinalGallerySnapshot,
            *, extract: Callable, config: dict[str, Any] | None = None) -> RunManifest:
        current = gallery
        blockers: list[str] = []
        assigned_count = 0
        unknown_count = 0
        pending_count = committed_count = rejected_count = 0
        records: list[dict[str, Any]] = []
        for session in sorted(sessions, key=lambda item: (item.recorded_at is None, item.recorded_at or "", item.session_uid)):
            start = current
            try:
                extracted = extract(session, start)
            except Exception as exc:
                blockers.append(f"{session.session_uid}:extract:{type(exc).__name__}")
                records.append({"session_uid": session.session_uid, "start_gallery_version": start.version, "end_gallery_version": start.version, "status": "FAILED_EXTRACT"})
                continue
            if len(extracted) == 2:
                tracklets, descriptors = extracted
                evidence_map = {}
            else:
                tracklets, descriptors, evidence_map = extracted
            scores = self._score_matrix(tracklets, descriptors, start)
            assignments = PersistentMatcher().assign(scores, ConflictGraphBuilder().build(tracklets), MatchingPolicy(model_version=start.encoder_fingerprint))
            assigned_count += sum(item.persistent_uid is not None for item in assignments)
            unknown_count += sum(item.persistent_uid is None for item in assignments)
            proposal_ids: list[str] = []
            for assignment in assignments:
                if assignment.persistent_uid is None or assignment.tracklet_uid not in evidence_map:
                    continue
                evidence = evidence_map[assignment.tracklet_uid]
                descriptor = evidence.get("descriptor") if isinstance(evidence, dict) else getattr(evidence, "descriptor", None)
                if descriptor is None:
                    continue
                quality = float(evidence.get("quality", 0.0) if isinstance(evidence, dict) else getattr(evidence, "quality", 0.0))
                windows = int(evidence.get("independent_windows", 0) if isinstance(evidence, dict) else getattr(evidence, "independent_windows", 0))
                row_index = scores.tracklet_uids.index(assignment.tracklet_uid)
                row_scores = sorted((float(x) for x in scores.values[row_index] if np.isfinite(x)), reverse=True)
                top1 = row_scores[0] if row_scores else float("-inf")
                top2 = row_scores[1] if len(row_scores) > 1 else float("-inf")
                profile = start.profiles[assignment.persistent_uid]
                # A first strong observation enters quarantine.  Only an
                # identity that already has a pending exemplar can be
                # promoted on a later session; this prevents a single frame
                # from silently mutating confirmed memory.
                current_role = "pending" if profile.pending else "confirmed"
                decision = self.gate.evaluate(top1_score=top1, top2_score=top2, tracklet_quality=quality,
                                              independent_window_count=windows, has_conflict=False,
                                              current_role=current_role)
                if decision.action not in {"promote", "accept_to_quarantine"}:
                    continue
                role = "confirmed" if decision.action == "promote" else "pending"
                proposal_ids.append(self.gallery_store.propose(
                    current.cohort_uid, start.version,
                    IdentityExemplar(
                        exemplar_uid=f"{role}:{session.session_uid}:{assignment.tracklet_uid}",
                        identity_uid=assignment.persistent_uid, role=role, descriptor=descriptor,
                        source_session_uid=session.session_uid, source_tracklet_uid=assignment.tracklet_uid,
                        quality=max(0.0, min(1.0, quality)), encoder_fingerprint=descriptor.encoder_fingerprint,
                        created_at=datetime.now(timezone.utc).isoformat(),
                    ),
                ))
                if role == "confirmed":
                    committed_count += 1
                else:
                    pending_count += 1
            if proposal_ids:
                current = self.gallery_store.commit(proposal_ids, start.version)
            records.append({"session_uid": session.session_uid, "start_gallery_version": start.version,
                            "end_gallery_version": current.version, "number_pending": pending_count,
                            "number_committed": committed_count, "number_rejected": rejected_count,
                            "unknown_count": unknown_count})
        run_id = hashlib.sha256(f"{gallery.cohort_uid}:{gallery.version}:{','.join(s.session_uid for s in sessions)}".encode()).hexdigest()[:16]
        status = "SUCCEEDED" if not blockers else "BLOCKED_MISSING_FRONTEND"
        return RunManifest(run_id, status, str((config or {}).get("protocol", {}).get("enrollment", "unknown")),
                           gallery.version, current.version, tuple(s.session_uid for s in sessions), tuple(blockers),
                           {"assigned_tracklets": assigned_count, "unknown_count": unknown_count,
                            "number_pending": pending_count, "number_committed": committed_count,
                            "number_rejected": rejected_count, "session_records": records,
                            "protocol": "end_of_session"})

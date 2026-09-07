from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import hashlib
import json

from mat.core.types import SessionSpec, LocalTracklet, IdentityDescriptor
from mat.enrollment.automatic import AutoRegistrar
from mat.enrollment.manual import ManualRegistrar
from mat.identity.gallery import GalleryStore, GallerySnapshot
from mat.identity.conflicts import ConflictGraphBuilder
from mat.identity.matching import PersistentMatcher, MatchingPolicy


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


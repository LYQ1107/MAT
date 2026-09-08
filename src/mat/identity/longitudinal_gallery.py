"""End-of-session, three-tier identity memory.

This store is intentionally separate from :mod:`mat.identity.gallery`, which
remains the immutable/static B0 registry.  A snapshot is read for the whole
session; proposals are committed only after that session finishes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Iterable
import uuid

import numpy as np

from mat.core.errors import ProtocolError, ValidationError
from mat.core.types import IdentityDescriptor, PersistentIdentity
from mat.enrollment.base import EnrollmentResult


@dataclass(frozen=True)
class IdentityExemplar:
    exemplar_uid: str
    identity_uid: str
    role: str
    descriptor: IdentityDescriptor
    source_session_uid: str
    source_tracklet_uid: str
    quality: float
    encoder_fingerprint: str
    created_at: str

    def __post_init__(self) -> None:
        if self.role not in {"anchor", "confirmed", "pending"}:
            raise ValidationError("exemplar role must be anchor, confirmed or pending")
        if not 0.0 <= float(self.quality) <= 1.0:
            raise ValidationError("exemplar quality must be in [0,1]")
        if self.encoder_fingerprint != self.descriptor.encoder_fingerprint:
            raise ValidationError("exemplar/descriptor fingerprint mismatch")


@dataclass(frozen=True)
class IdentityProfile:
    identity_uid: str
    anchors: tuple[IdentityExemplar, ...] = ()
    confirmed: tuple[IdentityExemplar, ...] = ()
    pending: tuple[IdentityExemplar, ...] = ()

    def all_scoring_exemplars(self) -> tuple[IdentityExemplar, ...]:
        # Pending samples never confirm or score another pending sample.
        return self.anchors + self.confirmed


@dataclass(frozen=True)
class LongitudinalGallerySnapshot:
    cohort_uid: str
    version: str
    profiles: dict[str, IdentityProfile]
    encoder_fingerprint: str


@dataclass(frozen=True)
class ExemplarProposal:
    proposal_uid: str
    cohort_uid: str
    base_version: str
    exemplar: IdentityExemplar
    status: str = "proposed"


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 0 else 0.0


def score_profile(query: IdentityDescriptor, profile: IdentityProfile, *, top_k: int = 3) -> float:
    """Score query against anchor/confirmed exemplars using top-k mean."""
    if top_k <= 0:
        raise ValidationError("top_k must be positive")
    scoring_exemplars = profile.all_scoring_exemplars()
    if scoring_exemplars and any(query.encoder_fingerprint != exemplar.encoder_fingerprint for exemplar in scoring_exemplars):
        raise ProtocolError("feature-space fingerprint mismatch")
    exemplars = scoring_exemplars
    if not exemplars:
        return float("-inf")
    anchor = [e for e in profile.anchors]
    confirmed = [e for e in profile.confirmed]

    def top_mean(items: list[IdentityExemplar]) -> float:
        if not items:
            return float("-inf")
        scores = sorted((_cosine(query.global_feature, e.descriptor.global_feature) for e in items), reverse=True)
        return float(np.mean(scores[: max(1, min(int(top_k), len(scores)))]))

    anchor_score = top_mean(anchor)
    confirmed_score = top_mean(confirmed)
    if not anchor:
        return confirmed_score
    if not confirmed:
        return anchor_score
    return float(0.6 * anchor_score + 0.4 * confirmed_score)


class LongitudinalGalleryStore:
    """Small auditable in-memory store with immutable version snapshots."""

    def __init__(self) -> None:
        self._snapshots: dict[tuple[str, str], LongitudinalGallerySnapshot] = {}
        self._heads: dict[str, str] = {}
        self._proposals: dict[str, ExemplarProposal] = {}

    @staticmethod
    def _fingerprint(descriptors: Iterable[IdentityDescriptor]) -> str:
        values = tuple(sorted({d.encoder_fingerprint for d in descriptors}))
        if len(values) != 1:
            raise ProtocolError("all gallery descriptors must share one encoder fingerprint")
        return values[0]

    def create(self, enrollment: EnrollmentResult) -> LongitudinalGallerySnapshot:
        if enrollment.status.startswith("BLOCKED"):
            raise ProtocolError(f"cannot create longitudinal gallery from {enrollment.status}")
        if enrollment.cohort_uid in self._heads:
            raise ProtocolError(f"cohort already exists: {enrollment.cohort_uid}")
        identities = {item.identity_uid: item for item in enrollment.identities}
        profiles: dict[str, IdentityProfile] = {}
        now = datetime.now(timezone.utc).isoformat()
        for uid in sorted(identities):
            descriptor = enrollment.descriptors[uid]
            exemplar = IdentityExemplar(
                exemplar_uid=f"anchor:{uid}", identity_uid=uid, role="anchor",
                descriptor=descriptor, source_session_uid=enrollment.reference_session_uid,
                source_tracklet_uid=identities[uid].anchor_refs[0] if identities[uid].anchor_refs else "",
                quality=1.0, encoder_fingerprint=descriptor.encoder_fingerprint, created_at=now,
            )
            profiles[uid] = IdentityProfile(uid, (exemplar,), (), ())
        fingerprint = self._fingerprint(enrollment.descriptors.values())
        snapshot = LongitudinalGallerySnapshot(enrollment.cohort_uid, "v0", profiles, fingerprint)
        self._snapshots[(snapshot.cohort_uid, snapshot.version)] = snapshot
        self._heads[snapshot.cohort_uid] = snapshot.version
        return snapshot

    def snapshot(self, cohort_uid: str, version: str | None = None) -> LongitudinalGallerySnapshot:
        version = version or self._heads.get(cohort_uid)
        if version is None or (cohort_uid, version) not in self._snapshots:
            raise ProtocolError(f"unknown longitudinal gallery {cohort_uid}/{version}")
        return self._snapshots[(cohort_uid, version)]

    def propose(self, cohort_uid: str, expected_version: str, exemplar: IdentityExemplar) -> str:
        current = self.snapshot(cohort_uid)
        if current.version != expected_version:
            raise ProtocolError("stale gallery version for proposal")
        if exemplar.identity_uid not in current.profiles:
            raise ProtocolError("proposal references unknown identity")
        if exemplar.role == "anchor":
            raise ProtocolError("anchors are immutable and cannot be proposed")
        if exemplar.encoder_fingerprint != current.encoder_fingerprint:
            raise ProtocolError("proposal encoder fingerprint mismatch")
        proposal_uid = str(uuid.uuid4())
        self._proposals[proposal_uid] = ExemplarProposal(proposal_uid, cohort_uid, expected_version, exemplar)
        return proposal_uid

    def commit(self, proposal_uids: list[str], expected_version: str) -> LongitudinalGallerySnapshot:
        if not proposal_uids:
            raise ValidationError("commit requires at least one proposal")
        proposals = [self._proposals.get(uid) for uid in proposal_uids]
        if any(item is None for item in proposals):
            raise ProtocolError("unknown exemplar proposal")
        first = proposals[0]
        assert first is not None
        if any(item.cohort_uid != first.cohort_uid or item.base_version != expected_version or item.status != "proposed" for item in proposals if item is not None):
            raise ProtocolError("proposal cohort/version/status mismatch")
        current = self.snapshot(first.cohort_uid)
        if current.version != expected_version:
            raise ProtocolError("stale expected gallery version")
        profiles = dict(current.profiles)
        for item in proposals:
            assert item is not None
            profile = profiles[item.exemplar.identity_uid]
            if item.exemplar.role == "confirmed":
                # Promotion consumes the quarantine evidence in the new head.
                # The old snapshot remains immutable/auditable, but a pending
                # exemplar must not remain in the active profile indefinitely.
                profiles[item.exemplar.identity_uid] = replace(
                    profile, confirmed=profile.confirmed + (item.exemplar,), pending=(),
                )
            elif item.exemplar.role == "pending":
                profiles[item.exemplar.identity_uid] = replace(profile, pending=profile.pending + (item.exemplar,))
            self._proposals[item.proposal_uid] = replace(item, status="committed")
        number = int(expected_version[1:]) + 1 if expected_version.startswith("v") and expected_version[1:].isdigit() else len(self._snapshots)
        snapshot = LongitudinalGallerySnapshot(first.cohort_uid, f"v{number}", profiles, current.encoder_fingerprint)
        self._snapshots[(snapshot.cohort_uid, snapshot.version)] = snapshot
        self._heads[snapshot.cohort_uid] = snapshot.version
        return snapshot

    def reject(self, proposal_uid: str, reason: str) -> None:
        proposal = self._proposals.get(proposal_uid)
        if proposal is None or proposal.status != "proposed":
            raise ProtocolError("proposal missing or finalized")
        self._proposals[proposal_uid] = replace(proposal, status=f"rejected:{reason}")

    def rollback(self, target_version: str, cohort_uid: str | None = None) -> LongitudinalGallerySnapshot:
        """Publish a new head copied from an older snapshot without deleting history."""
        if cohort_uid is None:
            matches = [uid for (uid, version) in self._snapshots if version == target_version]
            if len(matches) != 1:
                raise ProtocolError("target version is ambiguous or missing")
            cohort_uid = matches[0]
        target = self.snapshot(cohort_uid, target_version)
        current = self.snapshot(cohort_uid)
        number = int(current.version[1:]) + 1 if current.version.startswith("v") and current.version[1:].isdigit() else len(self._snapshots)
        rolled = LongitudinalGallerySnapshot(cohort_uid, f"v{number}", dict(target.profiles), target.encoder_fingerprint)
        self._snapshots[(cohort_uid, rolled.version)] = rolled
        self._heads[cohort_uid] = rolled.version
        return rolled


__all__ = [
    "IdentityExemplar", "IdentityProfile", "LongitudinalGallerySnapshot",
    "ExemplarProposal", "LongitudinalGalleryStore", "score_profile",
]
